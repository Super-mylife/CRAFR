"""Stage 3 physical execution: catalog identity, DDL, and real queries."""

import hashlib
import json
import re
import time
import uuid
from collections import Counter

import psycopg2
from psycopg2 import sql

from core.stage1 import WorkloadExtractor
from core.stage3 import _node_shape, configuration_identity

def load_workload_queries(path):
    extractor = WorkloadExtractor()
    extractor.load_workload(path)
    seen = {}
    queries = []
    for query in extractor.queries:
        qid = query["qid"]
        seen[qid] = seen.get(qid, 0) + 1
        label = qid if seen[qid] == 1 else "%s#%d" % (qid, seen[qid])
        queries.append({"qid": label, "sql": query["sql"], "weight": 1.0})
    return queries


def compact_plan(document, name_to_id=None):
    """Keep only plan fields needed for structural before/after comparison."""
    if isinstance(document, str):
        document = json.loads(document)
    if isinstance(document, list):
        document = document[0]
    envelope = document
    root = document.get("Plan", document)
    nodes = []
    name_to_id = name_to_id or {}

    def walk(node, depth=0, path="0"):
        children = list(node.get("Plans") or [])
        child_relations = set()
        child_relation_aliases = set()
        for position, child in enumerate(children):
            relations, relation_aliases = walk(
                child, depth + 1, "%s.%d" % (path, position),
            )
            child_relations.update(relations)
            child_relation_aliases.update(relation_aliases)
        relation = node.get("Relation Name")
        relations = set(child_relations)
        relation_aliases = set(child_relation_aliases)
        if relation:
            relations.add(relation)
            relation_aliases.add((relation, node.get("Alias") or relation))
        item = {
            "op": node.get("Node Type"),
            "depth": depth,
            "path": path,
            "rows": int(node.get("Plan Rows") or 0),
            "cost": round(float(node.get("Total Cost") or 0.0), 2),
        }
        if relation:
            item["rel"] = relation
        if node.get("Alias"):
            item["alias"] = node["Alias"]
        if node.get("Index Name"):
            raw = str(node["Index Name"])
            item["index"] = raw
            if raw in name_to_id:
                item["index_id"] = name_to_id[raw]
        if relations:
            item["relations"] = sorted(relations)
        if relation_aliases:
            item["relation_aliases"] = [list(value) for value in sorted(relation_aliases)]
        if "Actual Rows" in node:
            item["actual_rows"] = float(node["Actual Rows"])
        if "Actual Loops" in node:
            item["actual_loops"] = float(node["Actual Loops"])
        if "Actual Total Time" in node:
            item["actual_total_ms"] = float(node["Actual Total Time"])
        if "Rows Removed by Filter" in node:
            item["rows_removed_by_filter"] = float(node["Rows Removed by Filter"])
        buffers = {
            field: node[field]
            for field in (
                "Shared Hit Blocks", "Shared Read Blocks", "Shared Dirtied Blocks",
                "Shared Written Blocks", "Local Hit Blocks", "Local Read Blocks",
                "Temp Read Blocks", "Temp Written Blocks",
            ) if field in node
        }
        if buffers:
            item["buffers"] = buffers
        workers = int(node.get("Workers Planned") or 0)
        if workers:
            item["workers"] = workers
        if node.get("Parallel Aware"):
            item["parallel_aware"] = True
        if node.get("Join Type"):
            item["join"] = node["Join Type"]
        if node.get("Parent Relationship"):
            item["parent"] = node["Parent Relationship"]
        if node.get("Filter"):
            item["filter"] = node["Filter"]
        if node.get("Index Cond"):
            item["index_cond"] = node["Index Cond"]
        if node.get("Recheck Cond"):
            item["recheck_cond"] = node["Recheck Cond"]
        if node.get("Hash Cond") or node.get("Merge Cond") or node.get("Join Filter"):
            item["join_cond"] = " AND ".join(
                "(%s)" % node[field]
                for field in ("Hash Cond", "Merge Cond", "Join Filter")
                if node.get(field)
            )
        if node.get("Group Key"):
            item["group"] = node["Group Key"]
        if node.get("Sort Key"):
            item["sort"] = node["Sort Key"]
        nodes.append(item)
        return relations, relation_aliases

    walk(root)
    nodes.sort(key=lambda item: tuple(int(part) for part in item["path"].split(".")))
    operators = Counter(item["op"] for item in nodes)
    signature_items = [_node_shape(item) for item in nodes]
    result = {
        "root": root.get("Node Type"),
        "operators": dict(sorted(operators.items())),
        "workers": sum(item.get("workers", 0) for item in nodes),
        "nodes": nodes,
        "signature": json.dumps(signature_items, separators=(",", ":")),
    }
    if root.get("Actual Total Time") is not None:
        result["actual_total_ms"] = float(root["Actual Total Time"])
    if isinstance(envelope, dict) and "Planning Time" in envelope:
        result["planning_time_ms"] = float(envelope["Planning Time"])
    result["indexes"] = sorted({node.get("index_id") or node.get("index")
                                for node in nodes
                                if node.get("index_id") or node.get("index")})
    return result


def _is_read_only_sql(statement):
    """Conservative guard for EXPLAIN ANALYZE execution."""
    text = re.sub(r"/\*.*?\*/|--[^\n]*", " ", str(statement), flags=re.S).strip()
    if not re.match(r"^(select|with)\b", text, re.I):
        return False
    # Reject stacked statements even if each starts as a SELECT. A later SET
    # could otherwise undo the read-only/session limits of the collector.
    if ";" in re.sub(r"'(?:''|[^'])*'", "''", text.rstrip(";")):
        return False
    return not re.search(
        r"\b(insert|update|delete|merge|create|alter|drop|truncate|copy|call|do|"
        r"vacuum|refresh|grant|revoke)\b", text, re.I,
    )


class PhysicalEvaluator:
    def __init__(self, host, port, user, password, dbname, queries, specs):
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        self.dbname = dbname
        self.queries = queries
        self.specs = specs
        self._run_suffix = uuid.uuid4().hex[:10]
        self._physical_names = {
            index_id: "stage3_%s_%d" % (self._run_suffix, offset)
            for offset, index_id in enumerate(specs)
        }

    def _connect(self):
        kwargs = {
            "dbname": self.dbname,
            "user": self.user,
            "port": self.port,
            "application_name": "stage3-index-combination",
        }
        if self.host:
            kwargs["host"] = self.host
        if self.password:
            kwargs["password"] = self.password
        conn = psycopg2.connect(**kwargs)
        conn.autocommit = True
        return conn


    def _query(self, query_id):
        return next((item for item in self.queries if item["qid"] == query_id), None)

    def query_dependencies(self):
        """Resolve base relations using PostgreSQL's rewrite dependencies, without ANALYZE."""
        conn = self._connect()
        conn.autocommit = False
        cursor = conn.cursor()
        result = {}
        try:
            for query in self.queries:
                cursor.execute("SAVEPOINT stage3_dependencies")
                try:
                    self._ddl_timeout(cursor)
                    statement = query["sql"].strip().rstrip(";")
                    if not _is_read_only_sql(statement):
                        result[query["qid"]] = None
                        continue
                    cursor.execute(sql.SQL("CREATE TEMP VIEW stage3_dependencies AS {}")
                                   .format(sql.SQL(statement)))
                    cursor.execute("""
                        WITH RECURSIVE relations(oid) AS (
                            SELECT 'pg_temp.stage3_dependencies'::regclass::oid
                            UNION
                            SELECT d.refobjid FROM relations x
                            JOIN pg_rewrite r ON r.ev_class = x.oid
                            JOIN pg_depend d ON d.classid = 'pg_rewrite'::regclass
                                AND d.objid = r.oid AND d.refclassid = 'pg_class'::regclass
                        )
                        SELECT n.nspname, c.relname, c.relkind,
                            EXISTS (SELECT 1 FROM pg_rewrite r JOIN pg_depend d
                                ON d.classid = 'pg_rewrite'::regclass AND d.objid = r.oid
                                WHERE r.ev_class = c.oid
                                AND d.refclassid = 'pg_proc'::regclass)
                            OR EXISTS (SELECT 1 FROM pg_inherits
                                       WHERE inhrelid = c.oid OR inhparent = c.oid)
                        FROM relations x JOIN pg_class c ON c.oid = x.oid
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                    """)
                    relations = cursor.fetchall()
                    if any(kind not in {"r", "p", "m", "v"} or functions
                           for _, _, kind, functions in relations):
                        result[query["qid"]] = None
                    else:
                        result[query["qid"]] = {
                            name for schema, table, kind, _ in relations if kind != "v"
                            for name in (table, schema + "." + table)}
                except (psycopg2.Error, TimeoutError):
                    result[query["qid"]] = None
                finally:
                    cursor.execute("ROLLBACK TO SAVEPOINT stage3_dependencies")
                    cursor.execute("RELEASE SAVEPOINT stage3_dependencies")
        finally:
            conn.rollback()
            cursor.close()
            conn.close()
        return result

    def _physical_ddl(self, index_id):
        spec = self.specs[index_id]
        ddl = sql.SQL("CREATE INDEX {} ON {} ({})").format(
            sql.Identifier(self._physical_names[index_id]),
            sql.Identifier(spec.table),
            sql.SQL(", ").join(sql.Identifier(key) for key in spec.keys),
        )
        if spec.include:
            ddl += sql.SQL(" INCLUDE ({})").format(
                sql.SQL(", ").join(sql.Identifier(column) for column in spec.include)
            )
        return ddl

    def _apply_physical_delta(self, cursor, current, target):
        current, target = set(current), set(target)
        for index_id in sorted(current - target):
            self._ddl_timeout(cursor)
            cursor.execute(sql.SQL("DROP INDEX {}").format(
                sql.Identifier(self._physical_names[index_id])
            ))
        for index_id in sorted(target - current):
            self._ddl_timeout(cursor)
            cursor.execute(self._physical_ddl(index_id))

    def _timeout(self, limit_ms):
        remaining = getattr(self, "deadline", float("inf")) - time.monotonic()
        if remaining <= 0:
            return -1
        limits = []
        if limit_ms is not None and int(limit_ms) > 0:
            limits.append(int(limit_ms))
        if remaining != float("inf"):
            limits.append(max(1, int(remaining * 1000)))
        # PostgreSQL defines statement_timeout=0 as disabled.
        return min(limits) if limits else 0

    def _ddl_timeout(self, cursor):
        remaining = getattr(self, "deadline", float("inf")) - time.monotonic()
        cap_ms = int(getattr(self, "ddl_timeout_ms", 15 * 60 * 1000))
        limits = []
        if cap_ms > 0:
            limits.append(cap_ms)
        if remaining != float("inf"):
            limits.append(max(0, int(remaining * 1000)))
        timeout = min(limits) if limits else 0
        if remaining <= 0:
            raise TimeoutError("Tuning time budget exhausted before DDL")
        cursor.execute("SET LOCAL statement_timeout = %s", (timeout,))


    def _catalog_index_defs(self, cursor):
        cursor.execute("""
            SELECT pg_get_indexdef(i.indexrelid)
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY n.nspname, c.relname
        """)
        return [row[0] for row in cursor.fetchall()]

    def _set_baseline_identity(self, definitions):
        self.baseline_catalog_indexes = list(definitions)
        self.baseline_configuration_identity = hashlib.sha256(
            json.dumps(definitions, sort_keys=True).encode()
        ).hexdigest()[:16]

    def capture_baseline_configuration(self):
        """Record B's catalog identity before INIT, without planning/executing SQL."""
        conn = self._connect()
        cursor = conn.cursor()
        try:
            self._set_baseline_identity(self._catalog_index_defs(cursor))
            return {"configuration_identity": self.baseline_configuration_identity,
                    "existing_indexes": list(self.baseline_catalog_indexes)}
        finally:
            cursor.close()
            conn.close()


    def _actual_index_bytes_on_cursor(self, cursor, configuration):
        if not configuration:
            return 0
        cursor.execute(
            "SELECT COALESCE(sum(pg_relation_size(c.oid)), 0) "
            "FROM pg_class c WHERE c.relname = ANY(%s)",
            ([self._physical_names[index_id] for index_id in configuration],),
        )
        return int(cursor.fetchone()[0])

    def materialize(self, configuration, storage_budget_bytes=None):
        """Build the validated seed only; leave failed selections for later diagnosis."""
        configuration = list(dict.fromkeys(configuration))
        if (storage_budget_bytes is not None
                and sum(self.specs[key].size_bytes for key in configuration) > storage_budget_bytes):
            raise ValueError("Seed exceeds estimated storage budget")
        started = time.monotonic()
        conn = self._connect()
        conn.autocommit = False
        cursor = conn.cursor()
        built, skipped, index_sizes = [], [], {}
        try:
            definitions = self._catalog_index_defs(cursor)
            if not getattr(self, "baseline_configuration_identity", None):
                self._set_baseline_identity(definitions)
            actual_bytes = 0
            for index_id in configuration:
                cursor.execute("SAVEPOINT stage3_initial_index")
                try:
                    self._ddl_timeout(cursor)
                    cursor.execute(self._physical_ddl(index_id))
                    trial_bytes = self._actual_index_bytes_on_cursor(cursor, built + [index_id])
                    if storage_budget_bytes is not None and trial_bytes > storage_budget_bytes:
                        cursor.execute("ROLLBACK TO SAVEPOINT stage3_initial_index")
                        skipped.append({"index_id": index_id,
                                        "reason_code": "ACTUAL_BUDGET_EXCEEDED",
                                        "actual_bytes": trial_bytes - actual_bytes})
                    else:
                        built.append(index_id)
                        index_sizes[index_id] = trial_bytes - actual_bytes
                        actual_bytes = trial_bytes
                except (TimeoutError, psycopg2.Error) as exc:
                    cursor.execute("ROLLBACK TO SAVEPOINT stage3_initial_index")
                    skipped.append({
                        "index_id": index_id,
                        "reason_code": (
                            "TIME_BUDGET_EXHAUSTED"
                            if isinstance(exc, TimeoutError) else "BUILD_ERROR"
                        ),
                        "reason": exc.__class__.__name__,
                    })
                finally:
                    cursor.execute("RELEASE SAVEPOINT stage3_initial_index")
            conn.commit()
            return {
                "configuration": built,
                "skipped_indexes": skipped,
                "actual_index_bytes": actual_bytes,
                "index_sizes": index_sizes,
                "ddl_ms": round((time.monotonic() - started) * 1000, 3),
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    def _measure_one(self, cursor, configuration, query_id, role, timeout_ms):
        identity = configuration_identity(configuration)
        query = self._query(query_id)
        base = {
            "configuration_identity": identity,
            "configuration_role": role,
            "query_id": query_id,
            "timeout_ms": max(0, int(timeout_ms or 0)),
            "source": "DIRECT",
        }
        if query is None:
            return {**base, "status": "UNKNOWN", "reason": "QUERY_NOT_FOUND",
                    "actual_total_ms": None}
        statement = query["sql"].strip().rstrip(";")
        if not _is_read_only_sql(statement):
            return {**base, "status": "UNKNOWN", "reason": "QUERY_NOT_READ_ONLY",
                    "actual_total_ms": None}
        started = time.monotonic()
        cursor.execute("SAVEPOINT stage3_measure")
        try:
            cursor.execute("SET LOCAL statement_timeout = %s", (
                max(0, int(timeout_ms or 0)),
            ))
            cursor.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + statement)
            raw = cursor.fetchone()[0]
            plan = compact_plan(raw, name_to_id={name: index_id for index_id, name
                                                  in self._physical_names.items()})
            actual_total_ms = plan.get("actual_total_ms")
            if actual_total_ms is None:
                actual_total_ms = 0.0
            return {
                **base,
                "status": "OK",
                "actual_total_ms": actual_total_ms,
                "planning_time_ms": plan.get("planning_time_ms"),
                "waited_ms": round((time.monotonic() - started) * 1000, 3),
                "plan_summary": plan,
                "raw_plan": raw,
            }
        except psycopg2.errors.QueryCanceled:
            self._recover_measure(cursor)
            return {**base, "status": "TIMEOUT", "actual_total_ms": None,
                    "waited_ms": round((time.monotonic() - started) * 1000, 3),
                    "reason": "STATEMENT_TIMEOUT"}
        except psycopg2.Error as exc:
            self._recover_measure(cursor)
            return {**base, "status": "ENVIRONMENT_ERROR", "actual_total_ms": None,
                    "waited_ms": round((time.monotonic() - started) * 1000, 3),
                    "reason": exc.__class__.__name__, "error": str(exc)[:240]}
        finally:
            self._finish_measure(cursor)

    def _recover_measure(self, cursor):
        try:
            cursor.execute("ROLLBACK TO SAVEPOINT stage3_measure")
        except psycopg2.Error:
            pass
        try:
            cursor.execute("SET LOCAL statement_timeout = 0")
        except psycopg2.Error:
            pass

    def _finish_measure(self, cursor):
        try:
            cursor.execute("SET LOCAL statement_timeout = 0")
        except psycopg2.Error:
            pass
        try:
            cursor.execute("RELEASE SAVEPOINT stage3_measure")
        except psycopg2.Error:
            pass

    def _measure_many(self, cursor, configuration, query_ids, role, timeout_ms):
        observations = []
        for query_id in query_ids:
            limit = timeout_ms
            if isinstance(timeout_ms, dict):
                limit = timeout_ms.get(query_id)
                if limit is None:
                    limit = timeout_ms.get("__default__")
            timeout = self._timeout(limit)
            if timeout < 0:
                break
            observations.append(self._measure_one(
                cursor, configuration, query_id, role, timeout))
        return observations

    def measure(self, configuration, query_ids, timeout_ms):
        if not query_ids:
            return []
        conn = self._connect()
        conn.autocommit = False
        cursor = conn.cursor()
        try:
            measured = self._measure_many(
                cursor, configuration, query_ids, "current", timeout_ms,
            )
            conn.rollback()
            return measured
        finally:
            cursor.close()
            conn.close()


    def begin_trial(self, current_configuration, trial_configuration, query_ids,
                    timeout_ms, storage_budget_bytes=None):
        """Measure an uncommitted physical patch on the same session that applied it."""
        started = time.monotonic()
        conn = self._connect()
        conn.autocommit = False
        cursor = conn.cursor()
        try:
            cursor.execute("SAVEPOINT stage3_trial_base")
            self._apply_physical_delta(cursor, current_configuration, trial_configuration)
            ddl_ms = round((time.monotonic() - started) * 1000, 3)
            cursor.execute(
                "SELECT COALESCE(sum(pg_relation_size(c.oid)), 0) "
                "FROM pg_class c WHERE c.relname = ANY(%s)",
                ([self._physical_names[index_id] for index_id in trial_configuration],),
            )
            actual_index_bytes = int(cursor.fetchone()[0])
            observations, stopped_reason = [], None
            if storage_budget_bytes is not None and actual_index_bytes > storage_budget_bytes:
                stopped_reason = "STORAGE_BUDGET_EXCEEDED"
            for query_id in ([] if stopped_reason else query_ids):
                timeout = self._timeout(timeout_ms)
                if timeout < 0:
                    stopped_reason = "INCONCLUSIVE"
                    break
                observation = self._measure_one(
                    cursor, trial_configuration, query_id, "trial", timeout)
                observations.append(observation)
                # A statement timeout is recoverable at the per-query savepoint.
                # Keep its evidence and measure the rest; acceptance stays inconclusive.
                if observation.get("status") == "TIMEOUT":
                    continue
                # Environment failures may leave the session unusable.
                stopped_reason = "INCONCLUSIVE" if observation.get("status") != "OK" else None
                if stopped_reason:
                    break
            if not stopped_reason and any(row.get("status") != "OK" for row in observations):
                stopped_reason = "INCONCLUSIVE"
            index_sizes = self._index_sizes_on_cursor(cursor, trial_configuration)
            return {"connection": conn, "cursor": cursor,
                    "observations": observations, "ddl_ms": ddl_ms,
                    "stopped_reason": stopped_reason,
                    "actual_index_bytes": actual_index_bytes,
                    "index_sizes": index_sizes,
                    "storage_budget_bytes": storage_budget_bytes,
                    "configuration": list(trial_configuration)}
        except BaseException:
            conn.rollback()
            cursor.close()
            conn.close()
            raise

    def finish_trial(self, handle, accept):
        if handle.get("connection") is None:
            return
        try:
            handle["connection"].commit() if accept else handle["connection"].rollback()
        finally:
            handle["cursor"].close()
            handle["connection"].close()


    def _index_sizes_on_cursor(self, cursor, configuration):
        if not configuration:
            return {}
        cursor.execute("SELECT c.relname, pg_relation_size(c.oid) FROM pg_class c "
                       "WHERE c.relname = ANY(%s)",
                       ([self._physical_names[key] for key in configuration],))
        ids = {self._physical_names[key]: key for key in configuration}
        return {ids[name]: int(size) for name, size in cursor.fetchall()}

