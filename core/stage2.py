"""Stage 2 candidates: HypoPG builds the fixed pool. Size estimates live in stage2_size."""

import heapq
import json
import logging
import math
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import groupby, permutations, product
from typing import Any, List, Sequence

import psycopg2
import sqlparse
from sql_metadata import Parser as SqlParser
from sqlparse.tokens import Name

from core.common import half_cpu_workers, load_json as load_features
from core.stage1 import QID_MARKER, WorkloadExtractor, compact_bytes
from core.stage2_size import BTREE, SizeCalibrator, _SizeSpec


COST_RE = re.compile(r"cost=\d+(?:\.\d+)?\.\.(\d+(?:\.\d+)?)")
MAX_INDEX_COLUMNS = 4
MIN_STANDALONE_GAIN = 0.05

COLUMN_DELIMITER = ", "
MAX_INDEX_COLUMN_NUM = 4
MAX_CANDIDATE_COLUMNS = 40
_UNUSED_INDEX_PERM_CAP = 6

TABLE_CTX = {}
_CREATED_INDEXES = None
_FACTORY_LOCK = threading.Lock()

_INDEXDEF_KEYS = re.compile(
    r"(?:USING\s+\w+\s+)?\((.*)\)(?:\s+INCLUDE\s*\(.*\))?(?:\s+WHERE\s+.*)?\s*$",
    re.IGNORECASE | re.DOTALL,
)


class UniqueList(list):
    def append(self, item: Any) -> None:
        if item not in self:
            super().append(item)

    def extend(self, items: Sequence[Any]) -> None:
        for item in items:
            self.append(item)


class ExistingIndex:
    def __init__(self, schema, table, columns):
        self.__schema = schema
        self.__table = table
        self.__columns = columns
        self.__primary_key = False
        self.__index_type = ""

    def get_index_type(self):
        return self.__index_type

    def get_table(self):
        return self.__table

    def get_schema(self):
        return self.__schema

    def get_columns(self):
        return self.__columns

    def is_primary_key(self):
        return self.__primary_key

    def set_is_primary_key(self, is_primary_key: bool):
        self.__primary_key = is_primary_key


class AdvisedIndex:
    def __init__(self, tbl, cols, index_type=None):
        self.__table = tbl
        self.__columns = cols
        self.__index_type = index_type
        self.__source_index = None
        self.augmented = False
        self.benefit_positions = set()

    def set_source_index(self, source_index: ExistingIndex):
        self.__source_index = source_index

    def get_source_index(self):
        return self.__source_index

    def get_table(self):
        return self.__table

    def get_schema(self):
        return self.__table.split(".")[0]

    def get_columns(self):
        return self.__columns

    def get_columns_num(self):
        return len(self.get_columns().split(COLUMN_DELIMITER))

    def get_index_type(self):
        return self.__index_type

    def __eq__(self, other):
        if not isinstance(other, AdvisedIndex):
            return NotImplemented
        return (
            self.__table == other.__table
            and self.__columns == other.__columns
            and self.__index_type == other.__index_type
        )

    def __hash__(self):
        return hash((self.__table, self.__columns, self.__index_type))

    def __str__(self):
        return "table: %s columns: %s index_type: %s" % (
            self.__table, self.__columns, self.__index_type
        )

    def __repr__(self):
        return self.__str__()


def _singleton(cls):
    instances = {}

    def _wrap(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _wrap


@_singleton
class IndexItemFactory:
    def __init__(self):
        self.indexes = {}

    def get_index(self, tbl, cols, index_type):
        if COLUMN_DELIMITER not in cols:
            cols = cols.replace(",", COLUMN_DELIMITER)
        key = (tbl, cols, index_type)
        with _FACTORY_LOCK:
            if key not in self.indexes:
                self.indexes[key] = AdvisedIndex(tbl, cols, index_type=index_type)
            return self.indexes[key]


class QueryItem:
    def __init__(self, sql: str):
        self.__statement = sql

    def get_statement(self):
        return self.__statement


class WorkLoad:
    def __init__(self, queries: List[QueryItem]):
        self.__queries = queries

    def get_queries(self) -> List[QueryItem]:
        return self.__queries


@dataclass(eq=False)
class TableContext:
    schema: str
    table: str
    reltuples: int
    columns: List = field(default_factory=list)
    n_distincts: List = field(default_factory=list)

    @lru_cache(maxsize=None)
    def has_column(self, column):
        is_same_table = True
        if "." in column:
            if column.split(".")[0].upper() != self.table.split(".")[-1].upper():
                is_same_table = False
            column = column.split(".")[1].lower()
        return is_same_table and column in self.columns

    @lru_cache(maxsize=None)
    def get_n_distinct(self, column):
        column = column.split(".")[-1].lower()
        idx = self.columns.index(column)
        n_distinct = self.n_distincts[idx]
        if float(n_distinct) == float(0):
            return 1
        return 1 / (-n_distinct * self.reltuples) if n_distinct < 0 else 1 / n_distinct


def flatten(iterable):
    for item in iterable:
        if hasattr(item, "__iter__") and not isinstance(item, str):
            for nested in flatten(item):
                yield nested
        else:
            yield item


@lru_cache(maxsize=None)
def _get_tokens(query):
    return list(sqlparse.parse(query)[0].flatten())


@lru_cache(maxsize=None)
def has_dollar_placeholder(query):
    return any(item.ttype is Name.Placeholder for item in _get_tokens(query))


@lru_cache(maxsize=None)
def generate_placeholder_indexes(table_cxt, column):
    schema_table = "%s.%s" % (table_cxt.schema, table_cxt.table)
    return [IndexItemFactory().get_index(schema_table, column, "")]


def match_columns(column1, column2):
    return re.match(column1 + ",", column2 + ",")


def _result_rows(results, width):
    return [row for row in results if len(row) == width]


def _columns_from_indexdef(indexdef):
    if not indexdef:
        return None
    match = _INDEXDEF_KEYS.search(indexdef.strip())
    if not match:
        return None
    parts = []
    for part in match.group(1).split(","):
        token = part.strip().split()[0].strip('"') if part.strip() else ""
        if token:
            parts.append(token)
    return COLUMN_DELIMITER.join(parts) if parts else None


def prefetch_table_contexts(executor):
    schema = executor.get_schema().split(",")[0].strip()
    table_rows = _result_rows(
        executor.execute_sqls(
            [
                "SELECT c.relname, c.reltuples FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = '%s' AND c.relkind IN ('r', 'p', 'm')" % schema
            ]
        ),
        2,
    )
    stat_rows = _result_rows(
        executor.execute_sqls(
            [
                "SELECT tablename, attname, n_distinct FROM pg_stats "
                "WHERE schemaname = '%s' AND n_distinct IS NOT NULL" % schema
            ]
        ),
        3,
    )
    tables = {}
    for name, reltuples in table_rows:
        try:
            tables[str(name).lower()] = (str(name), int(float(reltuples)))
        except (TypeError, ValueError):
            continue

    columns = defaultdict(lambda: ([], []))
    for table, attname, n_distinct in stat_rows:
        names, values = columns[str(table).lower()]
        if attname not in names:
            names.append(attname)
            values.append(float(n_distinct))

    for key, (name, reltuples) in tables.items():
        if not reltuples:
            continue
        names, values = columns.get(key, ([], []))
        ctx = TableContext(schema, name, int(reltuples), names, values)
        TABLE_CTX[key] = ctx
        TABLE_CTX[name.lower()] = ctx


def get_table_context(origin_table, executor):
    key = origin_table.split(".")[-1].lower()
    cached = TABLE_CTX.get(key)
    if cached is not None:
        return cached
    if "." in origin_table:
        schema, table = origin_table.split(".", 1)
        schemas = [schema]
    else:
        table = origin_table
        schemas = [part.strip() for part in executor.get_schema().split(",") if part.strip()]
    for schema in schemas:
        info = _result_rows(
            executor.execute_sqls(
                [
                    "SELECT reltuples FROM pg_class "
                    "WHERE relname ILIKE '%s' AND relnamespace = "
                    "(SELECT oid FROM pg_namespace WHERE nspname = '%s')"
                    % (table, schema)
                ]
            ),
            1,
        )
        if not info:
            continue
        try:
            reltuples = int(float(info[0][0]))
        except (TypeError, ValueError):
            continue
        if not reltuples:
            continue
        stats = _result_rows(
            executor.execute_sqls(
                [
                    "SELECT n_distinct, attname FROM pg_stats "
                    "WHERE tablename ILIKE '%s' AND schemaname = '%s' "
                    "AND n_distinct IS NOT NULL" % (table, schema)
                ]
            ),
            2,
        )
        names, values = [], []
        for n_distinct, attname in stats:
            if attname not in names:
                names.append(attname)
                values.append(float(n_distinct))
        ctx = TableContext(schema, table, reltuples, names, values)
        TABLE_CTX[key] = ctx
        TABLE_CTX[ctx.table.lower()] = ctx
        return ctx
    return None


def fetch_created_indexes(executor):
    global _CREATED_INDEXES
    if _CREATED_INDEXES is not None:
        return _CREATED_INDEXES
    created = []
    schemas = [part.strip() for part in executor.get_schema().split(",") if part.strip()]
    for schema in schemas:
        rows = _result_rows(
            executor.execute_sqls(
                [
                    "SELECT n.nspname, c.relname, pg_get_indexdef(i.oid), p.contype "
                    "FROM pg_index x "
                    "JOIN pg_class c ON c.oid = x.indrelid "
                    "JOIN pg_class i ON i.oid = x.indexrelid "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "LEFT JOIN pg_constraint p ON i.oid = p.conindid AND p.contype = 'p' "
                    "WHERE c.relkind IN ('r', 'm', 'p') "
                    "AND i.relkind IN ('i', 'I') "
                    "AND n.nspname = '%s' "
                    "ORDER BY c.relname" % schema
                ]
            ),
            4,
        )
        for nsp, table, indexdef, pkey in rows:
            columns = _columns_from_indexdef(indexdef)
            if not columns:
                continue
            index = ExistingIndex(nsp, table, columns)
            if pkey:
                index.set_is_primary_key(True)
            created.append(index)
    _CREATED_INDEXES = created
    return created


def get_original_base_indexes(original_indexes: List[ExistingIndex]) -> List[AdvisedIndex]:
    original_base_indexes = []
    for index in original_indexes:
        if index.is_primary_key():
            continue
        table = "%s.%s" % (index.get_schema(), index.get_table())
        columns = index.get_columns().split(COLUMN_DELIMITER)
        index_type = index.get_index_type()
        for length in range(1, len(columns)):
            prefix = COLUMN_DELIMITER.join(columns[:length])
            original_base_indexes.append(IndexItemFactory().get_index(table, prefix, index_type))
        all_columns = IndexItemFactory().get_index(table, index.get_columns(), index_type)
        original_base_indexes.append(all_columns)
        all_columns.set_source_index(index)
    return original_base_indexes


def remove_unused_indexes(executor, statement, valid_indexes):
    if len(valid_indexes) > _UNUSED_INDEX_PERM_CAP:
        return valid_indexes
    least_indexes = valid_indexes
    for indexes in permutations(valid_indexes, len(valid_indexes)):
        cur_indexes, _cost = executor.hypo_check(statement, indexes)
        if len(cur_indexes) < len(least_indexes):
            least_indexes = cur_indexes
    return least_indexes


def generate_single_column_indexes(advised_indexes: List[AdvisedIndex]):
    single_column_indexes = []
    for index in advised_indexes:
        table = index.get_table()
        index_type = index.get_index_type()
        for column in index.get_columns().split(COLUMN_DELIMITER):
            single = IndexItemFactory().get_index(table, column, index_type)
            if single not in single_column_indexes:
                single_column_indexes.append(single)
    return single_column_indexes


def add_more_column_index(indexes, table, columns_info, single_col_info):
    columns, columns_index_type = columns_info
    single_column, _single_index_type = single_col_info
    current = IndexItemFactory().get_index(
        table, columns + COLUMN_DELIMITER + single_column, columns_index_type
    )
    if current not in indexes:
        indexes.append(current)


def filter_candidate_columns_by_cost(valid_indexes, statement, executor, max_candidate_columns):
    indexes = []
    for _table, index_group in groupby(valid_indexes, key=lambda x: x.get_table()):
        cost_index = []
        group = list(index_group)
        if len(group) <= max_candidate_columns:
            indexes.extend(group)
            continue
        for index in group:
            used, cost = executor.hypo_check(statement, [index])
            if used:
                heapq.heappush(cost_index, (cost if cost is not None else float("inf"), id(used[0]), used[0]))
        for _cost, _idx, index in heapq.nsmallest(max_candidate_columns, cost_index):
            indexes.append(index)
    return indexes


def set_source_indexes(indexes, source_indexes):
    for index in indexes:
        table = index.get_table()
        columns = index.get_columns()
        for source_index in source_indexes:
            if not source_index.get_source_index():
                continue
            if source_index.get_table() != table:
                continue
            if ("%s%s" % (columns, COLUMN_DELIMITER)).startswith(
                "%s%s" % (source_index.get_columns(), COLUMN_DELIMITER)
            ):
                index.set_source_index(source_index.get_source_index())


def get_valid_indexes(advised_indexes, original_base_indexes, statement, executor, **kwargs):
    single_column_indexes = generate_single_column_indexes(advised_indexes)
    valid_indexes, cost = executor.hypo_check(statement, single_column_indexes)
    valid_indexes = filter_candidate_columns_by_cost(
        valid_indexes,
        statement,
        executor,
        kwargs.get("max_candidate_columns", MAX_CANDIDATE_COLUMNS),
    )
    valid_indexes, cost = executor.hypo_check(statement, valid_indexes)
    pre_indexes = valid_indexes[:]
    need_check = False

    for column_num in range(2, MAX_INDEX_COLUMN_NUM + 1):
        for table, index_group in groupby(valid_indexes, key=lambda x: x.get_table()):
            base = [index for index in original_base_indexes if index.get_table() == table]
            for index in list(index_group) + base:
                columns = index.get_columns()
                index_type = index.get_index_type()
                if index.get_columns_num() != column_num - 1:
                    continue
                need_check = True
                for single in single_column_indexes:
                    if single.get_table() != table:
                        continue
                    single_column = single.get_columns()
                    if single_column not in columns.split(COLUMN_DELIMITER):
                        add_more_column_index(
                            valid_indexes,
                            table,
                            (columns, index_type),
                            (single_column, single.get_index_type()),
                        )
        if need_check:
            cur_indexes, cur_cost = executor.hypo_check(statement, valid_indexes)
            if cur_cost is not None and cost and cost / cur_cost < 1.05:
                set_source_indexes(pre_indexes, original_base_indexes)
                return pre_indexes
            valid_indexes = cur_indexes
            pre_indexes = valid_indexes[:]
            cost = cur_cost
            need_check = False
        else:
            break

    valid_indexes = remove_unused_indexes(executor, statement, valid_indexes)
    set_source_indexes(valid_indexes, original_base_indexes)
    return valid_indexes


def _index_sort_key(index):
    return index.get_table(), index.get_columns()


def filter_redundant_indexes_with_same_type(indexes: List[AdvisedIndex]):
    candidate_indexes = []
    for _table, table_group in groupby(
        sorted(indexes, key=lambda x: x.get_table()), key=lambda x: x.get_table()
    ):
        for _index_type, type_group in groupby(
            sorted(table_group, key=lambda x: x.get_index_type()),
            key=lambda x: x.get_index_type(),
        ):
            ordered = sorted(type_group, key=lambda x: x.get_columns())
            for i in range(len(ordered) - 1):
                if match_columns(ordered[i].get_columns(), ordered[i + 1].get_columns()):
                    continue
                candidate_indexes.append(ordered[i])
            candidate_indexes.append(ordered[-1])
    candidate_indexes.sort(key=_index_sort_key)
    return candidate_indexes


def reset_state():
    global _CREATED_INDEXES
    IndexItemFactory().indexes.clear()
    TABLE_CTX.clear()
    _CREATED_INDEXES = None
    TableContext.has_column.cache_clear()
    TableContext.get_n_distinct.cache_clear()
    generate_placeholder_indexes.cache_clear()
    has_dollar_placeholder.cache_clear()
    _get_tokens.cache_clear()

_EXTRACTOR = None
_TLS = threading.local()
_THREAD_EXECUTORS = []
_THREAD_EXECUTORS_LOCK = threading.Lock()


class PostgresExecutor:
    """PostgreSQL + hypopg session used by the candidate pipeline."""

    def __init__(self, dbname, user, password, host, port, schema="public"):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.schema = schema
        self.conn = None
        self.cur = None

    def get_schema(self):
        return self.schema

    def _connect(self):
        kwargs = dict(
            dbname=self.dbname,
            user=self.user,
            password=self.password or None,
            port=int(self.port),
            application_name="candidate-pool",
        )
        if self.host:
            kwargs["host"] = self.host
        self.conn = psycopg2.connect(**kwargs)
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        self.cur.execute("CREATE EXTENSION IF NOT EXISTS hypopg")
        self.cur.execute("SET search_path TO %s" % self.get_schema())

    def _ensure(self):
        if self.cur is None or self.cur.closed:
            self._connect()

    def _execute(self, sql):
        self._ensure()
        try:
            self.cur.execute(sql)
            if self.cur.description is None:
                return []
            return [(self.cur.statusmessage,)] + list(self.cur.fetchall())
        except Exception as exc:
            logging.warning("Found %s while executing SQL statement.", exc)
            return [("ERROR " + str(exc),)]

    def execute_sqls(self, sqls):
        results = []
        for sql in sqls:
            res = self._execute(sql)
            if res:
                results.extend(res)
        return results

    def hypo_check(self, query, indexes):
        """Reset, create an optional configuration, EXPLAIN, and always clean up."""
        if not indexes:
            return [], None
        indexes = sorted(indexes, key=lambda index: -len(index.get_columns()))
        self._ensure()
        cur = self.cur
        try:
            cur.execute("SELECT hypopg_reset()")
            created = []
            for index in indexes:
                ddl = "CREATE INDEX ON %s (%s)" % (
                    index.get_table(),
                    index.get_columns(),
                )
                cur.execute("SELECT * FROM hypopg_create_index(%s)", (ddl,))
                row = cur.fetchone()
                if row and len(row) >= 2 and str(row[0]).isdigit():
                    created.append((int(row[0]), str(row[1]), index))
            stmt = query.strip().rstrip(";")
            cur.execute("EXPLAIN %s" % stmt)
            plan = "\n".join(str(row[0]) for row in cur.fetchall())
        except Exception as exc:
            logging.warning("Found %s while executing SQL statement.", exc)
            return [], None

        cost = None
        match = COST_RE.search(plan)
        if match:
            cost = float(match.group(1))
        valid = []
        for oid, name, index in created:
            if ("<%s>" % oid) in plan or name in plan:
                if index not in valid:
                    valid.append(index)
        return valid, cost

    def close(self):
        if self.cur:
            try:
                self.cur.close()
            except Exception:
                pass
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
        self.cur = None
        self.conn = None

    @contextmanager
    def session(self):
        self._connect()
        try:
            yield
        finally:
            self.close()


def _parse_query_schema(query):
    parsed = SqlParser(query)
    tables = [table.lower() for table in parsed.tables]
    columns = []
    for position, _columns in parsed.columns_dict.items():
        if position.upper() not in ["SELECT", "INSERT", "UPDATE"]:
            columns.extend(_columns)
    flatten_columns = UniqueList()
    for column in flatten(columns):
        flatten_columns.append(column)
    return tuple(tables), tuple(flatten_columns)


def _parse_worker(args):
    """Runs in a forked process: pure SQL parsing, no database handle."""
    query, use_all_columns, want_feature = args
    parsed = None
    if use_all_columns or has_dollar_placeholder(query):
        try:
            parsed = _parse_query_schema(query)
        except (ValueError, AttributeError, KeyError) as exc:
            logging.warning("Found %s while parsing SQL statement.", exc)
    feature = None
    if want_feature and _EXTRACTOR is not None:
        try:
            feature = _EXTRACTOR.extract_expressions(query)
        except Exception as exc:
            logging.warning("Found %s while extracting predicates.", exc)
    return parsed, feature


def _advised_indexes_from_parsed(parsed, executor, n_distinct, reltuples):
    """Catalog-side half of generate_query_placeholder_indexes: same filter and order."""
    if not parsed:
        return []
    tables, flatten_columns = parsed
    indexes = []
    for table in tables:
        table_indexes = []
        table_context = get_table_context(table, executor)
        if not table_context or table_context.reltuples < reltuples:
            continue
        for column in flatten_columns:
            if (
                table_context.has_column(column)
                and table_context.get_n_distinct(column) <= n_distinct
            ):
                table_indexes.extend(
                    generate_placeholder_indexes(
                        table_context, column.split(".")[-1].lower()
                    )
                )
        indexes.extend(
            sorted(
                table_indexes,
                key=lambda x: table_context.get_n_distinct(x.get_columns()),
            )[:20]
        )
    return indexes


EQ_OPS = ("eq", "in", "isnull")
RANGE_OPS = ("range", "between")


def _columns_of(value):
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value or "").split(",")
    return [str(part).strip() for part in parts if str(part).strip()]


def _key_tuple(value):
    return tuple(part.lower() for part in _columns_of(value))


def _index_shape(index):
    table = str(index.get_table() or "")
    if "." in table:
        table = table.rsplit(".", 1)[-1]
    return table.lower(), _key_tuple(index.get_columns())


def _pk_only_key(keys, pk_cols):
    """True when a PK-only key adds no access path the PK does not already have.

    A strict leading prefix survives: it reaches rows through the same columns in
    the same order as the PK, only in a thinner index. Every other PK-only key is
    either the PK itself or a reordered subset, and a reordered subset opens a
    lookup path the PK cannot serve at all.
    """
    primary = _key_tuple(pk_cols)
    normalized = _key_tuple(keys)
    if not normalized or not primary:
        return False
    if not set(normalized) <= set(primary):
        return False
    if normalized != primary[: len(normalized)]:
        return True
    return len(normalized) == len(primary)


def _primary_key_columns(original_indexes):
    """Map table / schema.table -> primary-key columns."""
    by_table = {}
    for index in original_indexes or []:
        if not index.is_primary_key():
            continue
        keys = tuple(_columns_of(index.get_columns()))
        if not keys:
            continue
        table = str(index.get_table() or "")
        schema = str(index.get_schema() or "")
        by_table[table.lower()] = keys
        if schema:
            by_table[("%s.%s" % (schema, table)).lower()] = keys
    return by_table


def keep_existing_secondary_indexes(candidate_indexes, catalog_bases):
    """Add full non-PK catalog shapes produced by get_original_base_indexes."""
    by_shape = {_index_shape(index): index for index in candidate_indexes}
    added = 0
    for advised in catalog_bases or []:
        source = advised.get_source_index()
        if source is None:
            continue
        shape = _index_shape(advised)
        current = by_shape.get(shape)
        if current is None:
            candidate_indexes.append(advised)
            by_shape[shape] = advised
            added += 1
        elif current.get_source_index() is None:
            current.set_source_index(source)
    if added:
        print("existing_secondary : kept %d catalog indexes in pool" % added)
    return candidate_indexes


def drop_primary_key_candidates(candidate_indexes, pk_of):
    kept = []
    dropped = []
    for index in candidate_indexes:
        if _pk_only_key(
            index.get_columns(), _pk_for_table(index.get_table(), None, pk_of)
        ):
            dropped.append(
                "%s(%s)"
                % (index.get_table().split(".")[-1], index.get_columns())
            )
            continue
        kept.append(index)
    if dropped:
        print("dropped_primary_keys :", len(dropped), sorted(set(dropped)))
    return kept


def _pk_for_table(table, ctx, pk_of):
    names = [str(table or "").lower()]
    if names[0] and "." in names[0]:
        names.append(names[0].rsplit(".", 1)[-1])
    if ctx is not None:
        names.append(str(ctx.table or "").lower())
        if ctx.schema:
            names.append(("%s.%s" % (ctx.schema, ctx.table)).lower())
    for name in names:
        if name in pk_of:
            return pk_of[name]
    return ()


def _augmentation_keys(feature, table_ctx_of, max_cols, pk_of=None):
    """Bounded generic composites per (query, table).

    AutoIndex only extends a single adopted column.  Extra variants are
    cross-workload leading-key shapes, not dataset-specific recipes:

    1. equality/join by selectivity, then one range, then one sort
    2. a non-PK single join key, so a fact-table join path cannot disappear
       merely because a wider standalone composite was not adopted
    3. range-, join-, or order-leading, then the remaining query columns
    4. the first two equality columns and their reverse order

    Candidates whose keys are only PK columns (the PK, a permutation, or a
    subset) are omitted. A PK column may still lead a composite when a
    non-PK suffix adds covering, grouping, ordering, or filtering value.
    """
    pk_of = pk_of or {}
    eq_of = defaultdict(list)
    join_of = defaultdict(list)
    range_of = defaultdict(list)
    sort_of = defaultdict(list)
    group_of = defaultdict(list)

    for table, column, other_table, other_column in (
        (j[0], j[1], j[2], j[3]) for j in feature.get("joins") or []
    ):
        eq_of[table].append(column)
        eq_of[other_table].append(other_column)
        join_of[table].append(column)
        join_of[other_table].append(other_column)
    for pred in feature.get("filters") or []:
        if pred["op"] in EQ_OPS:
            eq_of[pred["table"]].append(pred["column"])
        elif pred["op"] in RANGE_OPS:
            range_of[pred["table"]].append(pred["column"])
    for item in (feature.get("groups") or []) + (feature.get("orders") or []):
        sort_of[item["table"]].append(item["column"])
    for item in feature.get("groups") or []:
        group_of[item["table"]].append(item["column"])

    keys = {}
    for table in sorted(set(eq_of) | set(range_of) | set(sort_of)):
        ctx = table_ctx_of(table)
        if ctx is None:
            continue

        def known(cols):
            out = []
            for col in cols:
                if col in ctx.columns and col not in out:
                    out.append(col)
            return out

        eq_cols = sorted(known(eq_of.get(table, [])), key=ctx.get_n_distinct)
        range_cols = known(range_of.get(table, []))[:1]
        join_cols = known(join_of.get(table, []))
        sort_cols = known(sort_of.get(table, []))[:1]
        group_cols = known(group_of.get(table, []))[: max(0, max_cols - 1)]
        pk_cols = _pk_for_table(table, ctx, pk_of)
        variants = []
        single_join_keys = {(col,) for col in join_cols}

        selective = list(eq_cols)
        for col in range_cols + sort_cols:
            if col not in selective:
                selective.append(col)
        variants.append(selective[:max_cols])
        variants.extend([[col] for col in join_cols])

        alt_leads = []
        for col in range_cols + join_cols + sort_cols:
            if col not in alt_leads:
                alt_leads.append(col)
        for lead in alt_leads:
            rest = []
            for col in eq_cols + join_cols + range_cols + sort_cols:
                if col != lead and col not in rest:
                    rest.append(col)
            variants.append(([lead] + rest)[:max_cols])
            # Join-entry + grouping, + ordering and + range are distinct covering
            # shapes. A single all-columns composite can be wider, exceed budget,
            # or lose the exact ordering benefit. Grouping keeps every column,
            # because a grouped aggregate only reads in order when the index
            # carries the whole GROUP BY list, while ordering needs one lead.
            if lead in join_cols:
                for suffix in (group_cols, sort_cols, range_cols):
                    compact = [lead] + [col for col in suffix if col != lead]
                    if len(compact) >= 2:
                        variants.append(compact[:max_cols])

        if len(eq_cols) >= 2:
            variants.append(eq_cols[:2])
            variants.append([eq_cols[1], eq_cols[0]])

        seen = []
        for raw in variants:
            key = tuple(raw)
            if (
                _pk_only_key(key, pk_cols)
                or (len(key) < 2 and key not in single_join_keys)
                or key in seen
            ):
                continue
            seen.append(key)
        if seen:
            keys[table] = seen
    return keys


def _verify_augmentations(
    candidates, statements, conn_args, min_gain, workers, max_probes=3
):
    """Keep only augmentations with standalone HypoPG gain of at least min_gain."""
    if not candidates:
        return []
    probes = {key: positions[:max_probes] for key, positions in candidates.items()}
    needed = sorted({pos for positions in probes.values() for pos in positions})

    def _baseline(pos):
        executor = _thread_executor(conn_args)
        executor._ensure()
        try:
            executor.cur.execute("EXPLAIN %s" % statements[pos].strip().rstrip(";"))
            plan = "\n".join(str(row[0]) for row in executor.cur.fetchall())
        except Exception as exc:
            logging.warning("Found %s while executing SQL statement.", exc)
            return pos, None
        match = COST_RE.search(plan)
        cost = float(match.group(1)) if match else None
        return pos, cost

    n_workers = max(1, min(int(workers or 1), len(needed) or 1))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        baselines = dict(pool.map(_baseline, needed))

    def _probe(item):
        (table, key), positions = item
        index = IndexItemFactory().get_index(table, COLUMN_DELIMITER.join(key), "")
        executor = _thread_executor(conn_args)
        for pos in positions:
            base = baselines.get(pos)
            used, cost = executor.hypo_check(statements[pos], [index])
            if used and cost and base and base / cost >= 1.0 + min_gain:
                return index
        return None

    items = list(probes.items())
    n_workers = max(1, min(int(workers or 1), len(items)))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        accepted = [index for index in pool.map(_probe, items) if index is not None]
    return accepted


def _thread_executor(conn_args):
    executor = getattr(_TLS, "executor", None)
    if executor is None:
        executor = PostgresExecutor(**conn_args)
        executor._connect()
        _TLS.executor = executor
        with _THREAD_EXECUTORS_LOCK:
            _THREAD_EXECUTORS.append(executor)
    return executor


def _close_thread_executors():
    with _THREAD_EXECUTORS_LOCK:
        for executor in _THREAD_EXECUTORS:
            try:
                executor.close()
            except Exception:
                pass
        _THREAD_EXECUTORS.clear()
    _TLS.executor = None


def generate_candidate_indexes_pipeline(
    workload,
    executor,
    n_distinct,
    reltuples,
    use_all_columns,
    workers=1,
    augment=False,
    augment_min_gain=0.05,
    max_index_cols=4,
    **kwargs,
):
    """Same candidate set as AutoIndex generate_candidate_indexes, pipelined:

    1. SQL parsing across processes (CPU bound, no database handle)
    2. one catalog prefetch, then the same per-table/column filter
    3. per-query hypo/EXPLAIN across connections
    4. union in original query order + same redundancy filter
    """
    queries = list(workload.get_queries())
    n_workers = max(1, min(int(workers or 1), len(queries) or 1))

    # Parse before any connection exists so the fork carries no libpq state.
    t0 = time.perf_counter()
    statements = [query.get_statement() for query in queries]
    args = [(stmt, use_all_columns, augment) for stmt in statements]
    if n_workers > 1 and len(statements) > 1:
        chunk = max(1, len(statements) // (n_workers * 4))
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            parsed = list(pool.map(_parse_worker, args, chunksize=chunk))
    else:
        parsed = [_parse_worker(item) for item in args]
    parsed_per_query = [item[0] for item in parsed]
    features_per_query = [item[1] for item in parsed]
    parse_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    with executor.session():
        prefetch_table_contexts(executor)
        original_indexes = fetch_created_indexes(executor)
        original_base_indexes = get_original_base_indexes(original_indexes)
        advised_per_query = []
        for parsed in parsed_per_query:
            advised = []
            for index in _advised_indexes_from_parsed(
                parsed, executor, n_distinct, reltuples
            ):
                if index not in advised:
                    advised.append(index)
            advised_per_query.append(advised)
    catalog_seconds = time.perf_counter() - t1

    conn_args = dict(
        dbname=executor.dbname,
        user=executor.user,
        password=executor.password,
        host=executor.host,
        port=executor.port,
        schema=executor.schema,
    )

    def _check(pos):
        return get_valid_indexes(
            advised_per_query[pos],
            original_base_indexes,
            queries[pos].get_statement(),
            _thread_executor(conn_args),
            **kwargs,
        )

    t2 = time.perf_counter()
    if n_workers <= 1:
        checked = [_check(pos) for pos in range(len(queries))]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            checked = list(pool.map(_check, range(len(queries))))
    for pos, valid_indexes in enumerate(checked):
        for index in valid_indexes:
            index.benefit_positions.add(pos)
    hypo_seconds = time.perf_counter() - t2

    t3 = time.perf_counter()
    all_indexes = []
    for valid_indexes in checked:
        for index in valid_indexes:
            if index not in all_indexes:
                all_indexes.append(index)
    candidate_indexes = filter_redundant_indexes_with_same_type(all_indexes)
    merge_seconds = time.perf_counter() - t3
    pk_of = _primary_key_columns(original_indexes)
    candidate_indexes = keep_existing_secondary_indexes(
        candidate_indexes, original_base_indexes
    )

    augment_seconds = 0.0
    extras = []
    if augment:
        t4 = time.perf_counter()
        covered = {_index_shape(index) for index in candidate_indexes}
        for table, pk_cols in pk_of.items():
            covered.add((table.rsplit(".", 1)[-1].lower(), _key_tuple(pk_cols)))

        wanted = {}
        for pos, feature in enumerate(features_per_query):
            if not feature:
                continue
            keys = _augmentation_keys(feature, TABLE_CTX.get, max_index_cols, pk_of)
            for table, variants in keys.items():
                ctx = TABLE_CTX.get(table)
                if ctx is None or ctx.reltuples < reltuples:
                    continue
                qualified = "%s.%s" % (ctx.schema, ctx.table)
                table_name = ctx.table.lower()
                for key in variants:
                    if (table_name, _key_tuple(key)) in covered:
                        continue
                    wanted.setdefault((qualified, key), []).append(pos)

        extras = _verify_augmentations(
            wanted, statements, conn_args, augment_min_gain, n_workers
        )
        extras = [
            index for index in extras if _index_shape(index) not in covered
        ]
        wanted_by_shape = {}
        for (qualified, key), positions in wanted.items():
            shape = (qualified.rsplit(".", 1)[-1].lower(), _key_tuple(key))
            wanted_by_shape.setdefault(shape, []).extend(positions)
        for index in extras:
            index.augmented = True
            index.benefit_positions.update(
                wanted_by_shape.get(_index_shape(index), [])
            )
        candidate_indexes = candidate_indexes + extras
        augment_seconds = time.perf_counter() - t4
        print(
            "augment : probed %d  kept %d  min_gain %.0f%%  seconds %.2f"
            % (len(wanted), len(extras), augment_min_gain * 100, augment_seconds)
        )
    _close_thread_executors()

    candidate_indexes = drop_primary_key_candidates(candidate_indexes, pk_of)

    print(
        "pipeline_seconds : parse %.2f  catalog %.2f  hypo %.2f  merge %.2f  augment %.2f  workers %d"
        % (
            parse_seconds,
            catalog_seconds,
            hypo_seconds,
            merge_seconds,
            augment_seconds,
            n_workers,
        )
    )
    return candidate_indexes


def reset_autoindex_state():
    reset_state()


def load_autoindex_workload(workload_file):
    extractor = WorkloadExtractor()
    text = open(workload_file, "r", errors="ignore").read()
    matches = list(QID_MARKER.finditer(text))
    blocks = []
    if matches:
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            blocks.append(text[match.end() : end])
    else:
        blocks.append(text)
    queries = []
    for body in blocks:
        sql = extractor._pick_sql(body)
        if sql:
            queries.append(QueryItem(sql))
    print("loaded queries :", len(queries))
    return WorkLoad(queries)

class CandidateGenerator:
    def __init__(self) -> None:
        self.features = None
        self.max_index_cols = MAX_INDEX_COLUMNS
        self.max_n_distinct = 0.01
        self.min_reltuples = 10000
        self.max_candidate_columns = 40
        self.use_all_columns = True
        self.workers = half_cpu_workers()
        self.candidates = []
        self.size_calibration_report = {}

    def load_features(self, feature_path):
        if feature_path and os.path.exists(feature_path):
            self.features = load_features(feature_path)
            print("loaded features :", feature_path)

    def generate(self, workload_file, host, port, user, password, dbname):
        global _EXTRACTOR, MAX_INDEX_COLUMN_NUM
        reset_autoindex_state()
        MAX_INDEX_COLUMN_NUM = self.max_index_cols
        print("seed heuristic : AutoIndex generate_candidate_indexes")
        print("pipeline : parse | catalog | hypo | merge | augment")
        print("workload :", workload_file)
        print("database :", dbname, "@", host)
        print("max_index_cols :", self.max_index_cols)
        print("max_n_distinct :", self.max_n_distinct)
        print("min_reltuples :", self.min_reltuples)
        print("workers :", self.workers)
        print("augment : enabled")

        _EXTRACTOR = None
        extractor = WorkloadExtractor()
        extractor.catalog.connect(host, port, user, password, dbname)
        _EXTRACTOR = extractor

        workload = load_autoindex_workload(workload_file)
        executor = PostgresExecutor(dbname, user, password, host, port, "public")
        started = time.perf_counter()
        candidates = generate_candidate_indexes_pipeline(
            workload,
            executor,
            self.max_n_distinct,
            self.min_reltuples,
            self.use_all_columns,
            workers=self.workers,
            augment=True,
            augment_min_gain=MIN_STANDALONE_GAIN,
            max_index_cols=self.max_index_cols,
            max_candidate_columns=self.max_candidate_columns,
        )
        elapsed = time.perf_counter() - started
        print("generate_seconds :", round(elapsed, 2))

        estimator = SizeCalibrator(host, port, user, password, dbname)
        estimator.connect()
        try:
            candidates = self._drop_unbuildable(candidates, estimator)
            self.candidates = self._assign_candidate_ids(candidates)
            self.size_calibration_report = self._estimate_pool_sizes(estimator)
        finally:
            estimator.close()
        print("queries :", len(workload.get_queries()))
        print("candidates :", len(self.candidates), "by_width", getattr(self, "_width_histogram", {}))
        print(
            "size_calibration :",
            self.size_calibration_report.get("method"),
            "formula",
            self.size_calibration_report.get("formula"),
        )
        return self.candidates

    def _drop_unbuildable(self, candidates, estimator):
        unbuildable = [
            index
            for index in candidates
            if estimator.max_key_bytes(
                index.get_table().split(".")[-1],
                _columns_of(index.get_columns()),
            )
            > BTREE.MAX_ITEM_BYTES
        ]
        if not unbuildable:
            return candidates
        dropped = {id(index) for index in unbuildable}
        print(
            "dropped_unbuildable :",
            len(unbuildable),
            sorted(
                "%s(%s)" % (i.get_table().split(".")[-1], i.get_columns())
                for i in unbuildable
            ),
        )
        return [index for index in candidates if id(index) not in dropped]

    def _assign_candidate_ids(self, candidates):
        id_of = {
            v: k for k, v in ((self.features or {}).get("column_ids") or {}).items()
        }
        query_features = (self.features or {}).get("query_features") or []
        candidates.sort(key=lambda index: (index.get_table(), index.get_columns()))
        items = []
        widths = {}
        for i, index in enumerate(candidates, 1):
            table = index.get_table().split(".")[-1]
            keys = _columns_of(index.get_columns())
            cids = [id_of.get(table + "." + col, col) for col in keys]
            if getattr(index, "augmented", False):
                source = "augment:verified"
            elif index.get_source_index():
                source = "existing:extend"
            else:
                source = "autoindex"
            widths[len(keys)] = widths.get(len(keys), 0) + 1
            benefit_queries = []
            for pos in sorted(index.benefit_positions):
                if pos < len(query_features) and query_features[pos].get("qid"):
                    benefit_queries.append(query_features[pos]["qid"])
            items.append(
                {
                    "id": "index%d" % i,
                    "table": table,
                    "keys": keys,
                    "cids": cids,
                    "est_size_bytes": 0,
                    "est_size": None,
                    "source": source,
                    "benefit_queries": benefit_queries,
                }
            )
        self._width_histogram = widths
        return items

    def _estimate_pool_sizes(self, estimator):
        size_specs = [
            _SizeSpec(
                source_id=item["id"],
                table=item["table"],
                keys=item["keys"],
                estimated_size=int(item.get("est_size_bytes") or 0),
            )
            for item in self.candidates
        ]
        report = estimator.calibrate_candidate_sizes(size_specs)
        calibrated_by_id = {item.id: item for item in size_specs}
        for item in self.candidates:
            calibrated = calibrated_by_id[item["id"]]
            item.update(
                {
                    "est_size_bytes": calibrated.estimated_size,
                    "est_size": compact_bytes(calibrated.estimated_size),
                }
            )
        return report

    def to_dict(self):
        out = {}
        for cand in self.candidates:
            item = {"keys": cand.get("cids") or cand["keys"]}
            if cand.get("include"):
                item["include"] = cand["include"]
            if cand.get("est_size_bytes") is not None:
                item["est_size_bytes"] = cand["est_size_bytes"]
            if cand.get("source"):
                item["source"] = cand["source"]
            item["benefit_queries"] = list(cand.get("benefit_queries") or [])
            out[cand["id"]] = item
        out["size_calibration"] = self.size_calibration_report
        return out

    def render(self):
        lines = ["CANDIDATE INDEXES"]
        if not self.candidates:
            lines.append("  ...")
            return "\n".join(lines)
        for cand in self.candidates:
            keys = cand.get("cids") or cand["keys"]
            extra = ""
            if cand.get("include"):
                extra = " INCLUDE (%s)" % ",".join(cand["include"])
            lines.append("  %s: (%s)%s" % (cand["id"], ",".join(keys), extra))
            if cand.get("est_size"):
                lines.append("       est_size = %s" % cand["est_size"])
        return "\n".join(lines)

    def merge_summary(self, summary_path):
        if os.path.exists(summary_path):
            text = open(summary_path, "r", encoding="utf-8").read()
            if "CANDIDATE INDEXES" in text:
                text = (
                    text.split("CANDIDATE INDEXES")[0].rstrip()
                    + "\n\n"
                    + self.render()
                    + "\n"
                )
            else:
                text = text.rstrip() + "\n\n" + self.render() + "\n"
        else:
            text = self.render() + "\n"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text


def generate_and_save(
    workload_file,
    feature_file,
    summary_file,
    output_file,
    host,
    port,
    user,
    password,
    dbname,
    workers=0,
):
    logging.basicConfig(level=logging.WARNING)
    gen = CandidateGenerator()
    if workers:
        gen.workers = workers
    gen.load_features(feature_file)
    gen.generate(workload_file, host, port, user, password, dbname)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    data = gen.to_dict()
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("candidates ->", output_file)
    gen.merge_summary(summary_file)
    print("summary ->", summary_file)
    return data

