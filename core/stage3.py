"""Stage 3 model: candidate specifications, evidence, and access opportunities."""

import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

MAX_FOCUS_PLAN_NODES = 24
MAX_PLAN_TEXT_CHARS = 160
_PLAN_NODE_FIELDS = (
    "op", "path", "rel", "alias", "index_id", "index", "index_cond",
    "rows", "actual_rows", "actual_loops", "join", "join_cond",
    "parent", "actual_total_ms", "rows_removed_by_filter",
)
_OPTIONAL_PLAN_NODE_FIELDS = (
    "buffers", "filter", "recheck_cond", "sort", "group",
)


def _truncate_plan_value(value, limit=MAX_PLAN_TEXT_CHARS):
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit - 1] + "…"
    if isinstance(value, list):
        return [_truncate_plan_value(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: _truncate_plan_value(item, limit)
                for key, item in value.items() if item not in (None, "", [], {}, 0)}
    return value


def _path_parts(path):
    try:
        return tuple(int(part) for part in str(path).split("."))
    except ValueError:
        return (10 ** 9, str(path))


def _ancestor_paths(path):
    parts = str(path).split(".")
    return [".".join(parts[:offset]) for offset in range(1, len(parts) + 1)]


def _access_key(node):
    return node.get("rel"), node.get("alias") or node.get("rel")


def _plan_node_score(node, preferred_accesses=()):
    score = 0
    if _access_key(node) in set(preferred_accesses):
        score += 80
    if node.get("index_id") or node.get("index"):
        score += 100
    if node.get("index_cond") or node.get("recheck_cond"):
        score += 50
    if node.get("join") or node.get("join_cond"):
        score += 30
    if node.get("rel"):
        score += 20
    if float(node.get("actual_loops") or 0) > 1:
        score += 15
    if node.get("filter"):
        score += 10
    return score


def _scan_work_ms(node):
    """Inclusive scan work across loops, not additive query wall time."""
    if not node.get("op", "").endswith("Scan"):
        return None
    elapsed, loops = node.get("actual_total_ms"), node.get("actual_loops")
    if elapsed is None or loops is None:
        return None
    work = float(elapsed) * float(loops)
    if float(elapsed) < 0 or float(loops) < 0 or not math.isfinite(work):
        return None
    return work


def _plan_node_priority(node, preferred_accesses=()):
    return (-(_scan_work_ms(node) or 0),
            -_plan_node_score(node, preferred_accesses),
            _path_parts(node.get("path")))


def focus_accesses(*observations, limit=8):
    """Share the most consequential relation/alias accesses across both plan sides."""
    ranked = []
    for observation in observations:
        for node in ((observation or {}).get("plan_summary") or {}).get("nodes") or []:
            if node.get("rel"):
                ranked.append((_plan_node_priority(node), _access_key(node)))
    result = []
    for _priority, key in sorted(ranked, key=lambda item: item[0]):
        if key not in result:
            result.append(key)
        if len(result) == limit:
            break
    return result


def _bounded_plan_nodes(plan_summary, limit=MAX_FOCUS_PLAN_NODES,
                        preferred_accesses=()):
    """Keep useful access nodes and their join ancestry within a prompt-only bound."""
    nodes = list((plan_summary or {}).get("nodes") or [])
    if not nodes or limit <= 0:
        return [], bool(nodes)
    by_path = {str(node.get("path")): node for node in nodes if node.get("path") is not None}
    if len(nodes) <= limit:
        selected_paths = set(by_path)
    else:
        ranked = sorted(
            nodes,
            key=lambda node: _plan_node_priority(node, preferred_accesses),
        )
        selected_paths = {"0"} if "0" in by_path else set()
        # Reserve the two busiest scans before ancestry can consume the bound.
        # Their immediate parents keep join/subplan context even in very deep plans.
        hot = [node for node in ranked if (_scan_work_ms(node) or 0) > 0][:2]
        for paths in ([str(node.get("path")) for node in hot],
                      [str(node.get("path")).rsplit(".", 1)[0] for node in hot]):
            for path in paths:
                if path in by_path and len(selected_paths) < limit:
                    selected_paths.add(path)
        for node in ranked:
            chain = [path for path in _ancestor_paths(node.get("path"))
                     if path in by_path and path not in selected_paths]
            if len(selected_paths) + len(chain) > limit:
                continue
            selected_paths.update(chain)
            if len(selected_paths) == limit:
                break
        if not selected_paths and "0" in by_path:
            selected_paths.add("0")
    result = []
    for path in sorted(selected_paths, key=_path_parts):
        node = by_path[path]
        item = {key: _truncate_plan_value(node[key]) for key in _PLAN_NODE_FIELDS
                if key in node and node[key] not in (None, "", [], {})}
        work = _scan_work_ms(node)
        if work is not None:
            item["scan_work_ms"] = round(work, 3)
        for key in _OPTIONAL_PLAN_NODE_FIELDS:
            value = node.get(key)
            if value not in (None, "", [], {}) and not (
                    key == "buffers" and not any(value.values())):
                item[key] = _truncate_plan_value(value)
        result.append(item)
    return result, len(selected_paths) < len(nodes)



def configuration_identity(configuration):
    ordered = sorted(configuration)
    return hashlib.sha256(json.dumps(ordered).encode()).hexdigest()[:16]


def _matches_role(item, role):
    if role != "configuration":
        return role is None or item.get("configuration_role") == role
    observed_role = item.get("configuration_role")
    if observed_role == "trial":
        return item.get("committed") is True
    return observed_role in {"current", "configuration"} and item.get("committed", True)


class EvidenceLedger:
    """Version observations once per completed trial or observation task."""

    def __init__(self):
        self.version = 0
        self._sequence = 0
        self.observations = []
        self.feedback = []
        self.candidate_coverage = {}
        self.query_coverage = {}

    def mark_coverage(self, counter, candidate_ids=(), query_ids=()):
        for coverage, ids in ((self.candidate_coverage, candidate_ids),
                              (self.query_coverage, query_ids)):
            for key in set(ids):
                counts = coverage.setdefault(key, {name: 0 for name in (
                    "seen_count", "partitioned_count", "tested_count", "accepted_count")})
                counts[counter] += 1

    def coverage_summary(self, candidate_ids=(), query_ids=()):
        result = {}
        for name, coverage, ids in (("candidates", self.candidate_coverage, candidate_ids),
                                    ("queries", self.query_coverage, query_ids)):
            result[name] = {"total": len(ids), **{
                counter: sum(item.get(counter, 0) > 0 for item in coverage.values())
                for counter in ("seen_count", "partitioned_count", "tested_count", "accepted_count")}}
        return result

    def record_feedback(self, record):
        self.version += 1
        record["feedback_id"] = "feedback_%d" % (len(self.feedback) + 1)
        record["feedback_evidence_version"] = self.version
        self.feedback.append(record)

    def add(self, observations):
        observations = list(observations or [])
        if not observations:
            return []
        self.version += 1
        added = []
        for observation in observations:
            self._sequence += 1
            item = dict(observation)
            item["evidence_id"] = "obs_%d" % self._sequence
            item["evidence_version"] = self.version
            self.observations.append(item)
            added.append(item)
        return added

    def latest(self, configuration, query_id, role=None):
        identity = configuration_identity(configuration)
        return next((item for item in reversed(self.observations)
                     if item.get("configuration_identity") == identity
                     and item.get("query_id") == query_id
                     and _matches_role(item, role)), None)

    def for_configuration(self, configuration, role=None):
        identity = configuration_identity(configuration)
        latest = {}
        for item in self.observations:
            if (item.get("configuration_identity") == identity
                    and _matches_role(item, role)):
                latest[item.get("query_id")] = item
        return latest


def summarize_observation(observation, include_plan_nodes=False,
                          preferred_accesses=()):
    """Keep factual fields; attach bounded compact nodes only for focus evidence."""
    result = {key: observation[key] for key in (
        "evidence_id", "evidence_version", "configuration_identity",
        "configuration_role", "query_id",
        "state_version", "committed", "reused_from_configuration_identity",
        "status", "actual_total_ms", "planning_time_ms", "timeout_ms", "waited_ms",
        "source", "reason", "error", "plan_summary",
    ) if key in observation}
    if "plan_summary" in result:
        result["plan_summary"] = {key: value for key, value in result["plan_summary"].items()
                                  if key not in {"signature", "nodes"}}
        signature = observation['plan_summary'].get('signature')
        if signature:
            result['plan_summary']['structure_id'] = hashlib.sha256(signature.encode()).hexdigest()[:16]
    if include_plan_nodes:
        nodes, truncated = _bounded_plan_nodes(
            observation.get("plan_summary"),
            preferred_accesses=preferred_accesses,
        )
        result["plan_nodes"] = nodes
        result["plan_truncated"] = truncated
    return result


def compact_query_status(pair, include_plan=False):
    """Keep every query's measured current facts without the raw observation objects."""
    current = pair.get("current") or {}
    row = {"query_id": pair.get("query_id") or current.get("query_id"),
           "current_status": current.get("status", "UNKNOWN"),
           "current_ms": current.get("actual_total_ms")}
    if current.get("status") != "OK":
        row["current_timeout_ms"] = current.get("timeout_ms")
        row["current_error"] = current.get("error") or current.get("reason")
    if include_plan:
        plan = current.get("plan_summary") or {}
        row.update(current_root=plan.get("root"), current_indexes=plan.get("indexes", []))
        if current.get("evidence_id"):
            row["current_evidence_id"] = current["evidence_id"]
    return row


def compact_query_statuses(pairs, include_plan=False):
    return [compact_query_status(pair, include_plan=include_plan) for pair in pairs or []]


def trial_feedback(trial_id, mode, state_version, move, outcome, reason_code,
                   target_results, collateral_results, unmeasured_query_ids,
                   spent_ms):
    return {
        "trial_id": trial_id,
        "mode": mode,
        "base_state_version": state_version,
        "add": list(move.get("add", [])),
        "remove": list(move.get("remove", [])),
        "outcome": outcome,
        "reason_code": reason_code,
        "target_results": target_results,
        "collateral_results": collateral_results,
        "unmeasured_query_ids": list(unmeasured_query_ids),
        "spent_ms": dict(spent_ms),
    }

MB = 1024 * 1024
MAX_ADDITIONS_PER_ROUND = 2
MAX_REMOVALS_PER_ROUND = 2
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexSpec:
    table: str
    keys: tuple
    include: tuple
    size_bytes: int
    benefit_queries: tuple

    def prompt_dict(self, focus_qids):
        item = {
            "table": self.table,
            "keys": list(self.keys),
            "size_bytes": self.size_bytes,
        }
        if self.include:
            item["include"] = list(self.include)
        relevant = [qid for qid in self.benefit_queries if qid in set(focus_qids)]
        if relevant:
            item["relevant_query_ids"] = relevant
        return item


@dataclass
class SearchState:
    version: int
    configuration: list
    move: dict = field(default_factory=dict)
    actual_index_bytes: int = 0


class ProposalError(ValueError):
    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


class LLMCallBudgetExhausted(RuntimeError):
    """Stop search before spending the calls needed to review a physical trial."""


def build_index_specs(candidate_pool, features):
    column_ids = features.get("column_ids") or {}
    feature_columns = features.get("columns") or {}
    existing_indexes = features.get("existing_indexes") or []
    specs = {}
    for index_id, candidate in candidate_pool.items():
        if not index_id.startswith("index") or not isinstance(candidate, dict):
            continue
        table = candidate.get("table")
        raw_keys = list(candidate.get("keys") or [])
        for key in raw_keys:
            if table:
                break
            qualified = column_ids.get(key)
            if qualified:
                table = qualified.split(".", 1)[0]
                break
            if "." in key:
                table = key.split(".", 1)[0]
                break
            matches = [name.split(".", 1)[0] for name in feature_columns
                       if name.endswith("." + key)]
            matches += [item.get("table") for item in existing_indexes
                        if key in (item.get("keys") or [])]
            matches = list(dict.fromkeys(name for name in matches if name))
            if len(matches) == 1:
                table = matches[0]
                break
        keys = []
        for key in raw_keys:
            qualified = column_ids.get(key, key)
            if "." in qualified:
                key_table, column = qualified.split(".", 1)
                if table is not None and table != key_table:
                    raise ValueError("%s contains columns from multiple tables" % index_id)
                table = key_table
                keys.append(column)
            else:
                keys.append(qualified)
        if not table or not keys:
            raise ValueError("%s has no valid keys" % index_id)
        size_bytes = int(candidate.get("est_size_bytes") or 0)
        if size_bytes <= 0:
            raise ValueError("%s has no estimated size" % index_id)
        if "benefit_queries" not in candidate:
            raise ValueError("%s has no benefit_queries; regenerate Stage 2 candidates" % index_id)
        benefit_queries = list(candidate.get("benefit_queries") or ())
        specs[index_id] = IndexSpec(
            table, tuple(keys),
            tuple(column_ids.get(column, column).split(".")[-1]
                  for column in candidate.get("include") or ()),
            size_bytes, tuple(benefit_queries),
        )
    return specs


def _node_shape(node):
    return (
        node.get("op"), node.get("path"), tuple(node.get("relations") or ()),
        node.get("rel"), node.get("alias"), node.get("index_id") or node.get("index"),
        node.get("workers", 0), node.get("parent"), node.get("join"),
        node.get("filter"), node.get("index_cond"), node.get("join_cond"),
        node.get("recheck_cond"), tuple(node.get("group") or ()),
        tuple(node.get("sort") or ()),
    )


def _configuration_size(configuration, specs):
    return sum(specs[index_id].size_bytes for index_id in configuration)


def _pack_by_priority(priority, specs, budget_bytes):
    """Keep items in priority order; skip any that do not fit the remaining estimate."""
    packed = []
    skipped = []
    used = 0
    for index_id in priority:
        size = specs[index_id].size_bytes
        if used + size <= budget_bytes:
            packed.append(index_id)
            used += size
        else:
            skipped.append({
                "index_id": index_id,
                "reason_code": "ESTIMATED_BUDGET",
                "estimated_mb": round(size / MB, 6),
                "remaining_mb": round(max(0, budget_bytes - used) / MB, 6),
            })
    return packed, skipped, used


def _ids(value, allowed, name):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(name + " must be a list of IDs")
    if not set(value) <= set(allowed):
        raise ProposalError(name + " contains out-of-scope IDs")
    return list(dict.fromkeys(value))


def _persist_state(state):
    return {
        "state_version": state.version,
        "configuration": list(state.configuration),
        "configuration_identity": configuration_identity(state.configuration),
        "actual_index_bytes": state.actual_index_bytes,
        "move": state.move,
    }


def _finite_ms(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value



def storage_budget(features):
    """Byte budget recorded by Stage 1 in the feature file."""
    storage = features.get("storage") or {}
    constraint_mb = storage.get("storage_constraint_mb")
    if constraint_mb is None:
        raise ValueError("workload features have no storage.storage_constraint_mb")
    return max(0, int(float(constraint_mb) * MB))

def table_matches(left, right):
    # Unqualified feature names may match resolved schema-qualified dependencies.
    return left == right or ("." not in left and right.endswith("." + left)) or (
        "." not in right and left.endswith("." + right))


class CandidateAccess:
    def __init__(self, specs, query_dependencies, ledger, view_limit=64):
        self.specs = specs
        self.query_dependencies = query_dependencies
        self.ledger = ledger
        self.view_limit = max(1, int(view_limit))
        self.families = {key: "%s:%s" % (spec.table, spec.keys[0])
                         for key, spec in specs.items()}
        self.tables = sorted({spec.table for spec in specs.values()} | {
            table for values in query_dependencies.values() for table in values or ()})
        self.candidate_families = sorted(set(self.families.values()))
        self._offsets = defaultdict(int)

    def structural_ids(self, query_ids=(), tables=(), families=()):
        query_ids, tables, families = set(query_ids), set(tables), set(families)
        for qid in query_ids:
            tables.update(self.query_dependencies.get(qid) or ())
        return [key for key, spec in self.specs.items()
                if query_ids.intersection(spec.benefit_queries)
                or any(table_matches(spec.table, table) for table in tables)
                or self.families[key] in families]

    def _slice(self, ids, limit, strategy):
        groups = defaultdict(list)
        for key in ids:
            groups[(self.specs[key].table, self.families[key])].append(key)
        for keys in groups.values():
            keys.sort(key=lambda key: (
                self.ledger.candidate_coverage.get(key, {}).get("seen_count", 0), key))
        group_keys = sorted(groups)
        signature = tuple(group_keys)
        offset = self._offsets[signature] % max(1, len(group_keys))
        group_keys = group_keys[offset:] + group_keys[:offset]
        if strategy == "DIVERSIFY":
            group_keys.sort(key=lambda group: (
                min(self.ledger.candidate_coverage.get(key, {}).get("seen_count", 0)
                    for key in groups[group]),
                sum(self.ledger.candidate_coverage.get(key, {}).get("partitioned_count", 0)
                    for key in groups[group])))
        result = []
        while group_keys and len(result) < limit:
            for group in list(group_keys):
                result.append(groups[group].pop(0))
                if not groups[group]:
                    group_keys.remove(group)
                if len(result) == limit:
                    break
        self._offsets[signature] += min(limit, len(groups))
        return result

    def build(self, route, selected, previous=None, expansion=None, required_ids=(), preferred_ids=()):
        selected = set(selected)
        source = expansion or route
        prefix = "expand_" if expansion else "focus_"
        structural = self.structural_ids(
            source.get(prefix + "query_ids", []), source.get(prefix + "tables", []),
            source.get("expand_candidate_families", []) if expansion else [])
        retained = list(dict.fromkeys(list((previous or {}).get("candidates", {}))
                                     + [key for key in required_ids if key in self.specs]))
        opportunities = [key for key in structural if key not in selected and key not in retained]
        preferred = list(dict.fromkeys(key for key in preferred_ids if key in opportunities))[:self.view_limit]
        opportunities = [key for key in opportunities if key not in preferred]
        query_ids = set(source.get(prefix + "query_ids", []))
        direct = [key for key in opportunities if query_ids.intersection(self.specs[key].benefit_queries)]
        retrieved = preferred + self._slice(direct, self.view_limit - len(preferred), route.get("search_strategy"))
        if len(retrieved) < self.view_limit:
            retrieved += self._slice([key for key in opportunities if key not in set(direct)],
                                     self.view_limit - len(retrieved), route.get("search_strategy"))
        visible = list(dict.fromkeys(sorted(selected) + retained + retrieved))
        rows = {key: self.specs[key].prompt_dict(self.specs[key].benefit_queries)
                for key in visible}
        self.ledger.mark_coverage("seen_count", retrieved + [key for key in selected if key not in retained])
        omitted = [key for key in opportunities if key not in retrieved]
        unseen = set(self.specs) - set(visible)
        return {
            "candidates": rows,
            "coverage": {
                "pool_count": len(self.specs), "selected_count": len(selected),
                "visible_count": len(visible), "structural_count": len(structural),
                "retrieved_count": len(retrieved), "omitted_count": len(omitted),
                "omitted_by_table": dict(Counter(self.specs[key].table for key in omitted)),
                "omitted_by_family": dict(Counter(self.families[key] for key in omitted)),
                "expansion_used": expansion is not None,
            },
            "expansion_options": {
                "query_ids": [qid for qid in self.query_dependencies
                              if unseen.intersection(self.structural_ids([qid]))],
                "tables": sorted({self.specs[key].table for key in unseen}),
                "candidate_families": sorted({self.families[key] for key in unseen}),
            },
        }

    def availability(self, selected):
        remaining = [key for key in self.specs if key not in set(selected)]
        return {
            "unselected_count": len(remaining),
            "by_query": {qid: len(set(self.structural_ids([qid])).intersection(remaining))
                         for qid in self.query_dependencies},
        }

def _columns(expression):
    # Ignore SQL string values. Unknown expressions simply get no lexical hint;
    # this is ranking evidence, never an applicability or performance validator.
    return set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*',
                          re.sub(r"'(?:''|[^'])*'", '', str(expression or ''))))


def _predicate_columns(expression):
    # Recognize only a bare left operand at a condition boundary. Arithmetic,
    # functions and unrecognized syntax retain their weaker lexical hints.
    text = re.sub(r"'(?:''|[^'])*'", "''", str(expression or ''))
    return set(re.findall(
        r'(?:^|\(|\bAND\b|\bOR\b)\s*(?:[A-Za-z_]\w*\.)?'
        r'([A-Za-z_]\w*)\s*(?:[<>=!]=?|\bBETWEEN\b|\bIN\b|\bIS\b)',
        text, re.IGNORECASE))


def _prefix_length(keys, columns):
    for position, column in enumerate(keys):
        if column not in columns:
            return position
    return len(keys)


def access_opportunities(specs, selected, observations, free_bytes, removable_bytes,
                         candidate_ids=None, limit=4, tested_ids=()):
    selected = set(selected)
    allowed = set(specs if candidate_ids is None else candidate_ids) - selected
    tested = set(tested_ids)
    result = []
    for observation in observations:
        if not observation or observation.get('status') != 'OK':
            continue
        nodes = (observation.get('plan_summary') or {}).get('nodes') or observation.get('plan_nodes') or []
        best = None
        for node in nodes:
            if not node.get('rel') or not node.get('op', '').endswith('Scan'):
                continue
            work = float(node.get('actual_total_ms') or 0) * float(node.get('actual_loops') or 0)
            if not math.isfinite(work) or work <= 0:
                continue
            probe = _columns(node.get('index_cond')) | _columns(node.get('recheck_cond'))
            predicates = (_predicate_columns(node.get('index_cond'))
                          | _predicate_columns(node.get('recheck_cond'))
                          | _predicate_columns(node.get('filter')))
            if node.get('op') == 'Bitmap Heap Scan':
                for child in nodes:
                    if (child.get('op') == 'Bitmap Index Scan'
                            and str(child.get('path', '')).startswith(str(node.get('path')) + '.')):
                        probe |= _columns(child.get('index_cond'))
                        predicates |= _predicate_columns(child.get('index_cond'))
            filters = _columns(node.get('filter'))
            columns = probe | filters
            ranked = []
            for key in allowed:
                spec = specs[key]
                if not table_matches(spec.table, node['rel']) or not spec.keys:
                    continue
                prefix = _prefix_length(spec.keys, columns)
                if not prefix or spec.size_bytes > free_bytes + removable_bytes:
                    continue
                rank = (key in tested, -_prefix_length(spec.keys, predicates),
                        -prefix, -len(set(spec.keys) & columns), -(spec.keys[0] in probe),
                        -(observation['query_id'] in spec.benefit_queries),
                        len(spec.keys), spec.size_bytes, key)
                ranked.append((rank, key))
            if not ranked:
                continue
            ids = [key for _, key in sorted(ranked)[:2]]
            item = dict(query_id=observation['query_id'], path=node.get('path'),
                        table=node['rel'], scan_work_ms=round(work, 3),
                        candidates=[dict(index_id=key, keys=list(specs[key].keys),
                                         required_removal_mb=round(max(0, specs[key].size_bytes - free_bytes) / 1024**2, 3),
                                         tested_in_patch=key in tested,
                                         predicate_prefix=_prefix_length(specs[key].keys, predicates))
                                    for key in ids])
            if best is None or _opportunity_priority(item) < _opportunity_priority(best):
                best = item
        if best:
            result.append(best)
    return sorted(result, key=_opportunity_priority)[:limit]


def _opportunity_priority(row):
    # Apply the same exploration preference before node/query truncation, so
    # tested hotspots cannot hide every untested path outside the hint quota.
    return (all(item['tested_in_patch'] for item in row['candidates']),
            -row['scan_work_ms'], row['query_id'])

