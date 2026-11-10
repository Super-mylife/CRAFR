"""Stage 3 context: compressed views sent to the three roles."""

import json
import re
from collections import Counter

from core.stage3 import MB

def _prompt_json(value):
    """Serialize evidence as compact JSON with literal values."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _object_schema(properties):
    return {
        "type": "object", "properties": properties,
        "required": list(properties), "additionalProperties": False,
    }


def _id_array(allowed, limit=None):
    schema = {"type": "array", "items": {"type": "string"}}
    if allowed:
        schema["items"]["enum"] = list(allowed)
        if limit is not None:
            schema["maxItems"] = limit
    else:
        schema["maxItems"] = 0
    return schema


def _response_format(name, properties, defs=None):
    schema = _object_schema(properties)
    if defs:
        schema["$defs"] = defs
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


OUTPUT_TOKENS = {
    "index_initializer": 6000,
    "index_router": 1200,
    "candidate_partitioner": 800,
    "index_builder": 800,
}

TRANSPORT_ATTEMPTS = 3


def _unwrap_singleton_object(result):
    """Some gateways return the required object as a one-element array."""
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        return result[0]
    return result


def _prose(value):
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        if parts:
            return "; ".join(parts)
    return None


def _normalize_initializer_reply(result):
    """Novamind does not enforce strict schema, so accept the fields Gemini actually sends."""
    folded = {str(key).lower(): value for key, value in result.items()}
    if "configuration" not in result and "configuration" in folded:
        result["configuration"] = folded["configuration"]
    if _prose(result.get("reason")) is None:
        for name in ("reason", "reasoning", "reasons"):
            prose = _prose(result.get(name) if name in result else folded.get(name))
            if prose:
                result["reason"] = prose
                break
    return result

TRIAL_DETAIL_QUERY_TARGET = 4


def _initialization_context(state):
    skipped, failures = [], []
    for item in state.get('initial_skipped_indexes') or []:
        if item.get('reason_code') == 'ESTIMATED_BUDGET':
            skipped.append(item['index_id'])
        else:
            failures.append(item)
    return {'InitialPackingSkipped': skipped, 'InitialBuildFailures': failures}


def _opportunity_view(rows, candidates=None):
    result = []
    for row in rows or []:
        items = [item for item in row['candidates']
                 if candidates is None or item['index_id'] in candidates]
        if items:
            result.append([row['query_id'], row['path'], row['table'], row['scan_work_ms'],
                           [[item['index_id'], item['keys'], item['required_removal_mb'],
                             item.get('tested_in_patch', False), item.get('predicate_prefix', 0)] for item in items[:2]]])
    return {'columns': ['query_id', 'path', 'table', 'scan_work_ms', 'candidates_id_keys_removal_mb_tested_in_patch_predicate_prefix'],
            'rows': result[:4]} if result else None


def _mechanism_feedback(feedback, patch):
    rows = feedback.get('target_results', []) + feedback.get('collateral_results', [])
    if not rows:
        return None
    add = set(patch.get('add_ids', patch.get('add', [])))
    used = {key for row in rows for key in ((row.get('after') or {}).get('plan_summary') or {}).get('indexes', [])}
    sources = Counter((side, (row.get(key) or {}).get('source') or 'UNKNOWN')
                      for row in rows for side, key in [('current', 'before'), ('trial', 'after')])
    return {'added_used_ids': sorted(add & used), 'added_not_observed_ids': sorted(add - used),
            'changed_focus_query_ids': [row['query_id'] for row in feedback.get('target_results', [])
                                       if _query_path_or_status_changed(row)],
            'measurement_sources': [[side, source, count] for (side, source), count in sorted(sources.items())]}


def _query_families(workload):
    """Canonical, readable query structures; parameter variants share one row."""
    families = {}
    for query in workload.get("queries", []):
        shape = {"tables": sorted(set(query.get("tables", [])))}
        for key in ("filters", "groups", "orders"):
            values = []
            for item in query.get(key, []):
                value = "%s.%s" % (item.get("table"), item.get("column"))
                suffix = item.get("op") if key == "filters" else item.get("direction") if key == "orders" else None
                values.append(value + (" " + str(suffix) if suffix else ""))
            if values:
                shape[key] = values if key == "orders" else sorted(set(values))
        if query.get("joins"):
            shape["joins"] = sorted({"%s.%s=%s.%s" % tuple(join) for join in query["joins"]})
        signature = _prompt_json(shape)
        families.setdefault(signature, {"query_ids": [], **shape})["query_ids"].append(query["qid"])
    return list(families.values())


def _family_shapes(workload):
    """One row per family: queries, tables, join edges, and predicates."""
    families = _query_families(workload)
    return {
        "columns": ["query_ids", "tables", "joins", "filters", "groups", "orders"],
        "rows": [[family.get("query_ids") or [], family.get("tables") or [],
                  family.get("joins") or [], family.get("filters") or [],
                  family.get("groups") or [], family.get("orders") or []]
                 for family in families],
    }


_CREATE_INDEX_ON = re.compile(
    r"\bON\s+(?:ONLY\s+)?(?:[\w]+\.)?([\w]+)(?:\s+USING\s+\w+)?\s*\(",
    re.I,
)


def _indexdef_keys(definition):
    match = _CREATE_INDEX_ON.search(definition)
    if not match:
        return []
    table = match.group(1)
    start = match.end()
    depth = 1
    index = start
    while index < len(definition) and depth:
        if definition[index] == "(":
            depth += 1
        elif definition[index] == ")":
            depth -= 1
        index += 1
    keys = []
    for part in definition[start:index - 1].split(","):
        token = part.strip().split()[0].strip('"') if part.strip() else ""
        if token and not token.startswith("("):
            keys.append("%s.%s" % (table, token))
    return keys


def _visible_existing_indexes(items):
    visible = []
    for item in items or []:
        if isinstance(item, dict):
            visible.append(item)
        elif isinstance(item, str) and " ON pg_toast." not in item:
            visible.append(item)
    return visible


def _used_columns(workload):
    used = set()
    for query in workload.get("queries") or []:
        if not isinstance(query, dict):
            continue
        for key in ("filters", "groups", "orders"):
            for item in query.get(key) or []:
                if isinstance(item, dict) and item.get("table") and item.get("column"):
                    used.add("%s.%s" % (item["table"], item["column"]))
        for join in query.get("joins") or []:
            if len(join) >= 4:
                used.add("%s.%s" % (join[0], join[1]))
                used.add("%s.%s" % (join[2], join[3]))
    for item in _visible_existing_indexes(workload.get("existing_indexes")):
        if isinstance(item, dict):
            table = item.get("table")
            used.update("%s.%s" % (table, key) if table else key
                        for key in item.get("keys") or [])
        else:
            used.update(_indexdef_keys(item))
    return used


def _column_stats(workload):
    used = _used_columns(workload)
    stats = workload.get("column_stats") or {}
    names = [name for name in stats if not used or name in used]
    return _rows_table([{"column": name, "ndv": stats[name].get("ndv"),
                         "top_value_fraction": stats[name].get("top_value_fraction")}
                        for name in names], ("column", "ndv", "top_value_fraction"))


def _tuning_workload(workload, query_ids=None, include_column_stats=True,
                     include_existing_indexes=True, include_sql=True):
    # Local SEARCH roles use SQL for the queries they actually reason about.
    # INIT uses family access patterns, not full-text SQL.
    scoped = workload
    if query_ids is not None:
        allowed = set(query_ids)
        scoped = {**workload, "queries": [query for query in workload.get("queries") or []
                                          if query.get("qid") in allowed]}
    tables = {table for query in scoped.get("queries", []) for table in query.get("tables", [])}
    view = {
        "table_rows": {key: value for key, value in workload.get("table_rows", {}).items()
                       if query_ids is None or key in tables},
    }
    if include_column_stats:
        view["column_stats"] = _column_stats(scoped)
    if include_existing_indexes:
        view["existing_indexes"] = _visible_existing_indexes(workload.get("existing_indexes"))
    queries = scoped.get("queries") or []
    if include_sql and queries and all(query.get("sql") for query in queries):
        view["queries"] = {"columns": ["query_id", "sql"], "rows": [
            [query["qid"], query["sql"].strip()] for query in queries]}
    else:
        view["query_families"] = _family_shapes(scoped)
        if include_sql:
            # Features-only callers lack the original SQL; mark that explicitly.
            view["sql_available"] = False
    return view


def _bottleneck_query_ids(state, limit=4):
    """SQL is only needed for the queries the model actually reasons about."""
    card = state.get("global_evidence") or {}
    ids = []
    for row in card.get("slow_query_plans") or []:
        if isinstance(row, dict) and row.get("query_id"):
            ids.append(row["query_id"])
    if ids:
        return list(dict.fromkeys(ids))
    queries = card.get("queries") or []
    if isinstance(queries, list):
        ranked = [row for row in queries if isinstance(row, dict) and row.get("query_id")]
        ranked.sort(key=lambda row: -(row.get("current_ms") or 0))
        return [row["query_id"] for row in ranked[:limit]]
    return None


def _state_view(state):
    return {"configuration": state["configuration"]} if "configuration" in state else {}


def _task_view(task):
    view = {"intent": task.get("intent", task.get("mode")), **{
        key: task[key] for key in ("focus_query_ids", "focus_tables", "focus_reason")
        if task.get(key)}}
    if task.get("diagnosis"):
        # Partitioner runs after commit/rollback: the retained side is now CURRENT.
        view["diagnosis"] = [{**item, "side": "current"} for item in task["diagnosis"]]
    return view


def _brief_error(value):
    return " ".join(str(value).split())[:240] if value else None


def _patch_view(patch):
    result = {"add_ids": patch.get("add_ids", patch.get("add", [])),
              "remove_ids": patch.get("remove_ids", patch.get("remove", [])),
              "reason": patch.get("reason", "")}
    for key in ("expected_improvements", "risk_query_ids"):
        value = patch.get(key, patch.get("probe_query_ids") if key == "expected_improvements" else None)
        if value:
            result[key] = value
    return result


def _index_definitions(indexes):
    return _rows_table([{"index_id": key, "table": value.get("table"),
                         "keys": value.get("keys"), "include": value.get("include")}
                        for key, value in (indexes or {}).items()],
                       ("index_id", "table", "keys", "include"))


def _comparison_table(rows, include_plans=True):
    facts = []
    for row in rows:
        item = {"query_id": row["query_id"]}
        for source, side in (("before", "current"), ("after", "trial")):
            observation = row.get(source) or {}
            plan = observation.get("plan_summary")
            plan = plan if isinstance(plan, dict) else {}
            item.update({side + "_time_ms": observation.get("actual_total_ms"),
                         side + "_status": observation.get("status", "UNKNOWN")})
            if include_plans:
                item.update({side + "_root": plan.get("root"), side + "_indexes": plan.get("indexes")})
            if observation.get("status") != "OK":
                item[side + "_timeout_ms"] = observation.get("timeout_ms")
                item[side + "_error"] = _brief_error(observation.get("error") or observation.get("reason"))
        facts.append(item)
    return _intern_table_values(_rows_table(facts, ("query_id", "current_time_ms", "trial_time_ms",
        "current_status", "trial_status", "current_timeout_ms", "trial_timeout_ms",
        "current_error", "trial_error", "current_root", "trial_root", "current_indexes", "trial_indexes")))


def _history_view(history):
    """One outcome and one patch rationale, with facts from every affected query."""
    result = []
    for record in history[-6:]:
        feedback, patch = record.get("feedback") or {}, record.get("patch") or {}
        item = {"outcome": record.get("outcome")}
        configuration = feedback.get("current_configuration", record.get("configuration"))
        if configuration is not None:
            item["current_configuration"] = configuration
        if patch:
            if patch.get("action") == "NOOP":
                item["reason"] = patch.get("reason")
            else:
                item["patch"] = _patch_view(patch)
        task, partition = record.get("task") or {}, record.get("partition") or {}
        focus = partition.get("query_ids", partition.get("focus_query_ids")) or task.get("focus_query_ids")
        if focus:
            item["focus"] = {"focus_query_ids": focus}
        elif task.get("focus_tables"):
            item["focus"] = {"focus_tables": task["focus_tables"]}
        if patch.get("action") == "NOOP" or not patch:
            region = {key: partition[key] for key in ("active_selected_ids", "active_unselected_ids")
                      if partition.get(key)}
            if region:
                item["region"] = region
        reason = feedback.get("reason_code", record.get("reason_code")) or record.get("failure_category")
        if reason and reason not in {"ROUTER_CURRENT", "ROUTER_TRIAL", record.get("outcome")}:
            item["reason_code"] = reason
        error = record.get("error") or record.get("reason")
        if error:
            item["error"] = _brief_error(error)
        if feedback.get("workload_comparison"):
            item["workload_comparison"] = _workload_comparison_prompt(feedback["workload_comparison"])
        indexes = record.get("patch_indexes") or feedback.get("patch_indexes")
        if indexes:
            item["indexes"] = _index_definitions(indexes)
        rows = feedback.get("target_results", []) + feedback.get("collateral_results", [])
        if rows:
            # Keep every historical timing/status; current plans and the pending
            # trial already carry access paths, so do not repeat old plan arrays.
            item["results"] = _comparison_table(rows, include_plans=False)
            item['mechanism'] = _mechanism_feedback(feedback, patch)
        storage = feedback.get("storage") or {}
        if storage:
            item["storage_bytes"] = {key: storage[key] for key in ("current_bytes", "trial_bytes") if key in storage}
        if feedback.get("unmeasured_query_ids"):
            item["unmeasured_query_ids"] = feedback["unmeasured_query_ids"]
        if record.get("expansion_request"):
            item["expansion"] = {key: value for key, value in record["expansion_request"].items()
                                 if key != "action" and value}
        if item not in result:
            result.append(item)
    return result


def _workload_comparison_prompt(comparison):
    # Per-query before/after times already express every gain; keep the stored
    # full map for analysis, without sending it again (including unaffected zeros).
    return {key: value for key, value in (comparison or {}).items()
            if key != "net_gain_by_query_ms"}


def _plan_side_identity(observation):
    """Access-path / status fingerprint; time changes do not count as a path change."""
    if not isinstance(observation, dict):
        return None
    plan = observation.get("plan_summary")
    plan = plan if isinstance(plan, dict) else {}
    indexes = plan.get("indexes") or []
    if isinstance(indexes, (list, tuple)):
        indexes = tuple(sorted(str(item) for item in indexes))
    return (
        observation.get("status", "UNKNOWN"),
        plan.get("root"),
        indexes,
        plan.get('structure_id'),
        bool(observation.get("error") or observation.get("reason")),
        bool(observation.get("plan_truncated")),
    )


def _query_path_or_status_changed(row):
    return _plan_side_identity(row.get("before")) != _plan_side_identity(row.get("after"))


def _trial_plan_rows(feedback):
    """Keep plans for focus queries and any affected query whose path or status changed."""
    focus = {row["query_id"] for row in feedback.get("target_results") or []}
    for key in ("expected_improvements", "risk_query_ids", "probe_query_ids"):
        focus.update(qid for qid in feedback.get(key) or [] if qid)
    selected, omitted = [], []
    for row in (feedback.get("target_results") or []) + (feedback.get("collateral_results") or []):
        qid = row.get("query_id")
        if not qid:
            continue
        if qid in focus or _query_path_or_status_changed(row):
            selected.append({"query_id": qid, "current": row.get("before"), "trial": row.get("after")})
        else:
            omitted.append(qid)
    # Explicit focus and risk evidence must not compete for a four-query quota.
    # Fill spare detail slots with the largest changed collateral measurements.
    def priority(row):
        before = (row.get('current') or {}).get('actual_total_ms')
        after = (row.get('trial') or {}).get('actual_total_ms')
        change = abs(after - before) if before is not None and after is not None else float('inf')
        return row['query_id'] not in focus, -change
    selected.sort(key=priority)
    limit = max(TRIAL_DETAIL_QUERY_TARGET, sum(row['query_id'] in focus for row in selected))
    omitted += [row['query_id'] for row in selected[limit:]]
    return selected[:limit], omitted


def _trial_feedback_prompt(feedback, opportunities=None):
    if not feedback:
        return None
    rows = feedback.get("target_results", []) + feedback.get("collateral_results", [])
    patch = _patch_view(feedback)
    patch["indexes"] = _index_definitions(feedback.get("patch_indexes"))
    plan_rows, omitted = _trial_plan_rows(feedback)
    plans = _focus_evidence_prompt(plan_rows)
    plans["omitted_query_ids"] = omitted
    return {"WorkloadComparison": _workload_comparison_prompt(feedback.get("workload_comparison")),
            "MechanismFeedback": _mechanism_feedback(feedback, patch),
            **({"TrialAccessOpportunities": _opportunity_view(opportunities)} if opportunities else {}),
            "CurrentConfiguration": feedback["current_configuration"],
            "TrialConfiguration": feedback["trial_configuration"], "TrialPatch": patch,
            "AffectedQueryFeedback": _comparison_table(rows),
            "Plans": plans,
            "Storage": feedback["storage"]}


def _rows_table(rows, columns):
    """Prompt-only table: header once, then aligned value rows."""
    used = [col for col in columns if any(
        isinstance(row, dict) and row.get(col) not in (None, "", [], {}) for row in rows)]
    if not used and columns:
        used = [columns[0]]
    return {
        "columns": used,
        "rows": [[row.get(col) if isinstance(row, dict) else None for col in used]
                 for row in rows],
    }


def _intern_table_values(table):
    """Losslessly share repeated long expressions and index lists in SEARCH tables."""
    counts = Counter(_prompt_json(value) for row in table['rows'] for value in row
                     if isinstance(value, (str, list)) and len(_prompt_json(value)) >= 64)
    shared = {value: str(i) for i, (value, count) in enumerate(counts.items()) if count > 1}
    if shared:
        table['value_dictionary'] = {key: json.loads(value) for value, key in shared.items()}
        table['rows'] = [[{'value_ref': shared[_prompt_json(value)]}
                          if isinstance(value, (str, list)) and _prompt_json(value) in shared
                          else value for value in row] for row in table['rows']]
    # Omitted trailing cells still mean null; avoid paying for placeholder runs.
    for row in table['rows']:
        while row and row[-1] is None:
            row.pop()
    return table


_QUERY_EVIDENCE_COLUMNS = (
    "query_id", "current_status", "current_ms", "current_timeout_ms", "current_error",
    "current_root", "current_indexes", "last_search", "unselected_count",
)
_SELECTED_INDEX_COLUMNS = (
    "index_id", "table", "keys", "include", "estimated_mb",
    "observed_query_ids", "observed_current_ms",
)

_PLAN_NODE_COLUMNS = (
    "query_id", "side", "path", "op", "actual_total_ms", "actual_loops", "actual_rows", "rows",
    "parent", "rel", "alias", "scan_work_ms", "index_id", "index", "index_cond",
    "join", "join_cond", "filter", "rows_removed_by_filter", "recheck_cond", "sort", "group",
)


def _query_evidence_table(rows):
    return _intern_table_values(_rows_table(rows or [], _QUERY_EVIDENCE_COLUMNS))


def _selected_index_table(rows):
    return _intern_table_values(_rows_table([{**row, "estimated_mb": round(row["size_bytes"] / MB, 6)
                        if row.get("size_bytes") is not None else None} for row in rows or []],
                       _SELECTED_INDEX_COLUMNS))

def _focus_evidence_prompt(rows):
    nodes, scans = [], []
    for row in rows or []:
        qid = row.get("query_id")
        blocks = [(side, row.get(side) or {}) for side in ("current", "trial")]
        if row.get("plan_nodes"):
            blocks.insert(0, ("current", row))
        for side, block in blocks:
            side_nodes = block.get("plan_nodes") or (block.get("plan_summary") or {}).get("nodes") or []
            for node in side_nodes:
                nodes.append({"query_id": qid, "side": side,
                              **{key: value for key, value in node.items() if key != "buffers"}})
            timed_scans = [node for node in side_nodes if node.get("scan_work_ms") is not None]
            if timed_scans:
                hot = max(timed_scans, key=lambda node: node["scan_work_ms"])
                scans.append({"query_id": qid, "side": side,
                              "query_ms": block.get("actual_total_ms", row.get("current_ms") if side == "current" else None),
                              **{key: hot[key] for key in ("path", "op", "rel", "alias", "actual_loops", "scan_work_ms")
                                 if key in hot}})
    return {
        # One bound row per query/side; collateral plans cannot grow this summary.
        "scan_focus": [item for side in ("current", "trial") for item in sorted(
            (item for item in scans if item["side"] == side),
            key=lambda item: -(item["query_ms"] or 0))[:4]],
        "plan_nodes": _intern_table_values(_rows_table(nodes, _PLAN_NODE_COLUMNS)),
        "truncated_plans": [
            {"query_id": row.get("query_id"), "side": "current"}
            for row in rows or [] if row.get("plan_truncated")
        ] + [
            {"query_id": row.get("query_id"), "side": side}
            for row in rows or [] for side in ("current", "trial")
            if (row.get(side) or {}).get("plan_truncated")],
    }


def _diagnosis_refs_schema():
    return {"type": "array", "maxItems": 4, "items": _object_schema({
        "query_id": {"type": "string"},
        "side": {"type": "string", "enum": ["current", "trial"]},
        "path": {"type": "string"},
    })}


def _table_records(table):
    values = table.get('value_dictionary', {})
    return [dict(zip(table.get("columns", []),
                     [values[value['value_ref']] if isinstance(value, dict) and set(value) == {'value_ref'}
                      else value for value in row] + [None] * (len(table.get('columns', [])) - len(row))))
            for row in table.get("rows", [])]


def _bind_diagnosis(reply, evidence):
    """Resolve only supplied plan facts; never infer an index's benefit or scope."""
    if not isinstance(reply, dict):
        raise ValueError("Diagnosis response must be an object")
    if "evidence_refs" not in reply:  # Older stored replies/test adapters.
        return reply
    refs = reply["evidence_refs"]
    if not isinstance(refs, list) or len(refs) > 4:
        raise ValueError("evidence_refs must contain at most four node references")
    card = evidence.get("GlobalEvidenceCard") or {}
    local = evidence.get("LocalProblem") or {}
    feedback = evidence.get("TrialFeedback") or {}
    comparisons = _table_records(feedback.get("AffectedQueryFeedback") or {})
    affected = {row["query_id"] for row in comparisons}
    times = {}
    for table in (card.get("queries"), evidence.get("QueryEvidence"), local.get("affected_evidence")):
        for row in _table_records(table or {}):
            times[("current", row["query_id"])] = row.get("current_ms")
    for row in comparisons:
        for side in ("current", "trial"):
            times[(side, row["query_id"])] = row.get(side + "_time_ms")
    nodes = {}
    for plans in (card.get("slow_query_plans"), evidence.get("Evidence"),
                  local.get("focus_plan_evidence"), feedback.get("Plans")):
        plans = plans or {}
        for item in plans.get("scan_focus", []):
            times.setdefault((item["side"], item["query_id"]), item["query_ms"])
        for node in _table_records(plans.get("plan_nodes") or {}):
            if all(isinstance(node.get(key), str) for key in ("side", "query_id", "path")):
                nodes[(node["side"], node["query_id"], node["path"])] = node
    retained = "trial" if reply.get("state_choice") == "TRIAL" else "current"
    focus = set(reply.get("focus_query_ids", (evidence.get("RegionCard") or {}).get("focus_query_ids", [])))
    required_focus = focus
    if reply.get("action") == "PATCH" and evidence.get("RegionCard"):
        claimed = set(reply.get("expected_improvements", [])) | set(reply.get("risk_query_ids", []))
        if claimed:
            required_focus = focus & claimed
    bound = []
    cited = set()
    for ref in refs:
        if (not isinstance(ref, dict) or set(ref) != {"query_id", "side", "path"}
                or any(not isinstance(value, str) for value in ref.values())):
            raise ValueError("Each evidence_ref requires query_id, side and path")
        qid, side = ref["query_id"], ref["side"]
        key = (side, qid, ref["path"])
        if key not in nodes:
            raise ValueError("Evidence reference is not in the supplied plans: %s" % (ref,))
        if side != retained and not (retained == "trial" and side == "current" and qid not in affected):
            raise ValueError("Evidence reference must use the retained %s plan: %s" % (retained, ref))
        if (focus and qid not in focus) or qid in cited:
            raise ValueError("Cite at most one node per focus query: %s" % qid)
        cited.add(qid)
        node = nodes[key]
        bound.append({"query_id": qid, "side": retained,
                      "query_ms": times.get((side, qid)),
                      **{field: node[field] for field in
                         ("path", "op", "rel", "alias", "actual_loops", "scan_work_ms")
                         if node.get(field) is not None}})
    needs = {qid for (side, qid, _), node in nodes.items()
             if qid in required_focus and node.get("scan_work_ms") is not None
             and times.get((side, qid)) is not None
             and (side == retained or (retained == "trial" and side == "current" and qid not in affected))}
    empty_region = (reply.get("action") == "FINAL" and reply.get("active_selected_ids") == []
                    and reply.get("active_unselected_ids") == [])
    if (not empty_region and reply.get("intent", reply.get("action")) not in {"STOP", "EXPAND", "NOOP"}
            and needs - cited):
        raise ValueError("Cite a supplied retained-state plan node for focus queries: " + ", ".join(sorted(needs - cited)))
    return {**reply, **({"diagnosis": bound} if bound else {})}


def _query_tables_table(query_tables):
    families = {}
    for qid, tables in (query_tables or {}).items():
        # Preserve unknown dependencies as null, distinct from an empty set.
        key = tuple(sorted(tables)) if tables is not None else None
        families.setdefault(key, []).append(qid)
    return {"columns": ["query_ids", "tables"],
            "rows": [[qids, list(tables) if tables is not None else None]
                     for tables, qids in families.items()]}


def _candidate_view(candidates, storage_free_bytes=None, compact=False):
    """Compact fixed definitions with the Stage 2 candidate-query relation."""
    has_include = any(item.get("include") for item in candidates.values())
    has_observed = any(item.get("observed_current_ms") is not None for item in candidates.values())
    has_observed_queries = any("observed_query_ids" in item for item in candidates.values())
    show_fit = storage_free_bytes is not None
    columns = ["index_id", "table", "keys"]
    if has_include:
        columns.append("include")
    columns.extend(["estimated_mb", "relevant_query_ids"])
    if has_observed_queries:
        columns.append("observed_query_ids")
    if has_observed:
        columns.append("observed_current_ms")
    if show_fit:
        columns.append("required_removal_mb")
    rows = []
    for index_id, item in candidates.items():
        row = [index_id, item.get("table"), item.get("keys", [])]
        if has_include:
            row.append(item.get("include") or [])
        estimated = item.get("size_bytes")
        row.extend([round(estimated / MB, 6) if estimated is not None else None,
                    item.get("relevant_query_ids", [])])
        if has_observed_queries:
            row.append(item.get("observed_query_ids"))
        if has_observed:
            row.append(item.get("observed_current_ms"))
        if show_fit:
            row.append(round(max(0, estimated - storage_free_bytes) / MB, 6)
                       if estimated is not None else None)
        rows.append(row)
    table = {"columns": columns, "rows": rows}
    return _intern_table_values(table) if compact else table


def _storage_prompt(constraints):
    budget = constraints.get("storage_budget_bytes")
    if budget is None:
        return None
    used = constraints.get("storage_used_bytes")
    free = constraints.get("storage_free_bytes")
    if free is None and used is not None:
        free = budget - used
    result = {"budget": budget / MB}
    if used is not None:
        result["used"] = used / MB
    if free is not None:
        result["free"] = free / MB
    if constraints.get('max_removable_bytes') is not None:
        result['max_releasable_with_removal_limit'] = constraints['max_removable_bytes'] / MB
    return result


def _with_index_evidence(candidates, state):
    observed = {item["index_id"]: item
                for item in (state.get("global_evidence") or {}).get("selected_indexes") or []}
    enriched = {}
    for key, item in candidates.items():
        enriched[key] = {**item, **{field: value for field, value in observed.get(key, {}).items()
                                  if field in {"observed_query_ids", "observed_current_ms"}}}
    return enriched
