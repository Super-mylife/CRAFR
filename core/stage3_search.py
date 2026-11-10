"""Stage 3 search: initialize S0 and choose indexes inside bounded transactions."""

import hashlib
import json
import math
import os
import time
from copy import deepcopy
from itertools import combinations

from core.stage3 import (
    CandidateAccess,
    EvidenceLedger,
    LLMCallBudgetExhausted,
    LOG,
    MAX_ADDITIONS_PER_ROUND,
    MAX_REMOVALS_PER_ROUND,
    MB,
    ProposalError,
    SearchState,
    _configuration_size,
    _finite_ms,
    _ids,
    _pack_by_priority,
    _persist_state,
    access_opportunities,
    compact_query_status,
    compact_query_statuses,
    configuration_identity,
    focus_accesses,
    summarize_observation,
    table_matches,
    trial_feedback,
)


class _RouteMixin:
    def _ordered(self, configuration):
        chosen = set(configuration)
        return [index_id for index_id in self.candidate_ids if index_id in chosen]

    def _validate_initial(self, reply):
        if not isinstance(reply, dict):
            raise ValueError("Initializer reply must be an object")
        priority = _ids(reply.get("configuration"), self.candidate_ids, "configuration")
        packed, skipped, estimated_bytes = _pack_by_priority(
            priority, self.specs, self.storage_budget_bytes)
        if not isinstance(reply.get("reason"), str) or not reply["reason"].strip():
            raise ValueError("Initializer requires an access-path reason for the seed")
        recommendation = {
            "priority": priority,
            "configuration": packed,
            "skipped_by_estimate": skipped,
            "estimated_index_bytes": estimated_bytes,
            "estimated_index_mb": round(estimated_bytes / MB, 2),
            "reason": reply["reason"],
        }
        return SearchState(1, packed, move={
            "kind": "llm_initialization", "add": packed, "remove": [],
            "reason": reply["reason"],
            "priority": priority,
            "skipped_indexes": skipped,
        }), recommendation

    def initial_configuration(self, workload):
        """Ask the LLM for a priority order; pack once and skip what does not fit."""
        state = {
            "state_version": 0,
            "configuration": [],
            "constraints": {
                "storage_budget_bytes": self.storage_budget_bytes,
            },
        }
        return self._ask("initialize", lambda reply: self._validate_initial(reply),
                         state, workload=workload, candidates=self.catalog)

    def _workload(self):
        features_by_id = {query["qid"]: query for query in self.features.get("query_features", [])}
        queries = []
        for query in self.evaluator.queries:
            qid = query["qid"]
            feature = features_by_id.get(qid, {})
            dependencies = self.query_dependencies.get(qid) or set()
            tables = set(feature.get("tables", [])) | {
                name for name in dependencies
                if "." not in name or name.rsplit(".", 1)[-1] not in dependencies}
            queries.append({**feature, "qid": qid, "sql": query["sql"], "tables": sorted(tables)})
        self.query_table_hints = {query["qid"]: query["tables"] for query in queries}
        return {
            "table_rows": {name: value.get("rows") for name, value in
                           self.features.get("tables", {}).items() if isinstance(value, dict)},
            "column_stats": self.features.get("columns", {}),
            "queries": queries,
            "existing_indexes": self.features.get("existing_indexes", []),
        }

    def _query_dependencies(self):
        resolved = self.evaluator.query_dependencies()
        dependencies = {qid: resolved.get(qid) for qid in self.query_ids}
        return dependencies

    def _remaining_seconds(self):
        if self.total_tuning_time_seconds is None:
            return math.inf
        return max(0.0, self.total_tuning_time_seconds - (time.monotonic() - self.started))

    def _remaining_llm_calls(self):
        if not self.max_llm_calls:
            return math.inf
        # Stage4Agents counts every API request, including failed requests/retries.
        used = getattr(self.agents, "usage", {}).get("calls", 0)
        return max(0, self.max_llm_calls - used)

    def _measurement_timeout_ms(self):
        """Return 0 for an unlimited DB statement, otherwise the effective cap."""
        limits = []
        if self.statement_timeout_ms > 0:
            limits.append(self.statement_timeout_ms)
        remaining = self._remaining_seconds()
        if math.isfinite(remaining):
            limits.append(max(1, int(remaining * 1000)))
        return min(limits) if limits else 0

    def _emit(self, phase, **extra):
        remaining = self._remaining_seconds() if hasattr(self, "started") else None
        if remaining is not None and not math.isfinite(remaining):
            remaining = None
        parts = ["STAGE3", phase]
        if remaining is not None:
            parts.append("remaining=%.0fs" % remaining)
        for key, value in extra.items():
            if value is None:
                continue
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
            if len(text) > 240:
                text = text[:240] + "..."
            parts.append("%s=%s" % (key, text))
        LOG.info(" ".join(parts))
        if not self.status_path:
            return
        payload = {
            "phase": phase,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "remaining_seconds": None if remaining is None else round(remaining, 1),
        }
        payload.update(extra)
        directory = os.path.dirname(self.status_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = self.status_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, self.status_path)


    def _fill_current(self, current, query_ids=None):
        """Measure missing observations once; retain completed and failed evidence."""
        query_ids = list(self.query_ids if query_ids is None else query_ids)
        if self._remaining_seconds() <= 0:
            return []
        rows = []
        missing = [qid for qid in query_ids
                   if self.ledger.latest(current.configuration, qid, "configuration") is None]
        if not missing:
            return rows
        started = time.monotonic()
        measured = self.evaluator.measure(
            current.configuration, missing,
            self._measurement_timeout_ms(),
        )
        self.timing["execution"] += (time.monotonic() - started) * 1000
        return rows + [{**row, "state_version": current.version} for row in measured]

    def _focus_plan_evidence(self, configuration, query_ids):
        rows = []
        for pair in self._pairs(configuration, query_ids, include_plan_nodes=True):
            current = pair.get("current") or {}
            row = compact_query_status(pair, include_plan=True)
            row["current"] = {key: current[key] for key in
                              ("status", "plan_nodes", "plan_truncated") if key in current}
            rows.append(row)
        return rows

    def _pairs(self, configuration, query_ids=None, include_plan_nodes=False):
        rows = []
        for qid in self.query_ids if query_ids is None else query_ids:
            current = self.ledger.latest(configuration, qid, "configuration")
            rows.append({"query_id": qid,
                         "current": summarize_observation(current,
                             include_plan_nodes=include_plan_nodes) if current else None})
        return rows

    def _focus_queries(self, task):
        if task.get("focus_query_ids"):
            return list(task["focus_query_ids"])
        tables = task.get("focus_tables") or []
        return [qid for qid in self.query_ids if any(
            table_matches(table, focus) for table in
            (self.query_dependencies.get(qid) or getattr(self, "query_table_hints", {}).get(qid, []))
            for focus in tables)][:4]

    def _access(self):
        if not hasattr(self, "candidate_access"):
            self.candidate_access = CandidateAccess(
                {key: self.specs[key] for key in self.candidate_ids},
                {qid: tables if tables is not None else getattr(self, "query_table_hints", {}).get(qid, [])
                 for qid, tables in self.query_dependencies.items()},
                self.ledger, self.candidate_view_limit)
        return self.candidate_access

    def _eligible_intents(self, current, observations=None):
        rows = (observations if observations is not None else
                [self.ledger.latest(current.configuration, qid, "configuration")
                 for qid in self.query_ids])
        if (current.actual_index_bytes > self.storage_budget_bytes
                or any(not row or row.get("status") != "OK" for row in rows)):
            return ["REPAIR", "STOP"]
        return ["IMPROVE", "COMPACT", "STOP"] if current.configuration else ["IMPROVE", "STOP"]

    def _search_status(self, current, history):
        """Diagnosis is scoped to an incumbent, not collateral measurement coverage."""
        status = {}
        for record in history:
            feedback = record.get("feedback") or {}
            outcome = record.get("outcome")
            configuration = record.get("configuration", feedback.get("current_configuration"))
            if outcome == "ACCEPTED":
                configuration = feedback.get("trial_configuration")
            if configuration is None or set(configuration) != set(current.configuration):
                continue
            partition = record.get("partition") or {}
            focus = partition.get("query_ids", partition.get("focus_query_ids", []))
            if not focus:
                continue
            result = {"NO_EXECUTABLE_PATCH": "NOOP", "EMPTY_REGION": "NOOP",
                      "NO_NEW_CONFIGURATION": "REPEATED"}.get(outcome, outcome)
            if outcome == "INCONCLUSIVE":
                result = record.get("reason_code", feedback.get("reason_code", outcome))
            if outcome in {"ACCEPTED", "REJECTED", "NO_EXECUTABLE_PATCH",
                           "EMPTY_REGION", "NO_NEW_CONFIGURATION", "INCONCLUSIVE"}:
                status.update(dict.fromkeys(focus, result))
        return status

    def _cooled_queries(self, current, history):
        status = self._search_status(current, history)
        cooled = {qid for qid, outcome in status.items()
                  if outcome in {"REJECTED", "NOOP", "REPEATED"}
                  and (self.ledger.latest(current.configuration, qid, "configuration") or {}).get("status") == "OK"}
        if not cooled:
            return set()
        availability = self._access().availability(current.configuration)["by_query"]
        fresh = [qid for qid in self.query_ids if qid not in status and availability.get(qid)]
        # Revisit remains possible once the other reachable queries have been diagnosed.
        if not fresh:
            return set()
        return cooled

    def _budget_repair_request(self, current, history):
        """Keep one oversized patch pending through proposal/protocol corrections."""
        if (not self.retry_over_budget_addition or not history or not current.configuration
                or not self.max_removals or not self.max_additions
                or self.physical_trials >= min(self.max_iterations, self.max_physical_trials)):
            return None
        previous = None
        for position in range(len(history) - 1, -1, -1):
            row = history[position]
            feedback = row.get("feedback") or {}
            if (row.get("reason_code", feedback.get("reason_code")) == "OVER_BUDGET"
                    and not row.get("budget_repair")
                    and set(row.get("configuration", [])) == set(current.configuration)
                    and row.get("state_version", current.version) == current.version):
                previous = row
                break
        if previous is None:
            return None
        feedback = previous.get("feedback") or {}
        patch = previous.get("patch") or {}
        add = patch.get("add", patch.get("add_ids", []))
        remove = patch.get("remove", patch.get("remove_ids", []))
        if (not add or not set(add) <= set(self.specs)
                or set(add) & set(current.configuration)
                or not set(current.configuration) - set(remove)):
            return None
        for row in history[position + 1:]:
            repair = row.get("budget_repair") or {}
            repaired_add = repair.get("candidate_ids", [repair.get("candidate_id")])
            if (set(row.get("configuration", [])) != set(current.configuration)
                    or set(repaired_add) != set(add)):
                continue
            if (row.get("ddl_outcome") or row.get("outcome") in {"ACCEPTED", "REJECTED"}
                    or (row.get("outcome") == "NO_EXECUTABLE_PATCH"
                        and set(add) <= set((row.get("partition") or {}).get("candidate_indexes", [])))):
                return None
        storage = feedback.get("storage") or {}
        excess = storage.get("trial_bytes", 0) - self.storage_budget_bytes
        if excess <= 0 or storage.get("budget_bytes") != self.storage_budget_bytes:
            return None
        return {
            "stage": "budget_repair",
            "error": (
                "The patch was physically built but rolled back for actual storage; "
                "its performance was NOT measured. Check one atomic joint replacement for this "
                "same addition set before stopping. Keep these candidates in the active region and their queries in the "
                "diagnosis; consider ALL selected indexes as possible donors. Builder chooses "
                "the removals and may return NOOP if no justified replacement exists. "
                "The measured excess is extra space beyond the failed patch's removals. "
                "Recommendation estimates are unchanged; physical size and whole-workload "
                "performance will still be checked before acceptance."
            ),
            "details": {
                **({"candidate_id": add[0]} if len(add) == 1 else {}),
                "candidate_ids": list(add),
                "failed_remove_ids": list(remove),
                "focus_query_ids": (previous.get("partition") or {}).get("query_ids", []),
                "actual_storage_bytes": storage,
                "extra_release_mb": round(excess / MB, 6),
            },
        }

    def _initial_diagnosis_needed(self, current, history):
        diagnosed = [row for row in history if row.get("partition") and row.get("outcome") in {
            "NO_EXECUTABLE_PATCH", "EMPTY_REGION", "ACCEPTED", "REJECTED", "INCONCLUSIVE"}]
        # One alternative candidate region protects against an uninformed first
        # NOOP without forcing a tour through every NEW query (DSB50 vs TPCH60).
        if (len(diagnosed) >= 2 or any(row["outcome"] not in {"NO_EXECUTABLE_PATCH", "EMPTY_REGION"}
                                     for row in diagnosed) or not self.max_additions
                or self.physical_trials >= min(self.max_iterations, self.max_physical_trials)
                or "REPAIR" in self._eligible_intents(current)):
            return False
        examined = {key for row in diagnosed for key in row["partition"].get("candidate_indexes", [])}
        capacity = self.storage_budget_bytes - _configuration_size(current.configuration, self.specs)
        capacity += sum(sorted((self.specs[key].size_bytes for key in current.configuration),
                               reverse=True)[:self.max_removals])
        return any(key not in current.configuration and key not in examined and self.specs[key].size_bytes <= capacity
                   for key in self._access().structural_ids(self.query_ids))

    def _route_issue(self, task, current, history):
        if self.physical_trials >= min(self.max_iterations, self.max_physical_trials):
            return None
        if task.get("intent", task.get("mode")) == "STOP" and self._initial_diagnosis_needed(current, history):
            return "Unexamined candidates fit an estimated add/replacement; diagnose a region before STOP (at most one alternative region)."
        return None

    def _validate_route(self, reply, current, eligible, trial=None, trial_eligible=None, history=()):
        if not isinstance(reply, dict):
            raise ProposalError("Router reply must be an object")
        choice = reply.get("state_choice")
        if choice not in ({"CURRENT", "TRIAL"} if trial else {"CURRENT"}):
            raise ProposalError("Router state_choice must be CURRENT, or TRIAL when valid TrialFeedback is present")
        retained = trial if choice == "TRIAL" else current
        allowed = trial_eligible if choice == "TRIAL" else eligible
        if reply.get("intent") not in allowed:
            raise ProposalError("Router intent must be eligible for the retained configuration")
        if reply.get("search_strategy") not in {"EXPLOIT", "DIVERSIFY"}:
            raise ProposalError("Router requires EXPLOIT or DIVERSIFY")
        focus = _ids(reply.get("focus_query_ids"), self.query_ids, "focus_query_ids")
        tables = _ids(reply.get("focus_tables"), self._access().tables, "focus_tables")
        if len(focus) > 4 or len(tables) > 8:
            raise ProposalError("Router focus exceeds the bounded workload region")
        if reply["intent"] != "STOP" and not focus and not tables:
            raise ProposalError("Router must identify focus queries or tables")
        if not isinstance(reply.get("focus_reason"), str):
            raise ProposalError("Router requires focus_reason")
        task = {**reply, "mode": reply["intent"], "focus_query_ids": focus,
                "focus_tables": tables, "reason": reply["focus_reason"],
                "state_version": retained.version, "evidence_version": self.ledger.version}
        # Resolve the incumbent first. Next-focus correction must not invalidate a
        # performance decision while the physical trial transaction is pending.
        issue = self._route_issue(task, retained, history) if trial is None else None
        if issue:
            raise ProposalError(issue)
        return task

    def _select_task(self, current, history, workload=None, trial=None, feedback=None):
        state = self._agent_state(current, history)
        eligible = self._eligible_intents(current)
        if trial is None and self._initial_diagnosis_needed(current, history):
            eligible = [intent for intent in eligible if intent != "STOP"]
        repair = self._budget_repair_request(current, history) if trial is None else None
        if repair:
            state["required_correction"] = repair
            eligible = [intent for intent in eligible if intent != "STOP"]
        trial_eligible = None
        if trial is not None:
            state.pop("allowed_focus_query_ids", None)
            state["trial_feedback"] = feedback
            compared = {row["query_id"]: row for row in
                        feedback["target_results"] + feedback["collateral_results"]}
            current_rows = [compared[qid]["before"] if qid in compared else
                            self.ledger.latest(current.configuration, qid, "configuration")
                            for qid in self.query_ids]
            trial_rows = [compared[qid]["after"] if qid in compared else row
                          for qid, row in zip(self.query_ids, current_rows)]
            current_total, trial_total = self._rows_total(current_rows), self._rows_total(trial_rows)
            query_gains = {}
            for qid, before, after in zip(self.query_ids, current_rows, trial_rows):
                before_ms, after_ms = self._rows_total([before]), self._rows_total([after])
                query_gains[qid] = (before_ms - after_ms
                                   if before_ms is not None and after_ms is not None else None)
            # Full-workload facts, using the same metric as the final result.
            # Only unaffected queries reuse CURRENT; failed measurements stay unknown.
            feedback["workload_comparison"] = {
                "query_count": len(self.query_ids),
                "affected_query_count": len(compared),
                "reused_unaffected_query_count": len(self.query_ids) - len(compared),
                "current_total_ms": current_total,
                "trial_total_ms": trial_total,
                "net_gain_ms": (current_total - trial_total
                                if current_total is not None and trial_total is not None else None),
                "net_gain_by_query_ms": query_gains,
            }
            trial_eligible = self._eligible_intents(trial, trial_rows)
            # This valid trial has been measured but is not in history until the
            # Router commits/rolls back. Its additions are already tested from CURRENT.
            state['global_evidence']['access_opportunities'] = self._access_opportunities(
                current, observations=current_rows, history=history,
                tested_additions=feedback.get('add_ids', feedback.get('add', [])))
            state['trial_access_opportunities'] = self._access_opportunities(
                trial, observations=trial_rows, history=history)
        state["eligible_intents_by_state"] = {"CURRENT": eligible}
        if trial_eligible is not None:
            state["eligible_intents_by_state"]["TRIAL"] = trial_eligible
        return self._ask("route", lambda reply: self._validate_route(
            reply, current, eligible, trial, trial_eligible, history), state,
            workload=workload or {},
            eligible_intents=list(dict.fromkeys(eligible + (trial_eligible or []))))

class _ProposalMixin:
    def _query_evidence(self, current):
        pairs = self._pairs(current.configuration)
        queries = compact_query_statuses(pairs, include_plan=True)
        queries.sort(key=lambda row: -(row.get("current_ms") or 0))
        return pairs, queries

    def _selected_evidence(self, current, pairs, queries):
        selected = []
        for key in current.configuration:
            spec = self.specs[key]
            observed = [pair["query_id"] for pair in pairs if key in
                        ((pair.get("current") or {}).get("plan_summary") or {}).get("indexes", [])]
            observed_ms = [row.get("current_ms") for row in queries if row.get("query_id") in observed]
            selected.append({
                "index_id": key, "table": spec.table, "keys": list(spec.keys), "include": list(spec.include),
                "size_bytes": spec.size_bytes,
                "relevant_query_ids": list(spec.benefit_queries),
                "observed_query_ids": observed,
                "observed_current_ms": (round(sum(value for value in observed_ms if value is not None), 3)
                                        if any(value is not None for value in observed_ms) else None),
            })
        return selected

    def _global_evidence(self, current, history):
        pairs, queries = self._query_evidence(current)
        selected = self._selected_evidence(current, pairs, queries)
        availability = self._access().availability(current.configuration)
        status = self._search_status(current, history)
        cooled = self._cooled_queries(current, history)
        for row in queries:
            row["last_search"] = status.get(row["query_id"], "NEW")
            row["unselected_count"] = availability["by_query"].get(row["query_id"], 0)
        opportunities = [row for row in queries if row["query_id"] not in cooled
                         and (row["unselected_count"] or row["current_status"] != "OK")]
        opportunities.sort(key=lambda row: row["current_status"] == "OK")
        plans = (opportunities or queries)[:1]
        remaining = [row for row in opportunities if row not in plans]
        fresh = [row for row in remaining if row["last_search"] == "NEW"]
        plans += (fresh or remaining)[:1]
        access = self._access_opportunities(current, history=history)
        # A rejected patch does not exhaust other candidates on the same scan.
        # Keep NOOP rotation; only untested direct matches override rejection cooling.
        fresh_access = [row for row in access
                        if any(not item['tested_in_patch'] for item in row['candidates'])
                        and (row['query_id'] not in cooled
                             or (status.get(row['query_id']) == 'REJECTED'
                                 and any(not item['tested_in_patch'] and item['predicate_prefix']
                                         for item in row['candidates'])))]
        if fresh_access:
            # Prefer actionable observed predicates; retain failure repair priority.
            ids = [row['query_id'] for row in queries if row['current_status'] != 'OK']
            ids += [row['query_id'] for row in fresh_access]
            ids += [row['query_id'] for row in plans]
            plans = [{'query_id': qid} for qid in dict.fromkeys(ids)][:2]
        return {
            "access_opportunities": access,
            "queries": queries,
            "current_total_ms": self._total_actual_ms(current.configuration),
            "slow_query_plans": self._focus_plan_evidence(
                current.configuration, [row["query_id"] for row in plans]),
            "query_tables": {qid: sorted(tables) if tables is not None else None
                             for qid, tables in self.query_dependencies.items()},
            "tables": self._access().tables,
            "selected_indexes": selected,
            "candidate_availability": availability,
            "coverage": self.ledger.coverage_summary(self.candidate_ids, self.query_ids),
        }

    def _access_opportunities(self, current, query_ids=None, candidate_ids=None, observations=None, history=(),
                              tested_additions=()):
        free = self.storage_budget_bytes - _configuration_size(current.configuration, self.specs)
        donors = sum(sorted((self.specs[key].size_bytes for key in current.configuration),
                            reverse=True)[:self.max_removals])
        if not self.max_additions:
            return []
        rows = (observations if observations is not None else
                [self.ledger.latest(current.configuration, qid, 'configuration')
                 for qid in (self.query_ids if query_ids is None else query_ids)])
        tested = {key for row in history if row.get('outcome') in {'ACCEPTED', 'REJECTED'}
                  and set(row.get('configuration', [])) == set(current.configuration)
                  for key in (row.get('patch') or {}).get('add', [])}
        tested.update(tested_additions)
        return access_opportunities(self.specs, current.configuration, rows, free, donors,
                                    candidate_ids, tested_ids=tested)

    def _agent_state(self, current, history, task=None, role="route", expansion=None):
        if role == "route":
            card = self._global_evidence(current, history)
        else:
            pairs, queries = self._query_evidence(current)
            # Local roles need candidate use, not Router availability/coverage or slow plans.
            card = {"selected_indexes": self._selected_evidence(current, pairs, queries)}
            if role == "partition":
                card["queries"] = queries
        state = {
            "state_version": current.version, "evidence_version": self.ledger.version,
            "configuration": list(current.configuration),
            "query_ids": self.query_ids, "global_evidence": card,
            "history": history[-6:], "task": task or {},
            "constraints": {
                "storage_budget_bytes": self.storage_budget_bytes,
                "storage_used_bytes": _configuration_size(current.configuration, self.specs),
                "storage_free_bytes": self.storage_budget_bytes - _configuration_size(current.configuration, self.specs),
                "max_add": self.max_additions, "max_remove": self.max_removals,
                "max_removable_bytes": sum(sorted((self.specs[key].size_bytes for key in current.configuration),
                                                   reverse=True)[:self.max_removals]),
                "remaining_seconds": (round(self._remaining_seconds(), 1)
                                      if math.isfinite(self._remaining_seconds()) else None),
                "remaining_physical_trials": max(0, min(self.max_iterations, self.max_physical_trials) - self.physical_trials),
            },
        }
        # Past query outcomes guide plan rotation, not a ban on a new hypothesis
        # or a demand to exhaust every structurally reachable query before STOP.
        state["allowed_focus_query_ids"] = list(self.query_ids)
        if role in {"route", "partition"}:
            state["initial_skipped_indexes"] = [
                {key: item[key] for key in ("index_id", "reason_code", "error") if key in item}
                for item in getattr(self, "initial_skipped_indexes", [])]
        if role == "partition":
            focus = self._focus_queries(task or {})
            if expansion:
                tables = list(expansion.get('expand_tables', []))
                families = set(expansion.get('expand_candidate_families', []))
                tables += [self.specs[key].table for key, family in self._access().families.items()
                           if family in families]
                # Broad candidate retrieval still uses a four-query SQL/plan focus.
                expanded = expansion.get('expand_query_ids', []) + self._focus_queries({'focus_tables': tables})
                focus = list(dict.fromkeys(expanded + focus))[:4]
                state['expansion_request'] = expansion
            state['access_opportunities'] = self._access_opportunities(current, focus, history=history)
            state["partitioner_evidence"] = self._focus_plan_evidence(
                current.configuration, focus)
            state["focus_query_ids"] = focus
        if role in {"partition", "build"}:
            repair = self._budget_repair_request(current, history)
            if repair:
                state["required_correction"] = repair
        return state

    def _validate_partition(self, reply, current, task, allowed_candidate_ids,
                            expansion_used=False, allowed_focus_query_ids=None, required_additions=()):
        if not isinstance(reply, dict):
            raise ProposalError("Partitioner reply must be an object")
        for key, expected in (("state_version", current.version), ("evidence_version", self.ledger.version)):
            if key in reply and reply[key] != expected:
                raise ProposalError("Partitioner refers to the wrong " + key)
        action = reply.get("action", "FINAL")
        if action == "EXPAND":
            if expansion_used:
                raise ProposalError("Only one structural expansion is allowed per round")
            request = {"action": "EXPAND"}
            for key, allowed in (("expand_query_ids", self.query_ids),
                                 ("expand_tables", self._access().tables),
                                 ("expand_candidate_families", self._access().candidate_families)):
                request[key] = _ids(reply.get(key, []), allowed, key)
                if len(request[key]) > 8:
                    raise ProposalError("Expansion selector exceeds limit")
            if not any(request[key] for key in request if key != "action"):
                raise ProposalError("Expansion requires a structural selector")
            additional = set(self._access().structural_ids(
                request["expand_query_ids"], request["expand_tables"], request["expand_candidate_families"]))
            if not additional - set(allowed_candidate_ids) - set(current.configuration):
                raise ProposalError("Expansion retrieves no unseen candidates; return FINAL or choose another region")
            request["reason"] = reply.get("reason", reply.get("region_reason", ""))
            return request
        if action != "FINAL":
            raise ProposalError("Partitioner action must be FINAL or EXPAND")
        selected = set(current.configuration)
        if "active_selected_ids" in reply or "active_unselected_ids" in reply:
            repair = _ids(reply.get("active_selected_ids"), selected.intersection(allowed_candidate_ids),
                          "active_selected_ids")
            candidates = _ids(reply.get("active_unselected_ids"), set(allowed_candidate_ids) - selected,
                              "active_unselected_ids")
            region = repair + candidates
        else:  # Read older stored proposals; every hard boundary remains enforced.
            region = _ids(reply.get("region"), allowed_candidate_ids, "region")
            repair = [key for key in region if key in selected]
            candidates = [key for key in region if key not in selected]
        if len(candidates) > self.local_candidate_limit:
            candidates = candidates[:self.local_candidate_limit]
        if not set(required_additions) <= set(candidates):
            raise ProposalError("Keep the pending oversized additions in the region for a replacement or explicit NOOP")
        region = repair + candidates
        query_ids = _ids(reply.get("focus_query_ids", reply.get("query_ids", task["focus_query_ids"])),
                         self.query_ids, "focus_query_ids")
        if len(query_ids) > 4:
            raise ProposalError("Partitioner focus exceeds limit")
        if not query_ids:
            query_ids = self._focus_queries(task)
            if allowed_focus_query_ids is not None:
                query_ids = [qid for qid in query_ids if qid in allowed_focus_query_ids]
        if allowed_focus_query_ids is not None and set(query_ids) - set(allowed_focus_query_ids):
            raise ProposalError("Partitioner must diagnose the new region, not return to rejected/NOOP queries")
        reason = reply.get("region_reason", reply.get("hypothesis"))
        if not isinstance(reason, str):
            raise ProposalError("Partitioner requires region_reason")
        return {**reply, "action": "FINAL", "state_version": current.version,
                "evidence_version": self.ledger.version, "region": region, "query_ids": query_ids,
                "focus_query_ids": query_ids,
                "hypothesis": reason, "active_selected_ids": repair,
                "active_unselected_ids": candidates,
                "repair_indexes": repair, "candidate_indexes": candidates}

    def _local_problem(self, current, partition, state):
        affected = self._affected_queries({"add": partition["candidate_indexes"],
                                           "remove": current.configuration})
        return {
            "access_opportunities": self._access_opportunities(current, partition['query_ids'], partition['candidate_indexes'],
                                                               history=state.get('history', [])),
            "budget_options": self._budget_options(current, partition["candidate_indexes"],
                                                    state.get("history", []), state.get("required_correction")),
            "fits_alone_ids": [key for key in partition["candidate_indexes"] if self.max_additions
                               and _configuration_size(current.configuration, self.specs)
                               + self.specs[key].size_bytes <= self.storage_budget_bytes],
            "affected_query_ids": affected,
            "query_tables": {qid: sorted(self.query_dependencies[qid])
                             if self.query_dependencies[qid] is not None else None for qid in affected},
            "affected_evidence": compact_query_statuses(
                self._pairs(current.configuration, affected), include_plan=True),
            "focus_plan_evidence": self._focus_plan_evidence(
                current.configuration, partition["query_ids"]),
        }

    def _budget_options(self, current, candidates, history=(), repair=None):
        """At most three capacity witnesses; performance is deliberately not inferred."""
        used = _configuration_size(current.configuration, self.specs)
        donors = sorted(current.configuration, key=lambda key: (-self.specs[key].size_bytes, key))
        capacity = sum(self.specs[key].size_bytes for key in donors[:self.max_removals])
        tried = set()
        for row in history:
            if (set(row.get("configuration", [])) == set(current.configuration)
                    and (row.get("ddl_outcome") or (row.get("feedback") or {}).get("storage")
                         or row.get("outcome") in {"ACCEPTED", "REJECTED"})):
                patch = row.get("patch") or {}
                tried.add(frozenset(set(current.configuration)
                                   - set(patch.get("remove", patch.get("remove_ids", [])))
                                   | set(patch.get("add", patch.get("add_ids", [])))))
        pending = set(((repair or {}).get("details") or {}).get("candidate_ids", []))
        options = []
        for count in range(1, min(self.max_additions, 2, len(candidates)) + 1):
            for add in combinations(candidates, count):
                added = _configuration_size(add, self.specs)
                needed = used + added - self.storage_budget_bytes
                if (needed <= 0 and set(add) != pending) or needed > capacity:
                    continue
                choices = [choice for count in range(1, min(self.max_removals, 2) + 1)
                           for choice in combinations(donors, count)
                           if _configuration_size(choice, self.specs) >= needed
                           and frozenset(set(current.configuration) - set(choice) | set(add)) not in tried]
                if not choices and self.max_removals > 2:
                    choice = tuple(donors[:self.max_removals])
                    if frozenset(set(current.configuration) - set(choice) | set(add)) not in tried:
                        choices = [choice]
                choices.sort(key=lambda choice: (len(choice), _configuration_size(choice, self.specs), choice))
                # Give each addition set one witness before showing alternate donors.
                for ordinal, removed in enumerate(
                        [choice for choice in choices if len(choice) == len(choices[0])][:2]):
                    options.append((ordinal, {"add_ids": list(add), "remove_ids": list(removed),
                                    "estimated_final_mb": (used + added - _configuration_size(removed, self.specs)) / MB}))
        options.sort(key=lambda item: (set(item[1]["add_ids"]) != pending if pending else False,
                                      item[0], len(item[1]["remove_ids"]), len(item[1]["add_ids"])))
        return [row for _, row in options[:3]]

    def _validate_move(self, reply, current, partition, mode, tried=(), required_additions=()):
        if not isinstance(reply, dict):
            raise ProposalError("Builder reply must be an object")
        add = _ids(reply.get("add_ids", reply.get("add", [])), partition["candidate_indexes"], "add_ids")
        remove = _ids(reply.get("remove_ids", reply.get("remove", [])), current.configuration, "remove_ids")
        if set(add).intersection(current.configuration) or not set(remove) <= set(current.configuration):
            raise ProposalError("Patch violates the incumbent binary assignment")
        if not set(add + remove) <= set(self.candidate_ids) or set(add) & set(remove):
            raise ProposalError("Patch references non-pool or overlapping IDs")
        if len(add) > self.max_additions or len(remove) > self.max_removals:
            raise ProposalError("Builder patch exceeds configured limits")
        if reply.get("expand_indexes"):
            raise ProposalError("Expansion belongs to Partitioner, before Builder")
        action = reply.get("action", "PATCH" if add or remove else "NOOP")
        if action not in {"PATCH", "NOOP"} or (action == "NOOP" and (add or remove)):
            raise ProposalError("Builder action conflicts with the patch")
        if action == "PATCH" and not add and not remove:
            action = "NOOP"
        if action == "PATCH" and required_additions and set(add) != set(required_additions):
            raise ProposalError("Repair the same oversized addition set by changing removals, or return an explicit NOOP")
        probes = _ids(reply.get("expected_improvements", reply.get("probe_query_ids", [])),
                      self.query_ids, "expected_improvements")
        risks = _ids(reply.get("risk_query_ids", []), self.query_ids, "risk_query_ids")
        if action == 'PATCH' and 'expected_improvements' in reply and hasattr(self, 'query_dependencies'):
            unrelated = set(probes) - set(self._affected_queries({'add': add, 'remove': remove}))
            if unrelated:
                raise ProposalError('Claimed improvements are outside the patch table dependencies: '
                                    + ', '.join(sorted(unrelated)), failure_category='UNRELATED_IMPROVEMENT')
        if not isinstance(reply.get("reason"), str):
            raise ProposalError("Builder requires a concise reason describing the patch hypothesis")
        configuration = self._ordered((set(current.configuration) - set(remove)) | set(add))
        estimated_final_bytes = _configuration_size(configuration, self.specs)
        if action != "NOOP" and estimated_final_bytes > self.storage_budget_bytes:
            raise ProposalError("Final estimated configuration exceeds storage budget",
                                failure_category="BUDGET_CONFLICT",
                                estimated_final_mb=round(estimated_final_bytes / MB, 3),
                                budget_mb=round(self.storage_budget_bytes / MB, 3),
                                excess_mb=round((estimated_final_bytes - self.storage_budget_bytes) / MB, 3))
        if action != "NOOP" and (frozenset(current.configuration), frozenset(configuration)) in tried:
            raise ProposalError("This configuration was already physically tried from CURRENT; choose a distinct patch or NOOP",
                                failure_category="REPEATED_CONFIGURATION")
        return configuration, {**reply, "action": action, "add": add, "remove": remove,
                               "add_ids": add, "remove_ids": remove,
                               "probe_query_ids": probes, "risk_query_ids": risks}

class _TrialMixin:
    def _record(self, history, record):
        outcome = record.get("outcome")
        categories = {"EMPTY_REGION": "INSUFFICIENT_REGION", "NO_QUERY_CANDIDATES": "INSUFFICIENT_REGION",
                      "NO_EXECUTABLE_PATCH": "NO_USEFUL_PATCH", "NO_NEW_CONFIGURATION": "NO_USEFUL_PATCH",
                      "PROTOCOL_ERROR": "INVALID_PROPOSAL", "DDL_OR_ENVIRONMENT_ERROR": "DDL_FAILURE",
                      "STORAGE_BUDGET_EXCEEDED": "BUDGET_CONFLICT"}
        feedback = record.get("feedback") or {}
        reason = record.get("reason_code", feedback.get("reason_code", outcome))
        if "failure_category" not in record and outcome not in {"ACCEPTED", "REJECTED", "STOP"}:
            record["failure_category"] = categories.get(reason, categories.get(outcome, reason or outcome))
        self.ledger.record_feedback(record)
        history.append(record)

    def _ask(self, stage, validate, state, **kwargs):
        for attempt in range(2):
            # SEARCH must leave room for downstream roles and two Router review
            # attempts (one answer plus one correction) before starting any DDL.
            reserve = (0 if stage == "initialize" or state.get("trial_feedback")
                       else {"route": 4, "partition": 3, "build": 2}[stage])
            if self._remaining_llm_calls() < 1 + reserve:
                raise LLMCallBudgetExhausted("Insufficient LLM calls for " + stage + " and trial review")
            reply = None
            try:
                started = time.monotonic()
                if self._remaining_seconds() <= 0:
                    raise TimeoutError("Tuning budget exhausted before " + stage)
                try:
                    reply = getattr(self.agents, stage)(state=state, **kwargs)
                finally:
                    self.timing["llm"] += (time.monotonic() - started) * 1000
                return validate(reply)
            except ValueError as exc:
                reply = getattr(exc, "proposal", reply)
                LOG.warning("STAGE3 invalid %s: %s reply=%s", stage, exc,
                            json.dumps(reply, ensure_ascii=False, default=str)[:500])
                if attempt:
                    raise
                previous = state.get("required_correction")
                state = {**state, "required_correction": {
                    "stage": stage, "error": str(exc), "proposal": reply,
                    "details": getattr(exc, "details", {}),
                    **({"pending_task": previous.get("pending_task", previous)} if previous else {}),
                }}

    def _noop_fingerprint(self, current, partition, mode, repair):
        # Feedback-only ledger increments and Router labels are not new evidence.
        # Changed SQL focus, mechanism, candidates, observations or a physical
        # attempt must still allow a fresh diagnosis.
        context = [current.version, sorted(current.configuration), mode,
                   sorted(partition["query_ids"]), sorted(partition["candidate_indexes"]),
                   sorted(partition["repair_indexes"]), partition["hypothesis"],
                   self.physical_trials, self.storage_budget_bytes, self.max_additions, self.max_removals,
                   (repair or {}).get("details"),
                   [(self.ledger.latest(current.configuration, qid, "configuration") or {}).get("evidence_id")
                    for qid in self.query_ids]]
        return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()

    def _affected_queries(self, move):
        changed_tables = {self.specs[index_id].table
                          for index_id in move["add"] + move["remove"]}
        return [qid for qid in self.query_ids
                if self.query_dependencies[qid] is None
                or any(table_matches(table, changed) for table in self.query_dependencies[qid]
                       for changed in changed_tables)]

    def _rows_total(self, rows):
        rows = list(rows)
        times = [_finite_ms((row or {}).get("actual_total_ms")) for row in rows]
        if (any(not row or row.get("status") != "OK" for row in rows)
                or any(value is None for value in times)):
            return None
        return sum(times)

    def _trial_validity(self, affected, current_rows, trial_rows, trial_bytes, stopped_reason=None):
        """Only physical validity gates; performance belongs exclusively to Router."""
        if trial_bytes > self.storage_budget_bytes or stopped_reason == "STORAGE_BUDGET_EXCEEDED":
            return False, "OVER_BUDGET"
        if any(row.get("status") == "TIMEOUT" for row in trial_rows):
            return False, "TIMEOUT"
        if any(row.get("status") in {"ERROR", "EXECUTION_FAILURE", "ENVIRONMENT_ERROR"} for row in trial_rows):
            return False, "EXECUTION_FAILURE"
        for rows in (current_rows, trial_rows):
            if len(rows) != len(affected) or {row.get("query_id") for row in rows} != set(affected):
                return False, "MISSING_MEASUREMENT"
        if any(row.get("status") != "OK" or _finite_ms(row.get("actual_total_ms")) is None
               for row in trial_rows):
            return False, "MISSING_MEASUREMENT"
        # CURRENT may contain a recorded timeout/error: Router can select a valid repair.
        if any(row.get("status") not in {"OK", "TIMEOUT", "ERROR", "ENVIRONMENT_ERROR"}
               or (row.get("status") == "OK" and _finite_ms(row.get("actual_total_ms")) is None)
               for row in current_rows):
            return False, "MISSING_MEASUREMENT"
        if stopped_reason:
            return False, (stopped_reason if stopped_reason in {"DDL_FAILURE", "EXECUTION_FAILURE"}
                           else "MISSING_MEASUREMENT")
        return True, None

    def _total_actual_ms(self, configuration):
        rows = self.ledger.for_configuration(configuration, "configuration")
        if any(qid not in rows for qid in self.query_ids):
            return None
        return self._rows_total(rows[qid] for qid in self.query_ids)

    def _initialize_search(self):
        """Capture the baseline catalog, build S0 and record its measurements."""
        self.started = time.monotonic()
        self.evaluator.deadline = (
            float("inf") if self.total_tuning_time_seconds is None
            else self.started + self.total_tuning_time_seconds
        )
        self.evaluator.statement_timeout_ms = self.statement_timeout_ms
        self.agents.deadline = self.evaluator.deadline
        if self.statement_timeout_ms == 0:
            self.evaluator.ddl_timeout_ms = 0
        initialization_started = self.started
        self._emit("start", queries=len(self.evaluator.queries),
                   candidates=len(self.candidate_ids),
                   budget_s=self.total_tuning_time_seconds or "unlimited",
                   statement_timeout_ms=self.statement_timeout_ms,
                   max_iterations=self.max_iterations,
                   max_llm_calls=self.max_llm_calls)
        baseline_configuration = self.evaluator.capture_baseline_configuration()
        self.query_ids = [query["qid"] for query in self.evaluator.queries]
        self.query_dependencies = self._query_dependencies()
        workload = self._workload()
        workload["existing_indexes"] = baseline_configuration["existing_indexes"]
        self._emit("baseline_configuration_recorded",
                   identity=baseline_configuration["configuration_identity"])

        self._emit("llm_initialization_start", queries=len(self.query_ids),
                   candidates=len(self.candidate_ids))
        initial, initial_recommendation = self.initial_configuration(workload)
        self.timing["initialization"] += (time.monotonic() - initialization_started) * 1000
        self._emit("llm_initialization_done", s0=list(initial.configuration),
                   priority=initial_recommendation.get("priority"),
                   skipped_by_estimate=len(initial_recommendation.get("skipped_by_estimate") or []),
                   estimated_mb=initial_recommendation.get("estimated_index_mb"),
                   init_ms=round(self.timing["initialization"], 1))

        materialized = self.evaluator.materialize(
            initial.configuration, self.storage_budget_bytes)
        self.timing["ddl"] += materialized.get("ddl_ms", 0)
        initial.configuration = list(materialized.get("configuration", []))
        initial.actual_index_bytes = int(materialized.get("actual_index_bytes", 0))
        self.initial_skipped_indexes = list(initial_recommendation.get("skipped_by_estimate") or [])
        self.initial_skipped_indexes.extend(materialized.get("skipped_indexes") or [])
        initial.move.update(
            add=list(initial.configuration),
            skipped_indexes=self.initial_skipped_indexes,
        )
        initial_recommendation.update(
            materialized_configuration=list(initial.configuration),
            skipped_indexes=self.initial_skipped_indexes,
            actual_index_bytes=initial.actual_index_bytes,
        )
        current = initial
        self._emit("s0_materialized", s0=list(current.configuration),
                   skipped=self.initial_skipped_indexes,
                   bytes=current.actual_index_bytes,
                   ddl_ms=materialized.get("ddl_ms", 0))
        self.ledger.add(self._fill_current(current))
        initial_latency = self._total_actual_ms(current.configuration)
        self._emit("initial_observe", current=list(current.configuration),
                   complete=initial_latency is not None)
        return baseline_configuration, workload, initial, initial_recommendation, current, initial_latency

    def _finalize_result(self, current, initial, initial_recommendation, initial_latency,
                         baseline_configuration, states, history, accepted_trials, stop_reason):
        """Summarize the retained state without making another search decision."""
        observations = self.ledger.for_configuration(current.configuration, "configuration")
        unresolved = [qid for qid in self.query_ids
                      if observations.get(qid, {}).get("status") != "OK"
                      or _finite_ms(observations.get(qid, {}).get("actual_total_ms")) is None]
        final_latency = self._total_actual_ms(current.configuration)
        self._emit("done", stop_reason=stop_reason,
                   current=list(current.configuration),
                   accepted_trials=accepted_trials,
                   unresolved=len(unresolved),
                   latency=final_latency,
                   trial_log=history)
        return {
            "architecture": "router-owned-incumbent-real-trials",
            "current_best": {**_persist_state(current),
                             "observations": list(observations.values())},
            "final_current_state": _persist_state(current),
            "coverage": self.ledger.coverage_summary(self.candidate_ids, self.query_ids),
            "candidate_coverage": self.ledger.candidate_coverage,
            "query_coverage": self.ledger.query_coverage,
            "evidence_version": self.ledger.version,
            "physical_trials": self.physical_trials,
            "routing_attempts": len(history),
            "accepted_trials": accepted_trials,
            "initial_total_latency_ms": initial_latency,
            "improvement_beyond_s0_ms": (0.0 if initial.configuration == current.configuration and initial_latency is not None
                else initial_latency - final_latency
                if initial_latency is not None and final_latency is not None else None),
            "accepted_trial_rate": accepted_trials / self.physical_trials if self.physical_trials else 0,
            "patch_counts": {kind: sum(1 for record in history if record.get("ddl_outcome") and
                (bool((record.get("patch") or {}).get("add")), bool((record.get("patch") or {}).get("remove"))) == flags)
                for kind, flags in (("replacement", (True, True)), ("addition", (True, False)), ("deletion", (False, True)))},
            "llm_usage": getattr(self.agents, "usage", {}),
            "max_llm_calls": self.max_llm_calls,
            "llm_decisions": getattr(self.agents, "decisions", []),
            "end_to_end_ms": round((time.monotonic() - self.started) * 1000, 3),
            "run_id": getattr(self.evaluator, "_run_suffix", None),
            "data_version": hashlib.sha256(
                json.dumps({"features": self.features, "queries": self.evaluator.queries},
                           sort_keys=True, default=str).encode()
            ).hexdigest()[:16],
            "baseline_configuration": baseline_configuration,
            "baseline_configuration_identity": baseline_configuration["configuration_identity"],
            "initial_configuration": list(initial.configuration),
            "initial_recommendation": initial_recommendation,
            "final_configuration": list(current.configuration),
            "validation_status": ("INCOMPLETE" if unresolved else "VERIFIED_INCREMENTAL"),
            "per_query_observations": self.ledger.observations,
            "final_total_latency_ms": final_latency,
            "actual_index_bytes": current.actual_index_bytes,
            "timing_breakdown_ms": {key: round(value, 3) for key, value in self.timing.items()},
            "unresolved_queries": unresolved,
            "trial_log": history,
            "states": states,
            "stop_reason": stop_reason,
        }

    def run(self):
        """Initialize S0, run bounded transactional revisions, then report the incumbent."""
        (baseline_configuration, workload, initial, initial_recommendation,
         current, initial_latency) = self._initialize_search()

        states = [_persist_state(current)]
        history, tried = [], set()
        accepted_trials = 0
        next_task = None
        stop_reason = "routing_budget_exhausted"

        # Empty/duplicate proposals may reroute without using a physical round.
        # Bound that extra work separately to avoid an unbounded LLM-only loop.
        for turn in range(2 * self.max_iterations):
            if self._remaining_seconds() <= 0:
                stop_reason = "time_budget_exhausted"
                break
            if self.physical_trials >= min(self.max_iterations, self.max_physical_trials):
                stop_reason = "physical_trial_budget_exhausted"
                break
            if self._remaining_llm_calls() < (5 if next_task is None else 4):
                stop_reason = "llm_call_budget_exhausted"
                break
            record = {"round": turn, "state_version": current.version,
                      "configuration_identity": configuration_identity(current.configuration),
                      "configuration": list(current.configuration), "evidence_version": self.ledger.version}
            repair = self._budget_repair_request(current, history)
            required_additions = repair["details"]["candidate_ids"] if repair else []
            if repair:
                record["budget_repair"] = repair["details"]
            try:
                task = next_task
                if task is None or self._route_issue(task, current, history):
                    task = self._select_task(current, history, workload)
                next_task = None
                record["task"] = task
                self._emit("task", turn=turn, mode=task["mode"],
                           strategy=task["search_strategy"], focus=task["focus_query_ids"],
                           reason=task["reason"], current=list(current.configuration))
                if task["mode"] == "STOP":
                    record["outcome"] = "STOP"
                    self._record(history, record)
                    stop_reason = "router_judgment"
                    break
                self.ledger.mark_coverage("seen_count", query_ids=task["focus_query_ids"])
                filled = self._fill_current(current, task["focus_query_ids"])
                if filled:
                    self.ledger.add(filled)
                    task["evidence_version"] = self.ledger.version
                hints = self._access_opportunities(current, self._focus_queries(task), history=history)
                preferred = [item['index_id'] for row in hints for item in row['candidates']]
                view = self._access().build(task, current.configuration, required_ids=required_additions,
                                            preferred_ids=preferred)
                record["candidate_view"] = deepcopy(view["coverage"])
                partition = None
                for pass_number in range(2):
                    state = self._agent_state(current, history, task, role="partition",
                                              expansion=record.get('expansion_request') if pass_number else None)
                    state["candidate_view"] = {key: value for key, value in view.items() if key != "candidates"}
                    state["expansion_used"] = bool(pass_number)
                    candidates = view["candidates"]
                    partition = self._ask(
                        "partition", lambda reply: self._validate_partition(
                            reply, current, task, candidates, expansion_used=bool(pass_number),
                            allowed_focus_query_ids=state["allowed_focus_query_ids"],
                            required_additions=required_additions), state,
                        task=task, workload=workload, candidates=candidates,
                        active_candidate_limit=self.local_candidate_limit)
                    if partition["action"] == "FINAL":
                        break
                    record["expansion_request"] = partition
                    view = self._access().build(task, current.configuration,
                                                previous=view, expansion=partition)
                    record["expanded_candidate_view"] = deepcopy(view["coverage"])
                record["candidate_scope"] = {"query_ids": task["focus_query_ids"],
                    "eligible_index_ids": list(candidates)}
                record["partition"] = partition
                if not partition["region"]:
                    record["outcome"] = "EMPTY_REGION"
                    self._record(history, record)
                    continue
                self.ledger.mark_coverage("partitioned_count", partition["region"], partition["query_ids"])
                state = self._agent_state(current, history, task, role="build")
                state["local_problem"] = self._local_problem(current, partition, state)
                fingerprint = self._noop_fingerprint(current, partition, task["mode"], repair)
                record["noop_fingerprint"] = fingerprint
                previous = next((row for row in reversed(history)
                                 if row.get("outcome") == "NO_EXECUTABLE_PATCH"
                                 and row.get("noop_fingerprint") == fingerprint), None)
                if previous is not None:
                    trial_configuration, move = list(current.configuration), deepcopy(previous["patch"])
                    record["reused_noop"] = True
                else:
                    trial_configuration, move = self._ask(
                        "build", lambda reply: self._validate_move(reply, current, partition, task["mode"], tried,
                                                                   required_additions),
                        state, partition=partition, workload=workload,
                        candidates={key: candidates[key] for key in
                                    dict.fromkeys(current.configuration + partition["candidate_indexes"])})
            except LLMCallBudgetExhausted as exc:
                record.update(outcome="STOP", reason_code="LLM_CALL_BUDGET_EXHAUSTED", reason=str(exc))
                self._record(history, record)
                stop_reason = "llm_call_budget_exhausted"
                break
            except ValueError as exc:
                category = getattr(exc, "details", {}).get("failure_category", "INVALID_PROPOSAL")
                record.update(outcome="NO_NEW_CONFIGURATION" if category == "REPEATED_CONFIGURATION" else "PROTOCOL_ERROR",
                              reason=str(exc), failure_category=category)
                self._record(history, record)
                continue
            except Exception as exc:
                record.update(outcome="INCONCLUSIVE", reason=str(exc)[:240],
                              failure_category="AGENT_FAILURE")
                self._record(history, record)
                stop_reason = "time_budget_exhausted" if isinstance(exc, TimeoutError) else "agent_failure"
                break
            record["patch"] = move
            record["patch_indexes"] = {
                key: {"table": self.specs[key].table, "keys": list(self.specs[key].keys),
                      "include": list(self.specs[key].include)}
                for key in dict.fromkeys(move["add"] + move["remove"])}
            self._emit("patch", turn=turn, mode=task["mode"],
                       region=partition.get("region"),
                       add=move.get("add"), remove=move.get("remove"),
                       probe=move.get("probe_query_ids"),
                       hypothesis=partition.get("hypothesis"),
                       reason=move.get("reason"))
            if not move["add"] and not move["remove"]:
                record.update(outcome="NO_EXECUTABLE_PATCH")
                self._record(history, record)
                self._emit("skip_patch", turn=turn, outcome=record["outcome"])
                continue
            transition = (frozenset(current.configuration), frozenset(trial_configuration))
            if transition[0] == transition[1] or transition in tried:
                record["outcome"] = "NO_NEW_CONFIGURATION"
                self._record(history, record)
                self._emit("skip_patch", turn=turn, outcome=record["outcome"])
                continue
            affected = self._affected_queries(move)
            order = list(dict.fromkeys(move["probe_query_ids"] + move.get("risk_query_ids", []) + partition["query_ids"]
                                       + task["focus_query_ids"] + affected))
            order = [qid for qid in order if qid in affected]
            current_rows = []
            newly_measured_current = []
            previous_configuration = list(current.configuration)
            for qid in order:
                if self._remaining_seconds() <= 0:
                    break
                row = self.ledger.latest(current.configuration, qid, "configuration")
                if row is None:
                    measured_started = time.monotonic()
                    measured = self.evaluator.measure(
                        current.configuration, [qid], self._measurement_timeout_ms())
                    self.timing["execution"] += (time.monotonic() - measured_started) * 1000
                    current_rows.extend(measured)
                    newly_measured_current.extend(measured)
                else:
                    current_rows.append(row)
            trial_timeout_ms = self._measurement_timeout_ms()
            if self._remaining_seconds() <= 0:
                stop_reason = "time_budget_exhausted"
                break
            tried.add(transition)
            self.physical_trials += 1
            self.ledger.mark_coverage("tested_count", move["add"] + move["remove"], affected)
            try:
                handle = self.evaluator.begin_trial(
                    current.configuration, trial_configuration, order, trial_timeout_ms,
                    storage_budget_bytes=self.storage_budget_bytes)
            except Exception as exc:
                added = self.ledger.add(newly_measured_current)
                record.update(outcome="INCONCLUSIVE", reason_code="DDL_FAILURE",
                              ddl_outcome="FAILED", affected_query_ids=affected,
                              error=str(exc)[:240],
                              evidence_ids=[item["evidence_id"] for item in added])
                self._record(history, record)
                continue
            try:
                trial_rows = handle["observations"]
                self.timing["ddl"] += handle.get("ddl_ms", 0)
                self.timing["execution"] += sum(row.get("waited_ms", 0) for row in trial_rows)
                trial_bytes = int(handle.get("actual_index_bytes", self.storage_budget_bytes + 1))
                valid, reason_code = self._trial_validity(
                    affected, current_rows, trial_rows, trial_bytes, handle.get("stopped_reason"))
                target = set(partition["query_ids"])
                current_by = {row["query_id"]: row for row in current_rows}
                trial_by = {row["query_id"]: row for row in trial_rows}
                comparisons = []
                for qid in affected:
                    before, after = current_by.get(qid, {}), trial_by.get(qid, {})
                    accesses = focus_accesses(before, after)
                    comparisons.append({"query_id": qid,
                        "before": summarize_observation(before, include_plan_nodes=True, preferred_accesses=accesses),
                        "after": summarize_observation(after, include_plan_nodes=True, preferred_accesses=accesses)})
                feedback = trial_feedback(
                    "trial_%d" % (turn + 1), task["mode"], current.version, move,
                    "PENDING" if valid else "INCONCLUSIVE", reason_code,
                    [row for row in comparisons if row["query_id"] in target],
                    [row for row in comparisons if row["query_id"] not in target],
                    [qid for qid in affected if qid not in trial_by],
                    {"ddl": handle.get("ddl_ms", 0),
                     "execution": sum(row.get("waited_ms", 0) for row in trial_rows)})
                feedback.update(patch_indexes=record["patch_indexes"],
                                reason=move["reason"],
                                expected_improvements=move["probe_query_ids"],
                                risk_query_ids=move["risk_query_ids"],
                                current_configuration=list(current.configuration),
                                trial_configuration=list(trial_configuration),
                                storage={"current_bytes": current.actual_index_bytes,
                                         "trial_bytes": trial_bytes,
                                         "budget_bytes": self.storage_budget_bytes})
            except BaseException:
                self.evaluator.finish_trial(handle, False)
                raise
            accepted = False
            review_error = None
            try:
                if valid:
                    trial = SearchState(current.version + 1, trial_configuration,
                                        actual_index_bytes=trial_bytes, move=move)
                    self._emit("trial_review", turn=turn, affected=len(affected))
                    next_task = self._select_task(current, history, workload, trial, feedback)
                    accepted = next_task["state_choice"] == "TRIAL"
                    reason_code = "ROUTER_" + next_task["state_choice"]
                    record["router_decision"] = next_task
            except Exception as exc:
                review_error = exc
                reason_code = "ROUTER_FAILURE"
                record["error"] = str(exc)[:240]
            finally:
                # Any failed/invalid Router response leaves the transaction rolled back.
                try:
                    self.evaluator.finish_trial(handle, accepted)
                except Exception as exc:
                    record.update(outcome="INCONCLUSIVE", reason_code="COMMIT_FAILURE", error=str(exc)[:240])
                    self._record(history, record)
                    raise RuntimeError("Trial completion failed; physical state requires recovery") from exc
            feedback.update(outcome=("ACCEPTED" if accepted else
                                    "REJECTED" if valid and review_error is None else "INCONCLUSIVE"),
                            reason_code=reason_code,
                            state_choice="TRIAL" if accepted else "CURRENT")
            reused = []
            if accepted:
                new_identity = configuration_identity(trial_configuration)
                for qid in self.query_ids:
                    if qid in affected:
                        continue
                    previous = self.ledger.latest(
                        previous_configuration, qid, "configuration")
                    if previous:
                        reused.append({**previous, "configuration_identity": new_identity,
                                       "configuration_role": "current",
                                       "source": "REUSED_UNCHANGED_DEPENDENCIES",
                                       "reused_from_configuration_identity": previous["configuration_identity"],
                                       "committed": True, "state_version": current.version + 1})
                current = SearchState(current.version + 1, trial_configuration,
                                      actual_index_bytes=trial_bytes, move=move)
                accepted_trials += 1
                self.ledger.mark_coverage("accepted_count", move["add"] + move["remove"], affected)
                states.append(_persist_state(current))
            observed = []
            for row in newly_measured_current + trial_rows:
                trial_row = row.get("configuration_role") == "trial"
                observed.append({**row, "committed": accepted if trial_row else True,
                                 "state_version": current.version if trial_row and accepted else record["state_version"]})
            added_evidence = self.ledger.add(observed + reused)
            record.update(outcome=feedback["outcome"], feedback=feedback,
                          state_choice=feedback["state_choice"], reason_code=reason_code,
                          ddl_outcome="COMMITTED" if accepted else "ROLLED_BACK",
                          affected_query_ids=affected,
                          storage_delta_bytes=trial_bytes - feedback["storage"]["current_bytes"],
                          evidence_ids=[item["evidence_id"] for item in added_evidence])
            self._record(history, record)
            if next_task is not None:
                next_task["evidence_version"] = self.ledger.version
            self._emit("trial", turn=turn, mode=task["mode"],
                       outcome=feedback["outcome"], reason_code=reason_code,
                       current=list(current.configuration), accepted_trials=accepted_trials)
            if review_error is not None:
                stop_reason = ("time_budget_exhausted" if isinstance(review_error, TimeoutError)
                               else "agent_failure")
                break
            if (next_task is not None and next_task["mode"] == "STOP"
                    and not self._route_issue(next_task, current, history)):
                stop_reason = "router_judgment"
                break

        return self._finalize_result(
            current, initial, initial_recommendation, initial_latency,
            baseline_configuration, states, history, accepted_trials, stop_reason)


class CombinationSearch(_RouteMixin, _ProposalMixin, _TrialMixin):
    """Router/Partitioner/Builder neighborhood search certified by physical trials."""
    def __init__(self, candidate_ids, specs, features, storage_budget_bytes, agents,
                 evaluator, max_iterations=3, local_candidate_limit=12,
                 total_tuning_time_seconds=0, statement_timeout_seconds=600,
                 status_path=None, candidate_view_limit=64, max_physical_trials=None,
                 max_additions=MAX_ADDITIONS_PER_ROUND,
                 max_removals=MAX_REMOVALS_PER_ROUND,
                 retry_over_budget_addition=True, max_llm_calls=16):
        self.specs = specs
        self.candidate_ids = sorted(candidate_ids, key=lambda index_id: (
            specs[index_id].table, specs[index_id].keys, specs[index_id].include, index_id
        ))
        self.features = features
        self.storage_budget_bytes = max(0, int(storage_budget_bytes))
        self.agents = agents
        self.evaluator = evaluator
        self.max_iterations = max(0, int(max_iterations))
        self.local_candidate_limit = max(1, int(local_candidate_limit))
        self.candidate_view_limit = max(1, int(candidate_view_limit))
        self.max_physical_trials = (self.max_iterations if max_physical_trials is None
                                    else max(0, int(max_physical_trials)))
        self.max_additions = max(0, int(max_additions))
        self.max_removals = max(0, int(max_removals))
        self.retry_over_budget_addition = bool(retry_over_budget_addition)
        self.max_llm_calls = max(0, int(max_llm_calls))
        self.physical_trials = 0
        total_limit = float(total_tuning_time_seconds)
        statement_limit = float(statement_timeout_seconds)
        # Zero is the conventional PostgreSQL/CLI spelling for "no timeout".
        self.total_tuning_time_seconds = None if total_limit <= 0 else total_limit
        self.statement_timeout_ms = 0 if statement_limit <= 0 else int(statement_limit * 1000)
        self.status_path = status_path
        self.catalog = {index_id: specs[index_id].prompt_dict(specs[index_id].benefit_queries)
                        for index_id in self.candidate_ids}
        self.ledger = EvidenceLedger()
        self.timing = {key: 0.0 for key in
                       ("initialization", "ddl", "execution", "llm")}
