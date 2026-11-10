"""Post-search timing. This is not a fourth stage.

After Stage 3: drop public secondary indexes, warm once, time B, then time ours.

Search observations are not reused. B is the catalog left after every
non-primary index in public has been dropped. One invocation measures that
baseline once, then builds and times each saved final configuration.
"""

import argparse
import json
import logging
import os
import time

from psycopg2 import sql

from core.common import cfg_get, load_config, load_json, normalize_candidate_pool, resolve_path
from core.stage3 import build_index_specs, storage_budget
from core.stage3_execute import PhysicalEvaluator, load_workload_queries


CONFIG = load_config()
PROTOCOL = "warm_once_then_B_then_ours"


def _cfg(*keys, fallback=""):
    return cfg_get(CONFIG, *keys, default=fallback)


def defaults():
    return {
        "workload": _cfg("workload", "workload_file", fallback="../test/{dbname}_test.sql"),
        "features": _cfg(
            "workload",
            "output_file",
            fallback="./feature/{dbname}_workload_features.json",
        ),
        "candidates": _cfg(
            "index_selector",
            "candidate_file",
            fallback="./candidate/{dbname}_candidate_indexes.json",
        ),
        "host": _cfg("database", "host", fallback=""),
        "port": _cfg("database", "port", fallback=5432),
        "user": _cfg("database", "user", fallback="postgres"),
        "password": _cfg("database", "password", fallback=""),
        "dbname": _cfg("database", "dbname", fallback="") or "SSB",
        "result": _cfg(
            "index_combination",
            "result_file",
            fallback="./results/{dbname}_index_combination.json",
        ),
        "index_storage_proportion": _cfg(
            "index_selector", "index_storage_proportion", fallback=0.5
        ),
        "statement_timeout_seconds": _cfg(
            "index_combination", "statement_timeout_seconds", fallback=600
        ),
    }


def public_secondary_indexes(evaluator):
    """Non-primary indexes in public. PK-only catalogs return []."""
    conn = evaluator._connect()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT c.relname
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND NOT i.indisprimary
            ORDER BY c.relname
        """)
        return [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()


def drop_non_pk_indexes(evaluator):
    """Drop every public secondary index, including leftovers from other runs."""
    conn = evaluator._connect()
    cursor = conn.cursor()
    dropped = []
    try:
        for name in public_secondary_indexes(evaluator):
            cursor.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(
                sql.Identifier(name)))
            dropped.append(name)
        return dropped
    finally:
        cursor.close()
        conn.close()


def _path(value, default_name, dbname):
    return resolve_path(value, default_name, dbname=dbname)


def final_configuration(payload):
    configuration = payload.get("final_configuration")
    if configuration is None:
        configuration = (payload.get("current_best") or {}).get("configuration") or []
    return list(configuration)


def output_path(result_path, explicit):
    if explicit:
        return explicit
    stem, ext = os.path.splitext(result_path)
    if ext.lower() != ".json":
        stem = result_path
    return stem + ".latency.json"


def _dump(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def summarize(observations):
    rows = []
    for item in observations:
        row = {
            "query_id": item.get("query_id"),
            "status": item.get("status"),
            "actual_total_ms": item.get("actual_total_ms"),
        }
        if item.get("reason"):
            row["reason"] = item["reason"]
        rows.append(row)
    incomplete = [
        row["query_id"]
        for row in rows
        if row["status"] != "OK" or row["actual_total_ms"] is None
    ]
    complete_ms = sum(
        row["actual_total_ms"]
        for row in rows
        if row["status"] == "OK" and row["actual_total_ms"] is not None
    )
    complete = not incomplete
    return {
        "total_ms": complete_ms if complete else None,
        "complete_ms": complete_ms,
        "complete": complete,
        "incomplete": incomplete,
        "queries": rows,
    }


def reduction_pct(baseline_ms, ours_ms):
    if baseline_ms in (None, 0) or ours_ms is None:
        return None
    return (baseline_ms - ours_ms) / baseline_ms * 100.0


def _measure(evaluator, query_ids, configuration, timeout_ms):
    started = time.monotonic()
    observations = evaluator.measure(configuration, query_ids, timeout_ms)
    wall_ms = round((time.monotonic() - started) * 1000, 3)
    return summarize(observations), wall_ms


def same_proportion(left, right):
    return abs(float(left) - float(right)) <= 1e-9


def storage_constraint(features, proportion):
    """Byte budget recorded when Stage 1 extracted features for this proportion."""
    storage = features.get("storage") or {}
    recorded = storage.get("index_storage_proportion")
    if recorded is None or not same_proportion(recorded, proportion):
        raise ValueError(
            "feature storage proportion is %s, config index_storage_proportion is %s"
            % (recorded, proportion)
        )
    budget = storage_budget(features)
    return storage, budget


def assert_result_constraint(payload, storage, source):
    recorded = payload.get("storage") or {}
    if not recorded:
        return
    if recorded.get("storage_constraint_mb") != storage.get("storage_constraint_mb"):
        raise ValueError(
            "%s was searched under %s MB, features constrain %s MB"
            % (source, recorded.get("storage_constraint_mb"), storage.get("storage_constraint_mb"))
        )
    recorded_proportion = recorded.get("index_storage_proportion")
    if recorded_proportion is not None and not same_proportion(
            recorded_proportion, storage.get("index_storage_proportion")):
        raise ValueError(
            "%s was searched at proportion %s, features are %s"
            % (source, recorded_proportion, storage.get("index_storage_proportion"))
        )


def build_parser(values):
    parser = argparse.ArgumentParser(
        description="Time B once after one discarded warm-up, then time each saved final configuration."
    )
    parser.add_argument("--result", action="append", default=None,
                        help="Stage 3 result JSON. Default: index_combination.result_file in config. "
                             "Repeat to share one warm-up and one baseline.")
    parser.add_argument("--output", default="",
                        help="Latency JSON for a single --result. Default: <result>.latency.json.")
    parser.add_argument("--workload", default=values["workload"])
    parser.add_argument("--features", default=values["features"])
    parser.add_argument("--candidates", default=values["candidates"])
    parser.add_argument("--host", default=values["host"])
    parser.add_argument("--port", default=str(values["port"]))
    parser.add_argument("--user", default=values["user"])
    parser.add_argument("--password", default=values["password"])
    parser.add_argument("--dbname", default=values["dbname"])
    parser.add_argument("--index_storage_proportion", type=float,
                        default=float(values["index_storage_proportion"]),
                        help="Must match storage.index_storage_proportion in the feature file.")
    parser.add_argument("--statement-timeout-seconds", type=float,
                        default=float(values["statement_timeout_seconds"]),
                        help="Per-query timeout in seconds (default 600); 0 disables it.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even when the latency JSON already exists.")
    return parser


def run(args):
    result_args = args.result or [defaults()["result"]]
    if args.output and len(result_args) != 1:
        raise ValueError("--output applies to exactly one --result")
    dbname = args.dbname
    loaded = []
    for raw in result_args:
        path = os.path.abspath(_path(
            raw, "results/{dbname}_index_combination.json", dbname))
        loaded.append((path, load_json(path)))
    named = {item.get("dbname") for _, item in loaded if item.get("dbname")}
    if named and named != {dbname}:
        raise ValueError("result files name %s, not %s" % (sorted(named), dbname))
    pending = []
    reports = []
    for path, payload in loaded:
        destination = os.path.abspath(output_path(path, args.output))
        if os.path.isfile(destination) and not args.force:
            logging.info("skip %s", destination)
            reports.append(load_json(destination))
            continue
        requested = final_configuration(payload)
        pending.append((path, destination, requested))
    if not pending:
        return reports

    features = load_json(_path(
        args.features, "feature/{dbname}_workload_features.json", dbname))
    storage, budget_bytes = storage_constraint(features, args.index_storage_proportion)
    for path, payload in loaded:
        assert_result_constraint(payload, storage, path)
    candidates = load_json(_path(
        args.candidates, "candidate/{dbname}_candidate_indexes.json", dbname))
    specs = build_index_specs(normalize_candidate_pool(candidates), features)
    queries = load_workload_queries(_path(args.workload, "../test/{dbname}_test.sql", dbname))
    query_ids = [query["qid"] for query in queries]
    timeout_ms = int(float(args.statement_timeout_seconds) * 1000)
    if timeout_ms < 0:
        raise ValueError("--statement-timeout-seconds must be >= 0")
    evaluator = PhysicalEvaluator(
        args.host, args.port, args.user, args.password, dbname, queries, specs,
    )
    evaluator.deadline = float("inf")
    evaluator.ddl_timeout_ms = timeout_ms
    for _, _, requested in pending:
        missing = [key for key in requested if key not in specs]
        if missing:
            raise ValueError("final configuration is outside the candidate pool: %s" % missing)

    logging.info("%s db=%s proportion=%s constraint_mb=%s pending=%d queries=%d",
                 PROTOCOL, dbname, storage.get("index_storage_proportion"),
                 storage.get("storage_constraint_mb"), len(pending), len(query_ids))
    dropped = drop_non_pk_indexes(evaluator)
    leftover = public_secondary_indexes(evaluator)
    if leftover:
        raise RuntimeError("%s still has secondary indexes: %s" % (dbname, leftover))
    logging.info("dropped %d secondary indexes", len(dropped))
    try:
        warm_started = time.monotonic()
        evaluator.measure([], query_ids, timeout_ms)
        warm_ms = round((time.monotonic() - warm_started) * 1000, 3)
        baseline, baseline_wall = _measure(evaluator, query_ids, [], timeout_ms)
        logging.info("B complete=%s total_ms=%s", baseline["complete"], baseline["total_ms"])
        fresh = []
        for source, destination, requested in pending:
            built = evaluator.materialize(requested, budget_bytes)
            try:
                ours, ours_wall = _measure(
                    evaluator, query_ids, built["configuration"], timeout_ms)
            finally:
                drop_non_pk_indexes(evaluator)
            ours.update({
                "requested": requested,
                "built": built["configuration"],
                "skipped": built.get("skipped_indexes") or [],
                "actual_index_bytes": built.get("actual_index_bytes"),
                "ddl_ms": built.get("ddl_ms"),
            })
            payload = {
                "protocol": PROTOCOL,
                "dbname": dbname,
                "storage": storage,
                "storage_budget_bytes": budget_bytes,
                "source": source,
                "dropped_before": dropped,
                "warm_ms": warm_ms,
                "baseline_ms": baseline["total_ms"],
                "baseline": baseline,
                "elapsed_b_wall_ms": baseline_wall,
                "ours_ms": ours["total_ms"],
                "ours_vs_b_pct": reduction_pct(baseline["total_ms"], ours["total_ms"]),
                "ours": ours,
                "elapsed_ours_wall_ms": ours_wall,
            }
            _dump(destination, payload)
            logging.info("ours %s B=%s ours=%s reduction=%s -> %s",
                         os.path.basename(source), payload["baseline_ms"],
                         payload["ours_ms"], payload["ours_vs_b_pct"], destination)
            fresh.append(payload)
        return reports + fresh
    finally:
        drop_non_pk_indexes(evaluator)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser(defaults()).parse_args()
    reports = run(args)
    print(json.dumps([
        {
            "source": item.get("source"),
            "baseline_ms": item.get("baseline_ms"),
            "ours_ms": item.get("ours_ms"),
            "ours_vs_b_pct": item.get("ours_vs_b_pct"),
        }
        for item in reports
    ], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
