"""Index tuning entry point. There are only three stages.

Stage 1 extracts workload features and writes the budget for the configured storage proportion.
Stage 2 builds the fixed candidate pool.
Stage 3 searches index configurations on real PostgreSQL.
evaluate warms once after Stage 3, times B, builds the final indexes, then times them. It is not a fourth stage.

`all` runs Stage 1, Stage 2, Stage 3, then that timing on the result just written.

Run from CRAFR/: python -m core.main all
"""

import argparse
import json
import logging
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.common import cfg_get, load_config, load_json, normalize_candidate_pool, resolve_path, save_result
from core.evaluate import main as evaluate_main, run as run_evaluation
from core.stage1 import extract_and_save
from core.stage2 import generate_and_save
from core.stage3 import build_index_specs, storage_budget
from core.stage3_execute import PhysicalEvaluator, load_workload_queries
from core.stage3_llm import Stage3Agents
from core.stage3_search import CombinationSearch


CONFIG = load_config()


def _cfg(*keys, fallback=""):
    return cfg_get(CONFIG, *keys, default=fallback)


def defaults():
    return {
        "workload": _cfg("workload", "workload_file", fallback="../test/{dbname}_test.sql"),
        "summary": _cfg(
            "index_selector", "workload_summary",
            fallback="./feature/{dbname}_workload_summary.txt",
        ),
        "features": _cfg(
            "workload", "output_file",
            fallback="./feature/{dbname}_workload_features.json",
        ),
        "candidates": _cfg(
            "index_selector", "candidate_file",
            fallback="./candidate/{dbname}_candidate_indexes.json",
        ),
        "result": _cfg(
            "index_combination", "result_file",
            fallback="./results/{dbname}_index_combination.json",
        ),
        "index_storage_proportion": _cfg(
            "index_selector", "index_storage_proportion", fallback=0.5
        ),
        "max_iterations": _cfg("index_combination", "max_iterations", fallback=3),
        "max_llm_calls": _cfg("index_combination", "max_llm_calls", fallback=16),
        "candidate_view_limit": _cfg("index_combination", "candidate_view_limit", fallback=64),
        "max_physical_trials": _cfg("index_combination", "max_physical_trials", fallback=None),
        "max_additions": _cfg("index_combination", "max_additions", fallback=2),
        "max_removals": _cfg("index_combination", "max_removals", fallback=2),
        "local_candidate_limit": _cfg(
            "index_combination", "local_candidate_limit", fallback=12
        ),
        "total_tuning_time_seconds": _cfg(
            "index_combination", "total_tuning_time_seconds", fallback=0
        ),
        "statement_timeout_seconds": _cfg(
            "index_combination", "statement_timeout_seconds", fallback=600
        ),
        "host": _cfg("database", "host", fallback=""),
        "port": _cfg("database", "port", fallback=5432),
        "user": _cfg("database", "user", fallback="postgres"),
        "password": _cfg("database", "password", fallback=""),
        "dbname": _cfg("database", "dbname", fallback="") or "SSB",
        "api_key": _cfg("llm", "api_key", fallback=""),
        "base_url": _cfg("llm", "base_url", fallback=""),
        "model": _cfg("llm", "model", fallback="gpt-4.1-mini"),
    }


def _path(value, default_name, dbname):
    return resolve_path(value, default_name, dbname=dbname)


def add_database(parser, values):
    parser.add_argument("--host", default=values["host"])
    parser.add_argument("--port", default=str(values["port"]))
    parser.add_argument("--user", default=values["user"])
    parser.add_argument("--password", default=values["password"])
    parser.add_argument("--dbname", default=values["dbname"])


def add_inputs(parser, values):
    parser.add_argument("--workload", default=values["workload"])
    parser.add_argument("--summary", default=values["summary"])
    parser.add_argument("--features", default=values["features"])
    parser.add_argument("--candidates", default=values["candidates"])


def add_search(parser, values):
    parser.add_argument("--result", default=values["result"])
    parser.add_argument("--max_iterations", type=int, default=int(values["max_iterations"]))
    parser.add_argument("--local-candidate-limit", type=int, default=int(values["local_candidate_limit"]))
    parser.add_argument("--max-llm-calls", type=int, default=int(values["max_llm_calls"]))
    parser.add_argument("--candidate-view-limit", type=int, default=int(values["candidate_view_limit"]))
    parser.add_argument("--max-physical-trials", type=int, default=values["max_physical_trials"])
    parser.add_argument("--max-additions", type=int, default=int(values["max_additions"]))
    parser.add_argument("--max-removals", type=int, default=int(values["max_removals"]))
    parser.add_argument("--retry-over-budget-addition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--api_key", default=values["api_key"])
    parser.add_argument("--base_url", default=values["base_url"])
    parser.add_argument("--model", default=values["model"])
    parser.add_argument("--llm-proxy", default="")
    parser.add_argument("--llm-extra-body", default="")
    parser.add_argument("--total-tuning-time-seconds", type=float,
                        default=float(values["total_tuning_time_seconds"]))
    parser.add_argument("--statement-timeout-seconds", type=float,
                        default=float(values["statement_timeout_seconds"]))


def input_paths(args):
    return {
        "workload": _path(args.workload, "../test/{dbname}_test.sql", args.dbname),
        "summary": _path(args.summary, "feature/{dbname}_workload_summary.txt", args.dbname),
        "features": _path(args.features, "feature/{dbname}_workload_features.json", args.dbname),
        "candidates": _path(args.candidates, "candidate/{dbname}_candidate_indexes.json", args.dbname),
    }


def run_stage1(args):
    paths = input_paths(args)
    print("== Stage 1 提取负载特征 ==")
    extract_and_save(
        paths["workload"], paths["features"], paths["summary"],
        args.host, args.port, args.user, args.password, args.dbname,
        args.index_storage_proportion,
    )
    return paths


def run_stage2(args):
    paths = input_paths(args)
    print("== Stage 2 生成候选池 ==")
    generate_and_save(
        paths["workload"], paths["features"], paths["summary"], paths["candidates"],
        args.host, args.port, args.user, args.password, args.dbname, args.workers,
    )
    return paths


def run_stage3(args):
    paths = input_paths(args)
    paths["result"] = _path(args.result, "results/{dbname}_index_combination.json", args.dbname)
    print("== Stage 3 搜索 ==")
    features = load_json(paths["features"])
    candidate_pool = normalize_candidate_pool(load_json(paths["candidates"]))
    specs = build_index_specs(candidate_pool, features)
    queries = load_workload_queries(paths["workload"])
    extra_body = json.loads(args.llm_extra_body) if args.llm_extra_body else None
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("--llm-extra-body must be a JSON object")
    logging.info("LLM route model=%s proxy=%s extra=%s",
                 args.model, args.llm_proxy or "direct",
                 sorted(extra_body) if extra_body else [])
    agents = Stage3Agents(
        args.api_key, args.base_url, args.model,
        proxy=args.llm_proxy or None, extra_body=extra_body,
    )
    result_path = paths["result"]
    status_path = result_path[:-5] + ".live.json" if result_path.endswith(".json") else result_path + ".live.json"
    evaluator = PhysicalEvaluator(
        args.host, args.port, args.user, args.password, args.dbname, queries, specs,
    )
    search = CombinationSearch(
        list(candidate_pool), specs, features, storage_budget(features),
        agents, evaluator, args.max_iterations, args.local_candidate_limit,
        args.total_tuning_time_seconds, args.statement_timeout_seconds, status_path,
        candidate_view_limit=args.candidate_view_limit,
        max_physical_trials=args.max_physical_trials,
        max_additions=args.max_additions, max_removals=args.max_removals,
        retry_over_budget_addition=args.retry_over_budget_addition,
        max_llm_calls=args.max_llm_calls,
    )
    result = search.run()
    result.update({
        "dbname": args.dbname,
        "model": args.model,
        "llm_proxy": args.llm_proxy or None,
        "llm_extra_body": extra_body,
        "candidate_pool": list(candidate_pool),
        "storage": features.get("storage") or {},
    })
    save_result(paths["result"], result)
    print(json.dumps(result["final_configuration"], ensure_ascii=False, indent=2))
    return result


def run_evaluate(args, force=False):
    """Time the Stage 3 result: one discarded warm-up, one baseline, then the built indexes."""
    print("== 搜索后计时：预热一次，测 B，再建最终索引并测延迟 ==")
    reports = run_evaluation(argparse.Namespace(
        result=[args.result],
        output="",
        workload=args.workload,
        features=args.features,
        candidates=args.candidates,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname,
        index_storage_proportion=args.index_storage_proportion,
        statement_timeout_seconds=args.statement_timeout_seconds,
        force=force,
    ))
    print(json.dumps([
        {
            "source": item.get("source"),
            "baseline_ms": item.get("baseline_ms"),
            "ours_ms": item.get("ours_ms"),
            "ours_vs_b_pct": item.get("ours_vs_b_pct"),
        }
        for item in reports
    ], ensure_ascii=False, indent=2))
    return reports


def build_parser(values):
    parser = argparse.ArgumentParser(
        description="索引调优只有三个阶段：Stage 1 特征，Stage 2 候选池，Stage 3 搜索。"
                    "evaluate 是搜索后的计时，不是第四阶段。数据库和存储比例默认来自 config.json。"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    stage1 = sub.add_parser("stage1", help="Stage 1：提取负载特征并写入存储约束")
    add_database(stage1, values)
    add_inputs(stage1, values)
    stage1.add_argument("--index_storage_proportion", type=float,
                        default=float(values["index_storage_proportion"]))

    stage2 = sub.add_parser("stage2", help="Stage 2：生成固定候选池")
    add_database(stage2, values)
    add_inputs(stage2, values)
    stage2.add_argument("--workers", type=int, default=0)

    stage3 = sub.add_parser("stage3", help="Stage 3：搜索物理索引配置")
    add_database(stage3, values)
    add_inputs(stage3, values)
    add_search(stage3, values)

    sub.add_parser("evaluate", help="搜索之后：预热一次，计时测 B，再测 Stage 3 的最终配置")

    entire = sub.add_parser(
        "all",
        help="按顺序运行 Stage 1、Stage 2、Stage 3，然后预热、测 B、建最终索引并测延迟",
    )
    add_database(entire, values)
    add_inputs(entire, values)
    add_search(entire, values)
    entire.add_argument("--index_storage_proportion", type=float,
                        default=float(values["index_storage_proportion"]))
    entire.add_argument("--workers", type=int, default=0)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "evaluate":
        sys.argv = [sys.argv[0]] + argv[1:]
        return evaluate_main()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser(defaults()).parse_args(argv)
    if args.command == "stage1":
        run_stage1(args)
    elif args.command == "stage2":
        run_stage2(args)
    elif args.command == "stage3":
        run_stage3(args)
    elif args.command == "all":
        run_stage1(args)
        run_stage2(args)
        run_stage3(args)
        run_evaluate(args, force=True)
    else:
        raise SystemExit("unknown command %s" % args.command)


if __name__ == "__main__":
    main()
