#!/usr/bin/env python3
"""
MiniSim CLI — Unified command-line interface.

Usage:
    python cli.py predict  -q "Will X happen?" --agents 20 --rounds 3
    python cli.py scan     --source all --edge 0.03
    python cli.py survey   --demo consumer_product --respondents 30
    python cli.py backtest --agents 30 --rounds 2
    python cli.py benchmark --model qwen2.5:14b --agents 10
    python cli.py eval     --agents 20 --rounds 2
    python cli.py dashboard
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
load_dotenv()


def cmd_predict(args):
    """Run a swarm prediction on a question."""
    from main import main as _main
    # Reconstruct sys.argv for main.py's parser
    argv = ["-q", args.question]
    if args.context:
        argv += ["-c", args.context]
    argv += ["-a", str(args.agents), "-r", str(args.rounds)]
    if args.market_price is not None:
        argv += ["-m", str(args.market_price)]
    if args.output:
        argv += ["-o", args.output]
    if args.llm:
        argv += ["--llm"]
    elif args.offline:
        argv += ["--offline"]
    if args.model:
        argv += ["--model", args.model]
    if args.web_research:
        argv += ["--web-research"]
    sys.argv = ["main.py"] + argv
    _main()


def cmd_scan(args):
    """Run the opportunity scanner."""
    from scanner import main as scan_main
    sys.argv = ["scanner.py"]
    if args.source:
        sys.argv += ["--source", args.source]
    sys.argv += ["--agents", str(args.agents)]
    sys.argv += ["--rounds", str(args.rounds)]
    sys.argv += ["--edge", str(args.edge)]
    sys.argv += ["--max-markets", str(args.max_markets)]
    if args.watch:
        sys.argv += ["--watch"]
        sys.argv += ["--interval", str(args.interval)]
    scan_main()


def cmd_survey(args):
    """Run a synthetic survey."""
    from run_survey import main as survey_main
    sys.argv = ["run_survey.py"]
    sys.argv += ["--demo", args.demo]
    sys.argv += ["--respondents", str(args.respondents)]
    if args.model:
        sys.argv += ["--model", args.model]
    if args.file:
        sys.argv += ["--file", args.file]
    survey_main()


def cmd_backtest(args):
    """Run backtest on curated or live markets."""
    if args.live:
        from live_backtest import run_live_backtest
        run_live_backtest(
            n_agents=args.agents,
            n_rounds=args.rounds,
            max_events=args.max_events,
            max_markets=args.max_markets,
        )
    else:
        from backtest import run_backtest
        run_backtest()


def cmd_benchmark(args):
    """Run head-to-head benchmark: swarm vs single LLM vs market."""
    from benchmark import run_benchmark
    run_benchmark(n_agents=args.agents, n_rounds=args.rounds, model=args.model)


def cmd_eval(args):
    """Run the eval suite."""
    from eval_runner import run_eval
    run_eval(n_agents=args.agents, n_rounds=args.rounds)


def cmd_dashboard(args):
    """Launch the Streamlit dashboard."""
    import subprocess
    subprocess.run(["streamlit", "run", "streamlit_app.py"])


def cmd_track_record(args):
    """Show the prediction track record."""
    from src.track_record import TrackRecord
    tr = TrackRecord()
    tr.print_summary()

    if args.resolve:
        n = tr.resolve_from_kalshi()
        print(f"\nResolved {n} predictions from Kalshi.")
        tr.print_summary()


def cmd_arbitrage(args):
    """Find cross-platform arbitrage opportunities."""
    from src.cross_platform import find_arbitrage
    arbs = find_arbitrage(min_spread=args.min_spread)
    if not arbs:
        print("No arbitrage opportunities found.")
        return
    print(f"\nArbitrage opportunities (>{args.min_spread*100:.0f}% spread): {len(arbs)}")
    for a in arbs:
        print(f"  Spread={a['spread']:.2f} | Buy {a['buy_on']}@{a['buy_price']:.2f} / "
              f"Sell {a['sell_on']}@{a['sell_price']:.2f}")
        print(f"    {a['question'][:65]}")


def main():
    parser = argparse.ArgumentParser(
        prog="minisim",
        description="MiniSim — Swarm Prediction Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # predict
    p = subparsers.add_parser("predict", help="Run a swarm prediction")
    p.add_argument("-q", "--question", required=True)
    p.add_argument("-c", "--context", default="")
    p.add_argument("-a", "--agents", type=int, default=20)
    p.add_argument("-r", "--rounds", type=int, default=4)
    p.add_argument("-m", "--market-price", type=float, default=None)
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--llm", action="store_true", help="Use local LLM (Ollama)")
    p.add_argument("--offline", action="store_true", help="Offline mode (no LLM)")
    p.add_argument("--model", default=None)
    p.add_argument("--web-research", action="store_true")

    # scan
    p = subparsers.add_parser("scan", help="Scan markets for opportunities")
    p.add_argument("--source", choices=["kalshi", "polymarket", "manifold", "predictit", "all"], default="all")
    p.add_argument("--agents", type=int, default=20)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--edge", type=float, default=0.05)
    p.add_argument("--max-markets", type=int, default=30)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=int, default=300)

    # survey
    p = subparsers.add_parser("survey", help="Run a synthetic survey")
    p.add_argument("--demo", choices=["consumer_product", "brand_perception", "policy"], default="consumer_product")
    p.add_argument("--file", default=None, help="Custom survey JSON file")
    p.add_argument("--respondents", type=int, default=30)
    p.add_argument("--model", default=None)

    # backtest
    p = subparsers.add_parser("backtest", help="Run backtest")
    p.add_argument("--live", action="store_true", help="Use real Kalshi markets")
    p.add_argument("--agents", type=int, default=30)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--max-events", type=int, default=500)
    p.add_argument("--max-markets", type=int, default=200)

    # benchmark
    p = subparsers.add_parser("benchmark", help="Head-to-head: swarm vs single LLM vs market")
    p.add_argument("--model", default=None)
    p.add_argument("--agents", type=int, default=15)
    p.add_argument("--rounds", type=int, default=2)

    # eval
    p = subparsers.add_parser("eval", help="Run eval suite")
    p.add_argument("--agents", type=int, default=30)
    p.add_argument("--rounds", type=int, default=2)

    # dashboard
    subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")

    # track-record
    p = subparsers.add_parser("track-record", help="Show prediction track record")
    p.add_argument("--resolve", action="store_true", help="Auto-resolve from Kalshi")

    # arbitrage
    p = subparsers.add_parser("arbitrage", help="Find cross-platform arbitrage")
    p.add_argument("--min-spread", type=float, default=0.05)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    commands = {
        "predict": cmd_predict,
        "scan": cmd_scan,
        "survey": cmd_survey,
        "backtest": cmd_backtest,
        "benchmark": cmd_benchmark,
        "eval": cmd_eval,
        "dashboard": cmd_dashboard,
        "track-record": cmd_track_record,
        "arbitrage": cmd_arbitrage,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
