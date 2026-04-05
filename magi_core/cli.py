#!/usr/bin/env python3
"""Command-line interface for MAGI: Multi-Agent Group Intelligence."""
import asyncio
import argparse
import json
import os
import sys
import yaml

from magi_core.core import Magi
from magi_core.utils import load_yaml, get_default_prompts_path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


METHODS = ["VoteYesNo", "VoteOptions", "Majority", "Consensus", "Minority", "Probability", "Compose", "Synthesis"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="magi",
        description="MAGI: Multi-Agent Group Intelligence — consult a council of LLMs on complex questions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Decision Modes:
  VoteYesNo    Democratic Yes/No/Abstain vote
  VoteOptions  Vote with custom options
  Majority     Summarise the prevailing opinion
  Consensus    Find common ground between views
  Minority     Surface dissenting and overlooked views
  Probability  Estimate likelihood of a statement being true
  Compose      Generate content and rank via blind peer review
  Synthesis    Comprehensively combine ALL perspectives into one unified response

Examples:
  magi "Is it ever justified to lie to protect someone's feelings?" --method VoteYesNo
  magi "A runaway trolley will kill five people. Should you pull the lever to divert it, killing one?" --method Synthesis
  magi "Who should receive the last organ during a shortage?" --method VoteOptions --options "Youngest patient,Longest waiting,Random lottery,Medical urgency"
        """,
    )
    parser.add_argument("prompt", help="The question or issue to deliberate on")
    parser.add_argument("--system-prompt", help="Optional context to prepend", default=None)
    parser.add_argument(
        "--llms",
        help="Comma-separated list of LLM model names (overrides config)",
        default=None,
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="VoteYesNo",
        help="Aggregation method (default: VoteYesNo)",
    )
    parser.add_argument(
        "--options",
        help="Comma-separated options for VoteOptions method",
        default=None,
    )
    parser.add_argument(
        "--vote-threshold",
        type=float,
        default=0.5,
        help="Fraction of votes needed to declare a winner (default: 0.5)",
    )
    parser.add_argument(
        "--no-abstain",
        action="store_true",
        help="Disallow abstaining in VoteYesNo or Probability",
    )
    parser.add_argument(
        "--deliberative",
        action="store_true",
        help="Enable a second deliberative round where agents review peer responses",
    )
    parser.add_argument(
        "--rapporteur-prompt",
        help="Custom instructions appended to the rapporteur prompt",
        default=None,
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Probe each configured model with a real API call and report availability, then exit",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' (default) for Markdown/plain text, 'json' for machine-readable JSON",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML file (default: config.yaml in current directory, then package default)",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="Path to prompts YAML file (default: bundled prompts.yaml)",
    )
    return parser


def _load_config(path: str | None) -> dict:
    candidates = []
    if path:
        candidates.append(path)
    candidates.append("config.yaml")

    for p in candidates:
        if os.path.exists(p):
            return load_yaml(p)

    print("Warning: No config.yaml found. Using empty config.", file=sys.stderr)
    return {"llms": [], "defaults": {}}


def _load_prompts(path: str | None) -> dict:
    candidates = []
    if path:
        candidates.append(path)
    candidates.append("prompts.yaml")
    candidates.append(get_default_prompts_path())

    for p in candidates:
        if os.path.exists(p):
            return load_yaml(p)

    print("Error: Could not locate prompts.yaml.", file=sys.stderr)
    sys.exit(1)


def _print_model_check(results: list) -> bool:
    """Print a health-check table. Returns True if all models are OK."""
    CATEGORY_LABEL = {
        "ok": "OK",
        "deprecated": "DEPRECATED",
        "not_found": "NOT FOUND",
        "auth": "AUTH ERROR",
        "rate_limit": "RATE LIMITED",
        "timeout": "TIMEOUT",
        "empty_response": "EMPTY RESP",
        "unknown": "ERROR",
    }
    print("\nModel Health Check")
    print("=" * 56)
    all_ok = True
    for slot in results:
        checks = slot["checks"]
        primary = checks[0]
        label = CATEGORY_LABEL.get(primary["category"], primary["category"].upper())
        if primary["ok"]:
            print(f"  [OK]          {primary['model']}")
        else:
            all_ok = False
            print(f"  [{label:<12}] {primary['model']}: {primary['message']}")
            for fb in checks[1:]:
                fb_label = CATEGORY_LABEL.get(fb["category"], fb["category"].upper())
                fb_status = "OK" if fb["ok"] else fb_label
                print(f"    └─ fallback [{fb_status:<12}] {fb['model']}: {fb['message']}")
    print()
    if all_ok:
        print("All models are available.")
    else:
        print("Warning: some models are unavailable. Update config.yaml or add fallbacks.")
    return all_ok


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = _load_config(args.config)
    prompts = _load_prompts(args.prompts)

    selected_llms = (
        [x.strip() for x in args.llms.split(",")]
        if args.llms
        else config.get("llms", [])
    )

    if not selected_llms:
        print(
            "Error: No LLMs configured. Specify via --llms or add them to config.yaml.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not (0 < args.vote_threshold <= 1.0):
        print("Error: --vote-threshold must be > 0 and ≤ 1.", file=sys.stderr)
        sys.exit(1)

    if args.method == "VoteOptions" and not args.options:
        print("Error: --options is required for the VoteOptions method.", file=sys.stderr)
        sys.exit(1)

    method_options = {
        "vote_threshold": args.vote_threshold,
        "allow_abstain": not args.no_abstain,
        "options": [x.strip() for x in args.options.split(",")] if args.options else None,
        "rapporteur_prompt": args.rapporteur_prompt,
    }

    magi = Magi(config, prompts)

    if args.check_models:
        results = asyncio.run(magi.check_models(selected_llms))
        all_ok = _print_model_check(results)
        sys.exit(0 if all_ok else 1)

    print(f"Method : {args.method}")
    display_models = [s[0] if isinstance(s, list) else s for s in selected_llms]
    print(f"Models : {', '.join(display_models)}")
    if args.deliberative:
        print("Mode   : Deliberative")
    print("-" * 50)

    try:
        if args.output_format == "json":
            result = asyncio.run(
                magi.run_structured(
                    user_prompt=args.prompt,
                    system_prompt=args.system_prompt,
                    selected_llms=selected_llms,
                    method=args.method,
                    method_options=method_options,
                    deliberative=args.deliberative,
                )
            )
            print(json.dumps(result, indent=2))
        else:
            result = asyncio.run(
                magi.run(
                    user_prompt=args.prompt,
                    system_prompt=args.system_prompt,
                    selected_llms=selected_llms,
                    method=args.method,
                    method_options=method_options,
                    deliberative=args.deliberative,
                )
            )
            print(result)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
