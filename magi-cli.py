#!/usr/bin/env python3
import asyncio
import argparse
import os
import yaml
from magi.core import Magi
from magi.utils import load_yaml
from dotenv import load_dotenv

# Load env variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Magi: Multi-LLM Aggregator")
    parser.add_argument("prompt", help="The user prompt")
    parser.add_argument("--system-prompt", help="Optional system prompt", default=None)
    parser.add_argument("--llms", help="Comma-separated list of LLMs to use", default=None)
    parser.add_argument("--method", help="Aggregation method (VoteYesNo, VoteOptions, Majority, Consensus, Minority, Probability)", default="VoteYesNo")
    parser.add_argument("--vote-threshold", help="Threshold for vote method (default 0.5)", type=float, default=0.5)
    parser.add_argument("--no-abstain", help="Disallow abstain in VoteYesNo/Probability method", action="store_true")
    parser.add_argument("--options", help="Comma-separated options for VoteOptions method", default=None)
    parser.add_argument("--config", help="Path to config file", default="config.yaml")
    parser.add_argument("--prompts", help="Path to prompts file", default="prompts.yaml")
    parser.add_argument("--rapporteur-prompt", help="Custom instructions to append to the rapporteur prompt", default=None)
    
    args = parser.parse_args()

    # Load config
    try:
        config = load_yaml(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        # Fallback default
        config = {'llms': [], 'defaults': {}}
    
    try:
        prompts = load_yaml(args.prompts)
    except FileNotFoundError:
        print(f"Prompts file not found: {args.prompts}")
        return

    # Override config with args
    if args.llms:
        selected_llms = [x.strip() for x in args.llms.split(',')]
    else:
        selected_llms = config.get('llms', [])

    if not selected_llms:
        print("No LLMs selected. Please specify via --llms or config file.")
        return

    method_options = {
        'vote_threshold': args.vote_threshold,
        'allow_abstain': not args.no_abstain,
        'options': [x.strip() for x in args.options.split(',')] if args.options else None,
        'rapporteur_prompt': args.rapporteur_prompt
    }

    magi = Magi(config, prompts)

    print(f"Running Magi with method: {args.method}")
    print(f"Selected LLMs: {selected_llms}")
    print("-" * 40)

    try:
        result = asyncio.run(magi.run(
            user_prompt=args.prompt,
            system_prompt=args.system_prompt,
            selected_llms=selected_llms,
            method=args.method,
            method_options=method_options
        ))
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
