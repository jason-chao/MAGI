
import pytest
import os
import sys
import yaml
import datetime
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Helper to find config files relative to this test file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Artifact Store Setup
RESULTS_DIR = os.path.join(BASE_DIR, "test_results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def save_artifact(method, deliberative, prompt, content):
    """Saves the LLM output to a markdown file for review."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "deliberative" if deliberative else "standard"
    # clean method name for filename
    safe_method = "".join(c for c in method if c.isalnum())
    filename = f"{timestamp}_{safe_method}_{mode}.md"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Test Artifact: {method} ({mode})\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write("-" * 40 + "\n\n")
        f.write(content)
    
    print(f"\n[Artifact Saved]: {filepath}")

from magi.core import Magi

def load_settings():
    config_path = os.path.join(BASE_DIR, "config.yaml")
    prompts_path = os.path.join(BASE_DIR, "magi", "prompts.yaml")
    
    if not os.path.exists(config_path) or not os.path.exists(prompts_path):
        pytest.skip("Configuration files not found. Skipping integration tests.")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(prompts_path, "r") as f:
        prompts = yaml.safe_load(f)
    return config, prompts

def has_api_keys():
    # Check for common API keys in environment variables
    keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
    return any(os.getenv(k) for k in keys)

@pytest.fixture(scope="module")
def magi_client():
    if not has_api_keys():
        pytest.skip("No API keys found in environment variables.")
    
    config, prompts = load_settings()
    
    # You can override LLMs here to use cheaper/faster models for testing
    # config['llms'] = ['gpt-4o-mini', 'gemini-1.5-flash']
    
    return Magi(config, prompts)

# Define the matrix of scenarios
METHODS = [
    "VoteYesNo",
    "VoteOptions",
    "Majority",
    "Consensus",
    "Minority",
    "Probability",
    "Compose"
]

DELIBERATIVE_OPTIONS = [False, True]

@pytest.mark.asyncio
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("deliberative", DELIBERATIVE_OPTIONS)
async def test_method_execution(magi_client, method, deliberative):
    """
    Matrix test for all Magi methods and deliberative modes.
    This runs actual LLM calls (integration test).
    """
    prompt = "Should code comments be mandatory?"
    options = {}

    # Customize prompt/options based on method
    if method == "VoteOptions":
        prompt = "What is the best color for a UI primary button?"
        options = {"options": ["Blue", "Green", "Red", "Black"]}
    elif method == "Compose":
        prompt = "Write a one-sentence haiku about coding tests."
    elif method == "Probability":
        prompt = "It will snow in Paris on January 1st, 2030."

    print(f"\n[Integration] Testing {method} (Deliberative={deliberative})...")

    try:
        result = await magi_client.run(
            user_prompt=prompt,
            method=method,
            method_options=options,
            deliberative=deliberative
        )
        
        # Save output for review
        save_artifact(method, deliberative, prompt, result)
        
        # 1. Basic Validation
        assert result is not None, "Result should not be None"
        assert len(result) > 20, "Result is suspiciously short"
        assert "Error:" not in result[:20], f"Method returned an error: {result}"

        # 2. Deliberative Mode Validation
        if deliberative:
            assert "Pre-Deliberation Results" in result, "Missing Pre-Deliberation section"
            assert "Post-Deliberation Results" in result, "Missing Post-Deliberation section"

        # 3. Method-Specific Validation (Check for key phrases in output)
        if method in ["VoteYesNo", "VoteOptions"]:
             # Vote breakdown usually present
             assert "Vote Breakdown" in result or "Result:" in result
        elif method == "Majority":
            assert "Majority View Summary" in result
        elif method == "Consensus":
            assert "Consensus/Common Ground" in result
        elif method == "Minority":
            assert "Minority/Gap Analysis" in result
        elif method == "Probability":
            assert "Probability Analysis" in result
        elif method == "Compose":
            assert "Compose Results" in result
            assert "Rank 1" in result

    except Exception as e:
        pytest.fail(f"Exception during test execution for {method} (Deliberative={deliberative}): {str(e)}")
