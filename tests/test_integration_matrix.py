"""
Integration tests — make real LLM calls across all methods and deliberative modes.

WARNING: These tests incur API costs. Run only when you have API keys set.
Outputs are saved in test_results/ as both Markdown (.md) and JSON (.json) for manual review.
"""
import json
import pytest
import os
import sys
import yaml
import datetime
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, "test_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _base_filename(method: str, deliberative: bool) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "deliberative" if deliberative else "standard"
    safe_method = "".join(c for c in method if c.isalnum())
    return os.path.join(RESULTS_DIR, f"{timestamp}_{safe_method}_{mode}")


def save_text_artifact(method: str, deliberative: bool, prompt: str, content: str) -> str:
    filepath = _base_filename(method, deliberative) + ".md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Test Artifact: {method} ({'deliberative' if deliberative else 'standard'})\n")
        f.write(f"Prompt: {prompt}\n")
        f.write("-" * 40 + "\n\n")
        f.write(content)
    print(f"\n[Markdown Saved]: {filepath}")
    return filepath


def save_json_artifact(method: str, deliberative: bool, structured: dict) -> str:
    filepath = _base_filename(method, deliberative) + ".json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    print(f"\n[JSON Saved]:     {filepath}")
    return filepath


from magi_core.core import Magi


def load_settings():
    config_path = os.path.join(BASE_DIR, "config.yaml")
    prompts_path = os.path.join(BASE_DIR, "magi_core", "prompts.yaml")
    if not os.path.exists(config_path) or not os.path.exists(prompts_path):
        pytest.skip("Configuration files not found.")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(prompts_path) as f:
        prompts = yaml.safe_load(f)
    return config, prompts


def has_api_keys():
    return any(os.getenv(k) for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"])


@pytest.fixture(scope="module")
def magi_client():
    if not has_api_keys():
        pytest.skip("No API keys found.")
    config, prompts = load_settings()
    return Magi(config, prompts)


# ---------------------------------------------------------------------------
# Moral-dilemma prompts per method
# ---------------------------------------------------------------------------
METHOD_SCENARIOS = {
    "VoteYesNo": {
        # Active killing vs. letting die: pulling the lever makes you causally
        # responsible for one death; not pulling makes you responsible for five.
        "prompt": (
            "A self-driving car's brakes have failed. It will kill five pedestrians "
            "unless its AI deliberately swerves onto the pavement, killing the single "
            "passenger inside. Should the AI be programmed to sacrifice its passenger?"
        ),
        "options": {},
    },
    "VoteOptions": {
        # Lifeboat ethics: scarce resource, incommensurable claims.
        "prompt": (
            "A hospital has one donor heart. Who should receive it?"
        ),
        "options": {
            "options": [
                "A 10-year-old child with decades ahead",
                "A 45-year-old surgeon who saves hundreds of lives per year",
                "The patient who has waited longest on the list",
                "Whoever has the highest chance of survival post-transplant",
            ]
        },
    },
    "Majority": {
        # Capital punishment: retributivism vs. irreversibility vs. deterrence.
        "prompt": (
            "Should capital punishment be re-introduced for terrorist attacks "
            "that cause mass civilian casualties, even knowing that wrongful "
            "convictions are statistically inevitable?"
        ),
        "options": {},
    },
    "Consensus": {
        # Abortion: bodily autonomy vs. moral status of the foetus — one of the
        # hardest genuine consensus problems in applied ethics.
        "prompt": (
            "At what point, if any, does terminating a pregnancy become morally "
            "impermissible, and who — if anyone — has the authority to enforce that line?"
        ),
        "options": {},
    },
    "Minority": {
        # Effective altruism: most people accept giving to charity is good,
        # but strong EA conclusions (e.g. you must donate until marginal utility
        # equalises) are deeply uncomfortable.
        "prompt": (
            "If you can save a drowning child at trivial cost, you are obligated to. "
            "Distance and anonymity are morally irrelevant. Therefore, anyone who "
            "spends money on luxuries while children die of preventable diseases is "
            "morally equivalent to letting the drowning child die. Is this argument sound?"
        ),
        "options": {},
    },
    "Probability": {
        # Moral luck: a drunk driver who kills a pedestrian vs. one who arrives home
        # safely — same decision, different outcome. How much should luck matter?
        "prompt": (
            "Two drivers drink the same amount and drive home. One kills a pedestrian "
            "by chance; the other arrives safely. The lucky driver deserves the same "
            "moral blame and legal punishment as the unlucky one."
        ),
        "options": {},
    },
    "Compose": {
        # Write a steel-man of a deeply uncomfortable position.
        "prompt": (
            "Write the strongest possible moral argument for the claim that "
            "wealthy nations have an absolute obligation to accept unlimited "
            "refugees, regardless of cultural, economic, or security consequences. "
            "Steel-man the position without hedging."
        ),
        "options": {},
    },
    "Synthesis": {
        # The classic repugnant conclusion in population ethics — there is no
        # comfortable resolution, making it ideal for a comprehensive synthesis.
        "prompt": (
            "Derek Parfit's Repugnant Conclusion: a world of ten billion people "
            "living very happy lives is morally inferior to a world of a trillion "
            "people whose lives are barely worth living, because the total sum of "
            "well-being is greater. Is this conclusion repugnant, unavoidable, or "
            "does it reveal a flaw in utilitarian reasoning itself?"
        ),
        "options": {},
    },
}

METHODS = list(METHOD_SCENARIOS.keys())
DELIBERATIVE_OPTIONS = [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("deliberative", DELIBERATIVE_OPTIONS)
async def test_method_execution(magi_client, method, deliberative):
    """Matrix test: all methods × deliberative modes with real LLM calls."""
    scenario = METHOD_SCENARIOS[method]
    prompt = scenario["prompt"]
    options = scenario["options"]

    print(f"\n[Integration] {method} (Deliberative={deliberative})")
    print(f"Prompt: {prompt[:80]}...")

    try:
        # Both formats share a single deliberation run via run_structured()
        structured = await magi_client.run_structured(
            user_prompt=prompt,
            method=method,
            method_options=options,
            deliberative=deliberative,
        )
        result = magi_client._render_markdown(structured)

        save_text_artifact(method, deliberative, prompt, result)
        save_json_artifact(method, deliberative, structured)

        # --- text assertions ---
        assert result is not None
        assert len(result) > 20, "Result suspiciously short"
        assert not result.startswith("Error:"), f"Method returned error: {result}"

        if deliberative:
            assert "Pre-Deliberation Results" in result
            assert "Post-Deliberation Results" in result

        if method in ("VoteYesNo", "VoteOptions"):
            assert "Vote Breakdown" in result or "Result:" in result
        elif method == "Majority":
            assert "Majority View Summary" in result
        elif method == "Consensus":
            assert "Consensus/Common Ground" in result
        elif method == "Minority":
            assert "Minority/Gap Analysis" in result
        elif method == "Probability":
            assert "Probability Analysis" in result or "All LLMs abstained" in result
        elif method == "Compose":
            assert "Compose Results" in result
            assert "Rank 1" in result
        elif method == "Synthesis":
            assert "Synthesis" in result

        # --- JSON / structured assertions ---
        assert "error" not in structured, f"structured result has top-level error: {structured.get('error')}"
        assert structured.get("schema_version") == "1.0"
        assert structured.get("method") == method
        assert structured.get("deliberative") == deliberative
        assert "rounds" in structured and len(structured["rounds"]) >= 1

        for rnd in structured["rounds"]:
            assert "responses" in rnd
            assert "errors" in rnd
            assert len(rnd["responses"]) > 0, f"Round {rnd.get('round')} has no valid responses"
            for resp in rnd["responses"]:
                assert "model" in resp
                assert "pseudonym" in resp
                assert resp["pseudonym"].startswith("Participant ")

        # JSON serialisable
        json.dumps(structured)

    except Exception as e:
        pytest.fail(f"{method} (Deliberative={deliberative}) raised: {e}")


@pytest.mark.asyncio
async def test_synthesis_language_directive_live(magi_client):
    """Live smoke test for the `language` parameter.

    Runs a single Synthesis call with `language="Japanese"` and asserts the
    rapporteur summary contains Japanese characters. Japanese is detected via
    Hiragana / Katakana / CJK Unicode ranges, which cannot appear in ordinary
    English text — so the heuristic is unambiguous without demanding any
    specific vocabulary.
    """
    scenario = METHOD_SCENARIOS["Synthesis"]

    structured = await magi_client.run_structured(
        user_prompt=scenario["prompt"],
        method="Synthesis",
        method_options=scenario["options"],
        deliberative=False,
        language="Japanese",
    )

    assert "error" not in structured, f"run failed: {structured.get('error')}"
    rapporteur = (structured["rounds"][0].get("rapporteur") or {})
    summary = rapporteur.get("summary", "") or ""
    assert len(summary) > 20, f"summary suspiciously short: {summary!r}"

    save_json_artifact("Synthesis_language_ja", False, structured)

    def _is_japanese_char(ch: str) -> bool:
        cp = ord(ch)
        return (
            0x3040 <= cp <= 0x309F  # Hiragana
            or 0x30A0 <= cp <= 0x30FF  # Katakana
            or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        )

    japanese_chars = sum(1 for ch in summary if _is_japanese_char(ch))
    assert japanese_chars >= 10, (
        f"summary contains only {japanese_chars} Japanese chars — directive likely ignored. "
        f"summary head: {summary[:400]!r}"
    )
