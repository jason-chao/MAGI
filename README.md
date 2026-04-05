# MAGI: Multi-Agent Group Intelligence

**What if you could consult a council of diverse artificial minds before making a tough decision?**

MAGI is a decision-support system inspired by the **MAGI supercomputer from *Neon Genesis Evangelion***. In the anime, three distinct AI personas deliberate to govern Tokyo-3. Similarly, this project orchestrates a council of multiple LLMs to deliberate, vote, and reason about complex moral, ethical, and practical questions. By aggregating diverse AI perspectives, MAGI simulates a more human-like deliberation process — revealing consensus, surfacing minority risks, and synthesising collective wisdom.

## How It Works

MAGI sends a question to all configured models in parallel. Each model responds in a structured JSON format (answer + reasoning + confidence score). A **Rapporteur** — the most confident model — then synthesises the group's findings into a final report. An optional **Deliberative Round** lets agents read each other's initial responses before finalising their own, mimicking human group deliberation. During deliberation, each model is assigned a random anonymous ID (e.g. `Participant X7K2`) to reduce brand bias in peer review.

## Decision Modes

| Mode | What it does |
|------|-------------|
| `VoteYesNo` | Democratic Yes / No / Abstain vote |
| `VoteOptions` | Vote on a custom set of options |
| `Majority` | Summarises the prevailing opinion |
| `Consensus` | Finds common ground across all views |
| `Minority` | Surfaces dissenting and overlooked perspectives |
| `Probability` | Estimates the likelihood of a statement being true |
| `Compose` | Generates content and ranks it via blind peer review |
| `Synthesis` | Comprehensively combines **all** perspectives into one unified response |

**Synthesis** is the most inclusive mode — unlike `Majority` (which amplifies the dominant view) or `Consensus` (which finds the lowest common denominator), `Synthesis` instructs the rapporteur to weave every argument, nuance, and disagreement into a single coherent narrative.

## Decision Flows

### Standard flow (all methods)

Every deliberation follows this pipeline regardless of mode:

```
  ┌──────────────────────────────────────────────────┐
  │                    User Prompt                   │
  └────────────────────────┬─────────────────────────┘
                           │  dispatched in parallel
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │   LLM  1    │  │   LLM  2    │  │   LLM  N    │
  │  response   │  │  response   │  │  response   │
  │  reason     │  │  reason     │  │  reason     │
  │  confidence │  │  confidence │  │  confidence │
  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
         └────────────────┼────────────────┘
                          │  results collected
                          ▼
             ┌────────────────────────┐
             │       Aggregate        │
             │   (method-dependent)   │
             └────────────┬───────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │  Rapporteur selected   │
             │  (by confidence score) │
             └────────────┬───────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │      Final Report      │
             │   text  │  JSON        │
             └────────────────────────┘
```

### Aggregation by method

The "Aggregate" step is what distinguishes each mode:

```
  Responses collected
         │
         ├─ VoteYesNo / VoteOptions ──► tally votes ──► declare winner (if > threshold)
         │                                                       │
         │                                           rapporteur summarises vote
         │
         ├─ Majority ──────────────────────────────► highest-confidence model
         │                                           summarises prevailing view
         │
         ├─ Consensus ─────────────────────────────► highest-confidence model
         │                                           identifies common ground
         │
         ├─ Minority ──────────────────────────────► lowest-confidence model
         │                                           surfaces dissent and gaps
         │
         ├─ Probability ───────────────────────────► compute average / median score
         │                                           median model writes analysis
         │
         ├─ Compose ───────────────────────────────► generate texts (Round 1)
         │                                                │
         │                                           blind peer rating (Round 2)
         │                                                │
         │                                           ranked output, no rapporteur
         │
         └─ Synthesis ─────────────────────────────► highest-confidence model
                                                     weaves ALL views into one narrative
```

### Deliberative mode (`--deliberative`)

An optional second round where each agent reads its peers' anonymous responses before finalising:

```
  ┌──────────────────────────────────────────────────────────────┐
  │  Round 1                                                     │
  │                                                              │
  │  Prompt ──► LLM 1 ──► response₁                             │
  │         ──► LLM 2 ──► response₂                             │
  │         ──► LLM N ──► responseₙ                             │
  │                   │                                          │
  │          Aggregate + Rapporteur ──► Pre-Deliberation Report  │
  └──────────────────────────────────────────────────────────────┘
                         │  responses shared anonymously
                         ▼  (agents see peers' views, not their names)
  ┌──────────────────────────────────────────────────────────────┐
  │  Round 2                                                     │
  │                                                              │
  │  Prompt + peers' Round 1 responses                          │
  │         ──► LLM 1  (sees 2…N)   ──► response₁'             │
  │         ──► LLM 2  (sees 1,3…N) ──► response₂'             │
  │         ──► LLM N  (sees 1…N-1) ──► responseₙ'             │
  │                   │                                          │
  │          Aggregate + Rapporteur ──► Post-Deliberation Report │
  └──────────────────────────────────────────────────────────────┘
```

## Installation

### From PyPI

```bash
pip install magi-core
```

### From source

```bash
git clone https://github.com/jason-chao/magi
cd magi
pip install -e .
```

## Configuration

### API Keys

You only need a key for each provider you actually use.

**CLI:** The `magi` command automatically loads a `.env` file from your working directory. Copy `.env.example` and fill in your keys:

```bash
cp .env.example .env
# then edit .env with your actual keys
```

Or set them directly as environment variables:

```bash
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here
export GEMINI_API_KEY=your-key-here
```

**Package:** API keys must be set as environment variables before calling `run()` or `run_structured()`. The package does **not** auto-load `.env` — call `load_dotenv()` yourself if needed:

```python
from dotenv import load_dotenv
load_dotenv()  # reads .env from the current directory
```

### `config.yaml`

**CLI:** Required (unless you pass `--llms` on the command line). Place it in your working directory:

```yaml
llms:
  - openai/gpt-4.1
  - anthropic/claude-haiku-4-5-20251001
  - gemini/gemini-2.5-flash

defaults:
  max_retries: 2
  min_models: 2       # abort if fewer than this many models respond
  request_timeout: 60 # seconds before an individual LLM call is abandoned
```

Fallback chains are also supported — if the primary model fails, MAGI automatically tries the next:

```yaml
llms:
  - - openai/gpt-4.1
    - openai/gpt-4o          # fallback if gpt-4.1 is unavailable
  - anthropic/claude-haiku-4-5-20251001
```

**Package:** Not required — pass the model list directly as a Python dict. No file needed:

```python
config = {
    "llms": [
        "openai/gpt-4.1",
        "anthropic/claude-haiku-4-5-20251001",
        "gemini/gemini-2.5-flash",
    ]
}
```

### `magi_core/prompts.yaml`

The bundled prompt templates are used by default and work out of the box. Override them only if you want to customise system prompts or method instructions.

## CLI Usage

After installation, the `magi` command is available:

```bash
magi "Your question here" --method Synthesis
```

Or run directly from the repository:

```bash
python magi-cli.py "Your question here" --method VoteYesNo
```

### Examples

**1. VoteYesNo — The Self-Driving Car Dilemma**
```bash
magi "A self-driving car's brakes have failed. It will kill five pedestrians unless its AI swerves onto the pavement, killing the single passenger inside. Should the AI be programmed to sacrifice its passenger?" \
  --method VoteYesNo
```

**2. VoteOptions — Organ Allocation**
```bash
magi "A hospital has one donor heart. Who should receive it?" \
  --method VoteOptions \
  --options "A 10-year-old child with decades ahead,A 45-year-old surgeon who saves hundreds of lives per year,The patient who has waited longest on the list,Whoever has the highest chance of survival post-transplant"
```

**3. Majority — Capital Punishment After Mass Atrocity**
```bash
magi "Should capital punishment be re-introduced for terrorist attacks that cause mass civilian casualties, even knowing that wrongful convictions are statistically inevitable?" \
  --method Majority
```

**4. Consensus — Abortion**
```bash
magi "At what point, if any, does terminating a pregnancy become morally impermissible, and who — if anyone — has the authority to enforce that line?" \
  --method Consensus
```

**5. Minority — The Demanding Conclusion of Effective Altruism**
```bash
magi "Anyone who spends money on luxuries while children die of preventable diseases is morally equivalent to letting a drowning child die. Is this argument sound?" \
  --method Minority
```

**6. Probability — Moral Luck**
```bash
magi "Two drivers drink the same amount and drive home. One kills a pedestrian by chance; the other arrives safely. The lucky driver deserves the same moral blame and legal punishment as the unlucky one." \
  --method Probability
```

**7. Compose — Steel-Manning Open Borders**
```bash
magi "Write the strongest possible moral argument for the claim that wealthy nations have an absolute obligation to accept unlimited refugees, regardless of cultural, economic, or security consequences." \
  --method Compose
```

**8. Synthesis — Parfit's Repugnant Conclusion**
```bash
magi "Derek Parfit's Repugnant Conclusion: a world of a trillion people living lives barely worth living is morally preferable to ten billion living very happy lives. Is this repugnant, unavoidable, or does it reveal a flaw in utilitarian reasoning?" \
  --method Synthesis
```

**9. Deliberative Round — Capital Punishment**
```bash
magi "Should capital punishment be re-introduced for terrorist attacks that cause mass civilian casualties, even knowing that wrongful convictions are statistically inevitable?" \
  --method VoteYesNo --deliberative
```

**10. JSON output — pipe into other tools**
```bash
magi "Should you pull the lever?" --method VoteYesNo --output-format json | jq '.rounds[0].aggregate'
```

### CLI Reference

| Argument | Description |
|----------|-------------|
| `prompt` | The question or issue to deliberate on |
| `--method` | `VoteYesNo` (default), `VoteOptions`, `Majority`, `Consensus`, `Minority`, `Probability`, `Compose`, `Synthesis` |
| `--llms` | Comma-separated model names (overrides `config.yaml`) |
| `--options` | Custom options for `VoteOptions` |
| `--vote-threshold` | Fraction of votes to declare a winner (default: `0.5`) |
| `--no-abstain` | Disallow abstaining in `VoteYesNo` / `Probability` |
| `--deliberative` | Enable deliberative second round |
| `--rapporteur-prompt` | Additional instructions for the rapporteur |
| `--system-prompt` | Context prepended to every agent's system prompt |
| `--output-format` | `text` (default) or `json` |
| `--config` | Path to a custom `config.yaml` |
| `--prompts` | Path to a custom `prompts.yaml` |
| `--check-models` | Probe each model with a live API call and report availability, then exit |

## JSON Output

Both the CLI and the Python package support structured JSON output, designed for integration with UIs and downstream applications.

### CLI

```bash
magi "Your question" --method VoteYesNo --output-format json
```

### Package

```python
result = await magi.run_structured(
    user_prompt="Your question",
    method="VoteYesNo",
)
import json
print(json.dumps(result, indent=2))
```

### JSON Schema

```json
{
  "schema_version": "1.0",
  "method": "VoteYesNo",
  "prompt": "Should you pull the lever?",
  "system_prompt": null,
  "deliberative": false,
  "models": ["openai/gpt-4.1", "anthropic/claude-haiku-4-5-20251001"],
  "rounds": [
    {
      "round": 1,
      "responses": [
        {
          "model": "openai/gpt-4.1",
          "pseudonym": "Participant X7K2",
          "response": "yes",
          "reason": "Utilitarian reasoning: saving five outweighs saving one.",
          "confidence_score": 0.9,
          "fallback_for": null
        }
      ],
      "errors": [
        {
          "model": "gemini/gemini-2.5-flash",
          "error": "Model not found",
          "error_category": "not_found",
          "attempted_fallbacks": ["gemini/gemini-2.5-flash", "gemini/gemini-1.5-pro"]
        }
      ],
      "aggregate": { },
      "rapporteur": {
        "model": "openai/gpt-4.1",
        "summary": "The council voted yes by a clear majority..."
      }
    }
  ]
}
```

#### `aggregate` by method

| Method | `aggregate` fields |
|--------|-------------------|
| `VoteYesNo`, `VoteOptions` | `votes` (object), `winner` (string), `threshold` (float) |
| `Probability` | `average`, `median`, `min`, `max` (all floats), `abstained_count` (int) |
| `Compose` | `ranked_candidates` (array — see below) |
| `Majority`, `Consensus`, `Minority`, `Synthesis` | `null` — result is in `rapporteur.summary` |

#### `ranked_candidates` (Compose)

```json
[
  {
    "rank": 1,
    "model": "openai/gpt-4.1",
    "pseudonym": "Participant X7K2",
    "average_score": 8.5,
    "text": "The composed paragraph text...",
    "peer_reviews": [
      {
        "reviewer_model": "anthropic/claude-haiku-4-5-20251001",
        "score": 8.0,
        "justification": "Well-structured and compelling."
      }
    ]
  }
]
```

#### Error responses

When deliberation cannot proceed (empty prompt, no models, quorum failure), `run_structured()` returns a dict with a top-level `error` key instead of `rounds`:

```json
{
  "error": "Quorum not met — 1 of 3 model(s) responded (minimum required: 2).",
  "failed_models": [
    { "model": "gemini/gemini-2.5-flash", "error_category": "not_found", "error": "Model not found" }
  ]
}
```

`run()` converts these to a plain `"Error: ..."` string.

## Package API

No `config.yaml` file needed — pass models as a dict. API keys must be set as environment variables (or loaded via `load_dotenv()`) before calling `run()` or `run_structured()`.

```python
import asyncio
from magi_core import Magi
from magi_core.utils import load_yaml, get_default_prompts_path

config = {
    "llms": [
        "openai/gpt-4.1",
        "anthropic/claude-haiku-4-5-20251001",
        "gemini/gemini-2.5-flash",
    ]
}
prompts = load_yaml(get_default_prompts_path())  # bundled prompts, no file needed

magi = Magi(config, prompts)

# Text output
result = asyncio.run(magi.run(
    user_prompt="A runaway trolley will kill five people. Should you pull the lever?",
    method="Synthesis",
    deliberative=True,
))
print(result)

# Structured JSON output
import json
result = asyncio.run(magi.run_structured(
    user_prompt="A runaway trolley will kill five people. Should you pull the lever?",
    method="VoteYesNo",
))
print(json.dumps(result, indent=2))
```

### `Magi(config, prompts)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `dict` | Configuration dict with `llms` list and optional `defaults` |
| `prompts` | `dict` | Prompt templates — load from bundled file via `load_yaml(get_default_prompts_path())` |

**`config` keys:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llms` | `list` | — | Model names or fallback lists |
| `defaults.max_retries` | `int` | `3` | Retries per model on transient errors |
| `defaults.min_models` | `int` | `1` | Minimum responding models before aborting |
| `defaults.request_timeout` | `float` | `60` | Seconds before an LLM call is abandoned |
| `defaults.vote_threshold` | `float` | `0.5` | Minimum vote fraction for a winner |
| `litellm_debug_mode` | `bool` | `false` | Enable verbose litellm logging |

Config is validated at construction time — invalid values raise `ValueError` immediately.

### `await magi.run(...)` / `await magi.run_structured(...)`

Both methods accept identical parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_prompt` | `str` | — | The question or statement to deliberate on |
| `system_prompt` | `str \| None` | `None` | Extra context prepended to the system prompt |
| `selected_llms` | `list \| None` | config value | Model names or fallback lists to use |
| `method` | `str` | `"VoteYesNo"` | One of the eight methods listed above |
| `method_options` | `dict` | `{}` | Method-specific options (see below) |
| `deliberative` | `bool` | `False` | Enable a second round where agents review peer responses |

**`method_options` keys:**

| Key | Type | Description |
|-----|------|-------------|
| `vote_threshold` | `float` | Minimum fraction of votes to declare a winner (default `0.5`) |
| `allow_abstain` | `bool` | Allow abstaining in `VoteYesNo` / `-1.0` probability in `Probability` (default `True`) |
| `options` | `list[str]` | Choices for `VoteOptions` |
| `rapporteur_prompt` | `str` | Additional instructions appended to the rapporteur prompt |

**Returns:**
- `run()` → `str` — Markdown-formatted report
- `run_structured()` → `dict` — JSON-serialisable structured result (see [JSON Schema](#json-schema))

### Method Quick Reference

```python
# Vote — trolley problem
await magi.run("Should you pull the lever?", method="VoteYesNo")

# Custom options — pandemic triage
await magi.run(
    "Who should receive the last ventilator?",
    method="VoteOptions",
    method_options={"options": ["Young Child", "Frontline Doctor", "Elderly Patient", "Random Lottery"]},
)

# Synthesis — comprehensive view of all perspectives
await magi.run(
    "Is it ever justified to lie to protect someone's feelings?",
    method="Synthesis",
)

# Probability — simulation hypothesis
await magi.run("We are living in a computer simulation.", method="Probability")

# Compose — generate and peer-review ethical arguments
await magi.run(
    "Write a paragraph arguing that lying is sometimes morally permissible.",
    method="Compose",
)

# Structured output for downstream use
result = await magi.run_structured("Should you pull the lever?", method="VoteYesNo")
winner = result["rounds"][0]["aggregate"]["winner"]
```

## Architecture

```
magi_core/
  __init__.py      Public API (Magi class, load_yaml, get_default_prompts_path)
  core.py          Orchestration, aggregation, rapporteur logic, and renderers
  cli.py           Console entry point (installed as `magi` command)
  utils.py         Pseudonym generation and YAML loading helpers
  prompts.yaml     System prompts and per-method instructions
config.yaml        Default model selection
```

**Key design decisions:**
- All LLM calls are async (`asyncio.gather`) — models are queried in parallel.
- Agents use random anonymous IDs (`Participant X7K2`) during deliberation to reduce brand bias.
- The rapporteur is selected by confidence score; ties are broken randomly.
- `Synthesis` uses the same rapporteur selection as `Majority` but with a prompt that mandates comprehensive inclusion of all perspectives.
- `run()` and `run_structured()` share a single `_deliberate()` engine; the only difference is whether the result dict is rendered to Markdown or returned as-is.
- Fallback chains trigger on permanent errors (model not found, deprecated, auth) and rate limits; timeouts and unknown errors retry the primary only.

### Fallback chain

Each slot in `config.yaml` can be a list; MAGI walks the list when a model is permanently unavailable:

```
  Slot: [primary, fallback-1, fallback-2, …]

  ┌───────────┐  deprecated /     ┌─────────────┐  error again  ┌─────────────┐
  │  Primary  │  not found /  ──► │  Fallback 1 │  ──────────►  │  Fallback 2 │
  │  Model    │  auth error /     │             │               │             │
  └───────────┘  rate limit       └─────────────┘               └──────┬──────┘
       │ ok                            │ ok                             │ all failed
       ▼                               ▼                                ▼
  result used                     result used +                error logged in
                                  fallback noted               round errors[]
                                  in report
```

Timeout and unknown errors retry the **primary only** — they do not burn through the fallback chain.

## Security

`litellm<1.44.12` is affected by **CVE-2024-8938** (arbitrary code execution via `eval()`). This package pins `litellm>=1.44.12`. Keep your dependencies up to date.

## Testing

### Unit tests (no API keys required)

```bash
pytest tests/test_core.py -v
```

### Integration tests (live LLM calls — incurs API costs)

```bash
pytest tests/test_integration_matrix.py -v
```

Integration test outputs are saved as Markdown in `test_results/`.

## License

MIT — see [LICENSE](LICENSE).
