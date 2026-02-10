# Using MAGI as a Package

MAGI is designed to be used not only as a CLI tool but also as a package within your own Python applications. This allows you to integrate multi-agent decision-making directly into your workflows.

## Installation

Ensure you have installed the package:

```bash
pip install -e .
```

## Basic Usage

To use MAGI programmatically, you need to:
1. Load your configuration and prompts.
2. Initialize the `Magi` class.
3. Call the `run` method.

### Example

```python
import asyncio
from magi.core import Magi
from magi.utils import load_yaml

# 1. Load Configuration
config = load_yaml('config.yaml')
prompts = load_yaml('prompts.yaml')

# Optional: Override configuration programmatically
config['llms'] = ['gpt-4o', 'claude-3-opus-20240229', 'gemini-1.5-pro-latest']

# 2. Initialize Magi
magi = Magi(config, prompts)

# 3. Define your prompt and method
user_prompt = "Should we adopt Rust for our new high-performance backend?"
method = "VoteYesNo"

# 4. Run the aggregation
async def main():
    result = await magi.run(
        user_prompt=user_prompt,
        method=method,
        method_options={'vote_threshold': 0.6},
        deliberative=True  # Enable the deliberative round
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## API Reference

### `Magi` Class

#### `__init__(self, config: Dict[str, Any], prompts: Dict[str, Any])`

- **config**: A dictionary containing configuration settings, primarily the list of LLMs.
- **prompts**: A dictionary containing the system prompts and method instructions (usually loaded from `prompts.yaml`).

#### `async run(...)`

Executes the multi-agent deliberation process.

**Parameters:**

- `user_prompt` (str): The main question or statement for the agents.
- `system_prompt` (Optional[str]): Additional context to prepend to the system prompt.
- `selected_llms` (Optional[List[str]]): A list of model names to use. If `None`, uses the models defined in `config`.
- `method` (str): The aggregation method. Options: `'VoteYesNo'`, `'VoteOptions'`, `'Majority'`, `'Consensus'`, `'Minority'`, `'Probability'`.
- `method_options` (Dict[str, Any]): Options specific to the chosen method.
    - `vote_threshold` (float): Threshold for winning a vote (default 0.5).
    - `allow_abstain` (bool): Whether agents can abstain (default True).
    - `options` (List[str]): Choices for `VoteOptions` method.
    - `rapporteur_prompt` (str): Additional instructions for the final summary.
- `deliberative` (bool): If `True`, enables a second round where agents review peer responses before finalizing their answer. Default is `False`.

**Returns:**

- `str`: A formatted string containing the results, vote breakdown, and rapporteur summary.
