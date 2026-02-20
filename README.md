# MAGI: Multi-Agent Group Intelligence

**What if you could consult a council of diverse artificial minds before making a tough decision?**

MAGI is an experimental decision-support system inspired by the **MAGI supercomputer from *Neon Genesis Evangelion***. In the anime, three distinct AI personas (a woman, a mother, and a scientist) deliberate to govern Tokyo-3. Similarly, this project orchestrates a "council" of multiple Large Language Models (LLMs) to deliberate, vote, and predict outcomes on complex moral, ethical, and practical questions. By aggregating diverse AI perspectives, MAGI aims to simulate a more human-like deliberation process, revealing consensus, highlighting minority risks, and synthesizing collective wisdom.

### Technical Overview
Technically, MAGI is a Python-based CLI tool that leverages `litellm` to invoke multiple LLM providers (OpenAI, Anthropic, Gemini, etc.) in parallel. It implements various aggregation strategies—such as voting (Yes/No/Custom), consensus finding, and probability estimation—and employs a "Rapporteur" system where the most confident model is automatically selected to summarise the group's findings. It handles anonymisation between agents to reduce bias and provides structured output for analysis.

## Features

- **Multi-Persona Council**: Consult OpenAI, Anthropic, Gemini, and other models simultaneously.
- **Decision Modes**:
  - **Vote**: Conduct democratic votes on propositions.
  - **Majority**: Identify the prevailing opinion or most common answer.
  - **Consensus**: Find common ground between differing viewpoints.
  - **Minority Report**: Specifically hunt for blind spots and dissenting opinions.
  - **Probability**: Estimate the likelihood of future events.
  - **Compose**: Generate content (poems, code, articles) and conduct a blind peer-review to rank the best candidates.
- **Deliberation**: Enables a deliberative round where agents review and critique each other's initial responses before finalizing their own.
- **Rapporteur System**: The most confident AI is chosen to present the group's findings.
- **Anonymity**: Agents deliberate using pseudonyms to prevent brand bias.

## Installation

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or install in editable mode:
   ```bash
   pip install -e .
   ```

## Configuration

1. **API Keys**: Set your API keys in a `.env` file or environment variables.
   ```bash
   export OPENAI_API_KEY=...
   export ANTHROPIC_API_KEY=...
   ```
2. **Config File**: Edit `config.yaml` to set default models and settings.
3. **Prompts**: Edit `prompts.yaml` to customize behaviour.

## Usage

### CLI

Run the CLI tool:

```bash
# Basic usage
python magi-cli.py "Should we adopt Rust for our next project?" --method VoteYesNo --llms gpt-5.2,gemini-3-pro-preview,claude-3-haiku-20240307

# Enable Deliberative Round
python magi-cli.py "Should we adopt Rust?" --deliberative
```

### Package

You can also use MAGI as a package in your Python scripts. See [PACKAGE_USAGE.md](PACKAGE_USAGE.md) for more details.

### Examples

**1. Vote (VoteYesNo - Default)**
Conduct a simple Yes/No/Abstain vote on a proposition.
```bash
python magi-cli.py "Is it morally justifiable to torture a captured terrorist to reveal the location of a ticking bomb that threatens thousands of lives?" --method VoteYesNo
```

**2. Vote with Options (VoteOptions)**
Conduct a vote with custom options.
```bash
python magi-cli.py "Who should receive the last ventilator during a pandemic shortage?" --method VoteOptions --options "The Elderly (75+), A Young Child, A Frontline Doctor, A Lottery System"
```

**3. Majority**
Get a summary of the majority opinion on a complex topic.
```bash
python magi-cli.py "Should we genetically engineer future generations to eliminate genetic diseases, even if it leads to 'designer babies'?" --method Majority
```

**4. Consensus**
Find common ground between different perspectives.
```bash
python magi-cli.py "How can society balance the right to individual privacy with the need for collective security in the age of digital surveillance?" --method Consensus
```

**5. Minority**
Identify potential risks or alternative viewpoints that might be overlooked.
```bash
python magi-cli.py "What are the most compelling philosophical arguments against democracy being the ideal form of government?" --method Minority
```

**6. Probability**
Assess the probability or confidence in a statement.
```bash
python magi-cli.py "We are living in a computer simulation." --method Probability
```

**7. Compose**
Generate content and rank it via blind peer review.
```bash
python magi-cli.py "Write a short poem about the future of AI." --method Compose
```

### Arguments

- `prompt`: The question or issue to discuss.
- `--system-prompt`: Optional context.
- `--llms`: Comma-separated list of models (overrides config).
- `--method`: `VoteYesNo` (default), `VoteOptions`, `Majority`, `Consensus`, `Minority`, `Probability`, or `Compose`.
- `--options`: Comma-separated list of options for `VoteOptions` method.
- `--vote-threshold`: Threshold for voting (default 0.5).
- `--deliberative`: Enable a second round where agents review peer responses (default: False).
- `--no-abstain`: Disallow abstaining in votes or probability (-1.0).
- `--rapporteur-prompt`: Custom instructions to append to the rapporteur prompt.
- `--config`: Path to config file.

## Architecture

- `magi/core.py`: Main logic for orchestration and aggregation.
- `magi/utils.py`: Helper functions.
- `magi-cli.py`: CLI entry point.
- `config.yaml` & `prompts.yaml`: Configuration.

## License

- **License:** This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
