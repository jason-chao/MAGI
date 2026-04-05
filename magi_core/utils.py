import yaml
import os
import random
from typing import Dict, Any

def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def get_default_prompts_path() -> str:
    """Returns the absolute path to the default prompts.yaml file bundled with the package."""
    return os.path.join(os.path.dirname(__file__), 'prompts.yaml')

# Unambiguous alphanumeric characters (O, 0, I, 1 excluded to avoid visual confusion).
_ID_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_ID_LENGTH = 4


def _random_id() -> str:
    return "".join(random.choices(_ID_CHARS, k=_ID_LENGTH))


def generate_pseudonyms(names: list[str]) -> Dict[str, str]:
    """
    Assign each model name a unique random ID label, e.g. 'Participant X7K2'.
    IDs use unambiguous alphanumeric characters so they carry no ordering or connotation.
    Returns a dict mapping each original name to its label.
    """
    used: set[str] = set()
    mapping: Dict[str, str] = {}
    for name in names:
        pid = _random_id()
        while pid in used:
            pid = _random_id()
        used.add(pid)
        mapping[name] = f"Participant {pid}"
    return mapping

def resolve_tie(items: list[Any]) -> Any:
    """Randomly choose one from the list in case of a tie."""
    if not items:
        return None
    return random.choice(items)
