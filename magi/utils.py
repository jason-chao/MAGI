import yaml
import os
import random
import string
from typing import Dict, Any

def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def get_default_prompts_path() -> str:
    """Returns the absolute path to the default prompts.yaml file bundled with the package."""
    return os.path.join(os.path.dirname(__file__), 'prompts.yaml')

def generate_pseudonyms(names: list[str]) -> Dict[str, str]:
    """
    Generate random pseudonyms for a list of names.
    Returns a dictionary mapping original names to pseudonyms.
    """
    adjectives = ["Happy", "Brave", "Calm", "Wise", "Swift", "Eager", "Bright", "Gentle"]
    nouns = ["Lion", "Eagle", "Dolphin", "Owl", "Tiger", "Fox", "Bear", "Wolf"]
    
    mapping = {}
    used_pseudos = set()
    
    for name in names:
        while True:
            pseudo = f"{random.choice(adjectives)} {random.choice(nouns)}"
            if pseudo not in used_pseudos:
                mapping[name] = pseudo
                used_pseudos.add(pseudo)
                break
    return mapping

def resolve_tie(items: list[Any]) -> Any:
    """Randomly choose one from the list in case of a tie."""
    if not items:
        return None
    return random.choice(items)
