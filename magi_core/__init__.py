"""MAGI: Multi-Agent Group Intelligence.

Orchestrate a council of LLMs to deliberate on complex questions using
voting, consensus-finding, minority reporting, probability estimation,
content composition, and comprehensive synthesis.
"""
from magi_core.core import Magi
from magi_core.utils import load_yaml, get_default_prompts_path

__version__ = "0.2.0"
__all__ = ["Magi", "load_yaml", "get_default_prompts_path"]
