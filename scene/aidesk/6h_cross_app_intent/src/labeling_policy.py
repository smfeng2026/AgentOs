"""Single source for the shared labelling policy used by agents and SFT exports."""
from src.utils import read_text

POLICY_PATH = "prompts/labeling_policy.txt"


def labeling_policy():
    return read_text(POLICY_PATH)


def sft_system_prompt():
    return read_text("prompts/sft_system.txt").rstrip() + "\n\n" + labeling_policy()
