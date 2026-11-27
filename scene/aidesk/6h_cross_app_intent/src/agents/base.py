import time

from src.utils import read_text, env


class BaseAgent:
    prompt_file = None
    name = "agent"
    temperature = 0.2

    def __init__(self, llm, recorder=None):
        self.llm = llm
        self.prompt = read_text(self.prompt_file)
        # Every LLM call appends {model, tokens, latency_s}; the orchestrator aggregates per agent.
        self.recorder = recorder if recorder is not None else []
        # Optional per-agent reasoning budget: LLM_<NAME>_THINKING (enabled|disabled|low|high|max).
        self.thinking = env(f"LLM_{self.name.upper()}_THINKING")

    def call(self, payload):
        t0 = time.time()
        result = self.llm.generate_json(
            self.prompt, payload, temperature=self.temperature, recorder=self.recorder, thinking=self.thinking
        )
        self.last_latency_s = round(time.time() - t0, 2)
        return result
