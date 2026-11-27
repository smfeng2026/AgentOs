from src.agents.base import BaseAgent


class GeneratorAgent(BaseAgent):
    prompt_file = "prompts/generator.txt"
    name = "generator"
    temperature = 0.9

    def run(self, spec, task):
        return self.call({"scenario_spec": spec, "generation_task": task})
