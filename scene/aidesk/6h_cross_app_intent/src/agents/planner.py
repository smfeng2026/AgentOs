from src.agents.base import BaseAgent


class PlannerAgent(BaseAgent):
    prompt_file = "prompts/planner.txt"
    name = "planner"
    temperature = 0.8

    def run(self, task):
        return self.call(task)
