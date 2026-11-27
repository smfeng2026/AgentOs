from src.agents.base import BaseAgent


class ReviewerAgent(BaseAgent):
    prompt_file = "prompts/reviewer.txt"
    name = "reviewer"
    temperature = 0.1

    def run(self, spec, labeled_events, blind_annotation):
        return self.call({
            "scenario_spec": spec,
            "labeled_events": labeled_events,
            "blind_annotation": blind_annotation
        })
