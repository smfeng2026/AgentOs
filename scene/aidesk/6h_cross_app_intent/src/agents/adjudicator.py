from src.agents.base import BaseAgent


class AdjudicatorAgent(BaseAgent):
    prompt_file = "prompts/adjudicator.txt"
    name = "adjudicator"
    temperature = 0.1

    def run(self, spec, labeled_events, model_input, blind_annotation, audit_report, taxonomy_brief=None):
        payload = {
            "scenario_spec": spec,
            "labeled_events": labeled_events,
            "model_input": model_input,
            "blind_annotation": blind_annotation,
            "audit_report": audit_report,
        }
        if taxonomy_brief:
            payload["label_taxonomy"] = taxonomy_brief
        return self.call(payload)
