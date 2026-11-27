from src.agents.base import BaseAgent


class AnnotatorAgent(BaseAgent):
    prompt_file = "prompts/annotator.txt"
    name = "annotator"
    temperature = 0.1

    def run(self, model_input, taxonomy_brief=None):
        # Important: only model_input + the label vocabulary are passed. No spec, no internal labels.
        payload = {"model_input": model_input}
        if taxonomy_brief:
            payload["label_taxonomy"] = taxonomy_brief
        return self.call(payload)
