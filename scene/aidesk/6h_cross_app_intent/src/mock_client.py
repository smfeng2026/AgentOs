"""Deterministic smoke-test client. Not suitable for dataset production."""
class MockClient:
    def generate_json(self, system_prompt, user_input, schema=None):
        # Smoke test deliberately returns a clear error rather than fake high-quality labels.
        raise RuntimeError(
            "MOCK_LLM is a placeholder. Configure a real OpenAI-compatible API. "
            "Do not use fabricated mock samples as training data."
        )
