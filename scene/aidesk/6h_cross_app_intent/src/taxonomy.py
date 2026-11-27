"""Label taxonomy loader — single source of truth for canonical categories / enums."""
from src.utils import load_yaml

DEFAULT_PATH = "config/label_taxonomy.yaml"


class Taxonomy:
    def __init__(self, data, path=None):
        self.data = data
        self.path = path
        self.categories = list(data["categories"])
        self.event_relations = list(data["event_relations"])
        self.intent_status = list(data["intent_status"])
        self.confidence = list(data["confidence"])
        self.purpose = {k: list(v) for k, v in data["purpose"].items()}

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        return cls(load_yaml(path), path=path)

    def valid_category(self, category):
        return category in self.categories

    def valid_purpose(self, category, purpose):
        return category in self.purpose and purpose in self.purpose[category]

    def purposeful_categories(self):
        """Categories that have a real purpose vocabulary (not just `unknown`)."""
        return [c for c, v in self.purpose.items() if set(v) - {"unknown"}]

    def brief(self):
        """Compact dict injected into planner / annotator / adjudicator payloads."""
        return {
            "intent_category": self.categories,
            "intent_status": self.intent_status,
            "confidence_level": self.confidence,
            "event_relation": self.event_relations,
            "purpose_by_category": self.purpose,
        }
