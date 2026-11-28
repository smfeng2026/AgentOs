from pathlib import Path

MODEL_EVAL_ROOT = Path(__file__).resolve().parent.parent
AIDESK_ROOT = MODEL_EVAL_ROOT.parent
STORE_ROOT = AIDESK_ROOT / "model_store"
SIXH_ROOT = AIDESK_ROOT / "6h_cross_app_intent"