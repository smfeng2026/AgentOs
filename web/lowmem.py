"""低内存加载 NandhaKishorM/laya(官方 laya 包 + 本地 convaiinnovations/laya checkpoint)。

本机 ~1.9GB 内存、无 GPU:官方 Agent.__init__ 用 fp32 `load_state_dict` 需要
权重两倍以上的连续内存,必然 OOM(实测被内核杀掉)。
这里以 laya.Agent.__new__ 手工组装,权重用 safetensors mmap 的 fp16 张量
`assign` 原地替换,峰值内存约 500MB。接口与官方 Agent 完全一致。

用法:
    from lowmem import load_agent
    agent = load_agent()
    agent.predict(state, questions)
"""

import json
import resource
from pathlib import Path

import torch
from safetensors.torch import safe_open

MODELS = Path(__file__).resolve().parent.parent / "models"
LAYA_DIR = MODELS / "laya-en"


def rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def _fp16_weights(path, keep_fp32=("temperature",)):
    """safetensors -> {name: fp16 tensor} 字典(mmap 读入,不复制 fp32)。"""
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if name not in keep_fp32:
                t = t.to(torch.float16)
            out[name] = t
    return out


def load_agent(laya_dir=LAYA_DIR, dtype=torch.float16):
    """组装官方 laya.Agent 实例,权重以 dtype(默认 fp16)assign 加载。"""
    import laya
    from laya.agent import Agent, _load_tokenizer, _fix_tokenizer_config, _verify_compatibility, _InferenceGate
    from laya.common import build_model, clamp_temperature

    laya_dir = Path(laya_dir)
    _fix_tokenizer_config(str(laya_dir))
    cfg = json.loads((laya_dir / "rl_agent_config.json").read_text())

    agent = Agent.__new__(Agent)
    agent.hooks = []
    agent.hooks_raise = True
    agent.hooks_concurrent = True
    agent.hooks_timeout = None
    agent._hooks_lock = None
    agent._hooks_mutex = __import__("threading").Lock()
    agent.model_id = str(laya_dir)
    agent.model_id_or_path = str(laya_dir)
    agent.subfolder = None
    agent.revision = None
    agent.device = torch.device("cpu")
    agent._backend = None
    agent._fast = None
    agent._compiled = False
    agent._reduce_overhead = False
    agent.parallel_options = False
    agent.cpu_fallback_count = 0
    agent.last_fallback_reason = None
    agent.amp_enabled = False
    agent.mps_amp_min_rows = 1
    agent._amp_failures = 0
    agent._gate = _InferenceGate()

    agent.cfg = cfg
    agent.tok = _load_tokenizer(str(laya_dir / "tokenizer"), cfg)

    enc_dir = laya_dir / "encoder"
    agent.model = build_model(cfg, encoder_dir=str(enc_dir) if enc_dir.exists() else None, pretrained=False)

    weights = _fp16_weights(laya_dir / "model.safetensors")
    _verify_compatibility(agent.model, cfg, weights, str(laya_dir))
    # assign=True:直接替换参数对象,避免 load_state_dict 的 fp32 复制
    agent.model.load_state_dict(weights, strict=True, assign=True)
    del weights
    try:
        agent.model.encoder.config.reference_compile = False
    except Exception:
        pass
    agent.model.eval()

    agent.temperature_raw = cfg.get("temperature", [1.0, 1.0, 1.0])
    agent.temperature_by_options_raw = cfg.get("temperature_by_options", {})
    agent.temperature = [clamp_temperature(t) for t in agent.temperature_raw]
    agent.temperature_by_options = {k: clamp_temperature(v) for k, v in agent.temperature_by_options_raw.items()}
    agent.binning_map = None
    agent.dtype = dtype

    torch.set_num_threads(1)
    return agent
