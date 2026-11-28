"""阶段一：transformers 服务端能力基线。torch/transformers 延迟导入 —— 未安装时其余 CLI 仍可用。"""
import time

from .base import BaseRunner
from ..store import unsupported_formats

DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


class HFRunner(BaseRunner):
    def __init__(self, resolved, device="cpu", dtype=None, decoding=None, allow_download=False):
        self.resolved = resolved
        self.device = device
        self.dtype = dtype or resolved["base_config"].get("torch_dtype", "bfloat16")
        self.decoding = decoding or {}
        self.allow_download = allow_download
        self.model = None
        self.tokenizer = None
        self._versions = {}

    def _source(self, weights_dir, repo_id):
        if weights_dir:
            return weights_dir
        if not self.allow_download:
            raise RuntimeError(
                f"本地权重缺失（{repo_id}）。下载须经用户确认后用 --allow-download 显式开启；"
                "或先把权重放入 model_store 对应目录。")
        return repo_id

    def load(self):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        bad = unsupported_formats(self.resolved)
        if bad:
            raise NotImplementedError(
                f"HF runner 不支持这些权重格式: {bad}；请等待 mnn/llamacpp 适配器（阶段三）。")
        cfg = self.resolved["base_config"]
        trust = bool(cfg.get("trust_remote_code", False))
        kind = self.resolved["kind"]
        if kind == "lora_adapter":
            from peft import PeftModel
            base_src = self._source(self.resolved["base_weights_dir"], self.resolved["base_repo_id"])
            model = AutoModelForCausalLM.from_pretrained(
                base_src, dtype=getattr(torch, DTYPES[self.dtype]), trust_remote_code=trust)
            model = PeftModel.from_pretrained(model, self.resolved["weights_dir"])
            tok_src = base_src
        else:
            src = self._source(self.resolved["weights_dir"], self.resolved["repo_id"])
            model = AutoModelForCausalLM.from_pretrained(
                src, dtype=getattr(torch, DTYPES[self.dtype]), trust_remote_code=trust)
            tok_src = src
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=trust)
        model.eval()
        model.to(self.device)
        self.model = model
        self._versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                          "device": self.device, "dtype": self.dtype}
        return self

    def generate(self, system, user, extra_user=None):
        import torch
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if extra_user:
            messages.append({"role": "user", "content": extra_user})
        kwargs = self.resolved["base_config"].get("chat_template_kwargs") or {}
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = {
            "max_new_tokens": int(self.decoding.get("max_new_tokens", 1600)),
            "do_sample": bool(self.decoding.get("do_sample", False)),
            "repetition_penalty": float(self.decoding.get("repetition_penalty", 1.0)),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        new_tokens = out[0][prompt_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {"text": text, "prompt_tokens": prompt_tokens,
                "completion_tokens": int(new_tokens.shape[0]), "latency_ms": latency_ms}

    def info(self):
        return {"backend": "hf", **self._versions}

    def unload(self):
        self.model = None
        self.tokenizer = None