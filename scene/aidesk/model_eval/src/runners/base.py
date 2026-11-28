"""推理后端统一接口。阶段一 hf_runner；阶段三 mnn / llamacpp 适配器实现同一接口。"""
from abc import ABC, abstractmethod


class BaseRunner(ABC):
    @abstractmethod
    def load(self):
        """加载权重与分词器。失败必须抛出明确异常，不得静默降级。"""

    @abstractmethod
    def generate(self, system, user, extra_user=None):
        """返回 {text, prompt_tokens, completion_tokens, latency_ms}。"""

    @abstractmethod
    def info(self):
        """后端与依赖版本，写入 run.json。"""

    def unload(self):
        pass