# SPDX-License-Identifier: Apache-2.0
# Encoding and result processing adapted from taobao-mnn/MNN-Pocket-0.3B-Dec.
# The runtime loader and forward pass below use MNN instead of PyTorch/PEFT.
from __future__ import annotations
import argparse
import json
import math
import re
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Any

CONTROL_TOKEN = re.compile('<\\|([A-Za-z0-9_]+)\\|>')

SPECIAL_TOKENS = ('<|fim_prefix|>', '<|fim_middle|>', '<|box_start|>', '<|box_end|>', '<|fim_suffix|>')

DEFAULT_REQUEST = {'state': 'I was charged twice for the same order. Please help.', 'questions': {'department': {'type': 'choice', 'instructions': 'Which team should handle this request?', 'criteria': {'billing': 'Charges and payment problems', 'shipping': 'Delivery and lost packages', 'returns': 'Returns and refunds'}}, 'needs_human': {'type': 'noul', 'instructions': 'Does this request require a human agent?'}, 'urgency': {'type': 'score', 'instructions': 'How urgent is this request?', 'criteria': ['No time pressure', 'Routine', 'Handle today']}}}

def render(value: Any, indent: int=0) -> str:
    """Preserve the training serializer, including insertion order and booleans."""
    pad = '  ' * indent
    if value is None:
        return ''
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return '\n'.join((f'{pad}- {render(item, indent + 1).lstrip()}' for item in value))
    if isinstance(value, dict):
        return '\n'.join((f'{pad}{key}:\n{render(item, indent + 1)}' if isinstance(item, (dict, list)) else f'{pad}{key}: {render(item)}' for key, item in value.items()))
    raise TypeError(f'unsupported JSON value: {type(value).__name__}')

def _json_value(value: Any) -> None:
    if isinstance(value, dict):
        if not all((isinstance(key, str) for key in value)):
            raise ValueError('JSON object keys must be strings')
        for item in value.values():
            _json_value(item)
    elif isinstance(value, list):
        for item in value:
            _json_value(item)
    elif value is not None and (not isinstance(value, (str, int, float, bool))):
        raise TypeError(f'unsupported JSON value: {type(value).__name__}')
    elif isinstance(value, float) and (not math.isfinite(value)):
        raise ValueError('JSON numbers must be finite')

def _question(question_id: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f'question {question_id!r} must be an object')
    kind, criteria = (value.get('type'), value.get('criteria'))
    instructions = value.get('instructions')
    _json_value(instructions)
    _json_value(criteria)

    def option(key: str, description: Any) -> str:
        return key if description is None or description == '' else key + ': ' + render(description)
    if kind == 'choice':
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f'question {question_id!r}: choice criteria must be a nonempty object')
        keys = list(criteria)
        options = [option(key, description) for key, description in criteria.items()]
    elif kind == 'noul':
        if criteria is not None and (not isinstance(criteria, dict) or not set(criteria).issubset({'false', 'true'})):
            raise ValueError(f'question {question_id!r}: noul criteria may contain only false and true')
        criteria = criteria or {}
        keys = ['false', 'true']
        options = [option('no', criteria.get('false')), option('yes', criteria.get('true'))]
    elif kind == 'score':
        if not isinstance(criteria, list) or not criteria:
            raise ValueError(f'question {question_id!r}: score criteria must be a nonempty ordered list')
        keys = [str(index) for index in range(len(criteria))]
        options = [render(description) for description in criteria]
    else:
        raise ValueError(f'question {question_id!r}: type must be choice, noul or score')
    if not 1 <= len(keys) <= 255:
        raise ValueError(f'question {question_id!r}: expected 1..255 candidates')
    return {'id': question_id, 'type': kind, 'keys': keys, 'options': options, 'instructions': render(instructions)}

class PocketDecisionModel:
    """Single-forward MNN decision inference with the checkpoint's JSON protocol."""

    def __init__(self, model_dir=".", model_file="decision-fp16.mnn", threads=4):
        import MNN.nn as nn
        import MNN.expr as F
        import numpy as np
        from tokenizers import Tokenizer
        root = Path(model_dir).expanduser().resolve()
        config = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))
        self._raw_tokenizer = Tokenizer.from_file(str(root / config["tokenizer_file"]))
        self.special_ids = [config["special_tokens"][token] for token in SPECIAL_TOKENS]
        if [self._raw_tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] != self.special_ids:
            raise ValueError("tokenizer control-token IDs do not match the checkpoint")
        class TokenizerAdapter:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
            def encode(self, text, add_special_tokens=False):
                return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
        self.tokenizer = TokenizerAdapter(self._raw_tokenizer)
        self.max_length = config["max_length"]
        self.temperature = self._temperature(config["temperature"])
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 64:
            raise ValueError("threads must be an integer in 1..64")
        path = root / model_file
        if not path.is_file():
            raise FileNotFoundError(path)
        self._F = F
        self._np = np
        self._input_names = list(config["inputs"])
        self._module = nn.load_module_from_file(
            str(path), self._input_names, ["logits"],
            dynamic=False, shape_mutable=True, rearrange=True,
            backend=F.CPU, memory_mode=F.MemoryMode.Normal,
            precision_mode=F.PrecisionMode.Normal, thread_num=threads,
        )
        if self._module is None:
            raise RuntimeError("MNN model loading failed")

    def _logits(self, encoded):
        np, F = self._np, self._F
        length = len(encoded["ids"])
        questions = len(encoded["questions"])
        candidates = max(len(q["keys"]) for q in encoded["questions"])
        options = np.zeros((1, questions, candidates), dtype=np.int32)
        for index, ends in enumerate(encoded["opt_idx"]):
            options[0, index, :len(ends)] = ends
        seg = np.asarray(encoded["seg"], dtype=np.int32)
        allow = np.tri(length, dtype=bool) & ((seg[:, None] == seg[None, :]) | (seg[None, :] == 0))
        allow &= seg[None, :] != -1
        np.fill_diagonal(allow, True)
        mask = np.zeros((1, 1, length, length), dtype=np.float32)
        mask[0, 0, ~allow] = np.float32(-3.3895313892515355e38)
        arrays = [
            np.asarray([encoded["ids"]], dtype=np.int32),
            np.asarray([encoded["pos"]], dtype=np.int32),
            mask,
            np.asarray([encoded["decide_idx"]], dtype=np.int32),
            options,
        ]
        inputs = [F.const(np.ascontiguousarray(a).reshape(-1), list(a.shape), F.NCHW,
                          F.float if a.dtype == np.float32 else F.int) for a in arrays]
        outputs = self._module.forward(inputs)
        values = np.asarray(outputs[0].read(), dtype=np.float32).reshape(questions, candidates)
        return [values[i, :len(q["keys"])].astype(float).tolist()
                for i, q in enumerate(encoded["questions"])]

    @staticmethod
    def _temperature(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('temperature must be a finite positive number')
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('temperature must be a finite positive number')
        return value

    def encode(self, state: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
        """Encode state and independent question branches without a chat template."""
        _json_value(state)
        if not isinstance(questions, dict) or not questions:
            raise ValueError('questions must be a nonempty object')
        if any((not isinstance(key, str) or not key for key in questions)):
            raise ValueError('question IDs must be nonempty strings')
    
        def tokens(text: str) -> list[int]:
            escaped = CONTROL_TOKEN.sub('<¦\\1¦>', text)
            return self.tokenizer.encode(escaped, add_special_tokens=False)
        state_id, question_id, option_id, close_id, decide_id = self.special_ids
        ids = [state_id] + tokens(render(state))
        segments = [0] * len(ids)
        positions = list(range(len(ids)))
        state_length = len(ids)
        encoded_questions = []
        decide_positions, option_positions = ([], [])
        for segment, (key, value) in enumerate(questions.items(), 1):
            question = _question(key, value)
            start = len(ids)
            branch = [question_id] + tokens(question['instructions'])
            ends = []
            for option in question['options']:
                branch += [option_id] + tokens(option) + [close_id]
                ends.append(start + len(branch) - 1)
            branch.append(decide_id)
            decide_positions.append(start + len(branch) - 1)
            option_positions.append(ends)
            ids += branch
            segments += [segment] * len(branch)
            positions += list(range(state_length, state_length + len(branch)))
            encoded_questions.append(question)
        if len(ids) > self.max_length:
            raise ValueError(f'input has {len(ids)} tokens; maximum is {self.max_length}; no truncation')
        return {'ids': ids, 'seg': segments, 'pos': positions, 'decide_idx': decide_positions, 'opt_idx': option_positions, 'questions': encoded_questions}

    def predict(self, state: Any, questions: Mapping[str, Any], temperature=None) -> dict[str, Any]:
        temperature = self.temperature if temperature is None else self._temperature(temperature)
        encoded = self.encode(state, questions)
        logits = self._logits(encoded)
        answers = {}
        for question, scores in zip(encoded['questions'], logits):
            if not all((math.isfinite(value) for value in scores)):
                raise RuntimeError('non-finite decision logits; check the selected device and dtype')
            maximum = max(scores)
            weights = [math.exp((value - maximum) / temperature) for value in scores]
            total = sum(weights)
            probabilities = [value / total for value in weights]
            winner = max(range(len(scores)), key=scores.__getitem__)
            answer = {'type': question['type'], 'probabilities': dict(zip(question['keys'], probabilities))}
            if question['type'] == 'choice':
                answer['choice'] = question['keys'][winner]
            elif question['type'] == 'noul':
                answer.update(noul=probabilities[1], decision=bool(winner))
            else:
                answer.update(score=sum((index * value for index, value in enumerate(probabilities))), legend=dict(zip(question['keys'], question['options'])))
            answers[question['id']] = answer
        return {'answers': answers, 'input_tokens': len(encoded['ids']), 'temperature': temperature}

def main():
    parser = argparse.ArgumentParser(description="MNN-Pocket-0.3B-Dec inference with MNN")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model", default="decision-fp16.mnn")
    parser.add_argument("--input", help="request JSON file, or - for stdin; omit for the original example")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, help="write clean result JSON to this file")
    args = parser.parse_args()
    try:
        if args.input is None:
            request = DEFAULT_REQUEST
        elif args.input == "-":
            request = json.load(sys.stdin)
        else:
            request = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if not isinstance(request, dict) or "state" not in request or "questions" not in request:
            raise ValueError("request must contain state and questions")
        model = PocketDecisionModel(args.model_dir, args.model, args.threads)
        result = model.predict(request["state"], request["questions"], args.temperature)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    except (ValueError, TypeError, OSError, RuntimeError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
