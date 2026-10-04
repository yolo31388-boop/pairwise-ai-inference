"""
游戏内AI推理与行为预测系统 - 核心模块

功能：
- 模型加载（多格式：JSON / 二进制 / 量化二进制，带结构与校验和验证）
- 前向传播（实际计算、权重预处理优化、INT8 量化降低内存）
- 行为预测（基于历史状态转移 + 置信度）
- 反作弊分析（特征提取 + 实时流式判定）
- 特征提取（归一化 + 降维）
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import json
import math
import os
import struct
import zlib

MODEL_VERSION = 1
MAGIC_BIN = b"AIMD"   # 二进制模型
MAGIC_QBIN = b"AIQ8"  # INT8 量化二进制模型
SUPPORTED_FORMATS = ("json", "bin", "qbin")
_ACTIVATIONS = ("linear", "relu", "tanh", "sigmoid", "softmax")

# 正常玩家各特征的 (均值, 标准差) 基线，用于反作弊 z-score 判定
DEFAULT_BASELINE = {
    "reaction_time_ms": (250.0, 60.0),
    "accuracy": (0.35, 0.12),
    "headshot_ratio": (0.25, 0.10),
    "apm": (180.0, 50.0),
    "movement_speed": (5.0, 1.0),
}


@dataclass
class Config:
    history_window: int = 50
    anticheat_threshold: float = 3.0
    n_components: int = 0
    feature_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    baseline_stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def _apply_activation(name, values):
    if name == "linear":
        return values
    if name == "relu":
        return [v if v > 0.0 else 0.0 for v in values]
    if name == "tanh":
        return [math.tanh(v) for v in values]
    if name == "sigmoid":
        return [1.0 / (1.0 + math.exp(-v)) for v in values]
    if name == "softmax":
        m = max(values)
        exps = [math.exp(v - m) for v in values]
        total = sum(exps)
        return [e / total for e in exps]
    raise ValueError(f"不支持的激活函数: {name}")


def _quantize_rows(rows):
    flat = [v for row in rows for v in row]
    bound = max(1e-12, max(abs(min(flat)), abs(max(flat))))
    scale = bound / 127.0
    qrows = [[max(-127, min(127, int(round(v / scale)))) for v in row] for row in rows]
    return qrows, scale


class AIInference:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {"time": 0.0, "model": None}
        self._history = defaultdict(
            lambda: deque(maxlen=int(self.config.get("history_window", 50)))
        )
        self._feature_ranges = {}
        self._feature_buffer = defaultdict(list)
        self._cheat_scores = defaultdict(list)

    def update(self, dt: float):
        self._state["time"] = self._state.get("time", 0.0) + float(dt)

    def reset(self):
        self._state = {"time": 0.0, "model": None}
        self._history = defaultdict(
            lambda: deque(maxlen=int(self.config.get("history_window", 50)))
        )
        self._feature_ranges = {}
        self._feature_buffer = defaultdict(list)
        self._cheat_scores = defaultdict(list)

    # ------------------------------------------------------------------
    # 模型加载：多格式 + 验证
    # ------------------------------------------------------------------
    @staticmethod
    def save_model(model: Dict, format: str = "json") -> bytes:
        """把模型字典序列化为指定格式的字节流（二进制格式带 CRC32 校验）。"""
        if format not in SUPPORTED_FORMATS:
            raise ValueError(f"不支持的模型格式: {format}")
        payload_model = dict(model)
        payload_model.setdefault("format_version", MODEL_VERSION)
        if format == "qbin":
            payload_model = AIInference._quantize_model_dict(payload_model)
        payload = json.dumps(payload_model).encode("utf-8")
        if format == "json":
            return payload
        magic = MAGIC_QBIN if format == "qbin" else MAGIC_BIN
        header = magic + struct.pack(">I", len(payload)) + struct.pack(">I", zlib.crc32(payload))
        return header + payload

    @staticmethod
    def _quantize_model_dict(model: Dict) -> Dict:
        qmodel = {k: v for k, v in model.items() if k != "layers"}
        qlayers = []
        for layer in model["layers"]:
            qweights, scale = _quantize_rows(layer["weights"])
            qlayers.append({
                "qweights": qweights,
                "scale": scale,
                "bias": list(layer["bias"]),
                "activation": layer.get("activation", "linear"),
            })
        qmodel["layers"] = qlayers
        qmodel["quantized"] = True
        return qmodel

    def load_model(self, source, format: Optional[str] = None) -> Dict:
        """加载模型。source 可为 dict、字节流或文件路径；format 缺省时自动探测。"""
        if isinstance(source, dict):
            model = dict(source)
        else:
            if isinstance(source, (str, os.PathLike)):
                with open(source, "rb") as fh:
                    raw = fh.read()
            elif isinstance(source, (bytes, bytearray)):
                raw = bytes(source)
            else:
                raise TypeError(f"不支持的模型来源类型: {type(source)!r}")
            fmt = format or self._detect_format(raw)
            if fmt in ("bin", "qbin"):
                model = self._parse_binary(raw, fmt)
            elif fmt == "json":
                try:
                    model = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"模型 JSON 解析失败: {exc}") from exc
            else:
                raise ValueError(f"不支持的模型格式: {fmt}")
        self._validate_model(model)
        prepared = self._prepare_model(model)
        self._state["model"] = prepared
        return prepared

    @staticmethod
    def _detect_format(raw: bytes) -> str:
        if raw[:4] == MAGIC_BIN:
            return "bin"
        if raw[:4] == MAGIC_QBIN:
            return "qbin"
        return "json"

    @staticmethod
    def _parse_binary(raw: bytes, fmt: str) -> Dict:
        magic = MAGIC_QBIN if fmt == "qbin" else MAGIC_BIN
        if len(raw) < 12 or raw[:4] != magic:
            raise ValueError("模型文件损坏：魔数不正确")
        (length,) = struct.unpack(">I", raw[4:8])
        (checksum,) = struct.unpack(">I", raw[8:12])
        payload = raw[12:]
        if len(payload) != length:
            raise ValueError("模型文件损坏：长度不匹配")
        if zlib.crc32(payload) != checksum:
            raise ValueError("模型文件损坏：CRC32 校验失败")
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"模型文件损坏：内容解析失败: {exc}") from exc

    @staticmethod
    def _validate_model(model: Dict):
        if not isinstance(model, dict):
            raise ValueError("模型必须是字典结构")
        layers = model.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("模型缺少有效的 layers")
        prev_out = None
        for idx, layer in enumerate(layers):
            weights = layer.get("qweights") or layer.get("weights")
            if not isinstance(weights, list) or not weights:
                raise ValueError(f"第 {idx} 层缺少权重")
            width = len(weights[0])
            for row in weights:
                if not isinstance(row, list) or len(row) != width:
                    raise ValueError(f"第 {idx} 层权重维度不一致")
                if not all(isinstance(v, (int, float)) for v in row):
                    raise ValueError(f"第 {idx} 层权重包含非数值")
            bias = layer.get("bias")
            if not isinstance(bias, list) or len(bias) != len(weights):
                raise ValueError(f"第 {idx} 层偏置与输出维度不匹配")
            if layer.get("activation", "linear") not in _ACTIVATIONS:
                raise ValueError(f"第 {idx} 层激活函数不支持")
            if "qweights" in layer and "scale" not in layer:
                raise ValueError(f"第 {idx} 层量化权重缺少 scale")
            if prev_out is not None and width != prev_out:
                raise ValueError(f"第 {idx} 层输入维度与上一层输出不匹配")
            prev_out = len(weights)

    @staticmethod
    def _prepare_model(model: Dict) -> Dict:
        layers = []
        for layer in model["layers"]:
            if "qweights" in layer:
                scale = float(layer["scale"])
                weights = [[q * scale for q in row] for row in layer["qweights"]]
                entry = {
                    "qweights": layer["qweights"],
                    "scale": scale,
                    "bias": [float(b) for b in layer["bias"]],
                    "activation": layer.get("activation", "linear"),
                }
            else:
                weights = [[float(v) for v in row] for row in layer["weights"]]
                entry = {
                    "bias": [float(b) for b in layer["bias"]],
                    "activation": layer.get("activation", "linear"),
                }
            entry["_w"] = weights  # 预处理后的浮点权重，前向传播直接用
            layers.append(entry)
        return {
            "format_version": model.get("format_version", MODEL_VERSION),
            "quantized": bool(model.get("quantized")),
            "layers": layers,
            "input_dim": len(layers[0]["_w"][0]),
            "output_dim": len(layers[-1]["_w"]),
        }

    def quantize(self) -> Dict:
        """把已加载模型量化为 INT8 存储，降低内存占用；前向传播时即时反量化。"""
        model = self._state.get("model")
        if model is None:
            raise RuntimeError("模型未加载，无法量化")
        if model.get("quantized"):
            return model
        for layer in model["layers"]:
            qweights, scale = _quantize_rows(layer["_w"])
            layer["qweights"] = qweights
            layer["scale"] = scale
            del layer["_w"]  # 只保留 INT8 权重，释放浮点权重内存
        model["quantized"] = True
        return model

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------
    def forward_pass(self, inputs) -> List[float]:
        model = self._state.get("model")
        if model is None:
            raise RuntimeError("模型未加载，无法执行前向传播")
        x = [float(v) for v in inputs]
        if len(x) != model["input_dim"]:
            raise ValueError(
                f"输入维度 {len(x)} 与模型输入维度 {model['input_dim']} 不匹配"
            )
        for layer in model["layers"]:
            weights = layer.get("_w")
            if weights is None:  # 量化模型：即时反量化
                scale = layer["scale"]
                weights = [[q * scale for q in row] for row in layer["qweights"]]
            bias = layer["bias"]
            out = []
            for row, b in zip(weights, bias):
                acc = b
                for w, v in zip(row, x):
                    acc += w * v
                out.append(acc)
            x = _apply_activation(layer["activation"], out)
        return x

    # ------------------------------------------------------------------
    # 行为预测：历史状态转移 + 置信度
    # ------------------------------------------------------------------
    def behavior_prediction(self, player_id, current_state: Optional[Dict] = None) -> Dict:
        hist = self._history[player_id]
        if current_state is not None:
            entry = dict(current_state)
            entry["t"] = self._state.get("time", 0.0)
            hist.append(entry)
        result = {
            "player_id": player_id,
            "predicted_action": None,
            "confidence": 0.0,
            "history_size": len(hist),
        }
        actions = [e["action"] for e in hist if e.get("action") is not None]
        if len(actions) < 2:
            return result
        transitions = defaultdict(lambda: defaultdict(int))
        for prev, nxt in zip(actions, actions[1:]):
            transitions[prev][nxt] += 1
        followers = transitions.get(actions[-1])
        if not followers:
            return result
        total = sum(followers.values())
        best_action, best_count = max(followers.items(), key=lambda kv: kv[1])
        # 拉普拉斯平滑置信度：历史越多、转移越集中，置信度越高
        confidence = (best_count + 1.0) / (total + len(transitions))
        result["predicted_action"] = best_action
        result["confidence"] = max(0.0, min(1.0, confidence))
        return result

    # ------------------------------------------------------------------
    # 反作弊分析：特征提取 + 实时流式判定
    # ------------------------------------------------------------------
    def anticheat_analysis(self, player_id, sample: Dict) -> Dict:
        if not isinstance(sample, dict) or not sample:
            raise ValueError("反作弊样本必须是非空字典")
        baseline = self.config.get("baseline_stats") or DEFAULT_BASELINE
        zscores = {}
        for name, value in sample.items():
            if not isinstance(value, (int, float)):
                continue
            stats = baseline.get(name)
            if stats is None:
                continue
            mean, std = stats
            zscores[name] = (float(value) - mean) / (std or 1e-6)
        score = max((abs(z) for z in zscores.values()), default=0.0)
        threshold = float(self.config.get("anticheat_threshold", 3.0))
        # 实时：每个样本到达即判定并累积，无需等对局结束
        self._cheat_scores[player_id].append(score)
        return {
            "player_id": player_id,
            "anomaly_score": score,
            "flagged": score > threshold,
            "feature_zscores": zscores,
            "samples_analyzed": len(self._cheat_scores[player_id]),
            "realtime": True,
            "timestamp": self._state.get("time", 0.0),
        }

    def match_report(self, player_id) -> Dict:
        scores = self._cheat_scores.get(player_id, [])
        threshold = float(self.config.get("anticheat_threshold", 3.0))
        return {
            "player_id": player_id,
            "samples_analyzed": len(scores),
            "max_anomaly_score": max(scores, default=0.0),
            "mean_anomaly_score": sum(scores) / len(scores) if scores else 0.0,
            "flagged": any(s > threshold for s in scores),
        }

    # ------------------------------------------------------------------
    # 特征提取：归一化 + 降维
    # ------------------------------------------------------------------
    def feature_extraction(self, raw_data, n_components: Optional[int] = None) -> List[float]:
        if isinstance(raw_data, dict):
            items = list(raw_data.items())
        elif isinstance(raw_data, (list, tuple)):
            items = [(f"f{i}", v) for i, v in enumerate(raw_data)]
        else:
            raise TypeError("特征提取输入必须是字典或序列")
        configured = self.config.get("feature_ranges") or {}
        normalized = []
        for name, value in items:
            v = float(value)
            if name in configured:
                lo, hi = configured[name]
            else:
                lo, hi = self._feature_ranges.get(name, (v, v))
                lo, hi = min(lo, v), max(hi, v)
                self._feature_ranges[name] = (lo, hi)
            span = hi - lo
            nv = 0.0 if span <= 0 else (v - lo) / span
            nv = max(0.0, min(1.0, nv))
            normalized.append((name, nv))
            buf = self._feature_buffer[name]
            buf.append(nv)
            if len(buf) > 100:
                buf.pop(0)
        k = n_components if n_components is not None else int(self.config.get("n_components", 0))
        if k and k < len(normalized):
            # 降维：按运行缓冲区中的方差保留信息量最大的 k 个特征
            def variance(item):
                buf = self._feature_buffer[item[0]]
                if len(buf) < 2:
                    return 0.0
                mean = sum(buf) / len(buf)
                return sum((x - mean) ** 2 for x in buf) / len(buf)
            normalized = sorted(normalized, key=variance, reverse=True)[:k]
        return [v for _, v in normalized]
