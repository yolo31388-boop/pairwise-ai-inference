"""
游戏内AI推理与行为预测系统 - 核心模块

功能：
- load_model():          多格式模型加载（json/pickle/bin），带完整性校验
- forward_pass():        前向传播推理，支持 int8 量化以压缩内存
- behavior_prediction(): 基于历史行为序列（马尔可夫转移）的下一步行为预测，带置信度
- anticheat_analysis():  基于特征提取的实时增量反作弊分析
- feature_extraction():  特征归一化（z-score）+ PCA 降维
"""
import hashlib
import json
import math
import os
import pickle
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

MAGIC = b"GAIM"
MODEL_FORMAT = "gameai-model"
MODEL_VERSION = 1
SUPPORTED_EXTENSIONS = (".json", ".pkl", ".pickle", ".bin")
ACTIVATIONS = ("relu", "sigmoid", "tanh", "linear")

# 正常玩家的合理上限，用于反作弊特征提取
ANTICHEAT_THRESHOLDS = {
    "reaction_ms": 100.0,
    "aim_snap_deg": 120.0,
    "speed": 10.0,
    "accuracy": 0.95,
    "actions_per_sec": 12.0,
}


@dataclass
class Config:
    quantize: bool = True
    n_components: Optional[int] = None


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _activation_fn(name: str):
    if name == "relu":
        return lambda v: v if v > 0.0 else 0.0
    if name == "sigmoid":
        return _sigmoid
    if name == "tanh":
        return math.tanh
    return lambda v: v  # linear


def _top_eigenvectors(matrix: List[List[float]], k: int, iterations: int = 200) -> List[List[float]]:
    """幂迭代 + 收缩法求协方差矩阵前 k 个主成分方向（纯标准库实现）。"""
    n = len(matrix)
    residual = [row[:] for row in matrix]
    vectors = []
    for _ in range(k):
        vec = [float(i + 1) for i in range(n)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        for _ in range(iterations):
            nxt = [sum(residual[i][j] * vec[j] for j in range(n)) for i in range(n)]
            norm = math.sqrt(sum(v * v for v in nxt))
            if norm < 1e-12:
                break
            vec = [v / norm for v in nxt]
        mv = [sum(residual[i][j] * vec[j] for j in range(n)) for i in range(n)]
        eigenvalue = sum(vec[i] * mv[i] for i in range(n))
        vectors.append(vec)
        for i in range(n):
            for j in range(n):
                residual[i][j] -= eigenvalue * vec[i] * vec[j]
    return vectors


class AIInference:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 模型加载：多格式 + 完整性验证
    # ------------------------------------------------------------------
    def load_model(self, path: str) -> Dict:
        ext = os.path.splitext(str(path))[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError("不支持的模型格式: %s（支持 %s）" % (ext, ", ".join(SUPPORTED_EXTENSIONS)))
        with open(path, "rb") as fh:
            raw = fh.read()
        if not raw:
            raise ValueError("模型文件为空: %s" % path)
        if ext == ".json":
            try:
                model = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("模型文件损坏（JSON 解析失败）: %s" % exc)
        elif ext in (".pkl", ".pickle"):
            try:
                model = pickle.loads(raw)
            except Exception as exc:
                raise ValueError("模型文件损坏（pickle 解析失败）: %s" % exc)
        else:
            model = self._decode_binary_model(raw)
        self._validate_model(model)
        self._state["model"] = model
        self._state["compiled"] = self._compile_model(model)
        return model

    @staticmethod
    def _decode_binary_model(raw: bytes) -> Dict:
        header = len(MAGIC) + 4
        if len(raw) < header + 4 or raw[:len(MAGIC)] != MAGIC:
            raise ValueError("模型文件损坏（magic 校验失败）")
        (payload_len,) = struct.unpack("<I", raw[len(MAGIC):header])
        payload = raw[header:header + payload_len]
        trailer = raw[header + payload_len:header + payload_len + 4]
        if len(payload) != payload_len or len(trailer) != 4:
            raise ValueError("模型文件损坏（长度校验失败）")
        (expected_crc,) = struct.unpack("<I", trailer)
        if zlib.crc32(payload) != expected_crc:
            raise ValueError("模型文件损坏（CRC 校验失败）")
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("模型文件损坏（payload 解析失败）: %s" % exc)

    @staticmethod
    def _validate_model(model) -> None:
        if not isinstance(model, dict):
            raise ValueError("模型验证失败：顶层结构必须是 dict")
        if model.get("format") != MODEL_FORMAT:
            raise ValueError("模型验证失败：format 字段缺失或不识别")
        layers = model.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("模型验证失败：layers 缺失或为空")
        checksum = model.get("checksum")
        if checksum is not None:
            digest = hashlib.sha256(
                json.dumps(layers, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if checksum != digest:
                raise ValueError("模型验证失败：checksum 不匹配，模型可能已损坏")
        prev_out = None
        for idx, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValueError("模型验证失败：第 %d 层结构非法" % idx)
            weights = layer.get("weights")
            bias = layer.get("bias")
            activation = layer.get("activation", "linear")
            if (not isinstance(weights, list) or not weights
                    or not all(isinstance(row, list) and row for row in weights)):
                raise ValueError("模型验证失败：第 %d 层 weights 非法" % idx)
            width = len(weights[0])
            for row in weights:
                if len(row) != width or not all(isinstance(v, (int, float)) for v in row):
                    raise ValueError("模型验证失败：第 %d 层 weights 维度不一致" % idx)
            if not isinstance(bias, list) or len(bias) != len(weights):
                raise ValueError("模型验证失败：第 %d 层 bias 维度与 weights 不匹配" % idx)
            if activation not in ACTIVATIONS:
                raise ValueError("模型验证失败：第 %d 层 activation 不支持: %s" % (idx, activation))
            if prev_out is not None and width != prev_out:
                raise ValueError("模型验证失败：第 %d 层输入维度与上一层输出不匹配" % idx)
            prev_out = len(weights)

    def _compile_model(self, model: Dict) -> List[Dict]:
        """加载时预处理：量化（int8）+ 激活函数绑定，加速后续推理。"""
        quantize = bool(self.config.get("quantize", True))
        compiled = []
        for layer in model["layers"]:
            weights = [[float(v) for v in row] for row in layer["weights"]]
            entry = {
                "bias": [float(v) for v in layer["bias"]],
                "activation": _activation_fn(layer.get("activation", "linear")),
                "in_dim": len(weights[0]),
                "scale": None,
                "quantized": False,
            }
            if quantize:
                max_abs = max(abs(v) for row in weights for v in row)
                scale = max_abs / 127.0 if max_abs > 0 else 1.0
                entry["weights"] = [
                    [max(-127, min(127, int(round(v / scale)))) for v in row]
                    for row in weights
                ]
                entry["scale"] = scale
                entry["quantized"] = True
            else:
                entry["weights"] = weights
            compiled.append(entry)
        return compiled

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------
    def forward_pass(self, inputs: List[float]) -> List[float]:
        compiled = self._state.get("compiled")
        if compiled is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        x = [float(v) for v in inputs]
        if len(x) != compiled[0]["in_dim"]:
            raise ValueError(
                "输入维度 %d 与模型期望维度 %d 不匹配" % (len(x), compiled[0]["in_dim"])
            )
        for layer in compiled:
            weights = layer["weights"]
            bias = layer["bias"]
            activation = layer["activation"]
            scale = layer["scale"]
            out = []
            if scale is not None:  # int8 量化权重，推理时反量化
                for j, row in enumerate(weights):
                    acc = bias[j]
                    for i, w in enumerate(row):
                        acc += w * scale * x[i]
                    out.append(acc)
            else:
                for j, row in enumerate(weights):
                    acc = bias[j]
                    for i, w in enumerate(row):
                        acc += w * x[i]
                    out.append(acc)
            x = [activation(v) for v in out]
        return x

    # ------------------------------------------------------------------
    # 行为预测：历史序列 + 置信度
    # ------------------------------------------------------------------
    def record_observation(self, player_id: str, state) -> None:
        """记录一次玩家行为/状态观测，作为行为预测的历史数据。"""
        self._history.append((player_id, state))

    def behavior_prediction(self, player_id: str, current_state=None) -> Dict:
        seq = [state for pid, state in self._history if pid == player_id]
        if current_state is None:
            current_state = seq[-1] if seq else None
        transitions: Dict = {}
        for prev, nxt in zip(seq, seq[1:]):
            if prev == current_state:
                transitions[nxt] = transitions.get(nxt, 0) + 1
        total = sum(transitions.values())
        if total == 0:
            return {
                "player_id": player_id,
                "current_state": current_state,
                "action": None,
                "confidence": 0.0,
                "samples": 0,
                "history_size": len(seq),
            }
        action, count = max(transitions.items(), key=lambda kv: kv[1])
        # 置信度 = 转移频率占比 × 样本量饱和因子（历史越多越自信）
        confidence = (count / total) * (1.0 - math.exp(-total / 5.0))
        return {
            "player_id": player_id,
            "current_state": current_state,
            "action": action,
            "confidence": confidence,
            "samples": total,
            "history_size": len(seq),
        }

    # ------------------------------------------------------------------
    # 反作弊分析：特征提取 + 实时增量
    # ------------------------------------------------------------------
    @staticmethod
    def _anticheat_features(event: Dict) -> List[Tuple[str, float]]:
        """从原始事件提取异常特征：超出正常阈值的程度（0 表示正常）。"""
        features = []
        for key, threshold in ANTICHEAT_THRESHOLDS.items():
            value = event.get(key)
            if isinstance(value, (int, float)) and threshold > 0 and value > 0:
                # 反应时间越低越可疑（低于人类极限）；其余指标越高越可疑
                ratio = threshold / float(value) if key == "reaction_ms" else float(value) / threshold
                excess = max(0.0, ratio - 1.0)
                if excess > 0.0:
                    features.append((key, excess))
        return features

    def anticheat_analysis(self, player_id: str, events: Optional[List[Dict]] = None) -> Dict:
        """实时分析：每来一批事件立即更新该玩家的作弊概率，无需等整局结束。"""
        profiles = self._state.setdefault("anticheat", {})
        profile = profiles.setdefault(player_id, {
            "events_processed": 0,
            "score_sum": 0.0,
            "flags": [],
        })
        if events:
            for event in events:
                features = self._anticheat_features(event)
                # 多特征加权后过 sigmoid，得到单事件作弊概率
                z = 3.0 * sum(excess for _, excess in features) - 1.0
                profile["score_sum"] += _sigmoid(z)
                profile["events_processed"] += 1
                for name, _ in features:
                    if name not in profile["flags"]:
                        profile["flags"].append(name)
        count = profile["events_processed"]
        probability = profile["score_sum"] / count if count else 0.0
        return {
            "player_id": player_id,
            "cheat_probability": probability,
            "is_cheater": probability >= 0.5,
            "flags": sorted(profile["flags"]),
            "events_processed": count,
            "realtime": True,
        }

    # ------------------------------------------------------------------
    # 特征提取：归一化 + 降维
    # ------------------------------------------------------------------
    def feature_extraction(self, raw_data, n_components: Optional[int] = None):
        single = bool(raw_data) and isinstance(raw_data[0], (int, float))
        matrix = [list(raw_data)] if single else [list(row) for row in raw_data]
        if not matrix or not matrix[0]:
            raise ValueError("特征提取失败：输入数据为空")
        dims = len(matrix[0])
        if any(len(row) != dims for row in matrix):
            raise ValueError("特征提取失败：各行维度不一致")

        n = len(matrix)
        means = [sum(row[j] for row in matrix) / n for j in range(dims)]
        stds = []
        for j in range(dims):
            var = sum((row[j] - means[j]) ** 2 for row in matrix) / n
            stds.append(math.sqrt(var) or 1.0)
        normalized = [
            [(row[j] - means[j]) / stds[j] for j in range(dims)]
            for row in matrix
        ]
        self._state["feature_stats"] = {"means": means, "stds": stds}

        if n_components is None:
            n_components = self.config.get("n_components", dims)
        n_components = max(1, min(int(n_components), dims))
        if n_components < dims:
            cov = [
                [sum(row[i] * row[j] for row in normalized) / n for j in range(dims)]
                for i in range(dims)
            ]
            components = _top_eigenvectors(cov, n_components)
            result = [
                [sum(row[j] * vec[j] for j in range(dims)) for vec in components]
                for row in normalized
            ]
        else:
            result = normalized
        return result[0] if single else result

    # ------------------------------------------------------------------
    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []
