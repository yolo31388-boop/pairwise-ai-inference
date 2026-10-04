"""
游戏内AI推理与行为预测系统 - 验收测试
运行方式：python -m pytest tests/test_inference.py -q
共 16 个测试用例
"""
import json
import math
import os
import pickle
import struct
import sys
import zlib

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.inference import AIInference, MAGIC


def make_model():
    return {
        "format": "gameai-model",
        "version": 1,
        "layers": [
            {"weights": [[0.5, -0.25], [0.1, 0.4]], "bias": [0.1, -0.2], "activation": "relu"},
            {"weights": [[0.3, 0.6]], "bias": [0.05], "activation": "linear"},
        ],
    }


def write_json_model(path, model):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(model, fh)


def write_bin_model(path, model):
    payload = json.dumps(model).encode("utf-8")
    data = MAGIC + struct.pack("<I", len(payload)) + payload
    data += struct.pack("<I", zlib.crc32(payload))
    with open(path, "wb") as fh:
        fh.write(data)


class TestAIInference:
    def setup_method(self):
        self.system = AIInference()

    # ---- 模型加载：多格式 + 验证 ----

    def test_case_01(self, tmp_path):
        path = tmp_path / "model.json"
        write_json_model(path, make_model())
        model = self.system.load_model(str(path))
        assert model["format"] == "gameai-model"
        assert len(model["layers"]) == 2

    def test_case_02(self, tmp_path):
        path = tmp_path / "model.pkl"
        with open(path, "wb") as fh:
            pickle.dump(make_model(), fh)
        model = self.system.load_model(str(path))
        assert len(model["layers"]) == 2

    def test_case_03(self, tmp_path):
        path = tmp_path / "model.bin"
        write_bin_model(path, make_model())
        model = self.system.load_model(str(path))
        assert model["layers"][0]["activation"] == "relu"

    def test_case_04(self, tmp_path):
        path = tmp_path / "model.bin"
        write_bin_model(path, make_model())
        raw = bytearray(path.read_bytes())
        raw[-6] ^= 0xFF  # 破坏 payload，CRC 校验应失败
        path.write_bytes(bytes(raw))
        with pytest.raises(ValueError):
            self.system.load_model(str(path))

    def test_case_05(self, tmp_path):
        path = tmp_path / "model.json"
        bad = make_model()
        bad["layers"][0]["bias"] = [0.1]  # bias 维度与 weights 不匹配
        write_json_model(path, bad)
        with pytest.raises(ValueError):
            self.system.load_model(str(path))

    def test_case_06(self, tmp_path):
        path = tmp_path / "model.txt"
        write_json_model(path, make_model())
        with pytest.raises(ValueError):
            self.system.load_model(str(path))

    # ---- 前向传播：实际生效 + 优化 + 量化 ----

    def test_case_07(self, tmp_path):
        path = tmp_path / "model.json"
        write_json_model(path, make_model())
        self.system.load_model(str(path))
        out = self.system.forward_pass([1.0, 2.0])
        assert isinstance(out, list) and len(out) == 1
        assert all(isinstance(v, float) for v in out)
        # 手算：layer1 -> [relu(0.1), relu(0.7)]，layer2 -> 0.3*0.1+0.6*0.7+0.05 = 0.5
        assert abs(out[0] - 0.5) < 0.05

    def test_case_08(self):
        with pytest.raises(RuntimeError):
            self.system.forward_pass([1.0, 2.0])

    def test_case_09(self, tmp_path):
        path = tmp_path / "model.json"
        write_json_model(path, make_model())
        self.system.load_model(str(path))
        compiled = self.system._state["compiled"]
        assert all(layer["quantized"] for layer in compiled)
        # int8 量化存储，内存占用远小于 float
        assert all(isinstance(w, int) and -127 <= w <= 127
                   for layer in compiled for row in layer["weights"] for w in row)

    # ---- 行为预测：历史数据 + 置信度 ----

    def test_case_10(self):
        for state in ["idle", "move", "idle", "move", "idle", "move", "idle", "move"]:
            self.system.record_observation("p1", state)
        pred = self.system.behavior_prediction("p1", current_state="idle")
        assert pred["action"] == "move"
        assert pred["samples"] >= 3

    def test_case_11(self):
        for state in ["idle", "attack"] * 2:
            self.system.record_observation("p2", state)
        few = self.system.behavior_prediction("p2")["confidence"]
        for state in ["idle", "attack"] * 20:
            self.system.record_observation("p3", state)
        many = self.system.behavior_prediction("p3")["confidence"]
        assert 0.0 <= few <= 1.0
        assert 0.0 <= many <= 1.0
        assert many > few  # 历史数据越多，置信度越高

    def test_case_12(self):
        pred = self.system.behavior_prediction("ghost")
        assert pred["action"] is None
        assert pred["confidence"] == 0.0

    # ---- 反作弊分析：特征提取 + 实时性 ----

    def test_case_13(self):
        events = [
            {"reaction_ms": 20, "aim_snap_deg": 175, "accuracy": 0.99, "actions_per_sec": 30}
            for _ in range(5)
        ]
        result = self.system.anticheat_analysis("hacker", events)
        assert result["is_cheater"]
        assert result["cheat_probability"] > 0.5
        assert result["flags"]  # 有具体异常特征标记

    def test_case_14(self):
        events = [
            {"reaction_ms": 250, "aim_snap_deg": 30, "accuracy": 0.45, "actions_per_sec": 4}
            for _ in range(5)
        ]
        result = self.system.anticheat_analysis("normal_player", events)
        assert not result["is_cheater"]
        assert result["cheat_probability"] < 0.5

    def test_case_15(self):
        # 实时性：对局进行中每来一个事件立即出结果，无需等整局结束
        event = {"reaction_ms": 15, "aim_snap_deg": 170, "accuracy": 0.99}
        first = self.system.anticheat_analysis("live", [event])
        assert first["events_processed"] == 1
        assert first["realtime"]
        second = self.system.anticheat_analysis("live", [event])
        assert second["events_processed"] == 2
        assert second["is_cheater"]  # 对局中途即可识别作弊

    # ---- 特征提取：归一化 + 降维 ----

    def test_case_16(self):
        data = [
            [1.0, 100.0, 5.0],
            [2.0, 200.0, 10.0],
            [3.0, 300.0, 15.0],
            [4.0, 400.0, 20.0],
        ]
        normalized = self.system.feature_extraction(data, n_components=3)
        for col in zip(*normalized):
            assert abs(sum(col) / len(col)) < 1e-9  # 每列均值≈0
            var = sum(v * v for v in col) / len(col)
            assert abs(math.sqrt(var) - 1.0) < 1e-9  # 每列标准差≈1
        reduced = self.system.feature_extraction(data, n_components=2)
        assert all(len(row) == 2 for row in reduced)  # 3 维降到 2 维


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
