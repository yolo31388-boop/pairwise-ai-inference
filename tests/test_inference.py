"""
游戏内AI推理与行为预测系统 - 验收测试
运行方式：python -m pytest tests/test_inference.py -q
共 16 个测试用例
"""
import math
import time

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.inference import AIInference


def make_model():
    return {
        "layers": [
            {"weights": [[0.5, -0.25], [0.1, 0.4]], "bias": [0.1, -0.2], "activation": "relu"},
            {"weights": [[0.3, 0.6]], "bias": [0.05], "activation": "sigmoid"},
        ]
    }


# 手工计算：x=[1,2] -> relu([0.1, 0.7]) -> sigmoid(0.5)
EXPECTED_OUTPUT = 1.0 / (1.0 + math.exp(-0.5))

CHEATER_SAMPLE = {"reaction_time_ms": 5.0, "accuracy": 0.99, "headshot_ratio": 0.95, "apm": 600.0}
NORMAL_SAMPLE = {"reaction_time_ms": 260.0, "accuracy": 0.34, "headshot_ratio": 0.24, "apm": 175.0}


class TestAIInference:
    def setup_method(self):
        self.system = AIInference()

    # ---- 模型加载：实际生效 + 多格式 + 验证 ----

    def test_case_01(self):
        model = self.system.load_model(make_model())
        assert model["input_dim"] == 2
        assert model["output_dim"] == 1
        assert self.system._state["model"] is model

    def test_case_02(self):
        raw = AIInference.save_model(make_model(), format="json")
        model = self.system.load_model(raw)
        assert model["input_dim"] == 2
        assert self.system.forward_pass([1.0, 2.0])[0] == pytest.approx(EXPECTED_OUTPUT)

    def test_case_03(self):
        raw = AIInference.save_model(make_model(), format="bin")
        assert raw[:4] == b"AIMD"
        model = self.system.load_model(raw)  # 自动探测格式
        assert self.system.forward_pass([1.0, 2.0])[0] == pytest.approx(EXPECTED_OUTPUT)

    def test_case_04(self):
        raw = AIInference.save_model(make_model(), format="qbin")
        assert raw[:4] == b"AIQ8"
        model = self.system.load_model(raw)
        assert model["quantized"] is True
        assert self.system.forward_pass([1.0, 2.0])[0] == pytest.approx(EXPECTED_OUTPUT, abs=0.05)

    def test_case_05(self):
        raw = bytearray(AIInference.save_model(make_model(), format="bin"))
        raw[-1] ^= 0xFF  # 篡改内容，破坏校验和
        with pytest.raises(ValueError):
            self.system.load_model(bytes(raw))
        with pytest.raises(ValueError):
            self.system.load_model(b"\x00\x01\x02\x03garbage")
        bad_structure = {"layers": [{"weights": [[1.0, 2.0]], "bias": [0.1, 0.2]}]}
        with pytest.raises(ValueError):
            self.system.load_model(bad_structure)

    # ---- 前向传播：实际生效 + 优化 + 量化 ----

    def test_case_06(self):
        self.system.load_model(make_model())
        out = self.system.forward_pass([1.0, 2.0])
        assert isinstance(out, list) and len(out) == 1
        assert out[0] == pytest.approx(EXPECTED_OUTPUT)

    def test_case_07(self):
        with pytest.raises(RuntimeError):
            self.system.forward_pass([1.0, 2.0])
        self.system.load_model(make_model())
        with pytest.raises(ValueError):
            self.system.forward_pass([1.0, 2.0, 3.0])

    def test_case_08(self):
        self.system.load_model(make_model())
        reference = self.system.forward_pass([1.0, 2.0])[0]
        model = self.system.quantize()
        assert model["quantized"] is True
        for layer in model["layers"]:
            assert "_w" not in layer  # 浮点权重已释放，只保留 INT8
            assert all(isinstance(q, int) and -127 <= q <= 127
                       for row in layer["qweights"] for q in row)
        out = self.system.forward_pass([1.0, 2.0])[0]
        assert out == pytest.approx(reference, abs=0.05)

    def test_case_09(self):
        self.system.load_model(make_model())
        start = time.perf_counter()
        for _ in range(2000):
            self.system.forward_pass([1.0, 2.0])
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0

    # ---- 行为预测：历史数据 + 置信度 ----

    def test_case_10(self):
        result = self.system.behavior_prediction("p1")
        assert result["predicted_action"] is None
        assert result["confidence"] == 0.0
        assert result["history_size"] == 0

    def test_case_11(self):
        for _ in range(6):
            self.system.behavior_prediction("p1", {"action": "move"})
            self.system.behavior_prediction("p1", {"action": "attack"})
        result = self.system.behavior_prediction("p1", {"action": "move"})
        assert result["predicted_action"] == "attack"
        assert 0.5 < result["confidence"] <= 1.0
        assert result["history_size"] == 13

    def test_case_12(self):
        self.system.behavior_prediction("p1", {"action": "move"})
        result = self.system.behavior_prediction("p1")
        assert 0.0 <= result["confidence"] <= 1.0
        # 历史越多、模式越稳定，置信度越高
        for _ in range(10):
            self.system.behavior_prediction("p1", {"action": "move"})
            result = self.system.behavior_prediction("p1", {"action": "move"})
        assert result["predicted_action"] == "move"
        assert result["confidence"] > 0.8

    # ---- 反作弊分析：特征提取 + 实时性 ----

    def test_case_13(self):
        result = self.system.anticheat_analysis("cheater", CHEATER_SAMPLE)
        assert result["flagged"] is True
        assert result["anomaly_score"] > 3.0
        assert "reaction_time_ms" in result["feature_zscores"]

    def test_case_14(self):
        result = self.system.anticheat_analysis("normal", NORMAL_SAMPLE)
        assert result["flagged"] is False
        assert result["anomaly_score"] < 3.0

    def test_case_15(self):
        # 实时性：对局进行中每个样本立即出结果并累积，无需等对局结束
        for i in range(3):
            self.system.update(0.1)
            result = self.system.anticheat_analysis("cheater", CHEATER_SAMPLE)
            assert result["realtime"] is True
            assert result["flagged"] is True
            assert result["samples_analyzed"] == i + 1
        report = self.system.match_report("cheater")
        assert report["samples_analyzed"] == 3
        assert report["flagged"] is True

    # ---- 特征提取：归一化 + 降维 ----

    def test_case_16(self):
        system = AIInference({
            "feature_ranges": {"reaction_time_ms": (0.0, 1000.0), "accuracy": (0.0, 1.0), "apm": (0.0, 400.0)},
        })
        features = system.feature_extraction(
            {"reaction_time_ms": 500.0, "accuracy": 0.5, "apm": 200.0}
        )
        assert features == pytest.approx([0.5, 0.5, 0.5])
        assert all(0.0 <= v <= 1.0 for v in features)
        # 降维：高维输入压缩到 2 维
        for i in range(5):
            reduced = system.feature_extraction(
                {"reaction_time_ms": 100.0 * i, "accuracy": 0.5, "apm": 50.0 * i},
                n_components=2,
            )
        assert len(reduced) == 2
        assert all(0.0 <= v <= 1.0 for v in reduced)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
