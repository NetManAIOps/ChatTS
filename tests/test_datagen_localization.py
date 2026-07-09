import copy
import importlib
import re
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from chatts.ts_generator.generate import generate_time_series
from chatts.ts_generator.local_changes import ChangeFactory


PLACEHOLDER_RE = re.compile(r"<\|-?\d+\|>")
POINT_RE = re.compile(r"\bpoint (-?\d+)\b")


def _base_attribute_pool(change_type, position_start, seq_len=256):
    return {
        "seasonal": {"type": "no periodic fluctuation"},
        "trend": {"type": "keep steady"},
        "frequency": {"type": "no periodicity"},
        "noise": {"type": "almost no noise"},
        "local": [
            {
                "type": change_type,
                "position_start": position_start,
                "amplitude": 10.0,
            }
        ],
        "overall_amplitude": 10.0,
        "overall_bias": 0.0,
        "seq_len": seq_len,
    }


class DatagenLocalizationAuditTests(unittest.TestCase):
    def assert_text_uses_raw_points(self, text, seq_len):
        self.assertNotRegex(text, PLACEHOLDER_RE)
        for point in POINT_RE.findall(text):
            point = int(point)
            self.assertGreaterEqual(point, 0, text)
            self.assertLess(point, seq_len, text)

    def test_all_local_changes_use_valid_raw_points_at_boundaries(self):
        seq_len = 256
        for change_type in ChangeFactory.get_supported_types():
            min_length = ChangeFactory.create_change(change_type).get_min_length()
            starts = [0, seq_len - min_length]
            for position_start in starts:
                with self.subTest(change_type=change_type, position_start=position_start):
                    np.random.seed(7)
                    import random

                    random.seed(7)
                    _, attribute_pool = generate_time_series(
                        copy.deepcopy(_base_attribute_pool(change_type, position_start, seq_len)),
                        seq_len,
                    )
                    self.assertEqual(len(attribute_pool["local"]), 1)
                    local = attribute_pool["local"][0]
                    self.assertGreaterEqual(local["position_start"], 0)
                    self.assertLess(local["position_start"], seq_len)
                    self.assertGreaterEqual(local["position_end"], local["position_start"])
                    self.assertLess(local["position_end"], seq_len)
                    self.assert_text_uses_raw_points(local["detail"], seq_len)

    def test_uts_template_clamps_question_points_and_keeps_raw_length_with_sp_encoding(self):
        import chatts.align.uts_template_qa as uts

        seq_len = 256
        raw_timeseries = np.arange(seq_len, dtype=float)
        generated_pool = _base_attribute_pool("sudden increase", 0, seq_len)
        generated_pool["local"][0].update(
            {
                "position_end": 1,
                "detail": "a sudden increase occurred between point 0 and point 1",
            }
        )

        randint_values = iter([-5, 255, 255, 255])

        def fake_randint(_low, _high):
            return next(randint_values)

        with patch.object(uts, "SEQ_LEN", seq_len), patch.object(uts, "ENCODING_METHOD", "sp"):
            with patch.object(uts, "generate_random_attributes", return_value={}):
                with patch.object(uts, "generate_time_series", return_value=(raw_timeseries, generated_pool)):
                    with patch.object(uts, "attribute_to_text", return_value="summary"):
                        with patch.object(uts.random, "randint", side_effect=fake_randint):
                            rows = uts.generate_single_dataset()

        self.assertIn(f"length {seq_len}", rows[0]["instruction"])
        self.assertEqual(rows[0]["timeseries"][0].shape[0], seq_len * 2)
        for row in rows:
            self.assert_text_uses_raw_points(row["question"], seq_len)
            self.assert_text_uses_raw_points(row["answer"], seq_len)

    def test_mts_local_llm_preserves_each_positive_cluster_position(self):
        fake_llm_utils = types.ModuleType("chatts.utils.llm_utils")
        fake_llm_utils.LLMClient = object

        with patch.dict(sys.modules, {"chatts.utils.llm_utils": fake_llm_utils}):
            import chatts.align.mts_local_llm_qa as mts

            mts = importlib.reload(mts)

        for cluster_count in [1, 2, 3]:
            with self.subTest(cluster_count=cluster_count):
                self._assert_mts_cluster_case(mts, cluster_count)

    def _assert_mts_cluster_case(self, mts, cluster_count):
        seq_len = 256
        positions = [40, 120, 210][:cluster_count]
        position_iter = iter(positions)

        mts.metric_config = [
            {
                "category": "database",
                "cluster": {
                    "cluster_a": ["a0", "a1", "a2"],
                    "cluster_b": ["b0", "b1", "b2"],
                    "cluster_c": ["c0", "c1", "c2"],
                    "unused": ["n0", "n1", "n2", "n3"],
                },
            }
        ]
        mts.all_prompt_idx = 0

        def fake_randint(low, high):
            if low == 1 and high == 3:
                return cluster_count
            if low == 0 and high == 5:
                return 0
            if low == 2:
                return 2
            return next(position_iter)

        def fake_np_choice(values, size=None, replace=True, p=None):
            values = sorted(list(values))
            if size is None:
                return values[0]
            return np.array(values[:size])

        def fake_positive_timeseries(cnt, change_position=None, seq_len=256):
            timeseries = [np.full(seq_len, change_position + offset, dtype=float) for offset in range(cnt)]
            attributes = []
            for _ in range(cnt):
                pool = _base_attribute_pool("sudden increase", change_position, seq_len)
                pool["local"][0].update(
                    {
                        "type": f"change_at_{change_position}",
                        "position_end": change_position + 1,
                        "detail": f"change around point {change_position}",
                    }
                )
                attributes.append(pool)
            return timeseries, attributes, change_position

        def fake_negative_timeseries(cnt, positive_positions, seq_len=256):
            self.assertEqual(cnt, 0)
            return [], []

        with patch.object(mts, "SEQ_LEN", seq_len), patch.object(mts, "ENCODING_METHOD", "no"):
            with patch.object(mts.random, "random", return_value=0.9):
                with patch.object(mts.random, "choice", side_effect=lambda seq: list(seq)[0]):
                    with patch.object(mts.random, "randint", side_effect=fake_randint):
                        with patch.object(mts.random, "sample", side_effect=lambda population, k: list(population)[:k]):
                            with patch.object(mts.np.random, "choice", side_effect=fake_np_choice):
                                with patch.object(mts.np.random, "permutation", side_effect=lambda n: np.arange(n)):
                                    with patch.object(mts, "generate_positive_timeseries", side_effect=fake_positive_timeseries):
                                        with patch.object(mts, "generate_negative_timeseries", side_effect=fake_negative_timeseries):
                                            with patch.object(mts, "timeseries_encoding", side_effect=lambda ts, _method: (ts[:, None], "<ts><ts/>", {})):
                                                with patch.object(mts, "attribute_to_text", return_value="description"):
                                                    (
                                                        _original_timeseries,
                                                        _combined_timeseries,
                                                        combined_metrics,
                                                        _combined_attributes,
                                                        _prompt,
                                                        _questions,
                                                        _answers,
                                                        _llm_prompts,
                                                        _fields,
                                                        _corr_pool,
                                                        label,
                                                    ) = mts.generate_prompt_data(seq_len)

        clusters = label["label"]["clusters"]
        self.assertEqual(len(clusters), cluster_count)
        self.assertEqual(label["label"]["positions"], positions)
        self.assertEqual([cluster["cluster_idx"] for cluster in clusters], list(range(cluster_count)))
        self.assertEqual([cluster["position"] for cluster in clusters], positions)

        prefix_to_position = {"a": 40, "b": 120, "c": 210}
        for correlation in label["label"]["correlations"]:
            if correlation["label"]:
                prefixes = {metric[0] for metric in correlation["pair"]}
                self.assertEqual(len(prefixes), 1, correlation)
                self.assertEqual(correlation["position"], prefix_to_position[prefixes.pop()])
                for metric in correlation["pair"]:
                    self.assertIn(metric, combined_metrics)


if __name__ == "__main__":
    unittest.main()
