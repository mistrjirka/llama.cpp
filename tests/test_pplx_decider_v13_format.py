"""Perplexity Decider v1.3 prompt regression tests, no GPU needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from jinja2 import Environment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversion"))
from pplx_decider_v4 import SYSTEM_PROMPT, is_v4, systemone_template_v4


class TestPplxDeciderV13(unittest.TestCase):
    def setUp(self):
        self.template = Environment().from_string(systemone_template_v4())

    def render(self, state, kind, options, instructions):
        return self.template.render(state=state, type=kind, options=options,
                                    instructions=instructions, images=[])

    def test_exact_choice_marker_readout(self):
        prompt = self.render("A worker carries a bag", "choice", [
            {"label": "A", "key": "carrying", "description": "Moving materials"},
            {"label": "B", "key": "waiting", "description": None},
        ], "What is the activity?")
        expected = (
            "<|im_start|>system\n" + SYSTEM_PROMPT
            + "<|im_end|>\n<|im_start|>user\nState:\nA worker carries a bag"
            + "<|fim_middle|>Question:\nWhat is the activity?"
            + "\nOptions:\nA: carrying: Moving materials\nB: waiting<|fim_suffix|>"
        )
        self.assertEqual(prompt, expected)
        self.assertEqual(prompt.count("<|fim_suffix|>"), 1)
        self.assertEqual(prompt.count("<|fim_middle|>"), 1)
        self.assertNotIn("<|im_start|>assistant", prompt)

    def test_score_and_yes_no(self):
        score = self.render("Some work", "score", [
            {"label": "A", "key": "0", "description": "No work"},
            {"label": "B", "key": "1", "description": "Heavy work"},
        ], None)
        self.assertIn("Question:\nChoose the best matching option.\nOptions:\nA: No work\nB: Heavy work<|fim_suffix|>", score)
        noul = self.render("Some work", "noul", [
            {"label": "A", "key": "false", "description": None},
            {"label": "B", "key": "true", "description": None},
        ], "Is anyone working?")
        self.assertTrue(noul.endswith("Options:\nA: No / false\nB: Yes / true<|fim_suffix|>"))

    def test_protocol_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            protocol = {"version": 4, "template": "joint-chunked-readout-v1",
                        "marker": "<|fim_suffix|>", "separator": "<|fim_middle|>",
                        "readout_offset": 0}
            self.assertFalse(is_v4(path, {"attention_mode": "noncausal_full_attention"}))
            (path / "joint_config.json").write_text(json.dumps(protocol))
            self.assertTrue(is_v4(path, {"attention_mode": "noncausal_full_attention"}))
            with self.assertRaises(ValueError):
                is_v4(path, {"attention_mode": "causal"})
            protocol["readout_offset"] = -1
            (path / "joint_config.json").write_text(json.dumps(protocol))
            with self.assertRaises(ValueError):
                is_v4(path, {"attention_mode": "noncausal_full_attention"})


if __name__ == "__main__":
    unittest.main()
