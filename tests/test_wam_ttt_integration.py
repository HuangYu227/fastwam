from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WAMTTTIntegrationContractTest(unittest.TestCase):
    def test_mmdit_exposes_opt_in_fast_weight_path(self) -> None:
        source = (ROOT / "lda/model/modules/action_model/flow_matching_head/mmdit/mmdit/mmdit_cross_attn.py").read_text(encoding="utf-8")
        self.assertIn("wam_ttt_enabled", source)
        self.assertIn("ttt_fast_weights", source)
        self.assertIn("return_ttt_inputs", source)
        self.assertIn("TTTMemoryBank", source)

    def test_action_head_can_represent_a_video_only_batch(self) -> None:
        source = (ROOT / "lda/model/modules/action_model/MMDiT_ActionHeader.py").read_text(encoding="utf-8")
        self.assertIn("if policy_indices", source)
        self.assertIn("ttt_fast_weights", source)
        self.assertIn("return_ttt_inputs", source)


if __name__ == "__main__":
    unittest.main()
