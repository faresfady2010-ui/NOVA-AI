"""Tests for instruction data masks and SFT pipeline primitives."""
import tempfile
import unittest
from pathlib import Path
import torch
from model import ByteTokenizer, ModelConfig, NOVATransformer
from training.instruction_dataset import InstructionBatcher, encode_example, load_examples
from training.train import save_checkpoint
from inference.generate import load_model

class InstructionTuningTests(unittest.TestCase):
    def setUp(self):
        self.example = {"messages": [
            {"role": "user", "content": "Explain briefly."},
            {"role": "assistant", "content": "A short answer."},
        ]}

    def test_loss_mask_supervises_answer_not_user_prompt(self):
        x, y = encode_example(self.example, ByteTokenizer(), 128)
        supervised = y[y != -100]
        self.assertGreater(supervised.numel(), 0)
        text = ByteTokenizer().decode(supervised.tolist())
        self.assertIn("A short answer.", text)
        self.assertNotIn("Explain briefly.", text)

    def test_jsonl_loader_and_batch_shapes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "examples.jsonl"
            path.write_text('{"messages":[{"role":"user","content":"Question?"},{"role":"assistant","content":"Answer."}]}\n', encoding="utf-8")
            examples = load_examples(path)
            batcher = InstructionBatcher(examples, ByteTokenizer(), 64, torch.device("cpu"))
            x, y = batcher.batch(2)
            self.assertEqual(x.shape, (2, 64))
            self.assertEqual(y.shape, (2, 64))
            self.assertTrue(torch.all(y[:, -1] == -100))

    def test_finetuned_checkpoint_reload(self):
        config = ModelConfig(context_length=32, d_model=16, n_heads=4, n_layers=1, dropout=0)
        model = NOVATransformer(config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.pt"
            save_checkpoint(path, model, config, 1)
            loaded, _, loaded_config, _ = load_model(path, torch.device("cpu"))
            self.assertEqual(config, loaded_config)
            for original, restored in zip(model.parameters(), loaded.parameters()):
                self.assertTrue(torch.equal(original, restored))

if __name__ == "__main__":
    unittest.main()
