"""Architecture, causality, loss, optimizer, and checkpoint tests."""
import tempfile, unittest
from pathlib import Path
import torch
from model import ModelConfig, NOVATransformer
from training.train import save_checkpoint

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7); self.config=ModelConfig(context_length=8,d_model=16,n_heads=4,n_layers=2,dropout=0)
        self.model=NOVATransformer(self.config)
    def test_dimensions_and_loss(self):
        x=torch.randint(0,259,(2,8)); logits,loss=self.model(x,x)
        self.assertEqual(logits.shape,(2,8,259)); self.assertTrue(torch.isfinite(loss))
    def test_future_tokens_do_not_change_past_logits(self):
        self.model.eval(); a=torch.tensor([[1,2,3,4]]); b=torch.tensor([[1,2,99,100]])
        with torch.no_grad(): la,_=self.model(a); lb,_=self.model(b)
        self.assertTrue(torch.allclose(la[:,:2],lb[:,:2],atol=1e-6))
    def test_optimizer_updates_parameters(self):
        x=torch.randint(0,259,(2,8)); before=self.model.token_embedding.weight.detach().clone()
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-3); _,loss=self.model(x,x); loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(before,self.model.token_embedding.weight))
    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"model.pt"; save_checkpoint(path,self.model,self.config,12)
            saved=torch.load(path,map_location="cpu",weights_only=False)
            copy=NOVATransformer(ModelConfig.from_dict(saved["config"])); copy.load_state_dict(saved["model"])
            self.assertEqual(saved["step"],12)
            for p,q in zip(self.model.parameters(),copy.parameters()): self.assertTrue(torch.equal(p,q))

if __name__=="__main__": unittest.main()
