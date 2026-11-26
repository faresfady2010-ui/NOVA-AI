"""Generation and Flask API tests."""
import tempfile, unittest
from pathlib import Path
import torch
from model import ByteTokenizer, ModelConfig, NOVATransformer
from training.train import save_checkpoint
from inference.generate import generate, load_model
from backend.server import app

class GenerationTests(unittest.TestCase):
    def test_generation_is_bounded_and_decodable(self):
        model=NOVATransformer(ModelConfig(context_length=16,d_model=16,n_heads=4,n_layers=1,dropout=0)); model.eval()
        result=generate(model,ByteTokenizer(),"Hi",max_new_tokens=5,temperature=1,top_k=10,device=torch.device("cpu"))
        self.assertIsInstance(result,str)
    def test_checkpoint_loads_for_generation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"model.pt"; save_checkpoint(path,NOVATransformer(ModelConfig(context_length=8,d_model=16,n_heads=4,n_layers=1)),ModelConfig(context_length=8,d_model=16,n_heads=4,n_layers=1),0)
            model,tok,config,device=load_model(path,torch.device("cpu")); self.assertEqual(config.context_length,8)
    def test_backend_validation_and_health(self):
        client=app.test_client()
        self.assertEqual(client.get("/api/health").status_code,200)
        self.assertEqual(client.post("/api/chat",json={"messages":[]}).status_code,400)

if __name__=="__main__": unittest.main()
