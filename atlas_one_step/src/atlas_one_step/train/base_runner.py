
from pathlib import Path
import json, torch
from torch.optim import Adam
from ..data.datamodules import build_dataloader
from ..models.model_factory import build_model
from ..losses.loss_factory import total_loss

class BaseRunner:
    def __init__(self,cfg): self.cfg=cfg
    def run(self, exp_id='EXP-BASE'):
        dl=build_dataloader(batch_size=int(self.cfg.dataset.batch_size), resolution=int(self.cfg.dataset.resolution))
        model=build_model(self.cfg.model)
        opt=Adam(model.parameters(), lr=float(self.cfg.train.lr))
        for _ in range(int(self.cfg.train.epochs)):
            for batch in dl:
                pred=model(batch['x0'])
                loss,_=total_loss(pred,batch['x0'],mu=float(self.cfg.train.mu),tau=float(self.cfg.train.tau))
                opt.zero_grad(); loss.backward(); opt.step(); break
        Path('checkpoints').mkdir(exist_ok=True)
        ck='checkpoints/last.pt'; torch.save(model.state_dict(),ck)
        rec={"exp_id":exp_id,"status":"ok","checkpoint":ck}
        Path('outputs').mkdir(exist_ok=True)
        Path('outputs/train_result.json').write_text(json.dumps(rec,indent=2))
        return rec
