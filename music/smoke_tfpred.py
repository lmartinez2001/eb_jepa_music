import sys, pathlib
_h=pathlib.Path(__file__).parent.resolve(); sys.path=[p for p in sys.path if pathlib.Path(p).resolve()!=_h]
import torch
from eb_jepa.training_utils import load_config
from music.datasets.decoder_dataset import build_models, load_jepa_checkpoint
from music.main import cls_state, encode_music
from music.models.rotation_utils import axis_angle_to_6d
import torch.nn.functional as F
dev="cuda"
cfg=load_config("/lustre/work/vivatech-4dudes/edugelay/checkpoints/music_jepa/dev_2026-06-20_13-53/exp_seed2025/config.yaml")
enc,menc,pred=build_models(cfg,dev)
load_jepa_checkpoint("/lustre/work/vivatech-4dudes/edugelay/checkpoints/music_jepa/dev_2026-06-20_13-53/exp_seed2025/epoch_10.pth.tar",enc,menc,pred,dev)
enc.eval();menc.eval();pred.eval()
B=2; W=cfg.data.window; H=cfg.data.get("pred_horizon",1)
S=cfg.data.audio_chunk_frames*(cfg.data.sample_rate//cfg.data.fps)
xt=torch.randn(B,W,25,3,device=dev)
xf=torch.randn(B,H,W,25,3,device=dev)
music=torch.randn(B,H,S,device=dev)
with torch.no_grad():
    zt=cls_state(enc,xt); print("z_t",tuple(zt.shape))
    me=encode_music(menc,music,cfg); print("music_emb",tuple(me.shape))
    zp=pred.generate(zt,me); print("generate",tuple(zp.shape),"-> take [:,0]")
    zp0=zp[:,0]
pose=xf[:,0].float()
rot6=axis_angle_to_6d(pose[...,:24,:]); trans6=F.pad(pose[...,24:25,:],(0,3))
poses6=torch.cat([rot6,trans6],-2)
print("poses6",tuple(poses6.shape),"z_pred0",tuple(zp0.shape))
assert poses6.shape==(B,W,25,6) and zp0.shape==(B,512)
print("SMOKE_TFPRED PASS")
