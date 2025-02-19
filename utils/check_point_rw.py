import os
import torch

def save_checkpoint(model, respth, name):
    """Save Checkpoint"""
    save_pth = os.path.join(respth, name)
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state, save_pth)
    print(save_pth)
