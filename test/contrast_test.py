'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development
'''

import pytest

import os.path
import torch

import src.vision_transformer as vits

class TestAttentionMap:

  # copied from https://github.com/facebookresearch/dino/hubconf.py
  def dino_vits8(pretrained=True, **kwargs):
    """
    ViT-Small/8x8 pre-trained with DINO.
    Achieves 78.3% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__["vit_small"](patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        # Modified: local load for state_dict instead of url download
        # Original download url: https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth
        absolutePath = os.path.dirname(__file__)
        path = os.path.normpath(os.path.join(absolutePath,
                                             '../model/dino_deitsmall8_pretrain.pth'))
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model
  
  def test_can_load_pretrained_model(self) -> None:
    model = self.dino_vits8()
    model.eval()
    pass
