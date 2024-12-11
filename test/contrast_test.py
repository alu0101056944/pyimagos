'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development
'''

import os.path
from PIL import Image

import matplotlib.pyplot as plt
import pytest
import torch
import torchvision.transforms as transforms

import src.vision_transformer as vits

class TestAttentionMap:

  # copied from https://github.com/facebookresearch/dino/hubconf.py
  # Modification: local load for state_dict instead of url download
  # Original download url: https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth
  # Modification: added patch_size parameter. Also added model.eval()
  def dino_vits8(pretrained=True, patch_size=8, **kwargs):
    """
    ViT-Small/8x8 pre-trained with DINO.
    Achieves 78.3% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__["vit_small"](patch_size=patch_size, num_classes=0, **kwargs)
    if pretrained:
        absolutePath = os.path.dirname(__file__)
        path = os.path.normpath(os.path.join(absolutePath,
                                             '../model/dino_deitsmall8_pretrain.pth'))
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    return model

  @pytest.fixture(scope="class")
  def model(self):
    yield self.dino_vits8(patch_size=8)

  @pytest.fixture(scope="class")
  def inputImage(self):
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Lambda(lambda x : x.repeat(3, 1, 1)) # Model needs 3 channels.
    ])
    image = Image.open('docs/radiography.jpg')
    tensorImage = transform(image)
    tensorImage = tensorImage.unsqueeze(0) # Makes it [1, C, H, W]
    yield tensorImage
  
  def test_can_load_pretrained_model(self, model) -> None:
    assert isinstance(model, vits.VisionTransformer)


  def test_can_show_attention_map(self, model, inputImage) -> None:
      with torch.no_grad():
        attention_maps = model.get_last_selfattention(inputImage)

      # Selection: attention head, batch, patchNi, patchNj.
      cls_attention = attention_maps[0, 0, 0, 1:] # dont include attention to self
      w0 = inputImage.shape[-1] // 8 # 8 = num_patches
      h0 = inputImage.shape[-2] // 8
      fit_size = w0 * h0
      attention_grid = cls_attention[:fit_size]
      attention_grid = attention_grid.reshape(h0, w0)
      plt.figure(figsize=(10, 10))
      plt.imshow(attention_grid)
      plt.title("Attention map")
      plt.show()

