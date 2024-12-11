'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development
'''

__authors__ = ["Marcos Jesús Barrios Lorenzo"]
__date__ = "2024/12/02"

import os.path
from PIL import Image

import click

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

import src.vision_transformer as vits

from constants import (
   MODEL_PATH
)

def loadInputImage(filename: str) -> torch.Tensor:
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x : x.repeat(3, 1, 1)) # Model needs 3 channels.
  ])
  image = Image.open(filename)
  tensor_image = transform(image)
  tensor_image = tensor_image.unsqueeze(0) # Makes it [1, C, H, W]
  return tensor_image

# copied from https://github.com/facebookresearch/dino/hubconf.py
# Modification: local load for state_dict instead of url download
# Original download url: https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth
# Modification: added patch_size parameter. Also added model.eval()
def dino_vits8(pretrained=True, patch_size=8, **kwargs) -> vits.VisionTransformer:
  """
  ViT-Small/8x8 pre-trained with DINO.
  Achieves 78.3% top-1 accuracy on ImageNet with k-NN classification.
  """
  model = vits.__dict__["vit_small"](patch_size=patch_size, num_classes=0, **kwargs)
  if pretrained:
      absolutePath = os.path.dirname(__file__)
      path = os.path.normpath(os.path.join(absolutePath, MODEL_PATH))
      state_dict = torch.load(path, map_location="cpu")
      model.load_state_dict(state_dict, strict=True)
      model.eval()
  return model

def showSelfAttentionMap(model: vits.VisionTransformer, inputImage: torch.Tensor) -> None:
  with torch.no_grad():
    selfAttentionMap = model.get_last_selfattention(inputImage)

  # Selection: [attention head, batch, patchNi, patchNj]
  cls_attention = selfAttentionMap[0, 0, 0, 1:] # dont include attention to self
  w0 = inputImage.shape[-1] // 8 # 8 = num_patches
  h0 = inputImage.shape[-2] // 8
  attention_grid = cls_attention.reshape(h0, w0)
  plt.figure(figsize=(10, 10))
  plt.imshow(attention_grid)
  plt.title("Attention map")
  plt.show()

@click.command()
@click.argument('filename')
def main(filename: str) -> None:
  inputImage = loadInputImage(filename)
  selfAttentionMapModel = dino_vits8()
  showSelfAttentionMap(selfAttentionMapModel, inputImage)

if __name__ == '__main__':
    main()
