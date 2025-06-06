'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development
'''

import os.path
from PIL import Image

import pytest
import torch
import torchvision.transforms as transforms

import src.vision_transformer as vits

class TestAttentionMap:

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
        path = os.path.normpath(os.path.join(absolutePath, '../model/dino_deitsmall8_pretrain.pth'))
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    return model

  @pytest.fixture(scope='class')
  def model(self):
    yield self.dino_vits8(patch_size=8)

  @pytest.fixture(scope='class')
  def input_image(self):
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Lambda(lambda x : x.repeat(3, 1, 1)) # Model needs 3 channels.
    ])
    try:
      with Image.open('docs/radiography.jpg') as imagefile:
        tensor_image = transform(imagefile)
    except Exception as e:
      print(f"Error opening image {'docs/radiography.jpg'}: {e}")
      raise
    tensor_image = tensor_image.unsqueeze(0) # Makes it [1, C, H, W]
    yield tensor_image

  @pytest.fixture(scope='class')
  def selfAttentionMap(self, model, input_image):
    with torch.no_grad():
      attention_maps = model.get_last_selfattention(input_image)
    return attention_maps
  
  def test_can_load_pretrained_model(self, model) -> None:
    assert isinstance(model, vits.VisionTransformer)

  def test_can_transform_input_image_to_tensor(self, input_image) -> None:
    assert isinstance(input_image, torch.Tensor)

  def test_can_get_attention_map(self, selfAttentionMap) -> None:
    assert isinstance(selfAttentionMap, torch.Tensor)
    assert len(selfAttentionMap.shape) == 4
    expected_num_patches = 5041
    assert selfAttentionMap.shape[-1] == expected_num_patches
    expected_num_heads = 6
    assert selfAttentionMap.shape[1] == expected_num_heads
