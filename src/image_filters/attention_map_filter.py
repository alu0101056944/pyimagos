'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Image processing step representation for the GUI. Uses visual transformer
  DINOViTS/8's self attention mask as a mask to contraste enchance. Outputs
  the self attention map scaled up.
'''

from typing import Union

import os.path

import numpy as np
import torch
import torchvision.transforms as transforms
import cv2 as cv

from src.image_filters.image_filter import ImageFilter
import src.vision_transformer as vits
from constants import CPU_CONTRAST_RESIZE_TARGET

# TODO update this to match ContrastEnhancement image filter

class AttentionMap(ImageFilter):
  def __init__(self, scale_up : bool = False, noresize: bool = False):
    self.scale_up = scale_up
    self.noresize = noresize

  def _loadInputImage(self, tensor_image: torch.Tensor) -> torch.Tensor:
    # TODO this assumes that the input is in format C H W but could be H W C due
    # to a previous conversion from array to tensor. Fix that.

    if tensor_image.ndim < 3:
      tensor_image = tensor_image.unsqueeze(0)
      tensor_image = torch.cat((tensor_image, tensor_image, tensor_image), dim=-2)
    elif tensor_image.ndim >= 3:
      if tensor_image.shape[0] < 3:
        tensor_image = tensor_image.flatten(0, -3)
        tensor_image = torch.cat((
            tensor_image, tensor_image, tensor_image), dim=0)
      elif tensor_image.shape[0] > 3:
        tensor_image = tensor_image.flatten(0, -3)[:3]
    tensor_image = tensor_image.unsqueeze(0) # Makes it [1, C, H, W]
    return tensor_image

  # copied from https://github.com/facebookresearch/dino/hubconf.py
  # Modification: local load for state_dict instead of url download
  # Original download url: https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth
  # Modification: added patch_size parameter. Also added model.eval()
  def _dino_vits8(self, pretrained=True, patch_size=8,
                  **kwargs) -> vits.VisionTransformer:
    """
    ViT-Small/8x8 pre-trained with DINO.
    Achieves 78.3% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__["vit_small"](patch_size=patch_size, num_classes=0,
                                        **kwargs)
    if pretrained:
      absolute_path = os.path.dirname(__file__)
      path = os.path.normpath(os.path.join(
        absolute_path,
        '../../model/dino_deitsmall8_pretrain.pth'
      ))
      state_dict = torch.load(path, map_location="cpu")
      model.load_state_dict(state_dict, strict=True)
      model.eval()
    return model

  def _getSelfAttentionMap(self, model: vits.VisionTransformer,
                            input_image: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      self_attention_map = model.get_last_selfattention(input_image)

    # Selection: [attention head, batch, patchNi, patchNj]
    cls_attention = self_attention_map[0, 0, 0, 1:] # dont include attention to self
    w0 = input_image.shape[-1] // 8 # 8 = num_patches
    h0 = input_image.shape[-2] // 8
    attention_grid = cls_attention.reshape(h0, w0)
    return attention_grid

  def _getThresholdedNdarray(self, self_attention_map: np.array) -> np.ndarray:
    self_attention_map = self_attention_map.numpy()
    self_attention_map = (
        self_attention_map > np.percentile(self_attention_map, 70)
    ).astype(np.uint8)
    return self_attention_map

  def process(self, image: Union[np.array, torch.Tensor]) -> np.array:
    if isinstance(image, np.ndarray):
      original_size = (image.shape[0], image.shape[1])
    else:
      original_size = (image.shape[-2], image.shape[-1])

    if self.noresize:
      target_size = (image.shape[0], image.shape[1])
    else:
      target_size = (CPU_CONTRAST_RESIZE_TARGET['height'],
                    CPU_CONTRAST_RESIZE_TARGET['width'])
  
    if not isinstance(image, torch.Tensor):
      image = cv.resize(image, (target_size[1], target_size[0]),
                        interpolation=cv.INTER_AREA)
      tensor_image = transforms.ToTensor()(image)
    else:
      tensor_image = torch.nn.functional.interpolate(
        image.unsqueeze(0), # C H W to 1 C H W
        size=(target_size[0], target_size[1]),
        mode='bilinear',
        align_corners=False,
      ).squeeze(0) # back to C 

    tensor_image = self._loadInputImage(tensor_image)
    model = self._dino_vits8()
    self_attention_map = self._getSelfAttentionMap(model, tensor_image)
    rough_mask = self._getThresholdedNdarray(self_attention_map)

    output_image = None
    if self.scale_up:
      scale_factor_x = tensor_image.shape[-1] / rough_mask.shape[-1]
      scale_factor_y = tensor_image.shape[-2] / rough_mask.shape[-2]
      scaled_mask = cv.resize(rough_mask, (0, 0), fx=scale_factor_x,
                              fy=scale_factor_y,
                              interpolation=cv.INTER_NEAREST) # to avoid non 0s and 1s

      scaled_mask = cv.normalize(scaled_mask, None, 0, 255, cv.NORM_MINMAX,
                                  cv.CV_8U)
      output_image = scaled_mask
    else:
      rough_mask = cv.normalize(rough_mask, None, 0, 255, cv.NORM_MINMAX,
                                cv.CV_8U)
      output_image = rough_mask
    
    output_image = cv.normalize(output_image, None, 0, 255, cv.NORM_MINMAX,
                          cv.CV_8U)
    return output_image

  def get_name(self):
      return 'AttentionMap'

  def get_params(self):
      return {"kernel_size": self.kernel_size, "sigma": self.sigma}
