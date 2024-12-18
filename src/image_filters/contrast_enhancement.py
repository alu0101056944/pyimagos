'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Image processing step representation for the GUI. Uses visual transformer
  DINOViTS/8's self attention mask as a mask to contraste enchance.
'''

from typing import Union

import os.path

import numpy as np
import torch
import torchvision.transforms as transforms

# from constants import MODEL_PATH
from src.image_filters.image_filter import ImageFilter
import src.vision_transformer as vits

class ContrastEnhancement(ImageFilter):
    def __init__(self):
        pass

    def loadInputImage_(self, tensor_image: torch.Tensor) -> torch.Tensor:
      if tensor_image.shape[0] <= 3:
        tensor_image = torch.cat(
          [tensor_image for i in range(3 - tensor_image.shape[0] + 1)], dim=0)
      elif tensor_image.shape[0] > 3:
        tensor_image = tensor_image[:3, :, :]
      tensor_image = tensor_image.unsqueeze(0) # Makes it [1, C, H, W]
      return tensor_image

    # copied from https://github.com/facebookresearch/dino/hubconf.py
    # Modification: local load for state_dict instead of url download
    # Original download url: https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth
    # Modification: added patch_size parameter. Also added model.eval()
    def dino_vits8_(self, pretrained=True, patch_size=8,
                    **kwargs) -> vits.VisionTransformer:
      """
      ViT-Small/8x8 pre-trained with DINO.
      Achieves 78.3% top-1 accuracy on ImageNet with k-NN classification.
      """
      model = vits.__dict__["vit_small"](patch_size=patch_size, num_classes=0, **kwargs)
      if pretrained:
          absolutePath = os.path.dirname(__file__)
          path = os.path.normpath(os.path.join(absolutePath, '../../model/dino_deitsmall8_pretrain.pth'))
          state_dict = torch.load(path, map_location="cpu")
          model.load_state_dict(state_dict, strict=True)
          model.eval()
      return model

    def getSelfAttentionMap_(self, model: vits.VisionTransformer,
                             inputImage: torch.Tensor) -> torch.Tensor:
      with torch.no_grad():
        selfAttentionMap = model.get_last_selfattention(inputImage)

      # Selection: [attention head, batch, patchNi, patchNj]
      cls_attention = selfAttentionMap[0, 0, 0, 1:] # dont include attention to self
      w0 = inputImage.shape[-1] // 8 # 8 = num_patches
      h0 = inputImage.shape[-2] // 8
      attention_grid = cls_attention.reshape(h0, w0)
      return attention_grid

    def getThresholdedNdarray_(self, selfAttentionMap: np.array) -> np.ndarray:
      selfAttentionMap = selfAttentionMap.numpy()
      selfAttentionMap = (selfAttentionMap > np.percentile(selfAttentionMap, 70)).astype(int)
      return selfAttentionMap

    def process(self, image: Union[np.array, torch.Tensor]) -> np.array:
      tensorImage = self.loadInputImage_(image)
      selfAttentionMapModel = self.dino_vits8_()
      selfAttentionMap = self.getSelfAttentionMap_(selfAttentionMapModel, tensorImage)
      roughMask = self.getThresholdedNdarray_(selfAttentionMap).astype(np.uint8)
      return roughMask

    def get_name(self):
        return 'ContrastEnhancement'

    def get_params(self):
        return {"kernel_size": self.kernel_size, "sigma": self.sigma}
