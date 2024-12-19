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

# from constants import MODEL_PATH
from src.image_filters.image_filter import ImageFilter
import src.vision_transformer as vits

class AttentionMap(ImageFilter):
    def __init__(self):
        pass

    def loadInputImage_(self, tensor_image: torch.Tensor) -> torch.Tensor:
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
      if not isinstance(image, torch.Tensor):
          image = transforms.ToTensor()(image)
      tensorImage = self.loadInputImage_(image)
      selfAttentionMapModel = self.dino_vits8_()
      selfAttentionMap = self.getSelfAttentionMap_(selfAttentionMapModel,
                                                   tensorImage)
      roughMask = self.getThresholdedNdarray_(selfAttentionMap).astype(np.uint8)

      scaleFactorX = tensorImage.shape[-1] / roughMask.shape[-1]
      scaleFactorY = tensorImage.shape[-2] / roughMask.shape[-2]
      scaledMask = cv.resize(roughMask, (0, 0), fx=scaleFactorX, fy=scaleFactorY,
                            interpolation=cv.INTER_NEAREST) # to avoid non 0s and 1s

      scaledMask = cv.normalize(scaledMask, None, 0, 255, cv.NORM_MINMAX,
                               cv.CV_8U)
      return scaledMask

    def get_name(self):
        return 'AttentionMap'

    def get_params(self):
        return {"kernel_size": self.kernel_size, "sigma": self.sigma}
