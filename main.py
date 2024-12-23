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
import numpy as np
import cv2 as cv

import src.vision_transformer as vits

from constants import (
   MODEL_PATH
)

def loadInputImage(filename: str) -> torch.Tensor:

  image = Image.open(filename)
  tensor_image = transforms.ToTensor()(image)
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

def getSelfAttentionMap(model: vits.VisionTransformer, inputImage: torch.Tensor) -> None:
  with torch.no_grad():
    selfAttentionMap = model.get_last_selfattention(inputImage)

  # Selection: [attention head, batch, patchNi, patchNj]
  cls_attention = selfAttentionMap[0, 0, 0, 1:] # dont include attention to self
  w0 = inputImage.shape[-1] // 8 # 8 = num_patches
  h0 = inputImage.shape[-2] // 8
  attention_grid = cls_attention.reshape(h0, w0)
  return attention_grid

def getThresholdedNdarray(selfAttentionMap) -> np.ndarray:
  selfAttentionMap = selfAttentionMap.numpy()
  selfAttentionMap = (selfAttentionMap > np.percentile(selfAttentionMap, 70)).astype(int)
  return selfAttentionMap

@click.command()
@click.argument('filename')
def main(filename: str) -> None:
  inputImage = loadInputImage(filename)

  selfAttentionMapModel = dino_vits8()
  selfAttentionMap = getSelfAttentionMap(selfAttentionMapModel, inputImage)
  roughMask = getThresholdedNdarray(selfAttentionMap).astype(np.uint8)

  erode_kernel = np.ones(4, np.uint8)
  cleanMask = cv.erode(roughMask, erode_kernel, iterations=1)
  # dilate_kernel = np.ones(3, np.uint8)
  # cleanMask = cv.dilate(roughMask, dilate_kernel, iterations=1)
  # erodeKernel = np.ones((5, 5), np.uint8)
  # cleanMask = cv.morphologyEx(np.copy(roughMask), cv.MORPH_OPEN, erodeKernel)

  scaleFactorX = inputImage.shape[-1] / cleanMask.shape[-1]
  scaleFactorY = inputImage.shape[-2] / cleanMask.shape[-2]
  scaledMask = cv.resize(cleanMask, (0, 0), fx=scaleFactorX, fy=scaleFactorY,
                        interpolation=cv.INTER_NEAREST) # to avoid non 0s and 1s

  inputImage = inputImage.numpy(force=True)[0, 0]

  maskedInputImage = inputImage * scaledMask
  maskedInputImage = np.ma.masked_equal(maskedInputImage, 0)
  maxValueWithinMask = maskedInputImage.max()
  minValueWithinMask = maskedInputImage.min()

  outputImage = np.clip((inputImage - minValueWithinMask) / (
     maxValueWithinMask - minValueWithinMask), 0, 1)

  ## Border detection
  gaussianBlurred = cv.GaussianBlur(outputImage, (5, 5), 0)
    
  block_size = (3, 3)
  pad_size = block_size[0] // 2 #Same as padding used by the Gaussian Blur
  padded_image = np.pad(gaussianBlurred, pad_size, mode='reflect')
  
  bordersDetected = np.zeros_like(gaussianBlurred, dtype=np.float32) # Using a float32 type for storing max-min differences

  for y in range(gaussianBlurred.shape[0]):
    for x in range(gaussianBlurred.shape[1]):
        window = padded_image[y:y + block_size[0], x:x + block_size[1]]
        max_min_diff = window.max() - window.min()

        bordersDetected[y, x] = max_min_diff

  # Normalize the image between 0 and 255 before showing to improve visualization.
  bordersDetected = cv.normalize(bordersDetected, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

  # showcase before and after
  f, axs = plt.subplots(1, 8)
  axs[0].imshow(inputImage, cmap='gray', vmin=0, vmax=1)
  axs[0].set_title('Input image')

  axs[1].imshow(roughMask, cmap='gray', vmin=0, vmax=1)
  axs[1].set_title('Unprocessed mask')

  axs[2].imshow(cleanMask, cmap='gray', vmin=0, vmax=1)
  axs[2].set_title('Clean mask')

  axs[3].imshow(scaledMask, cmap='gray', vmin=0, vmax=1)
  axs[3].set_title('Scaled up mask')

  axs[4].imshow(outputImage, cmap='gray')
  axs[4].set_title('HE enchanced')
  
  axs[5].imshow(gaussianBlurred, cmap='gray')
  axs[5].set_title('Gaussian blur')

  axs[6].imshow(bordersDetected, cmap='gray')
  axs[6].set_title('Border detection')
  plt.show()

if __name__ == '__main__':
    main()
