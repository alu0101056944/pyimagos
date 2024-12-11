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

  erodeKernel = np.ones((4, 4), np.uint8)
  cleanMask = cv.erode(roughMask, erodeKernel, iterations=1)
  dilateKernel = np.ones((6, 6), np.uint8)
  cleanMask = cv.dilate(cleanMask, dilateKernel, iterations=1)

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

  differenceImage = np.absolute(inputImage - outputImage)

  # f, axs = plt.subplots(2, 2)
  # axs[0, 0].imshow(inputImage, cmap='gray', vmin=0, vmax=1)
  # axs[0, 0].set_title('Input image')

  # axs[0, 1].imshow(outputImage, cmap='gray', vmin=0, vmax=1)
  # axs[0, 1].set_title('Output image')

  # im2 = axs[1, 0].imshow(differenceImage, cmap='gray')
  # axs[1, 0].set_title('Difference normalized')
  # plt.colorbar(im2, ax=axs[1, 0])

  # im3 = axs[1, 1].imshow(np.copy(differenceImage), cmap='gray', vmin=0, vmax=1)
  # axs[1, 1].set_title('Difference')
  # plt.colorbar(im3, ax=axs[1, 1])
  # plt.show()


  ## Border detection

  gaussianBlurred = cv.GaussianBlur(outputImage, (5, 5), 0)

  block_size = (3, 3)
  windows = np.lib.stride_tricks.sliding_window_view(
      gaussianBlurred, 
      window_shape=block_size
  )
  
  # Reshape to make blocks contiguous
  reshaped_windows = windows.reshape(-1, *block_size)
  
  # Calculate max and min for each block
  max_min_diff = reshaped_windows.max(axis=(1, 2)) - reshaped_windows.min(axis=(1, 2))
  
  bordersDetected = np.copy(gaussianBlurred)
  for i in range(windows.shape[0]):
    for j in range(windows.shape[1] ):
      # Calculate center indices
      center_y = i + block_size[0] // 2
      center_x = j + block_size[1] // 2
      
      # Replace center value with max-min difference
      bordersDetected[center_y, center_x] = max_min_diff[i * windows.shape[1] + j]
  
  bordersDetectedHEEnchanced = np.copy(bordersDetected)
  bordersDetectedHEEnchanced = (bordersDetectedHEEnchanced - bordersDetectedHEEnchanced.min()) / (
     bordersDetectedHEEnchanced.max() - bordersDetectedHEEnchanced.min())

  # showcase before and after
  f, axs = plt.subplots(1, 5)
  axs[0].imshow(inputImage, cmap='gray', vmin=0, vmax=1)
  axs[0].set_title('Input image')

  axs[1].imshow(outputImage, cmap='gray', vmin=0, vmax=1)
  axs[1].set_title('HE enchanced')
  
  axs[2].imshow(gaussianBlurred, cmap='gray', vmin=0, vmax=1)
  axs[2].set_title('Gaussian blur')

  axs[3].imshow(bordersDetected, cmap='gray', vmin=0, vmax=1)
  axs[3].set_title('Border detection')

  axs[4].imshow(bordersDetectedHEEnchanced, cmap='gray', vmin=0, vmax=1)
  axs[4].set_title('Border detection HE')
  plt.show()

if __name__ == '__main__':
    main()
