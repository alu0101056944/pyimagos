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
  # gaussianBlurred = cv.GaussianBlur(outputImage, (5, 5), 0)

  gaussianSigmas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  for i in range(11):
    gaussianBlurred = cv.GaussianBlur(outputImage, (5, 5), gaussianSigmas[i])

    blockSize = (3, 3)
    padSize = blockSize[0] // 2 # Same padding than Gaussian Blur filter
    paddedImage = np.pad(gaussianBlurred, padSize, mode='reflect')
    
    bordersDetected = np.zeros_like(gaussianBlurred, dtype=np.float32)

    for y in range(gaussianBlurred.shape[0]):
      for x in range(gaussianBlurred.shape[1]):
          window = paddedImage[y:y + blockSize[0], x:x + blockSize[1]]
          difference = window.max() - window.min()
          bordersDetected[y, x] = difference

    # Normalize the image between 0 and 255 before showing to improve visualization.
    outputImage = cv.normalize(outputImage, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    gaussianBlurred = cv.normalize(gaussianBlurred, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    bordersDetected = cv.normalize(bordersDetected, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    concatenated = np.concatenate((outputImage, gaussianBlurred, bordersDetected), axis=1)
    cv.imwrite(f'docs/{os.path.basename(filename)}' + (
                f'_bordersigma{gaussianSigmas[i]}.jpg'),
                concatenated)

  # cv.imwrite(f'docs/{os.path.basename(filename)}_bordersDetected.jpg', bordersDetected)

  # showcase before and after
  # f, axs = plt.subplots(1, 7)
  # axs[0].imshow(inputImage, cmap='gray', vmin=0, vmax=1)
  # axs[0].set_title('Input image')

  # axs[1].imshow(roughMask, cmap='gray', vmin=0, vmax=1)
  # axs[1].set_title('Unprocessed mask')

  # axs[2].imshow(cleanMask, cmap='gray', vmin=0, vmax=1)
  # axs[2].set_title('Clean mask')

  # axs[3].imshow(scaledMask, cmap='gray', vmin=0, vmax=1)
  # axs[3].set_title('Scaled up mask')

  # axs[4].imshow(outputImage, cmap='gray')
  # axs[4].set_title('HE enchanced')
  
  # axs[5].imshow(gaussianBlurred, cmap='gray')
  # axs[5].set_title('Gaussian blur')

  # axs[6].imshow(bordersDetected, cmap='gray')
  # axs[6].set_title('Border detection')
  # plt.show()

  _, bordersDetectedThresholded = cv.threshold(bordersDetected, 50, 255, cv.THRESH_BINARY)
  cv.imshow('binary', (bordersDetectedThresholded))

  erosionSizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
  for i in range(9):
    erosionKernel = np.ones(erosionSizes[i], np.uint8)
    erosionResult = cv.erode(bordersDetectedThresholded, erosionKernel, iterations=1)
    erosionResult2 = cv.erode(bordersDetectedThresholded, erosionKernel, iterations=2)
    erosionResult3 = cv.erode(bordersDetectedThresholded, erosionKernel, iterations=3)
    concatenated = np.concatenate((bordersDetectedThresholded, erosionResult,
                                   erosionResult2, erosionResult3), axis=1)
    cv.imwrite(f'docs/{os.path.basename(filename)}' + (
                f'_binarybordererosion{erosionSizes[i]}.jpg'),
                concatenated)

  openingSizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
  for i in range(9):
    openKernel = np.ones(openingSizes[i], np.uint8)
    openResult = cv.morphologyEx(bordersDetectedThresholded, cv.MORPH_OPEN, openKernel)
    concatenated = np.concatenate((bordersDetectedThresholded, openResult), axis=1)
    cv.imwrite(f'docs/{os.path.basename(filename)}' + (
                f'_binaryborderopening{openingSizes[i]}.jpg'),
                concatenated)

  closingSizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
  for i in range(9):
    closingKernel = np.ones(closingSizes[i], np.uint8)
    closingResult = cv.morphologyEx(bordersDetectedThresholded, cv.MORPH_OPEN, closingKernel)
    concatenated = np.concatenate((bordersDetectedThresholded, closingResult), axis=1)
    cv.imwrite(f'docs/{os.path.basename(filename)}' + (
                f'_binaryborderclosing{openingSizes[i]}.jpg'),
                concatenated)

  # numMinThresholds = np.linspace(0, 255, 16)
  # f, axs = plt.subplots(4, 12)
  # for i in range(4):
  #   for j in range(4):
  #     minThreshold = numMinThresholds[i * 4 + j]

  #     _, bordersDetectedThresholded = cv.threshold(bordersDetected, minThreshold, 255, cv.THRESH_BINARY)
  #     numLabels, markers = cv.connectedComponents(bordersDetectedThresholded)
  #     distanceTransform = cv.distanceTransform(bordersDetectedThresholded, cv.DIST_L2, 5)
  #     distanceTransform = cv.normalize(distanceTransform, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)

      # concatenated = np.concatenate(
      #   (bordersDetectedThresholded, markers * (255 / numLabels), distanceTransform), axis=1)
      # cv.imwrite(f'docs/{os.path.basename(filename)}_minThreshold{minThreshold}.jpg', concatenated)

  _, markers = cv.connectedComponents(bordersDetectedThresholded)
  cv.imshow('Markers', (markers * (255 / 3)).astype(np.uint8))
  distanceTransform = cv.distanceTransform(bordersDetectedThresholded, cv.DIST_L2, 5)
  distanceTransform = cv.normalize(distanceTransform, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
  cv.imshow('Distance transform', distanceTransform.astype(np.uint8))

  markers = markers.astype(np.int32)
  distanceTransform3Channel = cv.merge((distanceTransform, distanceTransform, distanceTransform))
  watershedImage = cv.watershed(distanceTransform3Channel, markers)
  watershedImage = watershedImage.astype(np.uint8)

  coloredImage = np.zeros((bordersDetected.shape[0], bordersDetected.shape[1], 3), dtype=np.uint8)
  uniqueLabels = np.unique(watershedImage)
  for label in uniqueLabels:
     if label == 0:
        continue
     
     mask = watershedImage == label
     b, g, r = np.random.randint(0, 256, 3)
     coloredImage[mask] = (b, g, r)

  cv.imshow('Borders Detected', bordersDetected)
  cv.imshow('Watershed segmentation', coloredImage)
  
  cv.waitKey(0)
  cv.destroyAllWindows()

 

if __name__ == '__main__':
    main()
