'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Processing steps of the radiograph
'''

import os.path
from PIL import Image

import torchvision.transforms as transforms
import numpy as np
import cv2 as cv

from src.image_filters.contrast_enhancement import ContrastEnhancement
from src.image_filters.border_detection_statistical_range import (
  BorderDetectionStatisticalRange
)

def process_radiograph(filename: str, write_images: bool = False,
                       show_images: bool = True) -> None:
  inputImage = Image.open(filename)
  inputImage = transforms.ToTensor()(inputImage)

  outputImage = ContrastEnhancement().process(inputImage)
  outputImage = cv.normalize(outputImage, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

  ## Border detection
  gaussianBlurred = cv.GaussianBlur(outputImage, (5, 5), 0)

  gaussianSigmas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  for i in range(11):
    gaussianBlurred = cv.GaussianBlur(outputImage, (5, 5), gaussianSigmas[i])

    bordersDetected = BorderDetectionStatisticalRange(5 // 2).process(gaussianBlurred)

    # Normalize the image between 0 and 255 before showing to improve visualization.
    gaussianBlurred = cv.normalize(gaussianBlurred, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    bordersDetected = cv.normalize(bordersDetected, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    concatenated = np.concatenate((outputImage, gaussianBlurred, bordersDetected), axis=1)
    if write_images:
      cv.imwrite(f'docs/local_images/{os.path.basename(filename)}' \
                  f'_bordersigma{gaussianSigmas[i]}.jpg',
                  concatenated)
    else:
      if show_images and i == 2:
        cv.imshow(f'{os.path.basename(filename)}_bordersigma{gaussianSigmas[i]}.jpg',
                  concatenated)
        cv.waitKey(0)
        cv.destroyAllWindows()
      else:
        continue
  
  ## Border detection, canny edge detector
  minThresholds = [30, 51, 70]
  maxThresholds = [135, 150, 165]
  kernelSizes = [3, 5, 7, 9]
  for minThreshold  in minThresholds:
    showedImage = False
    for maxThreshold in maxThresholds:
      for kernelSize in kernelSizes:
        gaussianBlurred = cv.GaussianBlur(outputImage, (kernelSize, kernelSize), 0)
        gaussianBlurred = cv.normalize(gaussianBlurred, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        bordersDetected = cv.Canny(gaussianBlurred, minThreshold, maxThreshold)
        bordersDetected = cv.normalize(bordersDetected, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        # contours, _ = cv.findContours(bordersDetected, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # largest_contour = max(contours, key=cv.contourArea)
        # mask = np.zeros_like(bordersDetected, dtype=np.uint8)
        # cv.drawContours(mask, [largest_contour], -1, 255, thickness=cv.FILLED)
        # mask = cv.bitwise_not(mask)
        # bordersDetected = cv.bitwise_and(bordersDetected, bordersDetected, mask=mask)

        numLabels, markers = cv.connectedComponents(bordersDetected)

        kernel = np.ones((kernelSize, kernelSize), np.uint8)
        closed_image = cv.morphologyEx(bordersDetected,
            cv.MORPH_CLOSE, kernel, iterations=1)

        distanceTransform = cv.distanceTransform(bordersDetected, cv.DIST_L2, 5)
        distanceTransform = cv.normalize(distanceTransform, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)

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

        # if numLabels > 1:
        #   cmap = plt.cm.get_cmap('hsv', numLabels)  
        #   colored_markers = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
        #   for label in range(1, numLabels): # Skip background label 0
        #       color = np.array(cmap(label)[:3]) * 255 # Get RGB color and convert to 0-255 range
        #       colored_markers[markers == label] = color.astype(np.uint8)
        # else:
        #     colored_markers = np.zeros_like(outputImage, dtype=np.uint8)

        concatenated = np.concatenate((cv.cvtColor(outputImage, cv.COLOR_GRAY2BGR),
                                                    cv.cvtColor(gaussianBlurred, cv.COLOR_GRAY2BGR),
                                                    cv.cvtColor(bordersDetected, cv.COLOR_GRAY2BGR),
                                                    cv.cvtColor(closed_image, cv.COLOR_GRAY2BGR),
                                                    coloredImage,
                                                    ), axis=1)
        if write_images:
          cv.imwrite(f'docs/local_images/{os.path.basename(filename)}' \
            f'_bordercanny_withfill_minThresh{minThreshold}' \
            f'_maxThresh{maxThreshold}_kernelSize{kernelSize}' \
            f'.jpg',
            concatenated)
        else:
          if show_images and not showedImage:
            showedImage = True
            cv.imshow(f'{os.path.basename(filename)}' \
              f'_bordercanny_withfill_minThresh{minThreshold}' \
              f'_maxThresh{maxThreshold}_kernelSize{kernelSize}' \
              f'.jpg', concatenated)
            cv.waitKey(0)
            cv.destroyAllWindows()

    showedImage = False
