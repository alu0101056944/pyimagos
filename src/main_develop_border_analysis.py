'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Output different intermediate results to study alternatives.
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

# TODO: review marker color related code and that the objective of 
# showing marker contour colors and then watershed areas both with
# black borders. Last review it seemed to be wrong in that watershed
# image did not have black borders (which represent boundaries)
def markerToColor(image: np.array, unique_labels: np.array) -> np.array:
  colored_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
  for label in unique_labels:
    if label <= 0:
        continue
    
    mask = image == label
    label = int(label)
    color = (label * 123 % 256, label * 456 % 256, label * 789 % 256)
    while color == (0, 0, 0): # save (0, 0, 0) for the background)
      color = (label * 123 % 256, label * 456 % 256, label * 789 % 256)
    b, g, r = color
    colored_image[mask] = (b, g, r)

  return colored_image

def calculateWatershed(border_image: np.array, markers: np.array) -> np.array:
  markers_local = np.copy(markers)
  markers_local = markers_local + 1 # to avoid 0 label which is unknown for watershed.

  border_image = cv.cvtColor(border_image, cv.COLOR_GRAY2BGR)

  watershed_image = cv.watershed(border_image, markers_local)
  watershed_image = watershed_image.astype(np.uint8)

  unique_labels = np.unique(watershed_image)
  colored_image = markerToColor(watershed_image, unique_labels)
  return colored_image

def borderFilterAlternativesAnalysis(filename: str,
                                        write_images: bool = False,
                                        show_images: bool = True) -> None:
  try:
    with Image.open(filename) as imagefile:
      if imagefile.mode == 'L':
        imagefile = imagefile.convert('RGB')
        input_image = np.array(imagefile)
      elif imagefile.mode == 'RGB':
        input_image = np.array(imagefile)
  except Exception as e:
    print(f"Error opening image {filename}: {e}")
    raise
  input_image = transforms.ToTensor()(input_image)

  output_image = ContrastEnhancement().process(input_image)
  output_image = output_image[:, :, 0]
  output_image = cv.normalize(output_image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

  kernel_sizes = [3, 5, 7, 9]
  gaussian_sigmas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  for kernel_size in kernel_sizes:
    for gaussian_sigma in gaussian_sigmas:
      gaussian_blurred = cv.GaussianBlur(output_image, (kernel_size, kernel_size),
                                        gaussian_sigma)

      # Border detection Method: Statistical range
      borders_detected = BorderDetectionStatisticalRange(
            padding=(5 // 2)
          ).process(gaussian_blurred)

      numLabels, markers = cv.connectedComponents(borders_detected)
      markers = markers.astype(np.int32)
      
      watershed_image = calculateWatershed(borders_detected, markers)
      markers = markerToColor(markers, np.unique(markers))

      concatenated = np.concatenate((cv.cvtColor(output_image, cv.COLOR_GRAY2BGR),
                                    cv.cvtColor(gaussian_blurred, cv.COLOR_GRAY2BGR),
                                    cv.cvtColor(borders_detected, cv.COLOR_GRAY2BGR),
                                    markers,
                                    watershed_image,
                                    ), axis=1)

      if write_images:
        cv.imwrite(f'docs/local_images/{os.path.basename(filename)}' \
                  f'_statrange_border' \
                  f'_sigma{gaussian_sigma}' \
                  f'_kernelsize{kernel_size}' \
                  f'.jpg',
                  concatenated)
      else:
        if show_images:
          cv.imshow(f'{os.path.basename(filename)}' \
            f'_statrange_border' \
            f'_sigma{gaussian_sigma}' \
            f'_kernelsize{kernel_size}'
            f'.jpg',
            concatenated)
          cv.waitKey(0)
          cv.destroyAllWindows()

      # Border detection Method: Canny Border Detector

      # I don't know if moving this to after gaussian blur image calculation
      # changes things up so temporarily leave this to not make too many changes.
      gaussian_blurred = cv.normalize(gaussian_blurred, None, 0, 255,
                                      cv.NORM_MINMAX, cv.CV_8U)

      min_thresholds = [30, 51, 70]
      max_thresholds = [135, 150, 165]
      for min_threshold  in min_thresholds:
        for max_threshold in max_thresholds:
          borders_detected = cv.Canny(gaussian_blurred, min_threshold,
                                      max_threshold)
          borders_detected = cv.normalize(borders_detected, None, 0, 255,
                                          cv.NORM_MINMAX, cv.CV_8U)

          numLabels, markers = cv.connectedComponents(borders_detected)
          markers = markers.astype(np.int32)

          watershed_image = calculateWatershed(borders_detected, markers)
          markers = markerToColor(markers, np.unique(markers))

          concatenated = np.concatenate((cv.cvtColor(output_image, cv.COLOR_GRAY2BGR),
                                         cv.cvtColor(gaussian_blurred, cv.COLOR_GRAY2BGR),
                                         cv.cvtColor(borders_detected, cv.COLOR_GRAY2BGR),
                                         markers,
                                         watershed_image,
                                         ), axis=1)
          if write_images:
            cv.imwrite(f'docs/local_images/{os.path.basename(filename)}' \
                       f'_canny_border' \
                       f'_sigma{gaussian_sigma}' \
                       f'_sigma{gaussian_sigma}' \
                       f'_kernelsize{kernel_size}' \
                       f'_minThresh{min_threshold}' \
                       f'_maxThresh{max_threshold}' \
                       f'.jpg',
                       concatenated)
          else:
            if show_images:
              cv.imshow(f'{os.path.basename(filename)}' \
                        f'_canny_border' \
                        f'_sigma{gaussian_sigma}' \
                        f'_kernelsize{kernel_size}' \
                        f'_minThresh{min_threshold}' \
                        f'_maxThresh{max_threshold}' \
                        f'.jpg',
                        concatenated)
              cv.waitKey(0)
              cv.destroyAllWindows()
