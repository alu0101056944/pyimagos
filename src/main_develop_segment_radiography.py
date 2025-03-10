'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Given a left hand radiography with clearly defined the exact contours
corresponding to each bone, and nothing more, segment those contours by relative
positions.

The algorithm is sensitive to contour amount. Meant for 21 contours:

distal 1, 2, 3, 4, 5
medial 1, 2, 3, 4
proximal 1, 2, 3, 4, 5
metacarpal 1, 2, 3, 4, 5
ulna, radius

1 = leftmost in the radiography, so the pinky finger first.
'''

from PIL import Image

import numpy as np
import cv2 as cv

def segment_radiography_main(filename: str):
  image = None
  try:
    with Image.open(filename) as imagefile:
      if imagefile.mode == 'L':
        imagefile = imagefile.convert('RGB')
        image = np.array(imagefile)
      elif imagefile.mode == 'RGB':
        image = np.array(imagefile)
  except Exception as e:
    print(f"Error opening image {filename}: {e}")
    raise

  image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  _, thresholded = cv.threshold(image, 40, 255, cv.THRESH_BINARY)

  contours, _ = cv.findContours(
    thresholded,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  )

  if len(contours) == 0:
    print('Empty contours. Stopping.')
    print('Stopped.')
    return

  segments_indices = {}
  contours = [
    np.reshape(contour, (-1, 2)) for contour in contours
  ]

  indexed_contours = list(enumerate(contours)) # (index, contour) list

  # get ulna and radius
  def get_max_y(indexed_contour):
    index, contour = indexed_contour
    all_y = contour[:, 1]
    return np.max(all_y) if len(all_y) > 0 else -np.inf # to the back
  
  contour_max_y_pairs = [
    (get_max_y(contour), contour) for contour in indexed_contours
  ]
  sorted_pairs = sorted(contour_max_y_pairs, key=lambda pair: pair[0],
                        reverse=True)
  ordered_contours = [pair[1] for pair in sorted_pairs]

  ulna_and_radius = ordered_contours[:2]

  def get_min_x(indexed_contour):
    index, contour = indexed_contour
    all_x = contour[:, 0]
    return np.min(all_x) if len(all_x) > 0 else -np.inf # to the back

  ulna_and_radius.sort()
  contour_min_x_pairs = [
    (get_min_x(contour), contour) for contour in ulna_and_radius
  ]
  sorted_pairs_2 = sorted(contour_min_x_pairs, key=lambda pair: pair[0])
  ordered_contours_2 = [pair[1] for pair in sorted_pairs_2]
  ulna_index = ordered_contours_2[0][0] # store index
  radius_index = ordered_contours_2[1][0]
  segments_indices['ulna'] = ulna_index
  segments_indices['radius'] = radius_index
  
  # get first four fingers
  def get_min_y(indexed_contour):
    index, contour = indexed_contour
    all_y = contour[:, 1]
    return np.min(all_y) if len(all_y) > 0 else -np.inf # to the back

  rest_of_contours = ordered_contours[2:]
  contour_min_x_pairs_2 = [
    (get_min_x(contour), contour) for contour in rest_of_contours
  ]
  sorted_pairs_3 = sorted(contour_min_x_pairs_2, key=lambda pair: pair[0])
  ordered_contours_3 = [pair[1] for pair in sorted_pairs_3]

  for i in range(4):
    finger_contours = ordered_contours_3[i * 4: i * 4 + 4]
    contour_min_y_pairs = [
      (get_min_y(contour), contour) for contour in finger_contours
    ]
    sorted_pairs_4 = sorted(contour_min_y_pairs, key=lambda pair: pair[0])
    ordered_contours_4 = [pair[1] for pair in sorted_pairs_4]
    segments_indices[f'distal_{i + 1}'] = ordered_contours_4[0][0] # Store index
    segments_indices[f'medial_{i + 1}'] = ordered_contours_4[1][0]
    segments_indices[f'proximal_{i + 1}'] = ordered_contours_4[2][0]
    segments_indices[f'metacarpal_{i + 1}'] = ordered_contours_4[3][0]

  # get fifth finger
  finger_contours = ordered_contours_3[16:19]
  contour_min_y_pairs_2 = [
    (get_min_y(contour), contour) for contour in finger_contours
  ]
  sorted_pairs_5 = sorted(contour_min_y_pairs_2, key=lambda pair: pair[0])
  ordered_pairs_5 = [pair[1] for pair in sorted_pairs_5]
  segments_indices[f'distal_5'] = ordered_pairs_5[0][0]
  segments_indices[f'proximal_5'] = ordered_pairs_5[1][0]
  segments_indices[f'metacarpal_5'] = ordered_pairs_5[2][0]

  with open('segmented_radiography.txt', 'w') as f:
    f.write(str(contours) + '\n' + str(segments_indices))
  print('Writing segmented_radiography.txt')
  print('Success.')
