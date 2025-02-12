'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Visualization of the search execution test
'''

from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from src.main_develop_test_distal_phalanx import (
  create_minimal_image_from_contours,
)
from src.expected_contours.distal_phalanx import ExpectedContourDistalPhalanx
from src.expected_contours.medial_phalanx import ExpectedContourMedialPhalanx
from src.expected_contours.expected_contour import ExpectedContour

def show_contours_position_restrictions(contours, contour_id,
                                        expected_contour: ExpectedContour, 
                                        padding=5,
                                        title='Position restrictions visualization',
                                        minimize_image: bool = True,
                                        branch_jump: bool = False,
                                        first_in_branch: ExpectedContour = None):
  all_points = np.concatenate(contours)
  all_points = np.reshape(all_points, (-1, 2))
  x_values = all_points[:, 0]
  y_values = all_points[:, 1]

  max_x = int(np.max(x_values))
  max_y = int(np.max(y_values))

  blank_image = np.zeros((max_y + 300, max_x + 25), dtype=np.uint8)

  if minimize_image:
    minimal_image, adjusted_contours = create_minimal_image_from_contours(
      blank_image,
      contours,
      padding
    )
    minimal_image = cv.cvtColor(minimal_image, cv.COLOR_GRAY2RGB)
    contours = adjusted_contours
  else:
    minimal_image = cv.cvtColor(blank_image, cv.COLOR_GRAY2RGB)

  for i, contour in enumerate(contours):
    color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
    if len(contour) > 0:
      cv.drawContours(minimal_image, contours, i, color, 1)

    
  expected_contour.prepare(contours[contour_id], minimal_image.shape[1],
                           minimal_image.shape[0])

  if not branch_jump:
    position_restrictions = expected_contour.next_contour_restrictions()
  else:
    position_restrictions = first_in_branch.branch_start_position_restrictions()

  for position_restriction in position_restrictions:
    point_a = position_restriction[0]
    point_b = position_restriction[1]
    direction = point_b - point_a
    cv.line(
      minimal_image,
      (point_a - direction * minimal_image.shape[1]).astype(np.int32),
      (point_b + direction * minimal_image.shape[1]).astype(np.int32),
      (255, 255, 0),
      1
    )

  fig = plt.figure()
  plt.imshow(minimal_image)
  plt.title(title)
  plt.axis('off')
  fig.canvas.manager.set_window_title(title)

def visualize_execute_tests():
  # borders_detected = Image.open('docs/composition_two.jpg')
  # borders_detected = np.array(borders_detected)
  # borders_detected = cv.cvtColor(borders_detected, cv.COLOR_RGB2GRAY)

  # _, thresh = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
  # contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,
  #                             cv.CHAIN_APPROX_SIMPLE)

  contours = [
    np.array([[[ 37,  97]],
      [[ 36,  98]],
      [[ 34,  98]],
      [[ 33,  99]],
      [[ 32,  99]],
      [[ 31, 100]],
      [[ 27, 100]],
      [[ 26,  99]],
      [[ 22,  99]],
      [[ 20, 101]],
      [[ 20, 106]],
      [[ 22, 108]],
      [[ 22, 109]],
      [[ 23, 110]],
      [[ 23, 115]],
      [[ 24, 116]],
      [[ 24, 117]],
      [[ 25, 118]],
      [[ 25, 123]],
      [[ 26, 124]],
      [[ 26, 132]],
      [[ 27, 133]],
      [[ 27, 134]],
      [[ 35, 134]],
      [[ 36, 133]],
      [[ 41, 133]],
      [[ 42, 132]],
      [[ 43, 132]],
      [[ 44, 131]],
      [[ 45, 131]],
      [[ 46, 130]],
      [[ 51, 130]],
      [[ 52, 129]],
      [[ 53, 129]],
      [[ 53, 126]],
      [[ 48, 121]],
      [[ 48, 120]],
      [[ 45, 117]],
      [[ 45, 116]],
      [[ 43, 114]],
      [[ 43, 113]],
      [[ 42, 112]],
      [[ 42, 105]],
      [[ 43, 104]],
      [[ 43, 102]],
      [[ 44, 101]],
      [[ 40,  97]]],
      dtype=np.int32
    ),
    np.array([[[25, 66]],
      [[24, 67]],
      [[21, 67]],
      [[19, 69]],
      [[19, 72]],
      [[20, 73]],
      [[20, 74]],
      [[21, 75]],
      [[21, 77]],
      [[22, 78]],
      [[22, 81]],
      [[23, 82]],
      [[23, 83]],
      [[22, 84]],
      [[22, 87]],
      [[21, 88]],
      [[21, 89]],
      [[20, 90]],
      [[20, 91]],
      [[19, 92]],
      [[19, 96]],
      [[20, 97]],
      [[31, 97]],
      [[32, 96]],
      [[35, 96]],
      [[36, 95]],
      [[39, 95]],
      [[40, 94]],
      [[41, 94]],
      [[40, 93]],
      [[40, 92]],
      [[41, 91]],
      [[40, 90]],
      [[39, 90]],
      [[34, 85]],
      [[34, 84]],
      [[32, 82]],
      [[32, 79]],
      [[31, 78]],
      [[31, 75]],
      [[32, 74]],
      [[32, 68]],
      [[31, 67]],
      [[28, 67]],
      [[27, 66]]],
      dtype=np.int32
    )
  ]
  expected_contour = ExpectedContourDistalPhalanx(1)
  show_contours_position_restrictions(contours, 1, expected_contour, padding=5,
               title='Composition.', minimize_image=False)

  contours = [
    np.array([[[ 37,  97]],
      [[ 36,  98]],
      [[ 34,  98]],
      [[ 33,  99]],
      [[ 32,  99]],
      [[ 31, 100]],
      [[ 27, 100]],
      [[ 26,  99]],
      [[ 22,  99]],
      [[ 20, 101]],
      [[ 20, 106]],
      [[ 22, 108]],
      [[ 22, 109]],
      [[ 23, 110]],
      [[ 23, 115]],
      [[ 24, 116]],
      [[ 24, 117]],
      [[ 25, 118]],
      [[ 25, 123]],
      [[ 26, 124]],
      [[ 26, 132]],
      [[ 27, 133]],
      [[ 27, 134]],
      [[ 35, 134]],
      [[ 36, 133]],
      [[ 41, 133]],
      [[ 42, 132]],
      [[ 43, 132]],
      [[ 44, 131]],
      [[ 45, 131]],
      [[ 46, 130]],
      [[ 51, 130]],
      [[ 52, 129]],
      [[ 53, 129]],
      [[ 53, 126]],
      [[ 48, 121]],
      [[ 48, 120]],
      [[ 45, 117]],
      [[ 45, 116]],
      [[ 43, 114]],
      [[ 43, 113]],
      [[ 42, 112]],
      [[ 42, 105]],
      [[ 43, 104]],
      [[ 43, 102]],
      [[ 44, 101]],
      [[ 40,  97]]],
      dtype=np.int32
    ),
    np.array([[[25, 66]],
      [[24, 67]],
      [[21, 67]],
      [[19, 69]],
      [[19, 72]],
      [[20, 73]],
      [[20, 74]],
      [[21, 75]],
      [[21, 77]],
      [[22, 78]],
      [[22, 81]],
      [[23, 82]],
      [[23, 83]],
      [[22, 84]],
      [[22, 87]],
      [[21, 88]],
      [[21, 89]],
      [[20, 90]],
      [[20, 91]],
      [[19, 92]],
      [[19, 96]],
      [[20, 97]],
      [[31, 97]],
      [[32, 96]],
      [[35, 96]],
      [[36, 95]],
      [[39, 95]],
      [[40, 94]],
      [[41, 94]],
      [[40, 93]],
      [[40, 92]],
      [[41, 91]],
      [[40, 90]],
      [[39, 90]],
      [[34, 85]],
      [[34, 84]],
      [[32, 82]],
      [[32, 79]],
      [[31, 78]],
      [[31, 75]],
      [[32, 74]],
      [[32, 68]],
      [[31, 67]],
      [[28, 67]],
      [[27, 66]]],
      dtype=np.int32
    ),
    np.array(
      [[[ 98,  35]],
      [[ 97,  36]],
      [[ 96,  36]],
      [[ 94,  38]],
      [[ 94,  39]],
      [[ 93,  40]],
      [[ 93,  42]],
      [[ 94,  43]],
      [[ 94,  44]],
      [[ 95,  45]],
      [[ 95,  47]],
      [[ 96,  48]],
      [[ 96,  50]],
      [[ 95,  51]],
      [[ 95,  54]],
      [[ 94,  55]],
      [[ 94,  56]],
      [[ 93,  57]],
      [[ 93,  58]],
      [[ 88,  63]],
      [[ 88,  67]],
      [[ 90,  67]],
      [[ 91,  68]],
      [[ 98,  68]],
      [[ 99,  69]],
      [[104,  69]],
      [[105,  68]],
      [[111,  68]],
      [[112,  67]],
      [[115,  67]],
      [[115,  63]],
      [[110,  58]],
      [[110,  57]],
      [[109,  56]],
      [[109,  55]],
      [[108,  54]],
      [[108,  50]],
      [[109,  49]],
      [[109,  46]],
      [[110,  45]],
      [[110,  44]],
      [[111,  43]],
      [[111,  41]],
      [[110,  40]],
      [[110,  37]],
      [[109,  37]],
      [[108,  36]],
      [[106,  36]],
      [[105,  35]]],
      dtype=np.int32
    )
  ]
  expected_contour = ExpectedContourDistalPhalanx(1)
  show_contours_position_restrictions(
    contours,
    1,
    expected_contour,
    padding=5,
    title='Composition with jump.',
    minimize_image=False,
    branch_jump=True,
    first_in_branch=expected_contour
  )

  contours = [
    np.array([[[ 39, 116]],
          [[ 38, 117]],
          [[ 36, 117]],
          [[ 35, 118]],
          [[ 34, 118]],
          [[ 33, 119]],
          [[ 29, 119]],
          [[ 28, 118]],
          [[ 24, 118]],
          [[ 22, 120]],
          [[ 22, 125]],
          [[ 24, 127]],
          [[ 24, 128]],
          [[ 25, 129]],
          [[ 25, 134]],
          [[ 26, 135]],
          [[ 26, 136]],
          [[ 27, 137]],
          [[ 27, 142]],
          [[ 28, 143]],
          [[ 28, 151]],
          [[ 29, 152]],
          [[ 29, 153]],
          [[ 37, 153]],
          [[ 38, 152]],
          [[ 43, 152]],
          [[ 44, 151]],
          [[ 45, 151]],
          [[ 46, 150]],
          [[ 47, 150]],
          [[ 48, 149]],
          [[ 53, 149]],
          [[ 54, 148]],
          [[ 55, 148]],
          [[ 55, 145]],
          [[ 50, 140]],
          [[ 50, 139]],
          [[ 47, 136]],
          [[ 47, 135]],
          [[ 45, 133]],
          [[ 45, 132]],
          [[ 44, 131]],
          [[ 44, 124]],
          [[ 45, 123]],
          [[ 45, 121]],
          [[ 46, 120]],
          [[ 42, 116]]],
          dtype=np.int32
    ),
          
    np.array([[[76, 89]],
          [[76, 91]],
          [[75, 92]],
          [[75, 95]],
          [[74, 96]],
          [[74, 97]],
          [[79, 97]],
          [[80, 96]],
          [[85, 96]],
          [[83, 94]],
          [[82, 94]],
          [[79, 91]],
          [[78, 91]]],
          dtype=np.int32
    ),

    np.array([[[ 27,  85]],
          [[ 26,  86]],
          [[ 23,  86]],
          [[ 21,  88]],
          [[ 21,  91]],
          [[ 22,  92]],
          [[ 22,  93]],
          [[ 23,  94]],
          [[ 23,  96]],
          [[ 24,  97]],
          [[ 24, 100]],
          [[ 25, 101]],
          [[ 25, 102]],
          [[ 24, 103]],
          [[ 24, 106]],
          [[ 23, 107]],
          [[ 23, 108]],
          [[ 22, 109]],
          [[ 22, 110]],
          [[ 21, 111]],
          [[ 21, 115]],
          [[ 22, 116]],
          [[ 33, 116]],
          [[ 34, 115]],
          [[ 37, 115]],
          [[ 38, 114]],
          [[ 41, 114]],
          [[ 42, 113]],
          [[ 43, 113]],
          [[ 42, 112]],
          [[ 42, 111]],
          [[ 43, 110]],
          [[ 42, 109]],
          [[ 41, 109]],
          [[ 36, 104]],
          [[ 36, 103]],
          [[ 34, 101]],
          [[ 34,  98]],
          [[ 33,  97]],
          [[ 33,  94]],
          [[ 34,  93]],
          [[ 34,  87]],
          [[ 33,  86]],
          [[ 30,  86]],
          [[ 29,  85]]],
          dtype=np.int32
    ),
    np.array([[[55, 72]],
          [[54, 73]],
          [[50, 73]],
          [[49, 74]],
          [[47, 74]],
          [[49, 76]],
          [[49, 77]],
          [[52, 80]],
          [[52, 81]],
          [[54, 83]],
          [[57, 83]],
          [[58, 82]],
          [[61, 82]],
          [[61, 81]],
          [[60, 80]],
          [[60, 79]],
          [[59, 78]],
          [[59, 76]],
          [[58, 75]],
          [[58, 74]],
          [[57, 73]],
          [[57, 72]]],
          dtype=np.int32
    ),
    np.array([[[74, 70]],
          [[73, 71]],
          [[72, 71]],
          [[71, 72]],
          [[70, 72]],
          [[70, 76]],
          [[71, 77]],
          [[71, 81]],
          [[78, 81]],
          [[78, 77]],
          [[79, 76]],
          [[79, 74]],
          [[76, 71]],
          [[75, 71]]],
          dtype=np.int32
    ),
    np.array([[[130,  54]],
          [[130,  56]],
          [[131,  57]],
          [[131,  61]],
          [[132,  62]],
          [[132,  64]],
          [[133,  64]],
          [[134,  63]],
          [[137,  63]],
          [[138,  62]],
          [[141,  62]],
          [[142,  61]],
          [[145,  61]],
          [[146,  60]],
          [[147,  60]],
          [[146,  60]],
          [[145,  59]],
          [[143,  59]],
          [[142,  58]],
          [[140,  58]],
          [[139,  57]],
          [[138,  57]],
          [[137,  56]],
          [[135,  56]],
          [[134,  55]],
          [[132,  55]],
          [[131,  54]]],
          dtype=np.int32
    ),
          
    np.array([
          [[ 98,  35]],
          [[ 97,  36]],
          [[ 96,  36]],
          [[ 94,  38]],
          [[ 94,  39]],
          [[ 93,  40]],
          [[ 93,  42]],
          [[ 94,  43]],
          [[ 94,  44]],
          [[ 95,  45]],
          [[ 95,  47]],
          [[ 96,  48]],
          [[ 96,  50]],
          [[ 95,  51]],
          [[ 95,  54]],
          [[ 94,  55]],
          [[ 94,  56]],
          [[ 93,  57]],
          [[ 93,  58]],
          [[ 88,  63]],
          [[ 88,  67]],
          [[ 90,  67]],
          [[ 91,  68]],
          [[ 98,  68]],
          [[ 99,  69]],
          [[104,  69]],
          [[105,  68]],
          [[111,  68]],
          [[112,  67]],
          [[115,  67]],
          [[115,  63]],
          [[110,  58]],
          [[110,  57]],
          [[109,  56]],
          [[109,  55]],
          [[108,  54]],
          [[108,  50]],
          [[109,  49]],
          [[109,  46]],
          [[110,  45]],
          [[110,  44]],
          [[111,  43]],
          [[111,  41]],
          [[110,  40]],
          [[110,  37]],
          [[109,  37]],
          [[108,  36]],
          [[106,  36]],
          [[105,  35]]],
          dtype=np.int32
    ),

    np.array([[[34, 11]],
          [[33, 12]],
          [[33, 13]],
          [[31, 15]],
          [[31, 16]],
          [[29, 18]],
          [[29, 19]],
          [[26, 22]],
          [[26, 23]],
          [[24, 25]],
          [[24, 26]],
          [[23, 27]],
          [[26, 27]],
          [[27, 26]],
          [[33, 26]],
          [[34, 25]],
          [[41, 25]],
          [[42, 24]],
          [[48, 24]],
          [[49, 23]],
          [[52, 23]],
          [[51, 22]],
          [[50, 22]],
          [[48, 20]],
          [[47, 20]],
          [[45, 18]],
          [[44, 18]],
          [[42, 16]],
          [[41, 16]],
          [[39, 14]],
          [[38, 14]],
          [[36, 12]],
          [[35, 12]]],
          dtype=np.int32
    ),
  ]
  expected_contour = ExpectedContourDistalPhalanx(1)
  show_contours_position_restrictions(
    contours,
    2,
    expected_contour,
    padding=5,
    title='Noisy with jump.',
    minimize_image=False,
    branch_jump=True,
    first_in_branch=expected_contour
  )

  contours = [
      np.array([[[ 39, 116]],
       [[ 38, 117]],
       [[ 36, 117]],
       [[ 35, 118]],
       [[ 34, 118]],
       [[ 33, 119]],
       [[ 29, 119]],
       [[ 28, 118]],
       [[ 24, 118]],
       [[ 22, 120]],
       [[ 22, 125]],
       [[ 24, 127]],
       [[ 24, 128]],
       [[ 25, 129]],
       [[ 25, 134]],
       [[ 26, 135]],
       [[ 26, 136]],
       [[ 27, 137]],
       [[ 27, 142]],
       [[ 28, 143]],
       [[ 28, 151]],
       [[ 29, 152]],
       [[ 29, 153]],
       [[ 37, 153]],
       [[ 38, 152]],
       [[ 43, 152]],
       [[ 44, 151]],
       [[ 45, 151]],
       [[ 46, 150]],
       [[ 47, 150]],
       [[ 48, 149]],
       [[ 53, 149]],
       [[ 54, 148]],
       [[ 55, 148]],
       [[ 55, 145]],
       [[ 50, 140]],
       [[ 50, 139]],
       [[ 47, 136]],
       [[ 47, 135]],
       [[ 45, 133]],
       [[ 45, 132]],
       [[ 44, 131]],
       [[ 44, 124]],
       [[ 45, 123]],
       [[ 45, 121]],
       [[ 46, 120]],
       [[ 42, 116]]],
       dtype=np.int32
    ),
    np.array([[[76, 89]],
       [[76, 91]],
       [[75, 92]],
       [[75, 95]],
       [[74, 96]],
       [[74, 97]],
       [[79, 97]],
       [[80, 96]],
       [[85, 96]],
       [[83, 94]],
       [[82, 94]],
       [[79, 91]],
       [[78, 91]]],
       dtype=np.int32
    ),
    np.array([[[ 27,  85]],
       [[ 26,  86]],
       [[ 23,  86]],
       [[ 21,  88]],
       [[ 21,  91]],
       [[ 22,  92]],
       [[ 22,  93]],
       [[ 23,  94]],
       [[ 23,  96]],
       [[ 24,  97]],
       [[ 24, 100]],
       [[ 25, 101]],
       [[ 25, 102]],
       [[ 24, 103]],
       [[ 24, 106]],
       [[ 23, 107]],
       [[ 23, 108]],
       [[ 22, 109]],
       [[ 22, 110]],
       [[ 21, 111]],
       [[ 21, 115]],
       [[ 22, 116]],
       [[ 33, 116]],
       [[ 34, 115]],
       [[ 37, 115]],
       [[ 38, 114]],
       [[ 41, 114]],
       [[ 42, 113]],
       [[ 43, 113]],
       [[ 42, 112]],
       [[ 42, 111]],
       [[ 43, 110]],
       [[ 42, 109]],
       [[ 41, 109]],
       [[ 36, 104]],
       [[ 36, 103]],
       [[ 34, 101]],
       [[ 34,  98]],
       [[ 33,  97]],
       [[ 33,  94]],
       [[ 34,  93]],
       [[ 34,  87]],
       [[ 33,  86]],
       [[ 30,  86]],
       [[ 29,  85]]],
       dtype=np.int32
    ),
    np.array([[[55, 72]],
       [[54, 73]],
       [[50, 73]],
       [[49, 74]],
       [[47, 74]],
       [[49, 76]],
       [[49, 77]],
       [[52, 80]],
       [[52, 81]],
       [[54, 83]],
       [[57, 83]],
       [[58, 82]],
       [[61, 82]],
       [[61, 81]],
       [[60, 80]],
       [[60, 79]],
       [[59, 78]],
       [[59, 76]],
       [[58, 75]],
       [[58, 74]],
       [[57, 73]],
       [[57, 72]]],
       dtype=np.int32
    ),
    np.array([[[74, 70]],
       [[73, 71]],
       [[72, 71]],
       [[71, 72]],
       [[70, 72]],
       [[70, 76]],
       [[71, 77]],
       [[71, 81]],
       [[78, 81]],
       [[78, 77]],
       [[79, 76]],
       [[79, 74]],
       [[76, 71]],
       [[75, 71]]],
       dtype=np.int32
    ),
    np.array([[[130,  54]],
       [[130,  56]],
       [[131,  57]],
       [[131,  61]],
       [[132,  62]],
       [[132,  64]],
       [[133,  64]],
       [[134,  63]],
       [[137,  63]],
       [[138,  62]],
       [[141,  62]],
       [[142,  61]],
       [[145,  61]],
       [[146,  60]],
       [[147,  60]],
       [[146,  60]],
       [[145,  59]],
       [[143,  59]],
       [[142,  58]],
       [[140,  58]],
       [[139,  57]],
       [[138,  57]],
       [[137,  56]],
       [[135,  56]],
       [[134,  55]],
       [[132,  55]],
       [[131,  54]]],
       dtype=np.int32),
    np.array([[[34, 11]],
       [[33, 12]],
       [[33, 13]],
       [[31, 15]],
       [[31, 16]],
       [[29, 18]],
       [[29, 19]],
       [[26, 22]],
       [[26, 23]],
       [[24, 25]],
       [[24, 26]],
       [[23, 27]],
       [[26, 27]],
       [[27, 26]],
       [[33, 26]],
       [[34, 25]],
       [[41, 25]],
       [[42, 24]],
       [[48, 24]],
       [[49, 23]],
       [[52, 23]],
       [[51, 22]],
       [[50, 22]],
       [[48, 20]],
       [[47, 20]],
       [[45, 18]],
       [[44, 18]],
       [[42, 16]],
       [[41, 16]],
       [[39, 14]],
       [[38, 14]],
       [[36, 12]],
       [[35, 12]]],
       dtype=np.int32
      )
    ]
  expected_contour = ExpectedContourDistalPhalanx(1)
  show_contours_position_restrictions(
    contours,
    2,
    expected_contour,
    padding=5,
    title='All invalid shape after first two ideal.',
    minimize_image=False,
    branch_jump=True,
    first_in_branch=expected_contour
  )

  ideal_medial_phalanx = np.array(
      [[[ 37,  97]],
      [[ 36,  98]],
      [[ 34,  98]],
      [[ 33,  99]],
      [[ 32,  99]],
      [[ 31, 100]],
      [[ 27, 100]],
      [[ 26,  99]],
      [[ 22,  99]],
      [[ 20, 101]],
      [[ 20, 106]],
      [[ 22, 108]],
      [[ 22, 109]],
      [[ 23, 110]],
      [[ 23, 115]],
      [[ 24, 116]],
      [[ 24, 117]],
      [[ 25, 118]],
      [[ 25, 123]],
      [[ 26, 124]],
      [[ 26, 132]],
      [[ 27, 133]],
      [[ 27, 134]],
      [[ 35, 134]],
      [[ 36, 133]],
      [[ 41, 133]],
      [[ 42, 132]],
      [[ 43, 132]],
      [[ 44, 131]],
      [[ 45, 131]],
      [[ 46, 130]],
      [[ 51, 130]],
      [[ 52, 129]],
      [[ 53, 129]],
      [[ 53, 126]],
      [[ 48, 121]],
      [[ 48, 120]],
      [[ 45, 117]],
      [[ 45, 116]],
      [[ 43, 114]],
      [[ 43, 113]],
      [[ 42, 112]],
      [[ 42, 105]],
      [[ 43, 104]],
      [[ 43, 102]],
      [[ 44, 101]],
      [[ 40,  97]]],
      dtype=np.int32
    )
  ideal_distal_phalanx = np.array(
      [[[25, 66]],
      [[24, 67]],
      [[21, 67]],
      [[19, 69]],
      [[19, 72]],
      [[20, 73]],
      [[20, 74]],
      [[21, 75]],
      [[21, 77]],
      [[22, 78]],
      [[22, 81]],
      [[23, 82]],
      [[23, 83]],
      [[22, 84]],
      [[22, 87]],
      [[21, 88]],
      [[21, 89]],
      [[20, 90]],
      [[20, 91]],
      [[19, 92]],
      [[19, 96]],
      [[20, 97]],
      [[31, 97]],
      [[32, 96]],
      [[35, 96]],
      [[36, 95]],
      [[39, 95]],
      [[40, 94]],
      [[41, 94]],
      [[40, 93]],
      [[40, 92]],
      [[41, 91]],
      [[40, 90]],
      [[39, 90]],
      [[34, 85]],
      [[34, 84]],
      [[32, 82]],
      [[32, 79]],
      [[31, 78]],
      [[31, 75]],
      [[32, 74]],
      [[32, 68]],
      [[31, 67]],
      [[28, 67]],
      [[27, 66]]],
      dtype=np.int32
    )
  contours = [
    ideal_distal_phalanx,
    ideal_medial_phalanx,
  ]
  show_contours_position_restrictions(
    contours,
    0,
    expected_contour,
    padding=5,
    title='Not enough candidate contours after first two ideal.',
    minimize_image=False,
    branch_jump=True,
    first_in_branch=expected_contour
  )

  contours = [
       np.array([[[ 53, 151]],
              [[ 51, 153]],
              [[ 49, 153]],
              [[ 48, 154]],
              [[ 46, 154]],
              [[ 45, 155]],
              [[ 37, 155]],
              [[ 36, 156]],
              [[ 35, 156]],
              [[ 35, 164]],
              [[ 38, 167]],
              [[ 38, 168]],
              [[ 40, 170]],
              [[ 40, 171]],
              [[ 42, 173]],
              [[ 42, 174]],
              [[ 43, 175]],
              [[ 43, 177]],
              [[ 44, 178]],
              [[ 44, 180]],
              [[ 45, 181]],
              [[ 45, 183]],
              [[ 46, 184]],
              [[ 46, 185]],
              [[ 47, 186]],
              [[ 47, 187]],
              [[ 48, 188]],
              [[ 48, 191]],
              [[ 49, 192]],
              [[ 49, 195]],
              [[ 50, 196]],
              [[ 50, 203]],
              [[ 49, 204]],
              [[ 49, 211]],
              [[ 51, 213]],
              [[ 59, 213]],
              [[ 60, 212]],
              [[ 61, 212]],
              [[ 62, 211]],
              [[ 64, 211]],
              [[ 65, 210]],
              [[ 67, 210]],
              [[ 68, 209]],
              [[ 71, 209]],
              [[ 72, 208]],
              [[ 83, 208]],
              [[ 84, 207]],
              [[ 85, 207]],
              [[ 85, 205]],
              [[ 86, 204]],
              [[ 86, 202]],
              [[ 85, 201]],
              [[ 85, 200]],
              [[ 84, 199]],
              [[ 83, 199]],
              [[ 70, 186]],
              [[ 70, 185]],
              [[ 68, 183]],
              [[ 68, 182]],
              [[ 67, 181]],
              [[ 67, 180]],
              [[ 65, 178]],
              [[ 65, 177]],
              [[ 64, 176]],
              [[ 64, 175]],
              [[ 63, 174]],
              [[ 63, 173]],
              [[ 62, 172]],
              [[ 62, 170]],
              [[ 60, 168]],
              [[ 60, 163]],
              [[ 59, 162]],
              [[ 59, 160]],
              [[ 60, 159]],
              [[ 60, 158]],
              [[ 59, 157]],
              [[ 59, 155]],
              [[ 58, 154]],
              [[ 57, 154]],
              [[ 54, 151]]], 
              dtype=np.int32
       ),
       np.array([[[ 39, 116]],
              [[ 38, 117]],
              [[ 36, 117]],
              [[ 35, 118]],
              [[ 34, 118]],
              [[ 33, 119]],
              [[ 29, 119]],
              [[ 28, 118]],
              [[ 24, 118]],
              [[ 22, 120]],
              [[ 22, 125]],
              [[ 24, 127]],
              [[ 24, 128]],
              [[ 25, 129]],
              [[ 25, 134]],
              [[ 26, 135]],
              [[ 26, 136]],
              [[ 27, 137]],
              [[ 27, 142]],
              [[ 28, 143]],
              [[ 28, 151]],
              [[ 29, 152]],
              [[ 29, 153]],
              [[ 37, 153]],
              [[ 38, 152]],
              [[ 43, 152]],
              [[ 44, 151]],
              [[ 45, 151]],
              [[ 46, 150]],
              [[ 47, 150]],
              [[ 48, 149]],
              [[ 53, 149]],
              [[ 54, 148]],
              [[ 55, 148]],
              [[ 55, 145]],
              [[ 50, 140]],
              [[ 50, 139]],
              [[ 47, 136]],
              [[ 47, 135]],
              [[ 45, 133]],
              [[ 45, 132]],
              [[ 44, 131]],
              [[ 44, 124]],
              [[ 45, 123]],
              [[ 45, 121]],
              [[ 46, 120]],
              [[ 42, 116]]],
              dtype=np.int32
       ),
       np.array([[[ 27,  85]],
              [[ 26,  86]],
              [[ 23,  86]],
              [[ 21,  88]],
              [[ 21,  91]],
              [[ 22,  92]],
              [[ 22,  93]],
              [[ 23,  94]],
              [[ 23,  96]],
              [[ 24,  97]],
              [[ 24, 100]],
              [[ 25, 101]],
              [[ 25, 102]],
              [[ 24, 103]],
              [[ 24, 106]],
              [[ 23, 107]],
              [[ 23, 108]],
              [[ 22, 109]],
              [[ 22, 110]],
              [[ 21, 111]],
              [[ 21, 115]],
              [[ 22, 116]],
              [[ 33, 116]],
              [[ 34, 115]],
              [[ 37, 115]],
              [[ 38, 114]],
              [[ 41, 114]],
              [[ 42, 113]],
              [[ 43, 113]],
              [[ 42, 112]],
              [[ 42, 111]],
              [[ 43, 110]],
              [[ 42, 109]],
              [[ 41, 109]],
              [[ 36, 104]],
              [[ 36, 103]],
              [[ 34, 101]],
              [[ 34,  98]],
              [[ 33,  97]],
              [[ 33,  94]],
              [[ 34,  93]],
              [[ 34,  87]],
              [[ 33,  86]],
              [[ 30,  86]],
              [[ 29,  85]]], 
              dtype=np.int32
       ),
       np.array( [[[ 98,  35]],
              [[ 97,  36]],
              [[ 96,  36]],
              [[ 94,  38]],
              [[ 94,  39]],
              [[ 93,  40]],
              [[ 93,  42]],
              [[ 94,  43]],
              [[ 94,  44]],
              [[ 95,  45]],
              [[ 95,  47]],
              [[ 96,  48]],
              [[ 96,  50]],
              [[ 95,  51]],
              [[ 95,  54]],
              [[ 94,  55]],
              [[ 94,  56]],
              [[ 93,  57]],
              [[ 93,  58]],
              [[ 88,  63]],
              [[ 88,  67]],
              [[ 90,  67]],
              [[ 91,  68]],
              [[ 98,  68]],
              [[ 99,  69]],
              [[104,  69]],
              [[105,  68]],
              [[111,  68]],
              [[112,  67]],
              [[115,  67]],
              [[115,  63]],
              [[110,  58]],
              [[110,  57]],
              [[109,  56]],
              [[109,  55]],
              [[108,  54]],
              [[108,  50]],
              [[109,  49]],
              [[109,  46]],
              [[110,  45]],
              [[110,  44]],
              [[111,  43]],
              [[111,  41]],
              [[110,  40]],
              [[110,  37]],
              [[109,  37]],
              [[108,  36]],
              [[106,  36]],
              [[105,  35]]], 
              dtype=np.int32
       )
    ]
  expected_contour = ExpectedContourMedialPhalanx(1)
  show_contours_position_restrictions(
    contours,
    1,
    expected_contour,
    padding=5,
    title='Proximal phalanx with jump.',
    minimize_image=False
  )


  plt.show()