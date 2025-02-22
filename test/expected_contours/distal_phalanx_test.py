'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development
'''

import pytest

import numpy as np
import cv2 as cv

from src.expected_contours.distal_phalanx import ExpectedContourDistalPhalanx
from src.main_execute import is_in_allowed_space

class TestDistalPhalanxExpectedContour:

  @pytest.fixture(scope='class')
  def distal_phalanx_contour(self):
    yield np.array(
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

  def test_empty_contour(self):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare([], 66, 151)
    shape_score = phalanx.shape_restrictions()
    assert shape_score == float('inf')

  def test_ideal_shape_accepted(self, distal_phalanx_contour):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    shape_score = phalanx.shape_restrictions()
    assert shape_score != float('inf')

  def test_shape_under_10_area(self):
    under_10_distal_phalanx = np.array(
      [[[0, 0]],
      [[0, 2]],
      [[1, 3]],
      [[0, 4]],
      [[3, 4]],
      [[3, 3]],
      [[2, 2]],
      [[3, 1]],
      [[3, 0]]],
      dtype=np.int32
    )
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(under_10_distal_phalanx, 40, 100)
    shape_score = phalanx.shape_restrictions()
    assert shape_score == float('inf')

  def test_bad_aspect_ratio(self):
    bad_aspect_ratio = np.array(
      [[[ 1,  3]],
      [[ 3,  5]],
      [[ 4,  5]],
      [[ 6,  7]],
      [[ 5,  8]],
      [[ 4,  8]],
      [[ 3,  9]],
      [[ 2,  9]],
      [[ 1, 10]],
      [[ 1, 14]],
      [[10, 14]],
      [[11, 13]],
      [[12, 13]],
      [[13, 12]],
      [[ 7,  6]],
      [[ 8,  5]],
      [[ 9,  5]],
      [[11,  3]],
      [[12,  3]]],
      dtype=np.int32
    )
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(bad_aspect_ratio, 66, 151)
    shape_score = phalanx.shape_restrictions()
    assert shape_score == float('inf')

  def test_second_occurence_aspect_ratio_tolerance_fault(
    self,
    distal_phalanx_contour
  ):
    distal_1 = ExpectedContourDistalPhalanx(1)
    distal_1.prepare(distal_phalanx_contour, 66, 151)
    distal_2 = ExpectedContourDistalPhalanx(2, distal_1)
    larger_aspect_contour = np.array(
      [[[ 41,  88]],
      [[ 40,  89]],
      [[ 37,  89]],
      [[ 35,  91]],
      [[ 35,  94]],
      [[ 36,  95]],
      [[ 36,  96]],
      [[ 37,  97]],
      [[ 37,  99]],
      [[ 38, 100]],
      [[ 38, 102]],
      [[ 39, 103]],
      [[ 39, 106]],
      [[ 40, 107]],
      [[ 40, 140]],
      [[ 39, 141]],
      [[ 39, 145]],
      [[ 40, 146]],
      [[ 51, 146]],
      [[ 52, 145]],
      [[ 55, 145]],
      [[ 56, 144]],
      [[ 59, 144]],
      [[ 60, 143]],
      [[ 61, 143]],
      [[ 60, 142]],
      [[ 60, 141]],
      [[ 61, 140]],
      [[ 60, 139]],
      [[ 59, 139]],
      [[ 54, 134]],
      [[ 54, 133]],
      [[ 52, 131]],
      [[ 52, 128]],
      [[ 51, 127]],
      [[ 51, 123]],
      [[ 50, 122]],
      [[ 50, 117]],
      [[ 49, 116]],
      [[ 49, 112]],
      [[ 48, 111]],
      [[ 48, 103]],
      [[ 47, 102]],
      [[ 47,  97]],
      [[ 48,  96]],
      [[ 48,  90]],
      [[ 47,  89]],
      [[ 44,  89]],
      [[ 43,  88]]],
      dtype=np.int32
    )
    distal_2.prepare(larger_aspect_contour, 61, 147)
    score = distal_2.shape_restrictions()
    assert score == float('inf')

  def test_solidity_too_high(self):
    high_solidity = np.array(
      [[[ 5,  4]],
      [[ 3,  6]],
      [[ 3,  7]],
      [[ 4,  8]],
      [[ 4, 10]],
      [[ 5, 11]],
      [[ 5, 12]],
      [[ 6, 13]],
      [[ 6, 15]],
      [[ 7, 16]],
      [[ 7, 17]],
      [[ 8, 18]],
      [[ 7, 19]],
      [[ 7, 20]],
      [[ 4, 23]],
      [[ 4, 24]],
      [[ 2, 26]],
      [[ 2, 35]],
      [[ 5, 38]],
      [[20, 38]],
      [[21, 37]],
      [[21, 36]],
      [[23, 34]],
      [[23, 33]],
      [[24, 32]],
      [[20, 28]],
      [[20, 27]],
      [[12, 19]],
      [[12, 18]],
      [[ 8, 14]],
      [[ 8,  4]]],
      dtype=np.int32
    )
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(high_solidity, 24, 38)
    shape_value = phalanx.shape_restrictions()
    assert shape_value == float('inf')

  def test_too_many_convexity_defects(self):
    over_convex_defects = np.array(
      [[[ 2,  0]],
      [[ 0,  2]],
      [[ 0,  6]],
      [[ 5, 11]],
      [[ 0, 16]],
      [[ 0, 26]],
      [[ 3, 29]],
      [[13, 29]],
      [[18, 27]],
      [[18, 25]],
      [[8, 20]],
      [[18, 16]],
      [[18, 15]],
      [[9,  8]],
      [[9,  7]],
      [[14,  4]],
      [[14,  0]]],
      dtype=np.int32
    )
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(over_convex_defects, 20, 30)
    shape_value = phalanx.shape_restrictions()
    assert shape_value == float('inf')

  def test_too_few_convexity_defects(self):
    under_convex_defects = np.array(
      [[[ 2,  0]],
      [[ 0,  2]],
      [[ 0,  6]],
      [[ 5, 11]],
      [[ 0, 16]],
      [[ 0, 26]],
      [[ 3, 29]],
      [[13, 29]],
      [[18, 24]],
      [[18,  1]],
      [[17,  0]]],
      dtype=np.int32
    )
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(under_convex_defects, 20, 30)
    shape_value = phalanx.shape_restrictions()
    assert shape_value == float('inf')

  def test_contour_fully_inside_allowed_area(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array(
      [[[14., 88.]],
       [[16., 88.]],
       [[16., 91.]],
       [[14., 91.]]]
    )
    assert is_in_allowed_space(square_contour, phalanx) == True

  def test_contour_partially_outside_allowed_area_left(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array(
      [[[-4., 89.]],
       [[ 2., 89.]],
       [[ 2., 91.]],
       [[-4., 91.]]]
    )
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_contour_fully_outside_allowed_area_left(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array(
      [[[-9., 93.]],
       [[-6., 93.]],
       [[-6., 95.]],
       [[-9., 95.]]]
    )
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_contour_fully_outside_allowed_area_right(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array(
      [[[73., 97.]],
       [[76., 97.]],
       [[76., 99.]],
       [[73., 99.]]]
    )
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_contour_partially_outside_allowed_area_right(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array(
      [[[ 67., 101.]],
       [[ 73., 101.]],
       [[ 73., 103.]],
       [[ 67., 103.]]]
    )
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_contour_fully_outside_allowed_area_top(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array(
      [[[14., 75.]],
       [[16., 75.]],
       [[16., 78.]],
       [[14., 78.]]]
    )
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_contour_partially_outside_allowed_area_top(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array(
      [[[14., 80.]],
       [[16., 80.]],
       [[16., 86.]],
       [[14., 86.]]]
    )
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_contour_fully_outside_allowed_area_bottom(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array(
      [[[ 64., 193.]],
       [[ 66., 193.]],
       [[ 66., 196.]],
       [[ 64., 196.]]]
    )
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_contour_partially_outside_allowed_area_bottom(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array(
      [[[ 64., 182.]],
       [[ 66., 182.]],
       [[ 66., 188.]],
       [[ 64., 188.]]]
    )
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_overlap_does_not_throw_error_for_convexity_defects(self):
    '''self-intercepting contours cause errors, but i expect overlap
    not to cause errors as inside-outside is non-ambiguous.'''
    contour_with_edge_overlap = np.array(
      [[ 4,  4],
      [ 4,  8],
      [ 8,  8],
      [ 8,  4],
      [16,  4],
      [20,  4],
      [16,  4],
      [ 8,  4]]
    )
    defects = cv.convexityDefects(
      contour_with_edge_overlap,
      cv.convexHull(contour_with_edge_overlap, returnPoints=False),
    )
    assert defects is not None

  def test_self_interception_contour_is_discarded(self):
    self_intercepting_contour = np.array(
      [[25, 66],
      [24, 67],
      [21, 67],
      [32, 68],
      [32, 82],
      [22, 84],
      [22, 87],
      [21, 88],
      [21, 89],
      [20, 91],
      [19, 92],
      [19, 96],
      [20, 97],
      [31, 97],
      [32, 82],
      [32, 68],
      [31, 67],
      [28, 67],
      [27, 66]],
      dtype=np.int32
    )

    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(self_intercepting_contour, 32, 97)
    score = phalanx.shape_restrictions()
    assert score == float('inf')

  def test_fifth_finger_aspect_ratio(self):
    pass
    # TODO fill fifth finger aspect ratio test
