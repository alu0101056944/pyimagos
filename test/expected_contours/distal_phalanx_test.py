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
    assert shape_score[0] == False
    assert shape_score[1] == -1

  def test_ideal_shape_accepted(self, distal_phalanx_contour):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    shape_score = phalanx.shape_restrictions()
    assert shape_score[0] == True
    assert shape_score != -1

  def test_shape_under_80_area(self):
    under_80_distal_phalanx = np.array(
      [[[28,  80]],
      [[27,  80]],
      [[26,  80]],
      [[26,  80]],
      [[27,  82]],
      [[27,  83]],
      [[28,  84]],
      [[28,  85]],
      [[27,  86]],
      [[27,  88]],
      [[26,  89]],
      [[26,  90]],
      [[26,  91]],
      [[27,  92]],
      [[28,  92]],
      [[29,  92]],
      [[30,  92]],
      [[31,  92]],
      [[32,  92]],
      [[32,  92]],
      [[33,  91]],
      [[33,  91]],
      [[35,  91]],
      [[36,  90]],
      [[36,  89]],
      [[34,  88]],
      [[33,  86]],
      [[32,  85]],
      [[32,  84]],
      [[32,  83]],
      [[33,  82]],
      [[33,  80]],
      [[31,  80]],
      [[30,  79]],
      [[29,  80]]],
      dtype=np.int32
    )
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(under_80_distal_phalanx, 40, 100)
    shape_score = phalanx.shape_restrictions()
    assert shape_score[0] == False
    assert shape_score != -1

  def test_bad_aspect_ratio(self):
    bad_aspect_ratio = np.array(
      [[[84, 120]],
      [[81, 120]],
      [[78, 120]],
      [[78, 120]],
      [[81, 123]],
      [[81, 123]],
      [[84, 126]],
      [[84, 126]],
      [[81, 129]],
      [[78, 132]],
      [[78, 137]],
      [[79, 138]],
      [[84, 138]],
      [[87, 138]],
      [[90, 138]],
      [[93, 138]],
      [[94, 138]],
      [[98, 135]],
      [[93, 126]],
      [[96, 123]],
      [[97, 120]],
      [[93, 120]],
      [[87, 120]]],
      dtype=np.int32
    )
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(bad_aspect_ratio, 66, 151)
    shape_score = phalanx.shape_restrictions()
    assert shape_score[0] == False
    assert shape_score != -1

  def test_second_occurence_aspect_ratio_tolerance_fault(
    self,
    distal_phalanx_contour
  ):
    distal_1 = ExpectedContourDistalPhalanx(1)
    distal_1.prepare(distal_phalanx_contour, 66, 151)
    distal_2 = ExpectedContourDistalPhalanx(2, distal_1)
    larger_aspect_contour = np.array([
      [[25, 59]],
      [[24, 60]],
      [[21, 60]],
      [[19, 62]],
      [[19, 65]],
      [[20, 66]],
      [[20, 67]],
      [[21, 68]],
      [[21, 73]],
      [[22, 74]],
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
      [[31, 68]],
      [[32, 67]],
      [[32, 61]],
      [[31, 60]],
      [[28, 60]],
      [[27, 59]]],
      dtype=np.int32
    )
    distal_2.prepare(larger_aspect_contour, 22, 38)
    score = distal_2.shape_restrictions()
    assert score[0] == False
    assert score[1] == -1

  def test_solidity_too_high(self):
    high_solidity = np.array(
      [[[ 2,  0]],
      [[ 0,  2]],
      [[ 0,  6]],
      [[ 5, 11]],
      [[ 0, 16]],
      [[ 0, 26]],
      [[ 3, 29]],
      [[13, 29]],
      [[18, 24]],
      [[16, 22]],
      [[16, 21]],
      [[11, 16]],
      [[11, 15]],
      [[ 8, 12]],
      [[ 8,  0]]],
      dtype=np.int32
    )
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(high_solidity, 20, 30)
    shape_value = phalanx.shape_restrictions()
    assert shape_value == [False, -1]

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
    assert shape_value == [False, -1]

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
    assert shape_value == [False, -1]

  def test_contour_fully_inside_allowed_area(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array([
      [[20, 106]],
      [[23, 106]],
      [[23, 108]],
      [[20, 108]],
    ])
    assert is_in_allowed_space(square_contour, phalanx) == True

  def test_contour_partially_outside_allowed_area_left(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array([
      [[12, 106]],
      [[16, 106]],
      [[16, 108]],
      [[12, 108]],
    ])
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_contour_fully_outside_allowed_area_left(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array([
      [[8, 106]],
      [[10, 106]],
      [[10, 108]],
      [[8, 108]],
    ])
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_contour_fully_outside_allowed_area_right(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array([
      [[48, 106]],
      [[50, 106]],
      [[50, 108]],
      [[48, 108]],
    ])
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_contour_partially_outside_allowed_area_right(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array([
      [[43, 116]],
      [[48, 116]],
      [[48, 118]],
      [[43, 118]],
    ])
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_contour_fully_outside_allowed_area_top(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array([
      [[32, 88]],
      [[38, 88]],
      [[38, 92]],
      [[32, 92]],
    ])
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_contour_partially_outside_allowed_area_top(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array([
      [[32, 98]],
      [[38, 98]],
      [[38, 102]],
      [[32, 102]],
    ])
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_contour_fully_outside_allowed_area_bottom(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array([
      [[32, 192]],
      [[38, 192]],
      [[38, 198]],
      [[32, 198]],
    ])
    assert is_in_allowed_space(square_contour, phalanx) == False

  def test_contour_partially_outside_allowed_area_bottom(
    self,
    distal_phalanx_contour
  ):
    phalanx = ExpectedContourDistalPhalanx(1)
    phalanx.prepare(distal_phalanx_contour, 66, 151)
    square_contour = np.array([
      [[32, 183]],
      [[38, 183]],
      [[38, 189]],
      [[32, 189]],
    ])
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
    is_valid, score = phalanx.shape_restrictions()
    assert is_valid == False
    assert score == -1
