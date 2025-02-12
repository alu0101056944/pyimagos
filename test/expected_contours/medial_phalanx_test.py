'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development
'''

import pytest

import numpy as np

from src.expected_contours.medial_phalanx import ExpectedContourMedialPhalanx

class TestMedialPhalanxExpectedContour:

  @pytest.fixture(scope='class')
  def medial_phalanx_contour(self):
    yield np.array(
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

  def test_empty_contour(self):
    phalanx = ExpectedContourMedialPhalanx(1)
    phalanx.prepare([], 66, 151)
    shape_score = phalanx.shape_restrictions()
    assert shape_score == float('inf')

  def test_ideal_shape_accepted(self, medial_phalanx_contour):
    phalanx = ExpectedContourMedialPhalanx(1)
    phalanx.prepare(medial_phalanx_contour, 66, 151)
    shape_score = phalanx.shape_restrictions()
    assert shape_score != float('inf')

  def test_shape_under_80_area(self):
    under_80_medial_phalanx = np.array(
      [[[ 0,  0]],
      [[ 0, 12]],
      [[ 5, 12]],
      [[ 7, 10]],
      [[ 6,  9]],
      [[ 6,  4]],
      [[ 5,  3]],
      [[ 6,  2]],
      [[ 6,  1]],
      [[ 7,  0]],
      [[ 6,  0]],
      [[ 5,  1]],
      [[ 2,  1]],
      [[ 1,  0]]],
      dtype=np.int32
    )
    phalanx = ExpectedContourMedialPhalanx(1)
    phalanx.prepare(under_80_medial_phalanx, 40, 100)
    shape_score = phalanx.shape_restrictions()
    assert shape_score == float('inf')

  def test_bad_aspect_ratio(self):
    bad_aspect_ratio = np.array(
      [[[0, 0]],
      [[0, 7]],
      [[5, 7]],
      [[6, 6]],
      [[7, 6]],
      [[6, 5]],
      [[6, 4]],
      [[5, 3]],
      [[6, 2]],
      [[6, 1]],
      [[7, 0]],
      [[6, 0]],
      [[5, 1]],
      [[2, 1]],
      [[1, 0]]],
      dtype=np.int32
    )
    phalanx = ExpectedContourMedialPhalanx(1)
    phalanx.prepare(bad_aspect_ratio, 66, 151)
    shape_score = phalanx.shape_restrictions()
    assert shape_score == float('inf')

  def test_second_occurence_aspect_ratio_tolerance_fault(
    self,
    medial_phalanx_contour
  ):
    distal_1 = ExpectedContourMedialPhalanx(1)
    distal_1.prepare(medial_phalanx_contour, 66, 151)
    distal_2 = ExpectedContourMedialPhalanx(2, distal_1)
    larger_aspect_contour = np.array(
      [[[ 32,  80]],
      [[ 31,  81]],
      [[ 29,  81]],
      [[ 28,  82]],
      [[ 27,  82]],
      [[ 26,  83]],
      [[ 22,  83]],
      [[ 21,  82]],
      [[ 17,  82]],
      [[ 15,  84]],
      [[ 15,  89]],
      [[ 17,  91]],
      [[ 17,  92]],
      [[ 19,  94]],
      [[ 19,  96]],
      [[ 20,  97]],
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
      [[ 42, 104]],
      [[ 41, 103]],
      [[ 41, 100]],
      [[ 40,  99]],
      [[ 40,  97]],
      [[ 39,  96]],
      [[ 39,  95]],
      [[ 37,  93]],
      [[ 37,  88]],
      [[ 38,  87]],
      [[ 38,  85]],
      [[ 39,  84]],
      [[ 35,  80]]],
      dtype=np.int32
    )
    distal_2.prepare(larger_aspect_contour, 60, 140)
    score = distal_2.shape_restrictions()
    assert score == float('inf')

  def test_solidity_too_high(self):
    high_solidity = np.array(
      [[[ 22,  99]],
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
      [[ 41, 114]],
      [[ 40, 113]],
      [[ 37, 113]],
      [[ 36, 112]],
      [[ 35, 112]],
      [[ 34, 111]],
      [[ 34, 109]],
      [[ 33, 108]],
      [[ 33, 103]],
      [[ 32, 102]],
      [[ 32,  99]],
      [[ 31, 100]],
      [[ 27, 100]],
      [[ 26,  99]]],
      dtype=np.int32
    )
    phalanx = ExpectedContourMedialPhalanx(1)
    phalanx.prepare(high_solidity, 60, 140)
    shape_value = phalanx.shape_restrictions()
    assert shape_value == float('inf')

  def test_too_many_convexity_defects(self):
    over_convex_defects = np.array(
      [[[ 22,  99]],
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
      [[ 52, 125]],
      [[ 43, 125]],
      [[ 42, 124]],
      [[ 43, 123]],
      [[ 43, 122]],
      [[ 45, 120]],
      [[ 45, 119]],
      [[ 46, 118]],
      [[ 45, 117]],
      [[ 45, 116]],
      [[ 43, 114]],
      [[ 41, 114]],
      [[ 40, 113]],
      [[ 37, 113]],
      [[ 36, 112]],
      [[ 35, 112]],
      [[ 34, 111]],
      [[ 34, 109]],
      [[ 33, 108]],
      [[ 33, 103]],
      [[ 32, 102]],
      [[ 32,  99]],
      [[ 31, 100]],
      [[ 27, 100]],
      [[ 26,  99]]],
      dtype=np.int32
    )
    phalanx = ExpectedContourMedialPhalanx(1)
    phalanx.prepare(over_convex_defects, 60, 140)
    shape_value = phalanx.shape_restrictions()
    assert shape_value == float('inf')

  def test_too_few_convexity_defects(self):
    under_convex_defects = np.array(
      [[[ 36,  97]],
      [[ 35,  98]],
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
      [[ 42, 111]],
      [[ 41, 110]],
      [[ 41, 108]],
      [[ 40, 107]],
      [[ 40, 106]],
      [[ 39, 105]],
      [[ 39, 104]],
      [[ 38, 103]],
      [[ 38, 101]],
      [[ 37, 100]],
      [[ 37,  99]],
      [[ 36,  98]]],
      dtype=np.int32
    )
    phalanx = ExpectedContourMedialPhalanx(1)
    phalanx.prepare(under_convex_defects, 60, 140)
    shape_value = phalanx.shape_restrictions()
    assert shape_value == float('inf')

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

    phalanx = ExpectedContourMedialPhalanx(1)
    phalanx.prepare(self_intercepting_contour, 32, 97)
    score = phalanx.shape_restrictions()
    assert score == float('inf')
