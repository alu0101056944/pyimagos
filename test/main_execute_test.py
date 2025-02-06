'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development
'''

import pytest

import numpy as np

from src.main_execute import search_complete_contours
from src.expected_contours.distal_phalanx import ExpectedContourDistalPhalanx

class TestMainExecute:

  def test_empty_expected_contours(self):
    contours = [
      np.array(
        [[[5, 5]],
        [[10, 5]],
        [[10, 10]],
        [[5, 10]]],
        dtype=np.int32
      ),
    ]
    expected_contours = []
    complete_contours = search_complete_contours(
      contours,
      expected_contours,
      5,
      50,
      50,
    )
    assert len(complete_contours) == 0

  def test_empty_contours(self):
    contours = []
    expected_contours = [ExpectedContourDistalPhalanx(1)]
    complete_contours = search_complete_contours(
      contours,
      expected_contours,
      5,
      50,
      50,
    )
    assert len(complete_contours) == 0

  def test_horizontal_line_contour(self):
    contours = [
      np.array(
        [[[2, 2]],
         [[4, 2]]],
         dtype=np.int32
      )
    ]
    expected_contours = [ExpectedContourDistalPhalanx(1)]
    complete_contours = search_complete_contours(
      contours,
      expected_contours,
      5,
      41,
      97,
    )
    assert len(complete_contours) == 0

  def test_vertical_line_contour(self):
    contours = [
      np.array(
        [[[2, 2]],
         [[2, 4]]],
         dtype=np.int32
      )
    ]
    expected_contours = [ExpectedContourDistalPhalanx(1)]
    complete_contours = search_complete_contours(
      contours,
      expected_contours,
      5,
      41,
      97,
    )
    assert len(complete_contours) == 0

  @pytest.fixture(scope='class')
  def ideal_distal_phalanx_contours(self):
    yield [
      np.array(
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
      ),
    ]


  def test_single_ideal_distal_phalanx(self, ideal_distal_phalanx_contours):
    contours = [
      ideal_distal_phalanx_contours
    ]
    expected_contours = [ExpectedContourDistalPhalanx(1)]
    complete_contours = search_complete_contours(
      contours,
      expected_contours,
      5,
      41,
      97,
    )
    assert len(complete_contours) == 1
    found_contours, total_score = complete_contours[0]
    assert len(found_contours) == 1
    assert np.array_equal(found_contours[0], contours[0])
    EPSILON = 1e-10
    assert np.absolute(total_score - 5.75452274621356e-09) < EPSILON

  def test_invalid_image_width(self, ideal_distal_phalanx_contours):
    contours = [
      ideal_distal_phalanx_contours
    ]
    expected_contours = [ExpectedContourDistalPhalanx(1)]
    with pytest.raises(ValueError) as excinfo:
      search_complete_contours(
        contours,
        expected_contours,
        5,
        21,
        97,
      )
    assert "Image width is not enough to cover the whole contour." \
      in str(excinfo.value)
  
  def test_invalid_image_height(self, ideal_distal_phalanx_contours):
    contours = [
      ideal_distal_phalanx_contours
    ]
    expected_contours = [ExpectedContourDistalPhalanx(1)]
    with pytest.raises(ValueError) as excinfo:
      search_complete_contours(
        contours,
        expected_contours,
        5,
        22,
        30,
      )
    assert "Image height is not enough to cover the whole contour." \
      in str(excinfo.value)
