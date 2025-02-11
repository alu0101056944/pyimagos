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
from src.expected_contours.medial_phalanx import ExpectedContourMedialPhalanx

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
  def ideal_distal_phalanx_contour(self):
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

  def test_single_ideal_distal_phalanx(self, ideal_distal_phalanx_contour):
    expected_contours = [ExpectedContourDistalPhalanx(1)]
    complete_contours = search_complete_contours(
      [ideal_distal_phalanx_contour],
      expected_contours,
      5,
      41,
      97,
    )
    assert len(complete_contours) == 1
    found_contours = complete_contours[0]['contours_committed']
    total_score = complete_contours[0]['committed_total_value']
    assert len(found_contours) == 1
    assert np.array_equal(found_contours[0], ideal_distal_phalanx_contour)
    EPSILON = 1e-8
    assert np.absolute(total_score - 5.75452274621356e-09) < EPSILON

  def test_invalid_image_width(self, ideal_distal_phalanx_contour):
    expected_contours = [ExpectedContourDistalPhalanx(1)]
    with pytest.raises(ValueError) as excinfo:
      search_complete_contours(
        [ideal_distal_phalanx_contour],
        expected_contours,
        5,
        21,
        97,
      )
    assert "Image width is not enough to cover the whole contour." \
      in str(excinfo.value)
  
  def test_invalid_image_height(self, ideal_distal_phalanx_contour):
    expected_contours = [ExpectedContourDistalPhalanx(1)]
    with pytest.raises(ValueError) as excinfo:
      search_complete_contours(
        [ideal_distal_phalanx_contour],
        expected_contours,
        5,
        22,
        30,
      )
    assert "Image height is not enough to cover the whole contour." \
      in str(excinfo.value)

  def test_ideal_distal_phalanx_and_medial_phalanx(self):
    expected_contours = [
      ExpectedContourDistalPhalanx(1),
      ExpectedContourMedialPhalanx(1),
    ]
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
    complete_contours = search_complete_contours(
      [ideal_distal_phalanx, ideal_medial_phalanx],
      expected_contours,
      20,
      60,
      140,
    )
    assert len(complete_contours) == 1
    found_contours = complete_contours[0]['contours_committed']
    total_score = complete_contours[0]['committed_total_value']
    assert len(found_contours) == 2
    assert (np.array_equal(found_contours[0], ideal_distal_phalanx) or
           np.array_equal(found_contours[1], ideal_distal_phalanx))
    assert (np.array_equal(found_contours[0], ideal_medial_phalanx) or
           np.array_equal(found_contours[1], ideal_medial_phalanx))
    EPSILON = 4.8549565553327885
    assert np.absolute(total_score - 4.854856561087312) < EPSILON

  def test_timeout(self):
    expected_contours = [
      ExpectedContourDistalPhalanx(1),
      ExpectedContourMedialPhalanx(1),
    ]
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
    complete_contours = search_complete_contours(
      [ideal_distal_phalanx, ideal_medial_phalanx],
      expected_contours,
      0,
      60,
      140,
    )
    assert len(complete_contours) == 0

  def test_ideal_finger_jump(self):
    expected_contours = [
      ExpectedContourDistalPhalanx(1),
    ]
    expected_contours.append(
      ExpectedContourMedialPhalanx(
        1,
        ends_branchs_sequence=True,
        first_in_branch=expected_contours[0]
      )
    )
    expected_contours.append(
      ExpectedContourDistalPhalanx(
        2,
        first_encounter=expected_contours[0]
      )
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
    ideal_distal_phalanx_second_finger = np.array(
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
    complete_contours = search_complete_contours(
      [
        ideal_distal_phalanx,
        ideal_medial_phalanx,
        ideal_distal_phalanx_second_finger
      ],
      expected_contours,
      20,
      120,
      140,
    )
    assert len(complete_contours) == 1
    found_contours = complete_contours[0]['contours_committed']
    total_score = complete_contours[0]['committed_total_value']
    assert len(found_contours) == 3
    assert (np.array_equal(found_contours[0], ideal_distal_phalanx) or
           np.array_equal(found_contours[1], ideal_distal_phalanx) or
           np.array_equal(found_contours[2], ideal_distal_phalanx))
    assert (np.array_equal(found_contours[0], ideal_medial_phalanx) or
           np.array_equal(found_contours[1], ideal_medial_phalanx) or 
           np.array_equal(found_contours[2], ideal_medial_phalanx))
    assert (
    np.array_equal(
      found_contours[0],
      ideal_distal_phalanx_second_finger
    ) or
    np.array_equal(
      found_contours[1],
      ideal_distal_phalanx_second_finger
    ) or
    np.array_equal(
      found_contours[2],
      ideal_distal_phalanx_second_finger
    ))
    EPSILON = 4.8549565553327885
    assert np.absolute(total_score - 4.854856561087312) < EPSILON
