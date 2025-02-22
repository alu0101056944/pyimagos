'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development
'''

import pytest

import numpy as np

from src.contour_operations.cut_contour import CutContour

class TestCutOperation:

  @pytest.fixture(scope='class')
  def contours(self):
    '''Two squares next to each other'''
    yield [
      np.array([[4, 4], [4, 8], [8, 8], [8, 4]]),
      np.array([[16, 4], [16, 8], [20, 8], [20, 4]])
    ]

  def test_cut_square(self, contours):
    cutOperation = CutContour(0, 1, 30, 30)
    contours = cutOperation.generate_new_contour(contours)
    assert len(contours) == 3

    expected_contour_a = np.array([[4, 4], [4, 8], [8, 4]], np.int64)
    expected_contour_b = np.array([[4, 8], [8, 8], [8, 4]], np.int64)
    assert np.array_equal(contours[0], expected_contour_a)
    assert np.array_equal(contours[2], expected_contour_b)

  def test_cut_two_points(self):
    contours = [
      np.array([[4, 4], [4, 8]]),
    ]

    cutOperation = CutContour(0, 0, 30, 30)
    contours = cutOperation.generate_new_contour(contours)
    assert contours is None

  def test_cut_one_point(self):
    contours = [
      np.array([[4, 4]]),
    ]

    cutOperation = CutContour(0, 0, 30, 30)
    contours = cutOperation.generate_new_contour(contours)
    assert contours is None

  def test_cut_cero_points(self):
    contours = [
      np.array([]),
    ]

    cutOperation = CutContour(0, 0, 30, 30)
    contours = cutOperation.generate_new_contour(contours)
    assert contours is None

  def test_cut_concave_contour(self):
    contours = [
      np.array([[5, 5], [10, 5], [10, 10], [7,7], [5, 10]])
    ]

    cutOperation = CutContour(0, 1, 30, 30)
    contours = cutOperation.generate_new_contour(contours)

    assert len(contours) == 2
    expected_contour_a = np.array([[5, 5], [10, 5], [ 7, 7], [5, 10]],
                                  np.int64)
    expected_contour_b = np.array([[10, 5], [10, 10], [ 7, 7]], np.int64)
    assert np.array_equal(contours[0], expected_contour_a)
    assert np.array_equal(contours[1], expected_contour_b)

    
  def test_cut_long_contour_1(self):
    contours = [
      np.array([
          [5, 5],
          [7, 8],
          [9, 8],
          [12, 9],
          [14, 8],
          [16, 7],
          [18, 7],
          [21, 6],
          [21, 5],
          [20, 4],
          [19, 4],
          [16, 4],
          [13, 3],
          [10, 3],
          [9, 2],
          [7, 4],
          [6, 5],
        ]
      )
    ]

    cutOperation = CutContour(0, 6, 30, 30)
    contours = cutOperation.generate_new_contour(contours)

    assert len(contours) == 2
    expected_contour_a = np.array(
      [[5,  5],
      [ 7,  8],
      [ 9,  8],
      [12,  9],
      [14,  8],
      [16,  7],
      [18,  7],
      [13,  3],
      [10,  3],
      [ 9,  2],
      [ 7,  4],
      [ 6,  5]],
      np.int64
    )
    expected_contour_b = np.array(
      [[18,  7],
       [21,  6],
       [21,  5],
       [20,  4],
       [19,  4],
       [16,  4],
       [13,  3]],
       np.int64
    )
    assert np.array_equal(contours[0], expected_contour_a)
    assert np.array_equal(contours[1], expected_contour_b)

  def test_cut_long_contour_2(self):
    contours = [
      np.array([
          [5, 5],
          [7, 8],
          [9, 8],
          [12, 9],
          [14, 8],
          [16, 7],
          [18, 7],
          [21, 6],
          [21, 5],
          [20, 4],
          [19, 4],
          [16, 4],
          [13, 3],
          [10, 3],
          [9, 2],
          [7, 4],
          [6, 5],
        ]
      )
    ]

    cutOperation = CutContour(0, 9, 30, 30)
    contours = cutOperation.generate_new_contour(contours)

    assert len(contours) == 2
    expected_contour_a = np.array(
      [[ 5,  5],
      [ 7,  8],
      [20,  4],
      [19,  4],
      [16,  4],
      [13,  3],
      [10,  3],
      [ 9,  2],
      [ 7,  4],
      [ 6,  5]],
      np.int64
    )
    expected_contour_b = np.array(
      [[ 7,  8],
      [ 9,  8],
      [12,  9],
      [14,  8],
      [16,  7],
      [18,  7],
      [21,  6],
      [21,  5],
      [20,  4]],
      np.int64
    )
    assert np.array_equal(contours[0], expected_contour_a)
    assert np.array_equal(contours[1], expected_contour_b)

  def test_cut_long_contour_3(self):
    contours = [
      np.array([
          [5, 5],
          [7, 8],
          [9, 8],
          [12, 9],
          [14, 8],
          [16, 7],
          [18, 7],
          [21, 6],
          [21, 5],
          [20, 4],
          [19, 4],
          [16, 4],
          [13, 3],
          [10, 3],
          [9, 2],
          [7, 4],
          [6, 5],
        ]
      )
    ]

    cutOperation = CutContour(0, 3, 30, 30)
    contours = cutOperation.generate_new_contour(contours)

    assert len(contours) == 2
    expected_contour_a = np.array(
      [[ 5,  5],
      [ 7,  8],
      [ 9,  8],
      [12,  9],
      [13,  3],
      [10,  3],
      [ 9,  2],
      [ 7,  4],
      [ 6,  5]],
      np.int64
    )
    expected_contour_b = np.array([[12,  9],
      [14,  8],
      [16,  7],
      [18,  7],
      [21,  6],
      [21,  5],
      [20,  4],
      [19,  4],
      [16,  4],
      [13,  3]],
      np.int64
    )
    assert np.array_equal(contours[0], expected_contour_a)
    assert np.array_equal(contours[1], expected_contour_b)


  def test_cut_long_contour_4(self):
    contours = [
      np.array([
          [5, 5],
          [7, 8],
          [9, 8],
          [12, 9],
          [14, 8],
          [16, 7],
          [18, 7],
          [21, 6],
          [21, 5],
          [20, 4],
          [19, 4],
          [16, 4],
          [13, 3],
          [10, 3],
          [9, 2],
          [7, 4],
          [6, 5],
        ]
      )
    ]

    cutOperation = CutContour(0, 1, 30, 30)
    contours = cutOperation.generate_new_contour(contours)

    assert len(contours) == 2
    expected_contour_a = np.array(
      [[ 5,  5],
      [ 7,  8],
      [16,  4],
      [13,  3],
      [10,  3],
      [ 9,  2],
      [ 7,  4],
      [ 6,  5]],
      np.int64
    )
    expected_contour_b = np.array(
      [[ 7,  8],
      [ 9,  8],
      [12,  9],
      [14,  8],
      [16,  7],
      [18,  7],
      [21,  6],
      [21,  5],
      [20,  4],
      [19,  4],
      [16,  4]],
      np.int64
    )
    assert np.array_equal(contours[0], expected_contour_a)
    assert np.array_equal(contours[1], expected_contour_b)

  def test_cut_complex_contour(self):
    contours = [
      np.array([[10, 10], [20, 10], [20, 20], [15, 15], [10, 20]])
    ]

    cutOperation = CutContour(0, 0, 30, 30)
    contours = cutOperation.generate_new_contour(contours)

    assert len(contours) == 2
    expected_contour_a = np.array(
      [[10, 10],
      [15, 15],
      [10, 20]],
      np.int64
    )
    expected_contour_b = np.array(
      [[10, 10],
      [20, 10],
      [20, 20],
      [15, 15]],
      np.int64
    )
    assert np.array_equal(contours[0], expected_contour_a)
    assert np.array_equal(contours[1], expected_contour_b)

  def test_cut_out_of_bounds(self):
    contours = [
      np.array([[1, 1],[1, 29],[29, 29],[29, 1]])
    ]

    cutOperation = CutContour(0, 0, 30, 30)
    contours = cutOperation.generate_new_contour(contours)

    assert len(contours) == 2
    expected_contour_a = np.array(
      [[ 1,  1],
      [29, 29],
      [29,  1]],
      np.int64
    )
    expected_contour_b = np.array(
      [[ 1,  1],
      [ 1, 29],
      [29, 29]],
      np.int64
    )
    assert np.array_equal(contours[0], expected_contour_a)
    assert np.array_equal(contours[1], expected_contour_b)