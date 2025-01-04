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
    assert np.array_equal(contours[1], expected_contour_b)

  def test_cut_two_points(self):
    contours = [
      np.array([[4, 4], [4, 8]]),
    ]

    cutOperation = CutContour(0, 0, 30, 30)
    contours = cutOperation.generate_new_contour(contours)
    assert len(contours) == 2

    expected_contour_a = np.array([[4, 4]], np.int64)
    expected_contour_b = np.array([[4, 8]], np.int64)
    assert np.array_equal(contours[0], expected_contour_a)
    assert np.array_equal(contours[1], expected_contour_b)

  def test_cut_one_point(self):
    contours = [
      np.array([[4, 4]]),
    ]

    cutOperation = CutContour(0, 0, 30, 30)
    contours = cutOperation.generate_new_contour(contours)
    assert len(contours) == 1

    expected_contour_a = np.array([[4, 4]], np.int64)
    assert np.array_equal(contours[0], expected_contour_a)

  def test_cut_cero_points(self):
    contours = [
      np.array([]),
    ]

    cutOperation = CutContour(0, 0, 30, 30)
    contours = cutOperation.generate_new_contour(contours)
    assert len(contours) == 1

    expected_contour_a = np.array([], np.int64)
    assert np.array_equal(contours[0], expected_contour_a)
