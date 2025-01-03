'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development
'''

import pytest

import numpy as np

from src.contour_operations.join_contour import JoinContour

class TestJoinOperation:

  @pytest.fixture(scope='class')
  def contours(self):
    '''Two squares next to each other'''
    yield [
      np.array([[4, 4], [4, 8], [8, 8], [8, 4]]),
      np.array([[16, 4], [16, 8], [20, 8], [20, 4]])
    ]

  def test_join_with_empty(self):
    contours = [
      np.array([[4, 4], [4, 8], [8, 8], [8, 4]]),
      np.array([])
    ]
    joinOperation = JoinContour(0, 1)
    contours = joinOperation.generate_new_contour(contours)
    assert len(contours) == 1

    expected_contour = np.array([[4, 4], [4, 8], [8, 8], [8, 4]], np.int64)
    assert np.array_equal(contours[0], expected_contour)
    
  def test_join_with_one_point(self):
    contours = [
      np.array([[4, 4], [4, 8], [8, 8], [8, 4]]),
      np.array([[16, 4]])
    ]
    joinOperation = JoinContour(0, 1)
    contours = joinOperation.generate_new_contour(contours)
    assert len(contours[1]) == 0

    expected_contour = np.array(
      [[4, 4], [4, 8], [8, 8], [8, 4], [16, 4], [16, 4], [8, 4]],
      np.int64
    )
    assert np.array_equal(contours[0], expected_contour)

 
  def test_join_with_two_points(self):
    contours = [
      np.array([[4, 4], [4, 8], [8, 8], [8, 4]]),
      np.array([[16, 4], [20, 4]])
    ]
    joinOperation = JoinContour(0, 1)
    contours = joinOperation.generate_new_contour(contours)
    assert len(contours[1]) == 0

    expected_contour = np.array(
      [[4, 4], [4, 8], [8, 8], [8, 4], [16, 4], [20, 4], [16, 4], [8, 4]],
      np.int64
    )
    assert np.array_equal(contours[0], expected_contour)

  
  def test_join_two_squares(self, contours):
    joinOperation = JoinContour(0, 1)
    contours = joinOperation.generate_new_contour(contours)
    assert len(contours[1]) == 0

    expected_contour = np.array([
      [4, 4],
      [4, 8],
      [8, 8],
      [16, 8],
      [20, 8],
      [20, 4],
      [16, 4],
      [16, 8],
      [8, 8],
      [8, 4]
    ], np.int64)
    assert np.array_equal(contours[0], expected_contour)

  def test_join_weird_shapes(self):
    contours = [
      np.array([[7, 7], [8, 8], [6, 8]]),
      np.array([[10, 10], [8, 10], [6, 10], [8, 12]])
    ]
    joinOperation = JoinContour(0, 1)
    contours = joinOperation.generate_new_contour(contours)
    assert len(contours[1]) == 0

    expected_contour = np.array([
      [7, 7],
      [8, 8],
      [8, 10],
      [6, 10],
      [8, 12],
      [10, 10],
      [8, 10],
      [8, 8],
      [6, 8]
    ], np.int64)
    assert np.array_equal(contours[0], expected_contour)