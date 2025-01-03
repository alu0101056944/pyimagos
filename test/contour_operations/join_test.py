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

  # TODO creo que todavia se pueden crear cruces pero porque intersecciona
  # al propio contorno en vez de al puente en si.
  def test_join_two_squares(self, contours):
    # joinOperation = JoinContour(0, 1, 1)
    # contours = joinOperation.generate_new_contour(contours)
    # assert len(contours[1]) == 0

    # expected_contour = np.array([
    #   [4, 4], [4, 8], [16, 4], [20, 4], [20, 8],  [16, 8], [8, 8], [8, 4]
    # ], np.int64)
    # assert np.array_equal(contours[0], expected_contour)
    pass

  def test_join_both_neighbors_positive(self):
    contours = [
      np.array([[7, 7], [8, 8], [6, 8]]),
      np.array([[10, 10], [8, 10], [6, 10], [8, 12]])
    ]
    joinOperation = JoinContour(0, 1, 1)
    contours = joinOperation.generate_new_contour(contours)
    assert len(contours[1]) == 0

    expected_contour = np.array([
      [10, 10], [8, 10], [8, 8], [7, 7], [6, 8], [6, 10], [8, 12]
    ], np.int64)
    assert np.array_equal(contours[0], expected_contour)