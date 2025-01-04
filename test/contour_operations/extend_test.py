'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development
'''

import pytest

import numpy as np

from src.contour_operations.extend_contour import ExtendContour

class TestExtendOperation:

  @pytest.fixture(scope='class')
  def contours(self):
    '''Two squares next to each other'''
    yield [
      np.array([[4, 12], [4, 16], [8, 16], [8, 12]]),
      np.array([
        [16, 4],
        [16, 8],
        [16, 12],
        [16, 16],
        [16, 20],
        [16, 24],
        [20, 24],
        [20, 20],
        [20, 16],
        [20, 12],
        [20, 8],
        [20, 4],
      ])
    ]

  def test_extend_into_rectangle(self, contours):
    invasion_count = 1
    extendOperation = ExtendContour(0, 1, 30, 30, invasion_count)
    contours = extendOperation.generate_new_contour(contours)

    expected_contour_a = np.array(
      [
        [ 4, 12],
        [ 4, 16],
        [ 8, 16],
        [16, 16],
        [16, 20],
        [20, 20],
        [20, 16],
        [20, 12],
        [16, 12],
        [16, 16],
        [ 8, 16],
        [ 8, 12]
      ],
      dtype=np.int64
    )
    expected_contour_b = np.array(
      [
        [16,  4],
        [16,  8],
        [16, 24],
        [20, 24],
        [20,  8],
        [20,  4]
      ],
      dtype=np.int64
    )

    assert np.array_equal(contours[0], expected_contour_a)
    assert np.array_equal(contours[1], expected_contour_b)

  def test_extend_into_cero_points(self):
    contours = [
      np.array([[4, 12], [4, 16], [8, 16], [8, 12]]),
      np.array([])
    ]

    invasion_count = 1
    extendOperation = ExtendContour(0, 1, 30, 30, invasion_count)
    contours = extendOperation.generate_new_contour(contours)

    expected_array_a = np.array(
      [
        [4, 12], [4, 16], [8, 16], [8, 12]
      ],
      dtype=np.int64
    )

    assert np.array_equal(contours[0], expected_array_a)
    assert len(contours) == 1
  
  def test_extend_into_one_point(self):
    contours = [
      np.array([[4, 12], [4, 16], [8, 16], [8, 12]]),
      np.array([[16, 4]])
    ]

    invasion_count = 1
    extendOperation = ExtendContour(0, 1, 30, 30, invasion_count)
    contours = extendOperation.generate_new_contour(contours)

    expected_array_a = np.array(
      [
        [4, 12], [4, 16], [8, 16], [8, 12], [16, 4], [16, 4], [8, 12]
      ],
      dtype=np.int64
    )

    assert np.array_equal(contours[0], expected_array_a)
    assert len(contours) == 1

  def test_extend_into_two_points(self):
    contours = [
      np.array([[4, 12], [4, 16], [8, 16], [8, 12]]),
      np.array([
        [16, 4],
        [16, 8],
      ])
    ]

    invasion_count = 1
    extendOperation = ExtendContour(0, 1, 30, 30, invasion_count)
    contours = extendOperation.generate_new_contour(contours)

    expected_array_a = np.array(
      [
        [4, 12], [4, 16], [8, 16], [8, 12], [16, 8], [16, 4], [16, 8], [8, 12]
      ],
      dtype=np.int64
    )

    assert np.array_equal(contours[0], expected_array_a)
    assert len(contours) == 1
