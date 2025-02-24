'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development
'''

import pytest

import numpy as np

from src.expected_contours.sesamoid import ExpectedContourSesamoid

class TestSesamoidExpectedContour:

  @pytest.fixture(scope='class')
  def sesamoid_contour(self):
    yield np.array(
      [[[294, 250]],
      [[293, 251]],
      [[293, 255]],
      [[294, 256]],
      [[299, 256]],
      [[300, 255]],
      [[301, 255]],
      [[302, 256]],
      [[304, 256]],
      [[305, 255]],
      [[305, 252]],
      [[304, 251]],
      [[303, 251]],
      [[302, 250]]],
      dtype=np.int32
    )

  def test_empty_contour(self):
    phalanx = ExpectedContourSesamoid()
    phalanx.prepare([], 66, 151)
    shape_score = phalanx.shape_restrictions()
    assert shape_score == float('inf')

  def test_ideal_shape_accepted(self, sesamoid_contour):
    phalanx = ExpectedContourSesamoid()
    phalanx.prepare(sesamoid_contour, 250, 500)
    shape_score = phalanx.shape_restrictions()
    assert shape_score != float('inf')
