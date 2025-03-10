'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Shape restrictions to apply to a contour and position restrictions
to apply to other contours for the sesamoid near the metacarpal.

This is meant to be used on the second search after the first search
of the fingers, on another filter level of the radiography that actually
includes the sesamoid. So the end estimate calculation is a double search,
with the second search being the one that segments the sesamoid near the
fifth metacarpal of the left hand.
'''

from src.expected_contours.expected_contour import (
  AllowedLineSideBasedOnYorXOnVertical
)
from src.expected_contours.metacarpal import (
  ExpectedContourMetacarpal
)
from constants import POSITION_FACTORS

class ExpectedContourSesamoidMetacarpal(ExpectedContourMetacarpal):

  def __init__(self):
    # For the specific use case of only expecting the metacarpal and the
    # sesamoid on the second search on the specific region of interest
    # (a clipped image focused on the fifth metacarpal)
    super().__init__(
      encounter_amount=5,
      first_encounter=None,
      first_in_branch=None,
      ends_branchs_sequence=False,
    )

  def prepare(self, contour: list, image_width: int, image_height: int) -> None:
    super().prepare(contour, image_width, image_height)
    # So that there is no fail on aspect_ratio tolerance relative to first
    # encounter
    class AnonymousFirstEncounter:
      pass
    self.first_encounter = AnonymousFirstEncounter()
    self.first_encounter._aspect_ratio = self._aspect_ratio

  def next_contour_restrictions(self) -> list:
    width = self.min_area_rect[1][0]
    height = self.min_area_rect[1][1]
    return [
      [
        self._add_factors_from_start_point(
          self.top_left_corner,
          restriction_index=0,
          direction_right=True,
          width=width,
          height=height,
        ),
        self._add_factors_from_start_point(
          self.bottom_left_corner,
          restriction_index=0,
          direction_right=True,
          width=width,
          height=height,
        ),
        [
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL, # m = +1
          AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL, # m = -1
          AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL, # m = 0
          AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL, # vertical
        ]
      ],
      [
        self._add_factors_from_start_point(
          self.top_left_corner,
          restriction_index=1,
          direction_right=True,
          width=width,
          height=height,
        ),
        self._add_factors_from_start_point(
          self.bottom_left_corner,
          restriction_index=1,
          direction_right=True,
          width=width,
          height=height,
        ),
        [
          AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL, # m = +1
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL, # m = -1
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL, # m = 0
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL, # vertical
        ]
      ],
      [
        self._add_factors_from_start_point(
          self.bottom_left_corner,
          restriction_index=2,
          direction_right=False,
          width=width,
          height=height,
        ),
        self._add_factors_from_start_point(
          self.bottom_right_corner,
          restriction_index=2,
          direction_right=False,
          width=width,
          height=height,
        ),
        [
          AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL, # m = +1
          AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL, # m = -1
          AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL, # m = 0
          AllowedLineSideBasedOnYorXOnVertical.LOWER_EQUAL, # vertical
        ]
      ],
      [
        self._add_factors_from_start_point(
          self.top_left_corner,
          restriction_index=3,
          direction_right=False,
          width=width,
          height=height,
        ),
        self._add_factors_from_start_point(
          self.top_right_corner,
          restriction_index=3,
          direction_right=False,
          width=width,
          height=height,
        ),
        [
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL, # m = +1
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL, # m = -1
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL, # m = 0
          AllowedLineSideBasedOnYorXOnVertical.GREATER_EQUAL, # vertical
        ]
      ],
    ]

  def shape_restrictions(self, criteria: dict = None) -> list:
    return super().shape_restrictions(criteria)

  def branch_start_position_restrictions(self) -> list:
    return super().branch_start_position_restrictions()


  def _add_factors_from_start_point(self, start_point: list,
                                    restriction_index: int,
                                    direction_right: bool,
                                    width: int,
                                    height: int,
                                    next_or_jump: str = 'next',
                                    encounter_n_or_default = 'default'):
    '''Applies the formula for using the POSITION_RESTRICTIONS_PADDING at
    constant.py. The goal is to define the actual values from that file.'''
    position_factors_array = (
      POSITION_FACTORS['metacarpal_sesamoid'][next_or_jump][encounter_n_or_default]
    )
    multiplier_factors = position_factors_array[restriction_index]['multiplier']
    additive_factor = position_factors_array[restriction_index]['additive']
    if direction_right:
      return start_point + (
          self.direction_right * width * multiplier_factors['width']
        ) + (
          self.direction_right * height * multiplier_factors['height']
        ) + (
          self.direction_right * multiplier_factors['constant']
        ) + (
          self.direction_right * additive_factor
        )
    else: # direction bottom
      return start_point + (
          self.direction_bottom * width * multiplier_factors['width']
        ) + (
          self.direction_bottom * height * multiplier_factors['height']
        ) + (
          self.direction_bottom * multiplier_factors['constant']
        ) + (
          self.direction_bottom * additive_factor
        )

  def measure(self) -> dict:
    return super().measure()
