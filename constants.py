#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Configuration constants
'''

CONTRAST_MODEL_PATH = './model/dino_deitsmall8_pretrain.pth'
SEARCH_EXECUTION_DURATION_SECONDS = 5

BONE_AGE_ATLAS = {
  '17.5': {
    'radius_length': 0.436,
    'radius_width': 1.000,
    'ulna_length': 0.252,
    'ulna_width': 0.488,
    'metacarpal_2_length': 0.158,
    'inter-sesamoid_distance': 0.000,
    'metacarpal_2_width': 0.479,
    'metacarpal_3_width': 0.487,
    'metacarpal_4_width': 0.383
  },
  '18.5': {
    'radius_length': 1.000,
    'radius_width': 1.000,
    'ulna_length': 0.552,
    'ulna_width': 0.184,
    'metacarpal_2_length': 0.325,
    'inter-sesamoid_distance': 0.000,
    'metacarpal_2_width': 0.323,
    'metacarpal_3_width': 0.364,
    'metacarpal_4_width': 0.000
  },
  '19.5': {
    'radius_length': 0.445,
    'radius_width': 1.000,
    'ulna_length': 0.287,
    'ulna_width': 0.495,
    'metacarpal_2_length': 0.187,
    'inter-sesamoid_distance': 0.000,
    'metacarpal_2_width': 0.437,
    'metacarpal_3_width': 0.497,
    'metacarpal_4_width': 0.316
  }
}

CRITERIA_DICT = {
  'distal': {
    'area': 28,
    'aspect_ratio_min': 1.3999999999999986,
    'aspect_ratio_max': 4.3100000000000005,
    'aspect_ratio_tolerance': 3.3499999999999996,
    'solidity': 2.81,
    'defect_area_ratio': 0.03
  },
  'medial': {
    'area': 9,
    'aspect_ratio_min': 1.8001000000000003,
    'aspect_ratio_max': 3.31,
    'aspect_ratio_tolerance': 3.0,
    'solidity': 2.65,
    'defect_area_ratio': 0.020009999999999323
  },
  'proximal': {
    'area': 28,
    'aspect_ratio_min': 1.8999999999999968,
    'aspect_ratio_max': 4.7,
    'aspect_ratio_tolerance': 3.1999999999999997,
    'solidity': 2.65,
    'defect_area_ratio': 0.03
  },
  'metacarpal': {
    'area': 10,
    'aspect_ratio_min': 2.499999999999995,
    'aspect_ratio_max': 17.400000000000002,
    'aspect_ratio_tolerance': 3.3,
    'solidity': 2.6999999999999997,
    'defect_area_ratio': 0.01
  },
  'radius': {
    'area': 8,
    'aspect_ratio_min': 1.0,
    'aspect_ratio_max': 6.2,
    'solidity': 2.8,
    'defect_area_ratio': 0.03499999999999865
  },
  'ulna': {
    'area': 460,
    'aspect_ratio_min': 1.399999999999998,
    'aspect_ratio_max': 10.200000000000001,
    'solidity': 2.8,
    'defect_area_ratio': 0.0
  },
  'sesamoid': {
    'solidity': 0.9
  }
}

CRITERIA_DICT_VARIATION_MAGNITUDES = {
  'distal': {
    'area': 10,
    'aspect_ratio_min': 0.1,
    'aspect_ratio_max': 5,
    'aspect_ratio_tolerance': 0.05,
    'solidity': 0.1,
    'defect_area_ratio': 0.01,
  },
  'medial': {
    'area': 10,
    'aspect_ratio_min': 0.1,
    'aspect_ratio_max': 5,
    'aspect_ratio_tolerance': 0.1,
    'solidity': 0.1,
    'defect_area_ratio': 0.01,
  },
  'proximal': {
    'area': 10,
    'aspect_ratio_min': 0.1,
    'aspect_ratio_max': 5,
    'aspect_ratio_tolerance': 0.1,
    'solidity': 0.1,
    'defect_area_ratio': 0.01,
  },
  'metacarpal': {
    'area': 10,
    'aspect_ratio_min': 0.1,
    'aspect_ratio_max': 5,
    'aspect_ratio_tolerance': 0.1,
    'solidity': 0.05,
    'defect_area_ratio': 0.01,
  },
  'radius': {
    'area': 100,
    'aspect_ratio_min': 0.1,
    'aspect_ratio_max': 5,
    'solidity': 0.1,
    'defect_area_ratio': 0.001,
  },
  'ulna': {
    'area': 100,
    'aspect_ratio_min': 0.1,
    'aspect_ratio_max': 5,
    'solidity': 0.1,
    'defect_area_ratio': 0.001,
  },
  'sesamoid': {
    'solidity': 0.1
  }
}

CPU_CONTRAST_RESIZE_TARGET = {
  'width': 640,
  'height': 480,
}

# Line position = start pos. + (
#   [
#     direction_bottom * multiplier(height)
#     | (or)
#     direction_right * width * multiplier
#   ] +
#   [
#     direction_bottom * constant
#     |
#     direction_right * constant
#   ] + additive
#   
# )
# Positive sign means towards the right or towards the bottom of the image.
# Negative sign means towards left or towards top of the image.
# Not all are used, it depends on whether it is used in the corresponding
# expected contour class.
POSITION_FACTORS = {
  'distal': {
    'next': {
      'default': [
        {
          'additive': 25,
          'multiplier': {
            'width': 0.5,
            'height': 0,
            'constant': 9.74,
          },
        },
        {
          'additive': -16,
          'multiplier': {
            'width': -0.5,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 0,
            'constant': -14,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 4,
            'constant': 0,
          },
        },
      ]
    },
    'jump': {
      'default': [
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': -2.5,
            'constant': -28.5,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 1,
            'constant': 21,
          },
        },
        {
          'additive': 12,
          'multiplier': {
            'width': 0.5,
            'height': 0,
            'constant': -13.8,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 3,
            'height': 0,
            'constant': 0,
          },
        },
      ],
      'encounter_4': [
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 0,
            'constant': 1,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 9,
            'constant': 55.67,
          },
        },
        {
          'additive': -3,
          'multiplier': {
            'width': 0.5,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 6,
            'height': 0,
            'constant': 18,
          },
        },
      ]
    }
  },
  'medial': {
    'next': {
      'default': [
        {
          'additive': 17,
          'multiplier': {
            'width': 0.5,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': -29,
          'multiplier': {
            'width': -0.5,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 0,
            'constant': -1,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 4,
            'constant': 10.024,
          },
        },
      ]
    },
    'jump': {
      'default': []
    }
  },
  'proximal': {
    'next': {
      'default': [
        {
          'additive': 20,
          'multiplier': {
            'width': 0.5,
            'height': 0,
            'constant': 5.094,
          },
        },
        {
          'additive': -24,
          'multiplier': {
            'width': -0.5,
            'height': 0,
            'constant': -2.84,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 0,
            'constant': -20,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 3,
            'constant': 53,
          },
        },
      ]
    },
    'jump': {
      'default': []
    }
  },
  'metacarpal': {
    'next': {
      'default': [
        { # Exception, does nothing.
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 0,
            'constant': 0,
          },
        },
        { # Exception, only width works and only on x coord.
          'additive': 0,
          'multiplier': {
            'width': -7,
            'height': 0,
            'constant': 0,
          },
        },
        { # Exception, only width works and only on x coord.
          'additive': 0,
          'multiplier': {
            'width': 1,
            'height': 0,
            'constant': 0,
          },
        },
      ]
    },
    'jump': {
      'default': []
    }
  },
  'ulna': {
    'next': {
      'default': [
        {
          'additive': 0,
          'multiplier': {
            'width': -0.3,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': -3,
          'multiplier': {
            'width': -0.5,
            'height': 0,
            'constant': 0,
          },
        },
      ]
    },
    'jump': {
      'default': []
    }
  },
  'metacarpal_sesamoid': {
    'next': {
      'default': [
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 0,
            'constant': 4,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': -1,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': -2/3,
            'constant': 0,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 1/8,
            'constant': -8.31,
          },
        },
      ]
    },
    'jump': {
      'default': []
    }
  }
}

PRE_EXPERIMENT_POSITION_FACTORS = {
  'distal': {
    'next': {
      'default': [
        {
          'additive': 25,
          'multiplier': {
            'width': 0.5,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': -16,
          'multiplier': {
            'width': -0.5,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 0,
            'constant': -14,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 4,
            'constant': 0,
          },
        },
      ]
    },
    'jump': {
      'default': [
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': -2.5,
            'constant': 0,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 1,
            'constant': 0,
          },
        },
        {
          'additive': 12,
          'multiplier': {
            'width': 0.5,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 3,
            'height': 0,
            'constant': 0,
          },
        },
      ],
      'encounter_4': [
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 0,
            'constant': 1,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 9,
            'constant': 0,
          },
        },
        {
          'additive': -3,
          'multiplier': {
            'width': 0.5,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 6,
            'height': 0,
            'constant': 0,
          },
        },
      ]
    }
  },
  'medial': {
    'next': {
      'default': [
        {
          'additive': 17,
          'multiplier': {
            'width': 0.5,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': -29,
          'multiplier': {
            'width': -0.5,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 4,
            'constant': 0,
          },
        },
      ]
    },
    'jump': {
      'default': []
    }
  },
  'proximal': {
    'next': {
      'default': [
        {
          'additive': 20,
          'multiplier': {
            'width': 0.5,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': -24,
          'multiplier': {
            'width': -0.5,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 0,
            'constant': -20,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 3,
            'constant': 0,
          },
        },
      ]
    },
    'jump': {
      'default': []
    }
  },
  'metacarpal': {
    'next': {
      'default': [
        { # Exception, does nothing.
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 0,
            'constant': 0,
          },
        },
        { # Exception, only width works and only on x coord.
          'additive': 0,
          'multiplier': {
            'width': -7,
            'height': 0,
            'constant': 0,
          },
        },
        { # Exception, only width works and only on x coord.
          'additive': 0,
          'multiplier': {
            'width': 1,
            'height': 0,
            'constant': 0,
          },
        },
      ]
    },
    'jump': {
      'default': []
    }
  },
  'ulna': {
    'next': {
      'default': [
        {
          'additive': 0,
          'multiplier': {
            'width': -0.3,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': -3,
          'multiplier': {
            'width': -0.5,
            'height': 0,
            'constant': 0,
          },
        },
      ]
    },
    'jump': {
      'default': []
    }
  },
  'metacarpal_sesamoid': {
    'next': {
      'default': [
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 0,
            'constant': 4,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': -1,
            'height': 0,
            'constant': 0,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': -2/3,
            'constant': 0,
          },
        },
        {
          'additive': 0,
          'multiplier': {
            'width': 0,
            'height': 1/8,
            'constant': 0,
          },
        },
      ]
    },
    'jump': {
      'default': []
    }
  }
}
