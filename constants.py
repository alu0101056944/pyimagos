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
    'area': 8,
    'aspect_ratio_min': 1.3999999999999986,
    'aspect_ratio_max': 2.700009999999999,
    'aspect_ratio_tolerance': 0.65,
    'solidity': 1.56,
    'defect_area_ratio': 0.030000000000000096,
    'positional_penalization': 0,
  },
  'medial': {
    'area': 8,
    'aspect_ratio_min': 1.80001,
    'aspect_ratio_max': 2.7000099999999985,
    'aspect_ratio_tolerance': 0.6,
    'solidity': 1.25,
    'defect_area_ratio': 0.019999999999999296,
    'positional_penalization': 0,
  },
  'proximal': {
    'area': 8,
    'aspect_ratio_min': 1.8999999999999981,
    'aspect_ratio_max': 3.3999999999999986,
    'aspect_ratio_tolerance': 0.8,
    'solidity': 1.3,
    'defect_area_ratio': 0.030000000000000006,
    'positional_penalization': 0,
  },
  'metacarpal': {
    'area': 8,
    'aspect_ratio_min': 2.499999999999997,
    'aspect_ratio_max': 9.699999999999994,
    'aspect_ratio_tolerance': 1.9999999999999991,
    'solidity': 1.3,
    'defect_area_ratio': 0.01,
    'positional_penalization': 0,
  },
  'radius': {
    'area': -140.0,
    'aspect_ratio_min': 1.0,
    'aspect_ratio_max': 5.3,
    'solidity': 1.4500000000000002,
    'defect_area_ratio': 0.03499999999999869,
    'positional_penalization': 0,
  },
  'ulna': {
    'area': 220,
    'aspect_ratio_min': 1.3999999999999988,
    'aspect_ratio_max': 6.399999999999998,
    'solidity': 1.75,
    'defect_area_ratio': 0.01299999999999912,
    'positional_penalization': 0,
  },
  'sesamoid': {
    'solidity': 1.4,
    'positional_penalization': 0,
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
