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
    'area': 10,
    'aspect_ratio': 1.1,
    'aspect_ratio_tolerance': 0.65,
    'solidity': 1.46,
    'defect_area_ratio': 0.08,
  },
  'medial': {
    'area': 10,
    'aspect_ratio': 1.2,
    'aspect_ratio_tolerance': 0.3,
    'solidity': 1.3,
    'defect_area_ratio': 0.06,
  },
  'proximal': {
    'area': 10,
    'aspect_ratio': 1.5,
    'aspect_ratio_tolerance': 0.5,
    'solidity': 1.3,
    'defect_area_ratio': 0.07,
  },
  'metacarpal': {
    'area': 10,
    'aspect_ratio': 2.5,
    'aspect_ratio_tolerance': 0.6,
    'solidity': 1.35,
    'defect_area_ratio': 0.06,
  },
  'radius': {
    'area': 250,
    'aspect_ratio': 1.2,
    'solidity': 1.6,
    'defect_area_ratio': 0.005,
  },
  'ulna': {
    'area': 250,
    'aspect_ratio': 2,
    'solidity': 1.6,
    'defect_area_ratio': 0.005,
  },
  'sesamoid': {
    'solidity': 1.3
  }
}

CRITERIA_DICT_VARIATION_MAGNITUDES = {
  'distal': {
    'area': 10,
    'aspect_ratio': 0.1,
    'aspect_ratio_tolerance': 0.05,
    'solidity': 0.1,
    'defect_area_ratio': 0.01,
  },
  'medial': {
    'area': 10,
    'aspect_ratio': 0.1,
    'aspect_ratio_tolerance': 0.1,
    'solidity': 0.1,
    'defect_area_ratio': 0.01,
  },
  'proximal': {
    'area': 10,
    'aspect_ratio': 0.1,
    'aspect_ratio_tolerance': 0.1,
    'solidity': 0.1,
    'defect_area_ratio': 0.01,
  },
  'metacarpal': {
    'area': 10,
    'aspect_ratio': 0.1,
    'aspect_ratio_tolerance': 0.1,
    'solidity': 0.05,
    'defect_area_ratio': 0.01,
  },
  'radius': {
    'area': 100,
    'aspect_ratio': 0.1,
    'solidity': 0.1,
    'defect_area_ratio': 0.001,
  },
  'ulna': {
    'area': 100,
    'aspect_ratio': 0.1,
    'solidity': 0.1,
    'defect_area_ratio': 0.001,
  },
  'sesamoid': {
    'solidity': 0.1
  }
}
