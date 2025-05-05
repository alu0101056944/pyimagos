'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Capture the cases where penalization needs to happen and adjust a penalization
factor so that the precision is highest.
'''

import copy
import time
import json

import numpy as np

from src.expected_contours.distal_phalanx import ExpectedContourDistalPhalanx
from src.expected_contours.metacarpal import ExpectedContourMetacarpal
from constants import CRITERIA_DICT

def case_004_metacarpal1():
  distal_phalanx_1 = ExpectedContourDistalPhalanx(
    encounter_amount=1,
  )
  distal_phalanx_1_contour = np.array([[[ 23,  96]],
       [[ 22,  97]],
       [[ 21,  97]],
       [[ 20,  98]],
       [[ 20, 104]],
       [[ 21, 105]],
       [[ 21, 110]],
       [[ 22, 111]],
       [[ 22, 120]],
       [[ 21, 121]],
       [[ 21, 122]],
       [[ 20, 123]],
       [[ 20, 128]],
       [[ 21, 129]],
       [[ 26, 129]],
       [[ 27, 128]],
       [[ 30, 128]],
       [[ 31, 127]],
       [[ 32, 127]],
       [[ 33, 126]],
       [[ 33, 125]],
       [[ 34, 124]],
       [[ 33, 123]],
       [[ 33, 122]],
       [[ 30, 119]],
       [[ 30, 117]],
       [[ 28, 115]],
       [[ 28, 112]],
       [[ 27, 111]],
       [[ 27, 104]],
       [[ 28, 103]],
       [[ 28, 100]],
       [[ 27,  99]],
       [[ 27,  98]],
       [[ 25,  96]]], dtype=np.int32)
  distal_phalanx_1.prepare(
    distal_phalanx_1_contour,
    image_width=301,
    image_height=462,
  )
  target_expected_contour = ExpectedContourMetacarpal(
    encounter_amount=1,
    ends_branchs_sequence=True,
    first_in_branch=distal_phalanx_1,
  )
  
  metacarpal_1_contour = np.array([[[ 47, 231]],
       [[ 46, 232]],
       [[ 44, 232]],
       [[ 43, 233]],
       [[ 42, 233]],
       [[ 41, 234]],
       [[ 41, 235]],
       [[ 40, 236]],
       [[ 40, 244]],
       [[ 41, 245]],
       [[ 41, 248]],
       [[ 42, 249]],
       [[ 42, 250]],
       [[ 43, 251]],
       [[ 43, 252]],
       [[ 44, 253]],
       [[ 44, 254]],
       [[ 45, 255]],
       [[ 45, 257]],
       [[ 46, 258]],
       [[ 46, 259]],
       [[ 47, 260]],
       [[ 47, 261]],
       [[ 48, 262]],
       [[ 48, 263]],
       [[ 49, 264]],
       [[ 49, 266]],
       [[ 50, 267]],
       [[ 50, 268]],
       [[ 51, 269]],
       [[ 51, 271]],
       [[ 52, 272]],
       [[ 52, 273]],
       [[ 53, 274]],
       [[ 53, 276]],
       [[ 54, 277]],
       [[ 54, 279]],
       [[ 55, 280]],
       [[ 55, 282]],
       [[ 56, 283]],
       [[ 56, 285]],
       [[ 57, 286]],
       [[ 57, 288]],
       [[ 58, 289]],
       [[ 58, 292]],
       [[ 59, 293]],
       [[ 59, 296]],
       [[ 60, 297]],
       [[ 60, 304]],
       [[ 61, 305]],
       [[ 61, 311]],
       [[ 62, 312]],
       [[ 62, 316]],
       [[ 63, 317]],
       [[ 63, 319]],
       [[ 64, 320]],
       [[ 64, 322]],
       [[ 65, 323]],
       [[ 65, 324]],
       [[ 66, 325]],
       [[ 66, 327]],
       [[ 67, 328]],
       [[ 67, 329]],
       [[ 68, 330]],
       [[ 68, 331]],
       [[ 69, 332]],
       [[ 70, 332]],
       [[ 71, 333]],
       [[ 72, 332]],
       [[ 73, 332]],
       [[ 76, 329]],
       [[ 77, 329]],
       [[ 80, 326]],
       [[ 81, 326]],
       [[ 84, 323]],
       [[ 85, 323]],
       [[ 87, 321]],
       [[ 87, 320]],
       [[ 85, 318]],
       [[ 85, 317]],
       [[ 84, 316]],
       [[ 84, 315]],
       [[ 80, 311]],
       [[ 80, 310]],
       [[ 79, 309]],
       [[ 79, 308]],
       [[ 77, 306]],
       [[ 77, 305]],
       [[ 76, 304]],
       [[ 76, 303]],
       [[ 74, 301]],
       [[ 74, 299]],
       [[ 73, 298]],
       [[ 73, 296]],
       [[ 72, 295]],
       [[ 72, 294]],
       [[ 71, 293]],
       [[ 71, 292]],
       [[ 70, 291]],
       [[ 70, 289]],
       [[ 69, 288]],
       [[ 69, 287]],
       [[ 68, 286]],
       [[ 68, 283]],
       [[ 67, 282]],
       [[ 67, 280]],
       [[ 66, 279]],
       [[ 66, 277]],
       [[ 65, 276]],
       [[ 65, 274]],
       [[ 64, 273]],
       [[ 64, 271]],
       [[ 63, 270]],
       [[ 63, 268]],
       [[ 62, 267]],
       [[ 62, 265]],
       [[ 61, 264]],
       [[ 61, 261]],
       [[ 60, 260]],
       [[ 60, 257]],
       [[ 59, 256]],
       [[ 59, 252]],
       [[ 58, 251]],
       [[ 58, 248]],
       [[ 59, 247]],
       [[ 59, 246]],
       [[ 60, 245]],
       [[ 60, 239]],
       [[ 59, 238]],
       [[ 59, 236]],
       [[ 58, 235]],
       [[ 58, 234]],
       [[ 56, 232]],
       [[ 54, 232]],
       [[ 53, 231]]], dtype=np.int32)
  metacarpal_2_contour = np.array([[[ 77, 216]],
       [[ 76, 217]],
       [[ 74, 217]],
       [[ 72, 219]],
       [[ 71, 219]],
       [[ 70, 220]],
       [[ 70, 226]],
       [[ 71, 227]],
       [[ 71, 231]],
       [[ 72, 232]],
       [[ 72, 233]],
       [[ 73, 234]],
       [[ 73, 239]],
       [[ 74, 240]],
       [[ 74, 243]],
       [[ 75, 244]],
       [[ 75, 249]],
       [[ 76, 250]],
       [[ 76, 253]],
       [[ 77, 254]],
       [[ 77, 257]],
       [[ 78, 258]],
       [[ 78, 261]],
       [[ 79, 262]],
       [[ 79, 265]],
       [[ 80, 266]],
       [[ 80, 269]],
       [[ 81, 270]],
       [[ 81, 273]],
       [[ 82, 274]],
       [[ 82, 278]],
       [[ 83, 279]],
       [[ 83, 283]],
       [[ 84, 284]],
       [[ 84, 289]],
       [[ 85, 290]],
       [[ 85, 305]],
       [[ 84, 306]],
       [[ 84, 308]],
       [[ 83, 309]],
       [[ 83, 311]],
       [[ 84, 312]],
       [[ 86, 312]],
       [[ 87, 313]],
       [[ 92, 313]],
       [[ 93, 314]],
       [[ 99, 314]],
       [[100, 315]],
       [[103, 315]],
       [[103, 308]],
       [[102, 307]],
       [[102, 303]],
       [[101, 302]],
       [[101, 301]],
       [[100, 300]],
       [[100, 299]],
       [[ 99, 298]],
       [[ 99, 297]],
       [[ 98, 296]],
       [[ 98, 295]],
       [[ 97, 294]],
       [[ 97, 292]],
       [[ 96, 291]],
       [[ 96, 288]],
       [[ 95, 287]],
       [[ 95, 284]],
       [[ 94, 283]],
       [[ 94, 278]],
       [[ 93, 277]],
       [[ 93, 271]],
       [[ 92, 270]],
       [[ 92, 263]],
       [[ 91, 262]],
       [[ 91, 255]],
       [[ 90, 254]],
       [[ 90, 246]],
       [[ 89, 245]],
       [[ 89, 239]],
       [[ 90, 238]],
       [[ 90, 229]],
       [[ 91, 228]],
       [[ 91, 222]],
       [[ 87, 218]],
       [[ 86, 218]],
       [[ 85, 217]],
       [[ 82, 217]],
       [[ 81, 216]]], dtype=np.int32)

  candidate_contours = [metacarpal_1_contour, metacarpal_2_contour]

  correct_candidate_index = 0

  return target_expected_contour, candidate_contours, correct_candidate_index

def case_004_distal2():
  distal_phalanx_1 = ExpectedContourDistalPhalanx(
    encounter_amount=1,
  )
  distal_phalanx_1_contour = np.array([[[ 23,  96]],
       [[ 22,  97]],
       [[ 21,  97]],
       [[ 20,  98]],
       [[ 20, 104]],
       [[ 21, 105]],
       [[ 21, 110]],
       [[ 22, 111]],
       [[ 22, 120]],
       [[ 21, 121]],
       [[ 21, 122]],
       [[ 20, 123]],
       [[ 20, 128]],
       [[ 21, 129]],
       [[ 26, 129]],
       [[ 27, 128]],
       [[ 30, 128]],
       [[ 31, 127]],
       [[ 32, 127]],
       [[ 33, 126]],
       [[ 33, 125]],
       [[ 34, 124]],
       [[ 33, 123]],
       [[ 33, 122]],
       [[ 30, 119]],
       [[ 30, 117]],
       [[ 28, 115]],
       [[ 28, 112]],
       [[ 27, 111]],
       [[ 27, 104]],
       [[ 28, 103]],
       [[ 28, 100]],
       [[ 27,  99]],
       [[ 27,  98]],
       [[ 25,  96]]], dtype=np.int32)
  distal_phalanx_1.prepare(
    distal_phalanx_1_contour,
    image_width=301,
    image_height=462,
  )
  target_expected_contour = ExpectedContourDistalPhalanx(
    encounter_amount=2,
    previous_encounter=distal_phalanx_1
  )

  medial_2_contour = np.array([[[ 66,  81]],
       [[ 65,  82]],
       [[ 64,  82]],
       [[ 64,  84]],
       [[ 63,  85]],
       [[ 63,  89]],
       [[ 64,  90]],
       [[ 64,  92]],
       [[ 65,  93]],
       [[ 65,  97]],
       [[ 66,  98]],
       [[ 66, 104]],
       [[ 65, 105]],
       [[ 65, 114]],
       [[ 64, 115]],
       [[ 64, 125]],
       [[ 63, 126]],
       [[ 63, 130]],
       [[ 64, 131]],
       [[ 66, 131]],
       [[ 67, 130]],
       [[ 71, 130]],
       [[ 72, 131]],
       [[ 73, 130]],
       [[ 77, 130]],
       [[ 78, 129]],
       [[ 82, 129]],
       [[ 82, 124]],
       [[ 81, 123]],
       [[ 81, 120]],
       [[ 80, 119]],
       [[ 80, 115]],
       [[ 79, 114]],
       [[ 79, 110]],
       [[ 78, 109]],
       [[ 78, 105]],
       [[ 77, 104]],
       [[ 77,  93]],
       [[ 78,  92]],
       [[ 78,  91]],
       [[ 79,  90]],
       [[ 79,  85]],
       [[ 78,  84]],
       [[ 78,  82]],
       [[ 77,  81]],
       [[ 72,  81]],
       [[ 71,  82]],
       [[ 68,  82]],
       [[ 67,  81]]], dtype=np.int32)
  medial_3_contour = np.array([[[102,  56]],
       [[101,  57]],
       [[ 96,  57]],
       [[ 95,  58]],
       [[ 95,  61]],
       [[ 94,  62]],
       [[ 94,  66]],
       [[ 95,  67]],
       [[ 95,  70]],
       [[ 96,  71]],
       [[ 96,  76]],
       [[ 97,  77]],
       [[ 97,  80]],
       [[ 96,  81]],
       [[ 96,  99]],
       [[ 95, 100]],
       [[ 95, 105]],
       [[ 94, 106]],
       [[ 94, 108]],
       [[ 95, 109]],
       [[ 95, 110]],
       [[107, 110]],
       [[108, 109]],
       [[115, 109]],
       [[116, 108]],
       [[116, 106]],
       [[117, 105]],
       [[116, 104]],
       [[116, 103]],
       [[114, 101]],
       [[114,  96]],
       [[113,  95]],
       [[113,  93]],
       [[112,  92]],
       [[112,  90]],
       [[111,  89]],
       [[111,  86]],
       [[110,  85]],
       [[110,  83]],
       [[109,  82]],
       [[109,  78]],
       [[108,  77]],
       [[108,  70]],
       [[109,  69]],
       [[109,  67]],
       [[110,  66]],
       [[110,  65]],
       [[111,  64]],
       [[111,  61]],
       [[110,  60]],
       [[110,  59]],
       [[109,  58]],
       [[109,  57]],
       [[108,  56]]], dtype=np.int32)
  distal_2_contour = np.array([[[69, 44]],
       [[68, 45]],
       [[67, 45]],
       [[65, 47]],
       [[65, 48]],
       [[64, 49]],
       [[64, 52]],
       [[65, 53]],
       [[65, 56]],
       [[66, 57]],
       [[66, 60]],
       [[67, 61]],
       [[67, 64]],
       [[66, 65]],
       [[66, 68]],
       [[65, 69]],
       [[65, 71]],
       [[64, 72]],
       [[64, 73]],
       [[63, 74]],
       [[63, 76]],
       [[64, 77]],
       [[63, 78]],
       [[64, 79]],
       [[76, 79]],
       [[77, 78]],
       [[79, 78]],
       [[79, 74]],
       [[78, 73]],
       [[78, 72]],
       [[76, 70]],
       [[76, 68]],
       [[75, 67]],
       [[75, 64]],
       [[74, 63]],
       [[74, 55]],
       [[75, 54]],
       [[75, 49]],
       [[74, 48]],
       [[74, 46]],
       [[73, 45]],
       [[70, 45]]], dtype=np.int32)

  candidate_contours = [
    medial_2_contour,
    medial_3_contour,
    distal_2_contour,
  ]

  correct_candidate_index = 2

  return target_expected_contour, candidate_contours, correct_candidate_index

def case_006_distal2():
  distal_phalanx_1 = ExpectedContourDistalPhalanx(
    encounter_amount=1,
  )
  distal_phalanx_1_contour = np.array([[[ 27,  98]],
       [[ 25, 100]],
       [[ 25, 102]],
       [[ 24, 103]],
       [[ 24, 111]],
       [[ 25, 112]],
       [[ 25, 119]],
       [[ 24, 120]],
       [[ 24, 123]],
       [[ 22, 125]],
       [[ 22, 126]],
       [[ 21, 127]],
       [[ 21, 128]],
       [[ 20, 129]],
       [[ 21, 130]],
       [[ 21, 132]],
       [[ 34, 132]],
       [[ 35, 131]],
       [[ 36, 131]],
       [[ 36, 127]],
       [[ 35, 126]],
       [[ 35, 125]],
       [[ 32, 122]],
       [[ 32, 112]],
       [[ 33, 111]],
       [[ 33, 106]],
       [[ 34, 105]],
       [[ 34, 103]],
       [[ 33, 102]],
       [[ 33, 101]],
       [[ 32, 100]],
       [[ 32,  99]],
       [[ 31,  99]],
       [[ 30,  98]]], dtype=np.int32)
  distal_phalanx_1.prepare(
    distal_phalanx_1_contour,
    image_width=301,
    image_height=462,
  )
  target_expected_contour = ExpectedContourDistalPhalanx(
    encounter_amount=2,
    previous_encounter=distal_phalanx_1
  )

  medial_2_contour = np.array([[[ 68,  85]],
       [[ 67,  86]],
       [[ 63,  86]],
       [[ 62,  87]],
       [[ 59,  87]],
       [[ 58,  86]],
       [[ 55,  86]],
       [[ 54,  87]],
       [[ 54,  90]],
       [[ 53,  91]],
       [[ 53,  95]],
       [[ 55,  97]],
       [[ 55,  98]],
       [[ 56,  99]],
       [[ 56, 101]],
       [[ 57, 102]],
       [[ 57, 105]],
       [[ 58, 106]],
       [[ 58, 119]],
       [[ 57, 120]],
       [[ 57, 131]],
       [[ 56, 132]],
       [[ 56, 138]],
       [[ 57, 139]],
       [[ 64, 139]],
       [[ 65, 140]],
       [[ 67, 140]],
       [[ 68, 139]],
       [[ 70, 139]],
       [[ 71, 138]],
       [[ 73, 138]],
       [[ 74, 137]],
       [[ 76, 137]],
       [[ 76, 135]],
       [[ 77, 134]],
       [[ 77, 133]],
       [[ 76, 132]],
       [[ 76, 131]],
       [[ 74, 129]],
       [[ 74, 126]],
       [[ 73, 125]],
       [[ 73, 122]],
       [[ 72, 121]],
       [[ 72, 118]],
       [[ 71, 117]],
       [[ 71, 115]],
       [[ 70, 114]],
       [[ 70, 111]],
       [[ 69, 110]],
       [[ 69,  97]],
       [[ 70,  96]],
       [[ 70,  89]],
       [[ 69,  88]],
       [[ 69,  86]]], dtype=np.int32)
  medial_3_contour = np.array([[[ 95,  60]],
       [[ 94,  61]],
       [[ 92,  61]],
       [[ 91,  62]],
       [[ 84,  62]],
       [[ 83,  63]],
       [[ 83,  68]],
       [[ 82,  69]],
       [[ 82,  70]],
       [[ 83,  71]],
       [[ 83,  73]],
       [[ 84,  74]],
       [[ 84,  75]],
       [[ 85,  76]],
       [[ 85,  79]],
       [[ 86,  80]],
       [[ 86,  86]],
       [[ 87,  87]],
       [[ 87,  92]],
       [[ 86,  93]],
       [[ 86, 107]],
       [[ 85, 108]],
       [[ 85, 117]],
       [[ 94, 117]],
       [[ 95, 118]],
       [[ 98, 118]],
       [[ 99, 117]],
       [[100, 117]],
       [[101, 116]],
       [[103, 116]],
       [[104, 115]],
       [[106, 115]],
       [[107, 114]],
       [[106, 113]],
       [[107, 112]],
       [[107, 109]],
       [[106, 108]],
       [[106, 107]],
       [[105, 106]],
       [[105, 105]],
       [[104, 104]],
       [[104, 102]],
       [[103, 101]],
       [[103,  98]],
       [[102,  97]],
       [[102,  95]],
       [[101,  94]],
       [[101,  92]],
       [[100,  91]],
       [[100,  89]],
       [[ 99,  88]],
       [[ 99,  82]],
       [[ 98,  81]],
       [[ 98,  76]],
       [[ 99,  75]],
       [[ 99,  72]],
       [[ 98,  71]],
       [[ 99,  70]],
       [[ 99,  69]],
       [[100,  68]],
       [[100,  64]],
       [[ 99,  63]],
       [[ 99,  62]],
       [[ 97,  60]]], dtype=np.int32)
  distal_2_contour = np.array([[[59, 44]],
       [[58, 45]],
       [[57, 45]],
       [[54, 48]],
       [[54, 49]],
       [[53, 50]],
       [[53, 57]],
       [[55, 59]],
       [[55, 60]],
       [[56, 61]],
       [[56, 72]],
       [[55, 73]],
       [[55, 75]],
       [[53, 77]],
       [[53, 79]],
       [[52, 80]],
       [[53, 81]],
       [[53, 84]],
       [[59, 84]],
       [[60, 85]],
       [[61, 85]],
       [[62, 84]],
       [[66, 84]],
       [[67, 83]],
       [[70, 83]],
       [[70, 78]],
       [[67, 75]],
       [[67, 74]],
       [[66, 73]],
       [[66, 71]],
       [[65, 70]],
       [[65, 60]],
       [[66, 59]],
       [[66, 56]],
       [[68, 54]],
       [[67, 53]],
       [[67, 49]],
       [[66, 48]],
       [[66, 47]],
       [[64, 45]],
       [[62, 45]],
       [[61, 44]]], dtype=np.int32)
  distal_4_contour = np.array([[[121,  35]],
       [[120,  36]],
       [[120,  37]],
       [[119,  38]],
       [[119,  43]],
       [[118,  44]],
       [[119,  45]],
       [[119,  48]],
       [[120,  49]],
       [[120,  61]],
       [[119,  62]],
       [[119,  64]],
       [[118,  65]],
       [[118,  70]],
       [[119,  71]],
       [[126,  71]],
       [[127,  70]],
       [[132,  70]],
       [[133,  69]],
       [[134,  69]],
       [[134,  66]],
       [[133,  65]],
       [[133,  63]],
       [[131,  61]],
       [[131,  60]],
       [[130,  59]],
       [[130,  57]],
       [[129,  56]],
       [[129,  47]],
       [[130,  46]],
       [[130,  42]],
       [[129,  41]],
       [[129,  39]],
       [[128,  38]],
       [[128,  37]],
       [[126,  35]]], dtype=np.int32)

  candidate_contours = [
    medial_2_contour,
    medial_3_contour,
    distal_2_contour,
    distal_4_contour,
  ]

  correct_candidate_index = 2

  return target_expected_contour, candidate_contours, correct_candidate_index

def case_023_distal2():
  distal_phalanx_1 = ExpectedContourDistalPhalanx(
    encounter_amount=1,
  )
  distal_phalanx_1_contour = np.array([[[ 22,  99]],
       [[ 21, 100]],
       [[ 21, 103]],
       [[ 20, 104]],
       [[ 20, 108]],
       [[ 21, 109]],
       [[ 21, 122]],
       [[ 20, 123]],
       [[ 20, 128]],
       [[ 21, 129]],
       [[ 27, 129]],
       [[ 28, 128]],
       [[ 29, 128]],
       [[ 30, 127]],
       [[ 32, 127]],
       [[ 33, 126]],
       [[ 34, 126]],
       [[ 34, 125]],
       [[ 32, 123]],
       [[ 32, 119]],
       [[ 30, 117]],
       [[ 30, 116]],
       [[ 28, 114]],
       [[ 28, 112]],
       [[ 27, 111]],
       [[ 27, 102]],
       [[ 26, 101]],
       [[ 26, 100]],
       [[ 25,  99]]], dtype=np.int32)
  distal_phalanx_1.prepare(
    distal_phalanx_1_contour,
    image_width=301,
    image_height=462,
  )
  target_expected_contour = ExpectedContourDistalPhalanx(
    encounter_amount=2,
    previous_encounter=distal_phalanx_1
  )

  medial_2_contour = np.array([[[ 86,  78]],
       [[ 85,  79]],
       [[ 75,  79]],
       [[ 75,  80]],
       [[ 74,  81]],
       [[ 74,  85]],
       [[ 73,  86]],
       [[ 74,  87]],
       [[ 74,  88]],
       [[ 75,  89]],
       [[ 75,  90]],
       [[ 76,  91]],
       [[ 76, 110]],
       [[ 75, 111]],
       [[ 75, 118]],
       [[ 74, 119]],
       [[ 74, 121]],
       [[ 73, 122]],
       [[ 73, 128]],
       [[ 74, 129]],
       [[ 82, 129]],
       [[ 83, 130]],
       [[ 84, 129]],
       [[ 87, 129]],
       [[ 88, 128]],
       [[ 90, 128]],
       [[ 91, 127]],
       [[ 96, 127]],
       [[ 96, 124]],
       [[ 95, 123]],
       [[ 95, 121]],
       [[ 93, 119]],
       [[ 93, 117]],
       [[ 92, 116]],
       [[ 92, 113]],
       [[ 91, 112]],
       [[ 91, 107]],
       [[ 90, 106]],
       [[ 90, 101]],
       [[ 89, 100]],
       [[ 89,  89]],
       [[ 90,  88]],
       [[ 90,  81]],
       [[ 89,  80]],
       [[ 89,  78]]], dtype=np.int32)
  distal_2_contour = np.array([[[81, 44]],
       [[80, 45]],
       [[77, 45]],
       [[77, 46]],
       [[76, 47]],
       [[76, 51]],
       [[77, 52]],
       [[77, 64]],
       [[76, 65]],
       [[76, 66]],
       [[75, 67]],
       [[75, 68]],
       [[73, 70]],
       [[73, 76]],
       [[77, 76]],
       [[78, 77]],
       [[84, 77]],
       [[85, 76]],
       [[90, 76]],
       [[90, 74]],
       [[91, 73]],
       [[91, 72]],
       [[90, 71]],
       [[90, 70]],
       [[89, 69]],
       [[89, 68]],
       [[87, 66]],
       [[87, 64]],
       [[86, 63]],
       [[86, 53]],
       [[87, 52]],
       [[87, 48]],
       [[86, 47]],
       [[86, 46]],
       [[85, 45]],
       [[84, 45]],
       [[83, 44]]], dtype=np.int32)

  candidate_contours = [
    medial_2_contour,
    distal_2_contour,
  ]

  correct_candidate_index = 1

  return target_expected_contour, candidate_contours, correct_candidate_index

def case_022_distal5():
  distal_phalanx_1 = ExpectedContourDistalPhalanx(
    encounter_amount=1,
  )
  distal_phalanx_1_contour = np.array([[[ 21, 102]],
       [[ 21, 103]],
       [[ 20, 104]],
       [[ 20, 109]],
       [[ 21, 110]],
       [[ 21, 114]],
       [[ 22, 115]],
       [[ 22, 125]],
       [[ 21, 126]],
       [[ 21, 128]],
       [[ 22, 129]],
       [[ 22, 131]],
       [[ 25, 131]],
       [[ 26, 130]],
       [[ 28, 130]],
       [[ 29, 129]],
       [[ 30, 129]],
       [[ 31, 128]],
       [[ 32, 128]],
       [[ 33, 127]],
       [[ 34, 127]],
       [[ 35, 126]],
       [[ 35, 124]],
       [[ 34, 123]],
       [[ 34, 122]],
       [[ 33, 122]],
       [[ 32, 121]],
       [[ 32, 120]],
       [[ 30, 118]],
       [[ 30, 117]],
       [[ 28, 115]],
       [[ 28, 112]],
       [[ 27, 111]],
       [[ 27, 104]],
       [[ 26, 104]],
       [[ 24, 102]]], dtype=np.int32)
  distal_phalanx_1.prepare(
    distal_phalanx_1_contour,
    image_width=301,
    image_height=462,
  )
  target_expected_contour = ExpectedContourDistalPhalanx(
    encounter_amount=5,
    previous_encounter=distal_phalanx_1,
  )

  proximal_5_contour = np.array([[[216, 189]],
       [[215, 190]],
       [[213, 190]],
       [[212, 191]],
       [[211, 191]],
       [[210, 192]],
       [[210, 201]],
       [[211, 202]],
       [[211, 208]],
       [[210, 209]],
       [[210, 212]],
       [[208, 214]],
       [[208, 215]],
       [[207, 216]],
       [[207, 217]],
       [[206, 218]],
       [[206, 219]],
       [[205, 220]],
       [[205, 221]],
       [[203, 223]],
       [[203, 224]],
       [[200, 227]],
       [[200, 228]],
       [[197, 231]],
       [[197, 232]],
       [[194, 235]],
       [[192, 235]],
       [[191, 236]],
       [[191, 237]],
       [[189, 239]],
       [[189, 244]],
       [[191, 244]],
       [[192, 245]],
       [[198, 245]],
       [[199, 246]],
       [[200, 246]],
       [[201, 247]],
       [[202, 247]],
       [[203, 248]],
       [[204, 248]],
       [[205, 249]],
       [[208, 249]],
       [[209, 250]],
       [[210, 249]],
       [[211, 250]],
       [[211, 251]],
       [[212, 252]],
       [[213, 251]],
       [[213, 250]],
       [[214, 249]],
       [[214, 248]],
       [[215, 247]],
       [[215, 239]],
       [[216, 238]],
       [[216, 234]],
       [[217, 233]],
       [[217, 230]],
       [[218, 229]],
       [[218, 226]],
       [[219, 225]],
       [[219, 222]],
       [[220, 221]],
       [[220, 219]],
       [[221, 218]],
       [[221, 216]],
       [[222, 215]],
       [[222, 213]],
       [[223, 212]],
       [[223, 211]],
       [[225, 209]],
       [[225, 208]],
       [[229, 204]],
       [[229, 203]],
       [[228, 202]],
       [[228, 201]],
       [[229, 200]],
       [[229, 199]],
       [[228, 198]],
       [[229, 197]],
       [[228, 196]],
       [[227, 196]],
       [[226, 195]],
       [[225, 195]],
       [[220, 190]],
       [[219, 190]],
       [[218, 189]]], dtype=np.int32)
  distal_5_contour = np.array([[[250, 162]],
       [[249, 163]],
       [[248, 163]],
       [[247, 164]],
       [[247, 165]],
       [[245, 167]],
       [[244, 167]],
       [[243, 168]],
       [[242, 168]],
       [[239, 171]],
       [[238, 171]],
       [[236, 173]],
       [[235, 173]],
       [[233, 175]],
       [[232, 175]],
       [[230, 177]],
       [[229, 177]],
       [[228, 178]],
       [[227, 178]],
       [[226, 179]],
       [[221, 179]],
       [[218, 182]],
       [[218, 186]],
       [[219, 187]],
       [[220, 187]],
       [[221, 188]],
       [[222, 188]],
       [[224, 190]],
       [[224, 191]],
       [[226, 193]],
       [[227, 193]],
       [[228, 194]],
       [[229, 194]],
       [[230, 195]],
       [[231, 195]],
       [[232, 196]],
       [[233, 196]],
       [[234, 195]],
       [[234, 193]],
       [[235, 192]],
       [[235, 191]],
       [[236, 190]],
       [[236, 189]],
       [[237, 188]],
       [[237, 187]],
       [[239, 185]],
       [[239, 184]],
       [[241, 182]],
       [[241, 181]],
       [[255, 167]],
       [[255, 163]],
       [[254, 162]]], dtype=np.int32)
  
  candidate_contours = [
    proximal_5_contour,
    distal_5_contour,
  ]

  correct_candidate_index = 1

  return target_expected_contour, candidate_contours, correct_candidate_index

def case_023_distal5():
  distal_phalanx_1 = ExpectedContourDistalPhalanx(
    encounter_amount=1,
  )
  distal_phalanx_1_contour = np.array([[[ 22,  99]],
       [[ 21, 100]],
       [[ 21, 103]],
       [[ 20, 104]],
       [[ 20, 108]],
       [[ 21, 109]],
       [[ 21, 122]],
       [[ 20, 123]],
       [[ 20, 128]],
       [[ 21, 129]],
       [[ 27, 129]],
       [[ 28, 128]],
       [[ 29, 128]],
       [[ 30, 127]],
       [[ 32, 127]],
       [[ 33, 126]],
       [[ 34, 126]],
       [[ 34, 125]],
       [[ 32, 123]],
       [[ 32, 119]],
       [[ 30, 117]],
       [[ 30, 116]],
       [[ 28, 114]],
       [[ 28, 112]],
       [[ 27, 111]],
       [[ 27, 102]],
       [[ 26, 101]],
       [[ 26, 100]],
       [[ 25,  99]]], dtype=np.int32)
  distal_phalanx_1.prepare(
    distal_phalanx_1_contour,
    image_width=301,
    image_height=462,
  )
  target_expected_contour = ExpectedContourDistalPhalanx(
    encounter_amount=5,
    previous_encounter=distal_phalanx_1,
  )

  metacarpal_5_contour = np.array([[[205, 280]],
       [[204, 281]],
       [[203, 281]],
       [[202, 282]],
       [[201, 282]],
       [[200, 283]],
       [[199, 283]],
       [[197, 285]],
       [[197, 292]],
       [[196, 293]],
       [[196, 295]],
       [[194, 297]],
       [[194, 298]],
       [[192, 300]],
       [[192, 301]],
       [[181, 312]],
       [[180, 312]],
       [[177, 315]],
       [[176, 315]],
       [[174, 317]],
       [[173, 317]],
       [[172, 318]],
       [[171, 318]],
       [[169, 320]],
       [[168, 320]],
       [[167, 321]],
       [[165, 321]],
       [[164, 322]],
       [[161, 322]],
       [[160, 323]],
       [[157, 323]],
       [[156, 324]],
       [[155, 324]],
       [[153, 326]],
       [[153, 328]],
       [[154, 329]],
       [[154, 332]],
       [[155, 333]],
       [[155, 336]],
       [[156, 337]],
       [[156, 340]],
       [[157, 341]],
       [[157, 344]],
       [[158, 345]],
       [[158, 347]],
       [[160, 349]],
       [[161, 349]],
       [[162, 350]],
       [[168, 350]],
       [[169, 349]],
       [[171, 349]],
       [[172, 348]],
       [[173, 348]],
       [[186, 335]],
       [[186, 334]],
       [[188, 332]],
       [[189, 332]],
       [[210, 311]],
       [[211, 311]],
       [[215, 307]],
       [[216, 307]],
       [[220, 303]],
       [[220, 302]],
       [[221, 301]],
       [[222, 301]],
       [[224, 299]],
       [[224, 297]],
       [[225, 296]],
       [[225, 292]],
       [[223, 290]],
       [[223, 289]],
       [[222, 288]],
       [[221, 288]],
       [[220, 287]],
       [[220, 286]],
       [[218, 284]],
       [[218, 283]],
       [[217, 283]],
       [[215, 281]],
       [[214, 281]],
       [[213, 280]]], dtype=np.int32)
  proximal_5_contour = np.array([[[249, 233]],
       [[248, 234]],
       [[246, 234]],
       [[243, 237]],
       [[243, 245]],
       [[242, 246]],
       [[242, 248]],
       [[239, 251]],
       [[239, 252]],
       [[237, 254]],
       [[236, 254]],
       [[232, 258]],
       [[231, 258]],
       [[230, 259]],
       [[229, 259]],
       [[227, 261]],
       [[226, 261]],
       [[225, 262]],
       [[223, 262]],
       [[221, 264]],
       [[219, 264]],
       [[218, 265]],
       [[216, 265]],
       [[215, 266]],
       [[211, 266]],
       [[209, 268]],
       [[209, 269]],
       [[208, 270]],
       [[208, 272]],
       [[209, 273]],
       [[209, 275]],
       [[210, 275]],
       [[211, 276]],
       [[212, 276]],
       [[213, 277]],
       [[214, 277]],
       [[215, 278]],
       [[216, 278]],
       [[222, 284]],
       [[222, 285]],
       [[225, 288]],
       [[225, 289]],
       [[226, 289]],
       [[227, 288]],
       [[228, 288]],
       [[230, 286]],
       [[230, 285]],
       [[232, 283]],
       [[232, 282]],
       [[233, 281]],
       [[233, 280]],
       [[234, 279]],
       [[234, 278]],
       [[237, 275]],
       [[237, 274]],
       [[240, 271]],
       [[240, 270]],
       [[245, 265]],
       [[245, 264]],
       [[256, 253]],
       [[257, 253]],
       [[258, 252]],
       [[259, 252]],
       [[259, 251]],
       [[260, 250]],
       [[261, 250]],
       [[261, 249]],
       [[262, 248]],
       [[262, 243]],
       [[259, 240]],
       [[259, 239]],
       [[258, 239]],
       [[256, 237]],
       [[256, 236]],
       [[253, 233]]], dtype=np.int32)
  distal_5_contour = np.array([[[283, 207]],
       [[282, 208]],
       [[281, 208]],
       [[280, 209]],
       [[279, 209]],
       [[278, 210]],
       [[277, 210]],
       [[277, 211]],
       [[273, 215]],
       [[272, 215]],
       [[270, 217]],
       [[269, 217]],
       [[268, 218]],
       [[267, 218]],
       [[264, 221]],
       [[262, 221]],
       [[261, 222]],
       [[259, 222]],
       [[258, 223]],
       [[254, 223]],
       [[252, 225]],
       [[252, 229]],
       [[254, 231]],
       [[255, 231]],
       [[256, 232]],
       [[256, 233]],
       [[258, 235]],
       [[258, 236]],
       [[259, 237]],
       [[260, 237]],
       [[261, 238]],
       [[261, 239]],
       [[263, 241]],
       [[267, 241]],
       [[268, 240]],
       [[268, 238]],
       [[269, 237]],
       [[269, 236]],
       [[270, 235]],
       [[270, 234]],
       [[271, 233]],
       [[271, 232]],
       [[273, 230]],
       [[273, 229]],
       [[275, 227]],
       [[275, 226]],
       [[286, 215]],
       [[288, 215]],
       [[289, 214]],
       [[289, 209]],
       [[287, 207]]], dtype=np.int32)

  candidate_contours = [
    metacarpal_5_contour,
    proximal_5_contour,
    distal_5_contour,
  ]

  correct_candidate_index = 2

  return target_expected_contour, candidate_contours, correct_candidate_index


def all_case_tuples() -> dict[list[list]]:
  (
    case_004_metacarpal1_target,
    case_004_metacarpal1_candidates,
    case_004_metacarpal1_correct_candidate_index,
  ) = case_004_metacarpal1()

  (
    case_004_distal2_target,
    case_004_distal2_candidates,
    case_004_distal2_correct_candidate_index,
  ) = case_004_distal2()

  (
    case_006_distal2_target,
    case_006_distal2_candidates,
    case_006_distal2_correct_candidate_index,
  ) = case_006_distal2()

  (
    case_023_distal2_target,
    case_023_distal2_candidates,
    case_023_distal2_correct_candidate_index,
  ) = case_023_distal2()

  (
    case_022_distal5_target,
    case_022_distal5_candidates,
    case_022_distal5_correct_candidate_index,
  ) = case_022_distal5()

  (
    case_023_distal5_target,
    case_023_distal5_candidates,
    case_023_distal5_correct_candidate_index,
  ) = case_023_distal5()

  expected_contour_to_cases = {
    'metacarpal_1': [
      [
        case_004_metacarpal1_target,
        case_004_metacarpal1_candidates,
        case_004_metacarpal1_correct_candidate_index,
        'case_004_metacarpal1',
      ],
    ],
    'distal_2': [
      [
        case_004_distal2_target,
        case_004_distal2_candidates,
        case_004_distal2_correct_candidate_index,
        'case_004_distal2',
      ],
      [
        case_006_distal2_target,
        case_006_distal2_candidates,
        case_006_distal2_correct_candidate_index,
        'case_006_distal2',
      ],
      [
        case_023_distal2_target,
        case_023_distal2_candidates,
        case_023_distal2_correct_candidate_index,
        'case_023_distal2',
      ],
    ],
    'distal_5': [
      [
        case_022_distal5_target,
        case_022_distal5_candidates,
        case_022_distal5_correct_candidate_index,
        'case_022_distal5',
      ],
      [
        case_023_distal5_target,
        case_023_distal5_candidates,
        case_023_distal5_correct_candidate_index,
        'case_023_distal5',
      ],
    ]
  }

  return expected_contour_to_cases

def experiment_penalization_main(debug_mode: bool, step: int = 0.0025,
                                 range: int = 40):
  start_time = time.time()

  output_string = ''

  criteria_dict = copy.deepcopy(CRITERIA_DICT)

  expected_contour_to_cases = all_case_tuples()

  expected_contour_to_case_to_contour_to_difference = {}
  for expected_contour_key in expected_contour_to_cases:
    expected_contour_to_case_to_contour_to_difference[expected_contour_key] = {}
    for case_info in expected_contour_to_cases[expected_contour_key]:
      target_expected_contour = case_info[0]
      candidate_contours = case_info[1]
      correct_candidate_index = case_info[2]
      case_title = case_info[3]
      
      expected_contour_to_case_to_contour_to_difference[
        expected_contour_key][case_title] = {}
      expected_contour_to_case_to_contour_to_difference[
        expected_contour_key][case_title]['original'] = {
        'correct_candidate_index': correct_candidate_index,
        'chosen_candidate_index': None,
        'candidate_contour_differences': []
      }

      scores = []
      for i, candidate_contour in enumerate(candidate_contours):
        target_expected_contour.prepare(
          candidate_contour,
          image_width=301,
          image_height=462,
        )
        score = target_expected_contour.shape_restrictions(criteria_dict)
        scores.append(score)

        expected_contour_to_case_to_contour_to_difference[
          expected_contour_key
          ][case_title]['original']['candidate_contour_differences'].append(score)

        chosen_candidate_index = int(np.argmin(scores))
        expected_contour_to_case_to_contour_to_difference[
          expected_contour_key
          ][case_title]['original']['chosen_candidate_index'] = chosen_candidate_index


  output_string = output_string + '# Stage 1 penalization factors:\n'

  expected_contour_to_factor_to_precision = {}
  for expected_contour_key in expected_contour_to_cases:
    expected_contour_to_factor_to_precision[expected_contour_key] = {}

  for expected_contour_key in expected_contour_to_cases:
    contour_type = expected_contour_key.split('_')[0]
    
    first_stage_penalization_factors = list(np.arange(1.0, 0.1, -0.1))
    first_stage_penalization_factors = [
      float(factor) for factor in first_stage_penalization_factors
    ]

    penalization_to_success_list = {}
    for penalization_factor in first_stage_penalization_factors:
      penalization_to_success_list[penalization_factor] = []

      original_penalization_factor = (
        criteria_dict[contour_type]['positional_penalization']
      )
      criteria_dict[contour_type]['positional_penalization'] = (
        penalization_factor
      )

      for case_info in expected_contour_to_cases[expected_contour_key]:
        target_expected_contour = case_info[0]
        candidate_contours = case_info[1]
        correct_candidate_index = case_info[2]
        case_title = case_info[3]

        scores = []
        for candidate_contour in candidate_contours:
          target_expected_contour.prepare(
            candidate_contour,
            image_width=301,
            image_height=462,
          )
          score = target_expected_contour.shape_restrictions(criteria_dict)
          scores.append(score)
        chosen_candidate_index = int(np.argmin(scores))
        if chosen_candidate_index == correct_candidate_index:
          penalization_to_success_list[penalization_factor].append(True)
        else:
          penalization_to_success_list[penalization_factor].append(False)

      criteria_dict[contour_type]['positional_penalization'] = (
        original_penalization_factor
      )
      
    expected_contour_to_penalization_to_precision = {}
    for penalization_factor in first_stage_penalization_factors:
      positive_amount = penalization_to_success_list[penalization_factor].count(True)
      length = len(penalization_to_success_list[penalization_factor])
      expected_contour_to_penalization_to_precision[penalization_factor] = positive_amount / length

    all_precisions_first_stage = list(expected_contour_to_penalization_to_precision.values())
    best_precision_penalization_factor_index = (
      len(all_precisions_first_stage) - 1 - np.argmax(all_precisions_first_stage[::-1])
    )
    best_precision_penalization_factor = first_stage_penalization_factors[
      best_precision_penalization_factor_index]

    output_string = output_string + f'expected_contour_key={expected_contour_key}, ' + (
      f'case_title={case_title}, ') + (
      f'best_precision_first_stage={best_precision_penalization_factor}\n')

    upper_bound = min(1, best_precision_penalization_factor + (step * (range / 2)))
    lower_bound = max(0, best_precision_penalization_factor - (step * (range / 2)))
    second_stage_penalization_factors = (
      list(np.arange(upper_bound, best_precision_penalization_factor, (-1) * step)) +
      list(np.arange(best_precision_penalization_factor, lower_bound, (-1) * step))
    )
    second_stage_penalization_factors = [
      float(factor) for factor in second_stage_penalization_factors
    ]

    penalization_to_success_list = {}
    for penalization_factor in second_stage_penalization_factors:
      penalization_to_success_list[penalization_factor] = []

      original_penalization_factor = (
        criteria_dict[contour_type]['positional_penalization']
      )
      criteria_dict[contour_type]['positional_penalization'] = (
        penalization_factor
      )

      for case_info in expected_contour_to_cases[expected_contour_key]:
        target_expected_contour = case_info[0]
        candidate_contours = case_info[1]
        correct_candidate_index = case_info[2]
        case_title = case_info[3]

        expected_contour_to_case_to_contour_to_difference[
          expected_contour_key][case_title][penalization_factor] = {}
        expected_contour_to_case_to_contour_to_difference[
          expected_contour_key][case_title][penalization_factor] = {
            'correct_candidate_index': correct_candidate_index,
            'chosen_candidate_index': None,
            'candidate_contour_differences': [],
          }

        scores = []
        for candidate_contour in candidate_contours:
          target_expected_contour.prepare(
            candidate_contour,
            image_width=301,
            image_height=462,
          )
          score = target_expected_contour.shape_restrictions(criteria_dict)
          scores.append(score)
          expected_contour_to_case_to_contour_to_difference[
            expected_contour_key][case_title][penalization_factor][
              'candidate_contour_differences'].append(score)

        chosen_candidate_index = int(np.argmin(scores))
        if chosen_candidate_index == correct_candidate_index:
          penalization_to_success_list[penalization_factor].append(True)
        else:
          penalization_to_success_list[penalization_factor].append(False)

        expected_contour_to_case_to_contour_to_difference[
          expected_contour_key
          ][case_title][penalization_factor]['chosen_candidate_index'] = chosen_candidate_index

      criteria_dict[contour_type]['positional_penalization'] = (
        original_penalization_factor
      )

    for penalization_factor in second_stage_penalization_factors:
      positive_amount = penalization_to_success_list[penalization_factor].count(True)
      length = len(penalization_to_success_list[penalization_factor])
      expected_contour_to_factor_to_precision[expected_contour_key][penalization_factor] = (
        positive_amount / length
      )

  output_string = output_string + f'step={step}, range={range}.\n\n'


  output_string = output_string + '# All expected_contour_to_factor_to_precision\n'
  output_string = output_string +  json.dumps(expected_contour_to_factor_to_precision, indent=2)
  output_string = output_string + '\n\n'

  # Different expected_contours of the same contour_type may have different factors
  # attempted. Make it homogeneous; same contour_type same penalization factors so
  # that an average precision per factor can be calculated for each contour type.
  for contour_type in criteria_dict.keys():
    relevant_expected_contour_keys = list(filter(
      lambda key : contour_type in key.split('_'),
      expected_contour_to_factor_to_precision.keys(),
    ))

    all_penalization_factor_keys = []
    for relevant_expected_contour_key in relevant_expected_contour_keys:
      all_penalization_factor_keys = all_penalization_factor_keys + (
        list(
          (
            expected_contour_to_factor_to_precision[relevant_expected_contour_key]
          ).keys()
        )
      )

    for relevant_expected_contour_key in relevant_expected_contour_keys:
      penalization_to_success_list = {}
      for penalization_factor in all_penalization_factor_keys:
        if penalization_factor not in expected_contour_to_factor_to_precision[relevant_expected_contour_key]:
          missing_penalization_factor = penalization_factor
          penalization_to_success_list[missing_penalization_factor] = []

          original_penalization_factor = (
            criteria_dict[contour_type]['positional_penalization']
          )
          criteria_dict[contour_type]['positional_penalization'] = (
            missing_penalization_factor
          )
          for case_info in expected_contour_to_cases[relevant_expected_contour_key]:
            target_expected_contour = case_info[0]
            candidate_contours = case_info[1]
            correct_candidate_index = case_info[2]
            case_title = case_info[3]

            expected_contour_to_case_to_contour_to_difference[
                      relevant_expected_contour_key][case_title][missing_penalization_factor] = {}
            expected_contour_to_case_to_contour_to_difference[
              relevant_expected_contour_key][case_title][missing_penalization_factor] = {}
            expected_contour_to_case_to_contour_to_difference[
              relevant_expected_contour_key][case_title][missing_penalization_factor] = {
                'correct_candidate_index': correct_candidate_index,
                'chosen_candidate_index': None,
                'candidate_contour_differences': [],
              }

            scores = []
            for candidate_contour in candidate_contours:
              target_expected_contour.prepare(
                candidate_contour,
                image_width=301,
                image_height=462,
              )
              score = target_expected_contour.shape_restrictions(criteria_dict)
              scores.append(score)
              expected_contour_to_case_to_contour_to_difference[
                relevant_expected_contour_key][case_title][missing_penalization_factor][
                  'candidate_contour_differences'].append(score)

            chosen_candidate_index = int(np.argmin(scores))

            if chosen_candidate_index == correct_candidate_index:
              penalization_to_success_list[penalization_factor].append(True)
            else:
              penalization_to_success_list[penalization_factor].append(False)

            expected_contour_to_case_to_contour_to_difference[
              relevant_expected_contour_key
              ][case_title][missing_penalization_factor]['chosen_candidate_index'] = chosen_candidate_index

          criteria_dict[contour_type]['positional_penalization'] = (
            original_penalization_factor
          )
        
      for penalization_factor in penalization_to_success_list.keys():
        positive_amount = penalization_to_success_list[penalization_factor].count(True)
        length = len(penalization_to_success_list[penalization_factor])
        expected_contour_to_factor_to_precision[relevant_expected_contour_key][penalization_factor] = (
          positive_amount / length
        )

  output_string = output_string + '# All expected_contour_to_factor_to_precision (homogeneous)\n'
  output_string = output_string + json.dumps(expected_contour_to_factor_to_precision, indent=2)
  output_string = output_string + '\n\n'

  output_string = output_string + '# Atomic differences \n'
  output_string = output_string + json.dumps(expected_contour_to_case_to_contour_to_difference, indent=2)
  output_string = output_string + '\n\n'

  contour_type_to_factor_to_average_precision = {}
  for contour_type in criteria_dict.keys():
    relevant_expected_contour_keys = list(filter(
      lambda key : contour_type in key.split('_'),
      expected_contour_to_factor_to_precision.keys(),
    ))

    if len(relevant_expected_contour_keys) > 0:
      contour_type_to_factor_to_average_precision[contour_type] = {}
      # penalization factors used are the same (relevant keys wise)
      penalization_factor_keys = (
        expected_contour_to_factor_to_precision[relevant_expected_contour_keys[0]]
      ).keys()
      
      for penalization_factor in penalization_factor_keys:
        relevant_precisions = [
          expected_contour_to_factor_to_precision[relevant_expected_contour_key][penalization_factor]
          for relevant_expected_contour_key in relevant_expected_contour_keys
        ]
        average_precision = sum(relevant_precisions) / len(relevant_precisions)
        contour_type_to_factor_to_average_precision[contour_type][penalization_factor] = average_precision

  output_string = output_string + '# Average penalization factors per contour type:\n'
  output_string = output_string + json.dumps(contour_type_to_factor_to_average_precision, indent=2)
  output_string = output_string + '\n\n'

  contour_type_to_best_factor = {}
  for contour_type in criteria_dict.keys():
    if contour_type in contour_type_to_factor_to_average_precision:
      local_average_precisions = list(
        contour_type_to_factor_to_average_precision[contour_type].values()
      )
      best_factor_index = (
        len(local_average_precisions) - 1 - np.argmax(local_average_precisions[::-1])
      )

      contour_type_to_best_factor[contour_type] = {
        'best_factor': (
          list(
            (
              contour_type_to_factor_to_average_precision[contour_type]
            ).keys()
          )[best_factor_index]
        ),
        'precision': local_average_precisions[best_factor_index]
      }
    else:
      contour_type_to_best_factor[contour_type] = 'No cases were given.'

  output_string = output_string + '# Best penalization factor per contour type:\n'
  output_string = output_string + json.dumps(contour_type_to_best_factor, indent=2)
  output_string = output_string + '\n\n'

  elapsed_time = time.time() - start_time
  print(f'Tiempo de ejecución: {elapsed_time}')

  if not debug_mode:
    with open('best_penalization_factors.txt', 'w') as f:
      f.write(output_string)
      print('Writing best_penalization_factors.txt')
      print('Success.')
  else:
    print('Debug mode is on. Not write best_penalization_factors.txt file.')
