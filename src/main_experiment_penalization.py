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
    ]
  }

  return expected_contour_to_cases

def experiment_penalization_main(debug_mode: bool):
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

      success_list = []
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


  expected_contour_to_factor_to_precision = {}
  for expected_contour_key in expected_contour_to_cases:
    expected_contour_to_factor_to_precision[expected_contour_key] = {}

  for expected_contour_key in expected_contour_to_cases:
    contour_type = expected_contour_key.split('_')[0]
    for case_info in expected_contour_to_cases[expected_contour_key]:

      # TODO algorithm: first try out big jumps to see where the precision changes most,
      # then use small jumps to fine tune.
      penalization_factors = list(np.arange(1, 0.0001, -0.0001))
      for penalization_factor in penalization_factors:
        original_penalization_factor = (
          criteria_dict[contour_type]['positional_penalization']
        )
        criteria_dict[contour_type]['positional_penalization'] = (
          penalization_factor
        )

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

        success_list = []
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

        expected_contour_to_case_to_contour_to_difference[
          expected_contour_key
          ][case_title][penalization_factor]['chosen_candidate_index'] = chosen_candidate_index

        if chosen_candidate_index == correct_candidate_index:
          success_list.append(True)
        else:
          success_list.append(False)
        precision = success_list.count(True) / len(success_list)

        expected_contour_to_factor_to_precision[expected_contour_key][penalization_factor] = precision

        criteria_dict[contour_type]['positional_penalization'] = (
          original_penalization_factor
        )

  output_string = output_string + '# All expected_contour_to_factor_to_precision\n'
  output_string = output_string +  json.dumps(expected_contour_to_factor_to_precision, indent=2)
  output_string = output_string + '\n\n'

  # Different expected_contours of the same contour_type may have different factors
  # attempted. Make it homogeneous; same contour_type same penalization factors so
  # that an average precision per factor can be calculated for each contour type.
  for contour_type in criteria_dict.keys():
    relevant_expected_contour_keys = filter(
      lambda key : contour_type in key.split('_'),
      expected_contour_to_factor_to_precision.keys(),
    )

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
      for penalization_factor in all_penalization_factor_keys:
        local_factors_dict = (
          expected_contour_to_factor_to_precision[relevant_expected_contour_key]
        )
        if penalization_factor not in local_factors_dict:
          missing_penalization_factor = penalization_factor
          for case_info in expected_contour_to_cases[relevant_expected_contour_key]:
            original_penalization_factor = (
              criteria_dict[contour_type]['positional_penalization']
            )
            criteria_dict[contour_type]['positional_penalization'] = (
              missing_penalization_factor
            )

            target_expected_contour = case_info[0]
            candidate_contours = case_info[1]
            correct_candidate_index = case_info[2]
            case_title = case_info[3]

            expected_contour_to_case_to_contour_to_difference[
                      expected_contour_key][case_title][missing_penalization_factor] = {}
            expected_contour_to_case_to_contour_to_difference[
              expected_contour_key][case_title][missing_penalization_factor] = {}
            expected_contour_to_case_to_contour_to_difference[
              expected_contour_key][case_title][missing_penalization_factor] = {
                'correct_candidate_index': correct_candidate_index,
                'chosen_candidate_index': None,
                'candidate_contour_differences': [],
              }

            success_list = []
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
                expected_contour_key][case_title][missing_penalization_factor][
                  'candidate_contour_differences'].append(score)

            chosen_candidate_index = int(np.argmin(scores))

            expected_contour_to_case_to_contour_to_difference[
              expected_contour_key
              ][case_title][missing_penalization_factor]['chosen_candidate_index'] = chosen_candidate_index

            if chosen_candidate_index == correct_candidate_index:
              success_list.append(True)
            else:
              success_list.append(False)
            precision = success_list.count(True) / len(success_list)

            expected_contour_to_factor_to_precision[expected_contour_key][missing_penalization_factor] = precision

            criteria_dict[contour_type]['positional_penalization'] = (
              original_penalization_factor
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
