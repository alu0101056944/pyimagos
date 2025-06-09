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

def case_022_distal2():
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
    encounter_amount=2,
    previous_encounter=distal_phalanx_1
  )

  medial_2_contour = np.array([[[ 66,  75]],
       [[ 65,  76]],
       [[ 64,  76]],
       [[ 63,  77]],
       [[ 60,  77]],
       [[ 59,  78]],
       [[ 55,  78]],
       [[ 55,  85]],
       [[ 56,  86]],
       [[ 56,  87]],
       [[ 58,  89]],
       [[ 58,  90]],
       [[ 59,  91]],
       [[ 59,  94]],
       [[ 60,  95]],
       [[ 60, 100]],
       [[ 61, 101]],
       [[ 61, 121]],
       [[ 62, 122]],
       [[ 62, 123]],
       [[ 61, 124]],
       [[ 62, 125]],
       [[ 62, 126]],
       [[ 64, 126]],
       [[ 65, 125]],
       [[ 73, 125]],
       [[ 74, 124]],
       [[ 75, 124]],
       [[ 76, 123]],
       [[ 77, 123]],
       [[ 78, 122]],
       [[ 82, 122]],
       [[ 82, 119]],
       [[ 81, 119]],
       [[ 80, 118]],
       [[ 80, 114]],
       [[ 78, 112]],
       [[ 78, 110]],
       [[ 77, 109]],
       [[ 77, 108]],
       [[ 76, 107]],
       [[ 76, 106]],
       [[ 75, 105]],
       [[ 75, 103]],
       [[ 74, 102]],
       [[ 74, 101]],
       [[ 73, 100]],
       [[ 73,  98]],
       [[ 72,  97]],
       [[ 72,  93]],
       [[ 71,  92]],
       [[ 71,  84]],
       [[ 72,  83]],
       [[ 72,  79]],
       [[ 71,  78]],
       [[ 71,  76]],
       [[ 70,  75]]], dtype=np.int32)
  medial_3_contour = np.array([[[ 95,  52]],
       [[ 94,  53]],
       [[ 93,  53]],
       [[ 92,  54]],
       [[ 91,  54]],
       [[ 90,  55]],
       [[ 88,  55]],
       [[ 87,  56]],
       [[ 83,  56]],
       [[ 83,  57]],
       [[ 82,  58]],
       [[ 82,  63]],
       [[ 84,  65]],
       [[ 84,  67]],
       [[ 85,  68]],
       [[ 85,  69]],
       [[ 86,  70]],
       [[ 86,  72]],
       [[ 87,  73]],
       [[ 87,  80]],
       [[ 88,  81]],
       [[ 88,  90]],
       [[ 89,  91]],
       [[ 89,  99]],
       [[ 91, 101]],
       [[100, 101]],
       [[102,  99]],
       [[103,  99]],
       [[104,  98]],
       [[106,  98]],
       [[108,  96]],
       [[108,  91]],
       [[106,  89]],
       [[106,  87]],
       [[105,  86]],
       [[105,  85]],
       [[104,  84]],
       [[104,  83]],
       [[103,  82]],
       [[103,  80]],
       [[101,  78]],
       [[101,  77]],
       [[100,  76]],
       [[100,  74]],
       [[ 99,  73]],
       [[ 99,  71]],
       [[ 98,  70]],
       [[ 98,  60]],
       [[ 99,  59]],
       [[ 99,  56]],
       [[ 98,  55]],
       [[ 98,  54]],
       [[ 96,  52]]], dtype=np.int32)
  distal_2_contour = np.array([[[55, 39]],
       [[54, 40]],
       [[53, 40]],
       [[52, 41]],
       [[52, 43]],
       [[51, 44]],
       [[51, 47]],
       [[52, 48]],
       [[52, 49]],
       [[53, 50]],
       [[53, 53]],
       [[54, 54]],
       [[54, 67]],
       [[53, 68]],
       [[53, 75]],
       [[62, 75]],
       [[63, 74]],
       [[64, 74]],
       [[65, 73]],
       [[66, 73]],
       [[67, 72]],
       [[70, 72]],
       [[70, 69]],
       [[69, 68]],
       [[69, 67]],
       [[68, 66]],
       [[68, 65]],
       [[66, 63]],
       [[66, 62]],
       [[65, 61]],
       [[65, 60]],
       [[64, 59]],
       [[64, 58]],
       [[63, 57]],
       [[63, 54]],
       [[62, 53]],
       [[62, 49]],
       [[61, 48]],
       [[61, 42]],
       [[59, 40]],
       [[57, 40]],
       [[56, 39]]], dtype=np.int32)

  candidate_contours = [
    medial_2_contour,
    medial_3_contour,
    distal_2_contour,
  ]

  correct_candidate_index = 2

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

def case_030_distal2():
  distal_phalanx_1 = ExpectedContourDistalPhalanx(
    encounter_amount=1,
  )
  distal_phalanx_1_contour = np.array([[[ 21,  87]],
       [[ 21,  88]],
       [[ 20,  89]],
       [[ 20,  93]],
       [[ 21,  94]],
       [[ 21,  97]],
       [[ 22,  98]],
       [[ 22, 101]],
       [[ 23, 102]],
       [[ 23, 105]],
       [[ 24, 106]],
       [[ 24, 107]],
       [[ 23, 108]],
       [[ 23, 113]],
       [[ 22, 114]],
       [[ 22, 116]],
       [[ 25, 116]],
       [[ 26, 115]],
       [[ 28, 115]],
       [[ 29, 114]],
       [[ 30, 114]],
       [[ 31, 113]],
       [[ 34, 113]],
       [[ 34, 112]],
       [[ 35, 111]],
       [[ 35, 110]],
       [[ 31, 106]],
       [[ 31, 105]],
       [[ 29, 103]],
       [[ 29, 101]],
       [[ 28, 100]],
       [[ 28,  91]],
       [[ 27,  90]],
       [[ 27,  88]],
       [[ 26,  87]]], dtype=np.int32)
  distal_phalanx_1.prepare(
    distal_phalanx_1_contour,
    image_width=301,
    image_height=462,
  )
  target_expected_contour = ExpectedContourDistalPhalanx(
    encounter_amount=2,
    previous_encounter=distal_phalanx_1,
  )

  medial_2_contour = np.array([[[ 74,  74]],
       [[ 73,  75]],
       [[ 71,  75]],
       [[ 70,  76]],
       [[ 67,  76]],
       [[ 66,  77]],
       [[ 62,  77]],
       [[ 62,  78]],
       [[ 61,  79]],
       [[ 61,  84]],
       [[ 62,  85]],
       [[ 62,  86]],
       [[ 64,  88]],
       [[ 64,  89]],
       [[ 65,  90]],
       [[ 65,  93]],
       [[ 66,  94]],
       [[ 66, 119]],
       [[ 67, 120]],
       [[ 78, 120]],
       [[ 79, 119]],
       [[ 80, 119]],
       [[ 81, 118]],
       [[ 82, 118]],
       [[ 83, 117]],
       [[ 84, 117]],
       [[ 85, 116]],
       [[ 86, 116]],
       [[ 86, 111]],
       [[ 85, 111]],
       [[ 84, 110]],
       [[ 84, 109]],
       [[ 82, 107]],
       [[ 82, 106]],
       [[ 81, 105]],
       [[ 81, 104]],
       [[ 80, 103]],
       [[ 80, 102]],
       [[ 79, 101]],
       [[ 79,  99]],
       [[ 78,  98]],
       [[ 78,  96]],
       [[ 77,  95]],
       [[ 77,  91]],
       [[ 76,  90]],
       [[ 76,  83]],
       [[ 75,  82]],
       [[ 75,  81]],
       [[ 76,  80]],
       [[ 77,  81]],
       [[ 76,  80]],
       [[ 77,  79]],
       [[ 76,  78]],
       [[ 76,  76]]], dtype=np.int32)
  distal_2_contour = np.array([[[61, 43]],
       [[59, 45]],
       [[59, 49]],
       [[60, 50]],
       [[60, 54]],
       [[61, 55]],
       [[61, 56]],
       [[62, 57]],
       [[62, 66]],
       [[61, 67]],
       [[61, 69]],
       [[60, 70]],
       [[60, 74]],
       [[61, 74]],
       [[62, 75]],
       [[63, 74]],
       [[69, 74]],
       [[70, 73]],
       [[72, 73]],
       [[73, 72]],
       [[75, 72]],
       [[75, 68]],
       [[72, 65]],
       [[72, 64]],
       [[70, 62]],
       [[70, 60]],
       [[69, 59]],
       [[69, 54]],
       [[68, 53]],
       [[68, 49]],
       [[69, 48]],
       [[69, 46]],
       [[68, 45]],
       [[68, 44]],
       [[67, 43]]], dtype=np.int32)
  
  candidate_contours = [
    medial_2_contour,
    distal_2_contour,
  ]

  correct_candidate_index = 1

  return target_expected_contour, candidate_contours, correct_candidate_index

def case_030_metacarpal1():
  target_expected_contour = ExpectedContourMetacarpal(
    encounter_amount=1,
  )
  metacarpal_1_contour = np.array([[[ 61, 209]],
       [[ 60, 210]],
       [[ 58, 210]],
       [[ 57, 211]],
       [[ 56, 211]],
       [[ 55, 212]],
       [[ 54, 212]],
       [[ 53, 213]],
       [[ 53, 214]],
       [[ 52, 215]],
       [[ 52, 218]],
       [[ 53, 219]],
       [[ 53, 220]],
       [[ 52, 221]],
       [[ 52, 229]],
       [[ 53, 230]],
       [[ 53, 233]],
       [[ 54, 234]],
       [[ 54, 235]],
       [[ 55, 236]],
       [[ 55, 238]],
       [[ 56, 239]],
       [[ 56, 241]],
       [[ 57, 242]],
       [[ 57, 244]],
       [[ 58, 245]],
       [[ 58, 247]],
       [[ 59, 248]],
       [[ 59, 251]],
       [[ 60, 252]],
       [[ 60, 255]],
       [[ 61, 256]],
       [[ 61, 259]],
       [[ 62, 260]],
       [[ 62, 262]],
       [[ 63, 263]],
       [[ 63, 265]],
       [[ 64, 266]],
       [[ 64, 268]],
       [[ 65, 269]],
       [[ 65, 271]],
       [[ 66, 272]],
       [[ 66, 274]],
       [[ 67, 275]],
       [[ 67, 277]],
       [[ 68, 278]],
       [[ 68, 286]],
       [[ 69, 287]],
       [[ 69, 296]],
       [[ 70, 297]],
       [[ 70, 298]],
       [[ 71, 299]],
       [[ 71, 300]],
       [[ 74, 303]],
       [[ 74, 304]],
       [[ 75, 304]],
       [[ 76, 303]],
       [[ 77, 303]],
       [[ 78, 302]],
       [[ 79, 302]],
       [[ 80, 301]],
       [[ 81, 301]],
       [[ 82, 300]],
       [[ 83, 300]],
       [[ 85, 298]],
       [[ 85, 297]],
       [[ 87, 295]],
       [[ 86, 294]],
       [[ 86, 291]],
       [[ 89, 288]],
       [[ 89, 285]],
       [[ 88, 284]],
       [[ 88, 280]],
       [[ 87, 279]],
       [[ 87, 278]],
       [[ 86, 277]],
       [[ 86, 276]],
       [[ 84, 274]],
       [[ 84, 273]],
       [[ 82, 271]],
       [[ 82, 269]],
       [[ 81, 268]],
       [[ 81, 267]],
       [[ 80, 266]],
       [[ 80, 265]],
       [[ 79, 264]],
       [[ 79, 262]],
       [[ 78, 261]],
       [[ 78, 259]],
       [[ 77, 258]],
       [[ 77, 255]],
       [[ 76, 254]],
       [[ 76, 250]],
       [[ 75, 249]],
       [[ 75, 246]],
       [[ 74, 245]],
       [[ 74, 242]],
       [[ 73, 241]],
       [[ 73, 237]],
       [[ 72, 236]],
       [[ 72, 220]],
       [[ 69, 217]],
       [[ 69, 216]],
       [[ 68, 215]],
       [[ 68, 212]],
       [[ 66, 210]],
       [[ 65, 210]],
       [[ 64, 209]]], dtype=np.int32)
  target_expected_contour.prepare(
    metacarpal_1_contour,
    image_width=301,
    image_height=462,
  )

  metacarpal_2_contour = np.array([[[ 89, 194]],
       [[ 88, 195]],
       [[ 86, 195]],
       [[ 85, 196]],
       [[ 84, 196]],
       [[ 82, 198]],
       [[ 82, 202]],
       [[ 83, 203]],
       [[ 83, 207]],
       [[ 82, 208]],
       [[ 82, 212]],
       [[ 83, 213]],
       [[ 83, 216]],
       [[ 84, 217]],
       [[ 84, 220]],
       [[ 85, 221]],
       [[ 85, 225]],
       [[ 86, 226]],
       [[ 86, 229]],
       [[ 87, 230]],
       [[ 87, 233]],
       [[ 88, 234]],
       [[ 88, 238]],
       [[ 89, 239]],
       [[ 89, 243]],
       [[ 90, 244]],
       [[ 90, 249]],
       [[ 91, 250]],
       [[ 91, 256]],
       [[ 92, 257]],
       [[ 92, 270]],
       [[ 93, 271]],
       [[ 93, 273]],
       [[ 92, 274]],
       [[ 92, 277]],
       [[ 91, 278]],
       [[ 91, 279]],
       [[ 90, 280]],
       [[ 90, 283]],
       [[ 91, 284]],
       [[ 91, 293]],
       [[ 92, 294]],
       [[ 92, 295]],
       [[ 93, 296]],
       [[ 93, 297]],
       [[ 94, 298]],
       [[ 97, 298]],
       [[ 99, 296]],
       [[102, 296]],
       [[103, 297]],
       [[106, 297]],
       [[108, 295]],
       [[108, 292]],
       [[109, 291]],
       [[109, 279]],
       [[108, 278]],
       [[108, 277]],
       [[107, 276]],
       [[107, 275]],
       [[106, 274]],
       [[106, 272]],
       [[105, 271]],
       [[105, 269]],
       [[104, 268]],
       [[104, 263]],
       [[103, 262]],
       [[103, 257]],
       [[102, 256]],
       [[102, 242]],
       [[101, 241]],
       [[101, 215]],
       [[102, 214]],
       [[102, 206]],
       [[100, 204]],
       [[100, 202]],
       [[ 99, 201]],
       [[ 99, 198]],
       [[ 98, 197]],
       [[ 98, 196]],
       [[ 97, 195]],
       [[ 95, 195]],
       [[ 94, 194]]], dtype=np.int32)
  
  candidate_contours = [
    metacarpal_2_contour,
    metacarpal_1_contour,
  ]

  correct_candidate_index = 1

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
    case_022_distal2_target,
    case_022_distal2_candidates,
    case_022_distal2_correct_candidate_index,
  ) = case_022_distal2()

  (
    case_022_distal5_target,
    case_022_distal5_candidates,
    case_022_distal5_correct_candidate_index,
  ) = case_022_distal5()

  (
    case_030_distal2_target,
    case_030_distal2_candidates,
    case_030_distal2_correct_candidate_index,
  ) = case_030_distal2()

  (
    case_030_metacarpal1_target,
    case_030_metacarpal1_candidates,
    case_030_metacarpal1_correct_candidate_index,
  ) = case_030_metacarpal1()

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
      # [
      #   case_030_metacarpal1_target,
      #   case_030_metacarpal1_candidates,
      #   case_030_metacarpal1_correct_candidate_index,
      #   'case_030_metacarpal1',
      # ]
    ],
    'distal_2': [
      [
        case_004_distal2_target,
        case_004_distal2_candidates,
        case_004_distal2_correct_candidate_index,
        'case_004_distal2',
      ],
      [
        case_022_distal2_target,
        case_022_distal2_candidates,
        case_022_distal2_correct_candidate_index,
        'case_022_distal2',
      ],
      [
        case_030_distal2_target,
        case_030_distal2_candidates,
        case_030_distal2_correct_candidate_index,
        'case_030_distal2',
      ],
      [
        case_023_distal2_target,
        case_023_distal2_candidates,
        case_023_distal2_correct_candidate_index,
        'case_023_distal2',
      ],
    ],
    # 'distal_5': [
    #   [
    #     case_022_distal5_target,
    #     case_022_distal5_candidates,
    #     case_022_distal5_correct_candidate_index,
    #     'case_022_distal5'
    #   ]
    # ]
  }

  return expected_contour_to_cases

def experiment_penalization_main(debug_mode: bool, step: int = 0.0025,
                                 range: int = 100):
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
