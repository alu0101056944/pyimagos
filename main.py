#! /usr/bin/env python3

'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Entrypoint. Handles the CLI.
'''

__authors__ = ["Marcos Jesús Barrios Lorenzo"]
__date__ = "2024/12/02"

import os.path
from PIL import Image

import cv2 as cv
from PIL import Image
import click
import numpy as np

from src.main_execute import process_radiograph
from src.main_estimate_ideal import estimate_age_from_ideal_contour
from src.main_develop_filter_gui import execute_ui
from src.main_develop_border_analysis import borderFilterAlternativesAnalysis
from src.main_develop_contours_gui import visualize_contours
from src.image_filters.attention_map_filter import AttentionMap
from src.main_develop_test_intersection import visualize_tests_intersection
from src.main_develop_test_normals import visualize_tests_normals
from src.main_develop_test_join import visualize_tests_join
from src.main_develop_test_cut import visualize_tests_cut
from src.main_develop_test_extend import visualize_tests_extend
from src.main_develop_corner_order import visualize_topleft_corner
from src.main_develop_find_contour_corner import find_contour_corner
from src.main_develop_find_sesamoid import find_sesamoid_main
from src.main_develop_test_distal_phalanx import visualize_distal_phalanx_shape
from src.main_develop_test_medial_phalanx import visualize_medial_phalanx_shape
from src.main_develop_test_proximal_phalanx import visualize_proximal_phalanx_shape
from src.main_develop_test_metacarpal import visualize_metacarpal_shape
from src.main_develop_test_radius import visualize_radius_shape
from src.main_develop_test_ulna import visualize_ulna_shape
from src.main_develop_test_sesamoid import visualize_sesamoid_shape
from src.main_develop_test_execute import visualize_execute_tests
from src.main_develop_test_search_two import visualize_execute_tests_2
from src.main_develop_measurement_match import visualize_shape_match
from src.main_develop_test_measurement_match_fourier import (
  visualize_shape_match_fourier
)

@click.group()
def cli() -> None:
  pass

@cli.command()
@click.argument('filename')
@click.option('--write_files', is_flag=True, default=False,
              help='Enable writing processed images to disk.')
@click.option('--show', is_flag=True, default=False,
              help='Show last image result. Opens a window.')
@click.option('--nofilter', is_flag=True, default=False,
              help='Skip image processing into borders detected.')
@click.option('--historygui', is_flag=True, default=False,
              help='Open history gui to show best contours.')
def execute(filename: str, write_files: bool, show: bool,
            nofilter: bool, historygui: bool) -> None:
  '''Left hand radiography segmentation.'''
  process_radiograph(
    filename,
    write_images=write_files,
    show_images=show,
    nofilter=nofilter,
    historygui=historygui
  )

@cli.command()
def estimate() -> None:
  '''Age estimation test from ideal image.'''
  estimate_age_from_ideal_contour()

@cli.group()
def develop() -> None:
   '''Developer-focused commands'''
   pass

@develop.command()
@click.argument('filename')
@click.option('--write_files', is_flag=True, default=False,
              help='Enable writing processed images to disk.')
@click.option('--noshow', is_flag=True, default=False,
              help='Skip image showing through new windows.')
def attmap(filename: str, write_files: bool, noshow: bool) -> None:
  '''Show the attention map of the given image.'''
  input_image = Image.open(filename)
  attention_map = AttentionMap().process(input_image)
  if write_files:
    cv.imwrite(f'docs/local_images/{os.path.basename(filename)}_attention_map.jpg',
               attention_map)
  else:
     if not noshow:
        cv.imshow(f'docs/local_images/{os.path.basename(filename)}_attention_map.jpg',
                  attention_map)
        cv.waitKey(0)
        cv.destroyAllWindows()

@develop.command()
@click.argument('filename')
def filters_gui(filename: str) -> None:
  '''GUI for filter application to an image.'''
  execute_ui(filename)

@develop.command()
@click.argument("filename")
@click.option('--write_files', is_flag=True, default=False,
              help='Enable writing processed images to disk.')
@click.option('--noshow', is_flag=True, default=False,
              help='Skip image showing through new windows.')
def border_analysis(filename: str, write_files: bool, noshow: bool) -> None:
  '''Show/write images border detection intermediate results.'''
  borderFilterAlternativesAnalysis(filename, write_images=write_files,
                       show_images=not noshow)

@develop.command()
@click.argument("filename")
def contours_gui(filename: str) -> None:
  '''Open a GUI that allows visualizing contours and step history.'''
  visualize_contours(filename)

@develop.command()
def test_intersection() -> None:
  '''Show image representations for each segment intersection test.'''
  visualize_tests_intersection()

@develop.command()
def test_normals() -> None:
  '''Show image representations for each normal test.'''
  visualize_tests_normals()

@develop.command()
def test_join() -> None:
  '''Show image representations for each join contour operation test.'''
  visualize_tests_join()

@develop.command()
def test_cut() -> None:
  '''Show image representations for each cut contour operation test.'''
  visualize_tests_cut()

@develop.command()
def test_extend() -> None:
  '''Show image representations for each extend contour operation test.'''
  visualize_tests_extend()

@develop.command()
def corner_order() -> None:
  '''Show image representations showing top lest corner on different
  rectangle orientations.''' 
  visualize_topleft_corner()

@develop.command()
def find_corner() -> None:
  '''Show image representations for corner match on specific bone contour.''' 
  find_contour_corner()

@develop.command()
def find_sesamoid() -> None:
  '''Show image representations for sesamoid search near fifth metacarpal.''' 
  find_sesamoid_main()

@develop.group()
def shape() -> None:
  '''Image representations for the different shapes.''' 
  pass

@shape.command()
def distal_phalanx():
  '''Image visualization of the distal phalanx 1 (leftmost) shape'''
  visualize_distal_phalanx_shape()

@shape.command()
def medial_phalanx():
  '''Image visualization of the medial phalanx 1 (leftmost) shape'''
  visualize_medial_phalanx_shape()

@shape.command()
def proximal_phalanx():
  '''Image visualization of the proximal phalanx 1 (leftmost) shape'''
  visualize_proximal_phalanx_shape()

@shape.command()
def metacarpal():
  '''Image visualization of the metacarpal 1 (leftmost) shape'''
  visualize_metacarpal_shape()

@shape.command()
def radius():
  '''Image visualization of the radius shape'''
  visualize_radius_shape()

@shape.command()
def ulna():
  '''Image visualization of the ulna shape'''
  visualize_ulna_shape()

@shape.command()
def sesamoid():
  '''Image visualization of the sesamoid shape'''
  visualize_sesamoid_shape()

@develop.command()
def test_execute():
  '''Image visualization of the execute tests'''
  visualize_execute_tests()

@develop.command()
def test_execute_2():
  '''Image visualization of the execute tests 2'''
  visualize_execute_tests_2()

@develop.command()
def test_shape_match():
  '''Image visualization of the execute tests 2'''
  visualize_shape_match()

@develop.command()
def test_shape_match_fourier():
  '''Image visualization of the execute tests 2'''
  visualize_shape_match_fourier()

@develop.command()
@click.argument("filename")
@click.option('--write_file', is_flag=True, default=False,
              help='Output to a file instead of to console.')
def contour(filename: str, write_file: bool):
  '''Given a binary image, print its contour.'''
  input_image = Image.open(filename)
  borders_detected = np.array(input_image)
  borders_detected = cv.cvtColor(borders_detected, cv.COLOR_RGB2GRAY)
  _, thresholded = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(
    thresholded,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  )

  if not contours:
    output_string = "No contours found in the image."
  else:
    output_string = "Contours found in the image:\n"
    for contour in contours:
      output_string += str(contour) + "\n"

  if write_file:
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    output_filename = f"{base_filename}_contours.txt"
    with open(output_filename, 'w') as f:
      f.write(output_string)
    click.echo(f"Contours written to: {output_filename}")
  else:
    click.echo(output_string)

@develop.command()
@click.argument("filename")
def hu_moments(filename: str):
  '''Given a binary image, print its hu moments.'''
  input_image = Image.open(filename)
  borders_detected = np.array(input_image)
  borders_detected = cv.cvtColor(borders_detected, cv.COLOR_RGB2GRAY)
  _, thresholded = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(
    thresholded,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  )

  moments = cv.moments(contours[0])
  hu_moments = cv.HuMoments(moments)
  hu_moments = np.absolute(hu_moments)
  hu_moments_no_zeros = np.where( # to avoid DivideByZero
    hu_moments == 0,
    np.finfo(float).eps,
    hu_moments
  )
  hu_moments = (np.log10(hu_moments_no_zeros)).flatten()

  print('Hu moments:')
  print(hu_moments)

@develop.command()
@click.argument("filename")
def fourier(filename: str):
  '''Given a binary image, print the fourier transforms of each contour in it.
  10 descriptors.'''
  input_image = Image.open(filename)
  borders_detected = np.array(input_image)
  borders_detected = cv.cvtColor(borders_detected, cv.COLOR_RGB2GRAY)
  _, thresholded = cv.threshold(borders_detected, 40, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(
    thresholded,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  )

  descriptors = []
  for contour in contours:
    contour = contour.astype(np.float32)
    contour = contour.reshape((-1, 2))
    complex_contour = contour[:, 0] + 1j * contour[:, 1]
    
    # Calculate Discrete Fourier Transform
    fourier = np.fft.fft(complex_contour)
    
    fourier[0] = 0  # Remove translation information
    magnitudes = np.abs(fourier)
    normalized = magnitudes / np.max(magnitudes[1:])  # Scale invariance done

    sliced = normalized[1:10 + 1]
    descriptors.append(sliced)
    
  print('Fourier descriptors (No DC component (traslacion info)):')
  print(descriptors)

if __name__ == '__main__':
    cli()
