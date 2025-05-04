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
import torchvision.transforms as transforms

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
from src.main_experiment import main_experiment
from src.main_develop_criteria_study import test_criteria_parameters
from src.image_filters.contrast_enhancement import ContrastEnhancement
from src.main_study_cpu_scale_factor import execute_resize_study
from src.main_canny_study import make_composition
from src.main_print_positional_differences import positional_differences_main
from src.main_print_shape_differences import shape_differences_main
from src.main_develop_show_segment_tags import visualize_tags_main
from src.main_experiment_positions import write_position_experiment
from src.main_experiment_shape import write_shape_experiment
from src.main_experiment_penalization import experiment_penalization_main

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
@click.option('--all', is_flag=True, default=False,
              help='Print the exact estimated age and the measurement ' \
                'dictionary.')
@click.option('--gpu', is_flag=True, default=False,
              help='Use gpu for the contrast enchancement step, which uses a' \
                'vision transformer')
@click.option('--noresize', is_flag=True, default=False,
              help='Do not resize the image before applying contrast ' \
                'enhancement.')
@click.option('--input2',
              type=click.Path(
                exists=True, file_okay=True, dir_okay=False, writable=True
              ),
              help='Input image to use on second stage when nofilter is set.' \
                'Fallsback to using input image 1.')
def execute(filename: str, write_files: bool, show: bool,
            nofilter: bool, all: bool, gpu: bool, noresize: bool,
            input2: click.Path) -> None:
  '''Left hand radiography age estimation.'''
  
  process_radiograph(
    filename,
    write_images=write_files,
    show_images=show,
    nofilter=nofilter,
    all=all,
    use_cpu=not gpu,
    noresize=noresize,
    input_image_2=input2
  )

@cli.command()
def estimate() -> None:
  '''Age estimation test from ideal image.'''
  estimate_age_from_ideal_contour()

@cli.command()
@click.option('--single', is_flag=True, default=False,
              help='Print experiment result for only one group.')
@click.option('--nofilter', is_flag=True, default=False,
              help='Skip image processing and use the input image directly.')
@click.option('--group17_5',
              type=click.Path(
                exists=True, file_okay=False, dir_okay=True, writable=True
              ),
              help='Folder path with radiographies for group of age 17.5.')
@click.option('--group18_5',
              type=click.Path(
                exists=True, file_okay=False, dir_okay=True, writable=True
              ),
              help='Folder path with radiographies for group of age 18.5.')
@click.option('--group19_5',
              type=click.Path(
                exists=True, file_okay=False, dir_okay=True, writable=True
              ),
              help='Folder path with radiographies for group of age 19.5.')
@click.option('--groupcontrol',
              type=click.Path(
                exists=True, file_okay=False, dir_okay=True, writable=True
              ),
              help='Folder path with radiographies for control group.')
@click.option('--gpu', is_flag=True, default=False,
              help='Use gpu for the contrast enchancement step, which uses a' \
                'vision transformer')
@click.option('--noresize', is_flag=True, default=False,
              help='Do not resize the image before applying contrast ' \
                'enhancement.')
@click.option('--useinput2', is_flag=True, default=False,
              help='To search for a x_stage_2.jpg image per image to use for' \
                ' the second stage search')
@click.option('--use_starts_file', is_flag=True, default=False,
              help='To look for a .json with the index of the start contour.' + (
                'The search will use that as starting contour.'
              ))
def experiment(nofilter: bool, single: bool, group17_5: str, group18_5: str,
               group19_5: str, groupcontrol: str, gpu: bool,
               noresize: bool, useinput2: bool, use_starts_file: bool) -> None:
  '''Estimate age and show measurement fit for three different groups and
      a control group. If --single option was not used then four options
      group17_5, group18_5 and group19_5, groupcontrol with the folder
      paths are required. Otherwise just a single group is required (will
       fail if passed more than one)'''
  main_experiment(single, group17_5, group18_5, group19_5, groupcontrol,
                  nofilter, use_cpu=not gpu, noresize=noresize,
                  useinput2=useinput2, use_starts_file=use_starts_file)

@cli.command()
@click.argument('filename')
@click.argument('outputfilename')
@click.option('--nofilter', is_flag=True, default=False,
              help='Skip image processing and use the input image directly.')
@click.option('--gpu', is_flag=True, default=False,
              help='Use gpu for the contrast enchancement step, which uses a' \
                'vision transformer')
@click.option('--noresize', is_flag=True, default=False,
              help='Do not resize the image before applying contrast ' \
                'enhancement.')
def criteria_study(nofilter: bool, filename: str, outputfilename: str,
                   gpu: bool, noresize: bool):
  '''Estimate age of a given radiography image file with different criteria
  parameter variations.'''
  test_criteria_parameters(filename, outputfilename, nofilter, use_cpu=not gpu,
                           noresize=noresize)

@cli.group()
def develop() -> None:
   '''Developer-focused commands'''
   pass

# TODO add gpu flag to remaining commands, like attmap, filters_gui

@develop.command()
@click.argument('filename')
@click.option('--write_files', is_flag=True, default=False,
              help='Enable writing processed images to disk.')
@click.option('--noshow', is_flag=True, default=False,
              help='Skip image showing through new windows.')
@click.option('--noresize', is_flag=True, default=False,
help='Do not resize the image before applying contrast ' \
  'enhancement.')
def attmap(filename: str, write_files: bool, noshow: bool,
            noresize: bool) -> None:
  '''Show the attention map of the given image.'''
  input_image = None
  try:
    with Image.open(filename) as image:
      if image.mode == 'L':
        image = image.convert('RGB')
        input_image = np.array(image)
      elif image.mode == 'RGB':
        input_image = np.array(image)
  except Exception as e:
    print(f"Error opening image {filename}: {e}")
    raise
  input_image = transforms.ToTensor()(input_image)
  attention_map = AttentionMap(noresize=noresize).process(input_image)
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
  '''Image visualization of the execute tests (second stage search)'''
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
  try:
    with Image.open(filename) as imagefile:
      if imagefile.mode == 'L':
        imagefile = imagefile.convert('RGB')
        input_image = np.array(imagefile)
      elif imagefile.mode == 'RGB':
        input_image = np.array(imagefile)
  except Exception as e:
    print(f"Error opening image {filename}: {e}")
    raise

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
  try:
    with Image.open(filename) as imagefile:
      if imagefile.mode == 'L':
        imagefile = imagefile.convert('RGB')
        input_image = np.array(imagefile)
      elif imagefile.mode == 'RGB':
        input_image = np.array(imagefile)
  except Exception as e:
    print(f"Error opening image {filename}: {e}")
    raise

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
  try:
    with Image.open(filename) as imagefile:
      if imagefile.mode == 'L':
        imagefile = imagefile.convert('RGB')
        input_image = np.array(imagefile)
      elif imagefile.mode == 'RGB':
        input_image = np.array(imagefile)
  except Exception as e:
    print(f"Error opening image {filename}: {e}")
    raise

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

@develop.command()
@click.argument("filename")
@click.option('--lower_thresh',
              help='Lower threshold for the canny binarization')
@click.option('--higher_thresh',
              help='Higher threshold for the canny binarization')
@click.option('--gpu', is_flag=True, default=False,
              help='Use gpu for the contrast enchancement step, which uses a' \
                'vision transformer')
@click.option('--noresize', is_flag=True, default=False,
              help='Do not resize the image before applying contrast ' \
                'enhancement.')
def canny(filename: str, lower_thresh: str, higher_thresh: str, gpu: bool,
          noresize: bool):
  '''Process the radiography image file to border detect it.'''

  if lower_thresh is None or higher_thresh is None:
    raise ValueError('Missing either lower_threshold or higher_threshold option')

  input_image = None
  try:
    with Image.open(filename) as image:
      if image.mode == 'L':
        image = image.convert('RGB')
        input_image = np.array(image)
      elif image.mode == 'RGB':
        input_image = np.array(image)
  except Exception as e:
    print(f"Error opening image {filename}: {e}")
    raise

  input_image = transforms.ToTensor()(input_image)
  he_enchanced = ContrastEnhancement(
    use_cpu=not gpu,
    noresize=noresize
  ).process(input_image)
  he_enchanced = cv.normalize(he_enchanced, None, 0, 255, cv.NORM_MINMAX,
                              cv.CV_8U)

  # gaussian_blurred = cv.GaussianBlur(he_enchanced, (3, 3), 0)

  scaled_image = cv.resize(
    he_enchanced,
    (0, 0),
    fx=0.3,
    fy=0.3,
    interpolation=cv.INTER_AREA
  )

  borders_detected = cv.Canny(scaled_image, float(lower_thresh),
                              float(higher_thresh))
  
  borders_detected = cv.normalize(borders_detected, None, 0, 255,
                                    cv.NORM_MINMAX, cv.CV_8U)

  cv.imwrite('canny_output.jpg', borders_detected)
  print('Sucesfully written image canny_output.jpg')

@develop.command()
def study_resize():
  '''Show contrast enhancement execution times for image sizes 360p,
  480p and 720p.'''
  execute_resize_study()

@develop.command()
@click.argument('filename')
def study_canny(filename: str):
  '''Write images to be able to be able to study the best combination of
  filters such that all the fingers are fully present.'''
  make_composition(filename)

@develop.command()
@click.argument('filename')
def validate_contours(filename: str):
  '''Count contour amount'''
  image = None
  try:
    with Image.open(filename) as imagefile:
      if imagefile.mode == 'L':
        imagefile = imagefile.convert('RGB')
        image = np.array(imagefile)
      elif imagefile.mode == 'RGB':
        image = np.array(imagefile)
  except Exception as e:
    print(f"Error opening image {filename}: {e}")
    raise

  image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  _, thresholded = cv.threshold(image, 40, 255, cv.THRESH_BINARY)
  contours, _ = cv.findContours(
    thresholded,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  )

  # TODO remove these two commented parts meant for debug

  # image = np.zeros((thresholded.shape[0], thresholded.shape[1], 3), dtype=np.uint8)
  # for i, contour in enumerate(contours):
  #   color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
  #   if len(contour) > 0:
  #     cv.drawContours(image, contours, i, color, 1)

  # fig = plt.figure()
  # plt.imshow(image)
  # plt.title('title')
  # plt.axis('off')
  # fig.canvas.manager.set_window_title('title')
  # plt.show()

  print('Contour amount:')
  print(len(contours))

@develop.command()
def positional_differences():
  '''Calculates the positional differences for a bunch of radiography countour
  cases based on real radiographies.'''
  positional_differences_main()

@develop.command()
def shape_differences():
  '''Calculates the shape differences for a bunch of radiography countour
  cases based on real radiographies.'''
  shape_differences_main()

@develop.command()
def visualize_tags():
  '''Visualize the segment tags of a bunch of radiographies.'''
  visualize_tags_main()

@develop.command()
def experiment_positions():
  '''Given a set of clean contour radiographies, calculate per position
  restriction the furthest distance into the wrong side globally (all
  radiographies).'''
  write_position_experiment()

@develop.command()
@click.option('--debug', is_flag=True, default=False,
              help='Don\'t write the results to the output file.')
def experiment_shapes(debug: bool):
  '''Given a set of clean contour radiographies, calculate per position
  restriction the furthest distance into the wrong side globally (all
  radiographies).'''
  write_shape_experiment(debug)

@develop.command()
@click.option('--debug', is_flag=True, default=False,
              help='Don\'t write the results to the output file.')
def experiment_penalization(debug: bool):
  '''Given a set of position penalization factor values calculate the precision
  for especific cases where the wrong contour was chosen.'''
  experiment_penalization_main(debug)

if __name__ == '__main__':
    cli()
