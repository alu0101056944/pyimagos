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
@click.option('--nosearch', is_flag=True, default=False,
              help='Skip image contours search.')
@click.option('--historygui', is_flag=True, default=False,
              help='Open history gui to show best contours.')
def execute(filename: str, write_files: bool, show: bool,
            nofilter: bool, nosearch: bool, historygui: bool) -> None:
  '''Left hand radiography segmentation.'''
  process_radiograph(filename, write_images=write_files,
                      show_images=show, nofilter=nofilter,
                      nosearch=nosearch, historygui=historygui)

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

if __name__ == '__main__':
    cli()
