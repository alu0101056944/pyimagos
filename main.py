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
from src.main_gui import execute_ui
from src.main_develop_border_analysis import borderFilterAlternativesAnalysis
from src.main_develop_contours import visualize_contours
from src.image_filters.attention_map_filter import AttentionMap

@click.group()
def cli() -> None:
  pass

@cli.command()
@click.argument('filename')
@click.option('--write_files', is_flag=True, default=False,
              help='Enable writing processed images to disk.')
@click.option('--noshow', is_flag=True, default=False,
              help='Skip image showing through new windows.')
def execute(filename: str, write_files: bool, noshow: bool) -> None:
    '''Left hand radiography segmentation.'''
    process_radiograph(filename, write_images=write_files,
                       show_images=not noshow)

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

if __name__ == '__main__':
    cli()
