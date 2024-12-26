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

from src.image_processing import process_radiograph
from src.border_experiment import execute_ui
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
def process(filename: str, write_files: bool, noshow: bool) -> None:
    process_radiograph(filename, write_images=write_files,
                       show_images=not noshow)

@cli.command()
@click.argument('filename')
def experiment(filename: str) -> None:
  execute_ui(filename)

@cli.command()
@click.argument('filename')
def attmap(filename: str) -> None:
  inputImage = Image.open(filename)
  attentionMap = AttentionMap().process(inputImage)
  cv.imwrite(f'docs/local_images/{os.path.basename(filename)}_attention_map.jpg',
             attentionMap)
  

if __name__ == '__main__':
    cli()
