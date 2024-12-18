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

import click

from src.image_processing import process_radiograph
from src.border_experiment import execute_ui

@click.group()
def cli() -> None:
  pass

@cli.command()
@click.argument('filename')
def process(filename: str) -> None:
  process_radiograph(filename)

@cli.command()
@click.argument('filename')
def experiment(filename: str) -> None:
  execute_ui(filename)

if __name__ == '__main__':
    cli()
