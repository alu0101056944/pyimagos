'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

GUI for contour work visualization:
  - Zoom the image to see a specific contour's points
  - See step history of contour modification
'''

from PIL import Image, ImageTk
from tkinter import (
  Tk, Checkbutton, Scale, HORIZONTAL, IntVar, BooleanVar,
  Canvas, NW, Y, X, Scrollbar
)

import numpy as np
import cv2 as cv

import src.rulesets.distal_phalanx as dist_phalanx

expected_contours = [
  {
    'relative_distance': (0, 0),
    'ruleset': dist_phalanx.ruleset
  }
]

def draw_contour_points(image: np.array, contours: np.array,
                        draw_points: bool = True) -> np.array:
  colored_image = np.copy(image)
  contour_colors = [(0, 0, 0) for _ in range(len(contours))]

  if draw_points:
    for i, contour in enumerate(contours):
      for point in contour:
        color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
        contour_colors[i] = color

        b, g, r = color
        x = point[0][0]
        y = point[0][1]
        cv.circle(colored_image, (x, y), radius=1, color=(b, g, r), thickness=-1)
  else:
    for i, contour in enumerate(contours):
      for point in contour:
        color = ((i + 1) * 123 % 256, (i + 1) * 456 % 256, (i + 1) * 789 % 256)
        contour_colors[i] = color


  return (colored_image, contour_colors)

def draw_contours(image: np.array, contours: np.array,
                  contour_colors: np.array) -> np.array:
  colored_image = np.copy(image)
  for i, contour_color in enumerate(contour_colors):
    cv.drawContours(colored_image, contours, i, contour_color, thickness=1)
  return colored_image

class ContourViewer:
  def __init__(self, image: np.array):
    self.input_image = image
    contours, _ = cv.findContours(self.input_image, cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)
    self.contours = contours

    self.root = Tk()
    self.root.title("Contour visualization")
    initial_width = self.input_image.shape[1] + 25
    initial_height = self.input_image.shape[0]
    self.root.geometry(f"{initial_width}x{initial_height}")

    self.zoom_level = 1.0
    self.zoom_var = IntVar(value=100)  # Percentage
    
    self.show_points = BooleanVar(value=True)
    self.draw_contours_var = BooleanVar(value=False)

    self._setup_gui()
    self.update_image()
    self.root.mainloop()

  def _setup_gui(self):
    self.canvas = Canvas(self.root, width=self.input_image.shape[1],
                         height=self.input_image.shape[0], bg="white")
    self.canvas.pack(side='left', expand="yes", fill="both")

    self.scrollbar_y = Scrollbar(self.canvas, orient="vertical",
                                 command=self.canvas.yview)
    self.scrollbar_y.pack(side="right", fill=Y)
    self.scrollbar_x = Scrollbar(self.canvas, orient="horizontal",
                                 command=self.canvas.xview)
    self.scrollbar_x.pack(side="bottom", fill=X)

    self.canvas.config(yscrollcommand=self.scrollbar_y.set,
                       xscrollcommand=self.scrollbar_x.set)

    # Zoom slider
    zoom_scale = Scale(self.root, label="Zoom %", from_=10, to=500,
                       orient=HORIZONTAL, variable=self.zoom_var,
                       command=self._change_zoom)
    zoom_scale.pack()

    points_check = Checkbutton(self.root, text="Show Points",
                               variable=self.show_points,
                               command=self.update_image)
    points_check.pack()

    contours_check = Checkbutton(self.root, text="Draw Contours",
                                 variable=self.draw_contours_var,
                                 command=self.update_image)
    contours_check.pack()

  def _change_zoom(self, zoom_value):
    self.zoom_level = int(zoom_value) / 100
    self.update_image()

  def update_image(self):
    display_image = cv.cvtColor(self.input_image, cv.COLOR_RGB2BGR)
    
    if self.show_points.get() and not self.draw_contours_var.get():
      display_image, _ = draw_contour_points(display_image, self.contours)
    elif self.draw_contours_var.get():
      display_image, contour_colors = draw_contour_points(
        display_image,
        self.contours,
        draw_points=True if self.show_points.get() else False
      )
      display_image = draw_contours(display_image, self.contours, contour_colors)

    # Scale Image
    pil_image = Image.fromarray(display_image)
    scaled_size = (int(pil_image.width * self.zoom_level),
                   int(pil_image.height * self.zoom_level))
    resized_image = pil_image.resize(scaled_size, Image.Resampling.LANCZOS)

    self.tk_image = ImageTk.PhotoImage(resized_image)
    self.canvas.create_image(0, 0, anchor=NW, image=self.tk_image)
    self.canvas.config(scrollregion=self.canvas.bbox('all'))

def visualize_contours(filename: str) -> None:
  input_image = Image.open(filename)
  input_image = np.asarray(input_image)
  gaussian_blurred = cv.GaussianBlur(input_image, (5, 5), 0)

  borders_detected = cv.Canny(gaussian_blurred, 40, 135)
  borders_detected = cv.normalize(borders_detected, None, 0, 255, cv.NORM_MINMAX,
                                  cv.CV_8U)

  visualizer = ContourViewer(borders_detected)
