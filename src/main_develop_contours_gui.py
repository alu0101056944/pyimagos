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
  Canvas, NW, Y, X, Scrollbar, Frame, Label, Button, LEFT
)

import numpy as np
import cv2 as cv

from src.contour_operations.join_contour import JoinContour

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
  def __init__(self, image: np.array, contour_history: np.ndarray):
    self.input_image = image
  
    self.contour_history = contour_history
    self.step_index = 0

    self.root = Tk()
    self.root.title("Contour visualization")
    initial_width = self.input_image.shape[1] + 250
    initial_height = self.input_image.shape[0] + 100
    self.root.geometry(f"{initial_width}x{initial_height}")

    self.zoom_level = 1.0
    self.zoom_var = IntVar(value=100)  # Percentage
    
    self.show_points = BooleanVar(value=True)
    self.draw_contours_var = BooleanVar(value=False)
    self.show_changes = BooleanVar(value=False)

    self._setup_gui()
    self.update_image()
    self.root.mainloop()

  def _setup_gui(self):
    canvas_frame = Frame(self.root)

    canvas_frame.pack(side='left', expand="yes", fill="both")

    self.canvas = Canvas(canvas_frame, width=self.input_image.shape[1],
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

    controls_frame = Frame(self.root)
    controls_frame.pack(side='right', fill=Y, padx=10)

    # Zoom slider
    zoom_scale = Scale(controls_frame, label="Zoom %", from_=10, to=500,
                       orient=HORIZONTAL, variable=self.zoom_var,
                       command=self._change_zoom)
    zoom_scale.pack(pady=5)

    points_check = Checkbutton(controls_frame, text="Show Points",
                               variable=self.show_points,
                               command=self.update_image)
    points_check.pack(pady=5)

    contours_check = Checkbutton(controls_frame, text="Draw Contours",
                                 variable=self.draw_contours_var,
                                 command=self.update_image)
    contours_check.pack(pady=5)

    contour_frame = Frame(controls_frame)
    contour_frame.pack(pady=10, fill=X)

    self.contour_info_label = Label(
      contour_frame,
      text=f"Step: {self.step_index + 1}/{len(self.contour_history)}"
    )
    self.contour_info_label.pack()

    # Step History Scale
    self.step_scale = Scale(contour_frame, from_=0,
                            to=max(0, len(self.contour_history) - 1),
                            orient=HORIZONTAL,
                            command=self._change_step)
    self.step_scale.pack(fill=X)

    button_frame = Frame(contour_frame)
    button_frame.pack(fill=X)
    
    prev_button = Button(button_frame, text="<", command=self._prev_step)
    prev_button.pack(side=LEFT)

    next_button = Button(button_frame, text=">", command=self._next_step)
    next_button.pack(side=LEFT)

    show_changes_check = Checkbutton(controls_frame, text="Highlight changes",
                              variable=self.show_changes,
                              command=self.update_image)
    show_changes_check.pack(pady=5)


  def _change_zoom(self, zoom_value):
    self.zoom_level = int(zoom_value) / 100
    self.update_image()

  def _update_contour_info(self):
    self.contour_info_label.config(
      text=f"Step: {self.step_index + 1}/{len(self.contour_history)}")
    self.step_scale.config(to=max(0, len(self.contour_history) - 1))
    self.step_scale.set(self.step_index)

  def _change_step(self, step_value: int):
    self.step_index = int(step_value)
    self._update_contour_info()
    self.update_image()

  def _prev_step(self):
    if self.step_index > 0:
      self.step_index -= 1
      self._update_contour_info()
      self.update_image()

  def _next_step(self):
    if self.step_index < len(self.contour_history) - 1:
      self.step_index += 1
      self._update_contour_info()
      self.update_image()

  def update_image(self):
    display_image = cv.cvtColor(self.input_image, cv.COLOR_RGB2BGR)
    contours = self.contour_history[self.step_index]

    if self.show_points.get() and not self.draw_contours_var.get():
      display_image, _ = draw_contour_points(display_image, contours)
    elif self.draw_contours_var.get():
      display_image, contour_colors = draw_contour_points(
        display_image,
        contours,
        draw_points=True if self.show_points.get() else False
      )
      display_image = draw_contours(display_image, contours, contour_colors)

    if self.show_changes.get() and len(self.contour_history) > 1 and (
      self.step_index > 0
    ):
      def __to_structured(a: np.array):
        '''So that each row is treated as a single element so that np.isin() 
        works correctly'''
        points_array = np.dtype(list([('x', a.dtype), ('y', a.dtype)]))
        return a.view(points_array)

      def __set_difference(a: np.array, b: np.array):
        a1 = __to_structured(a)
        a2 = __to_structured(b)
        mask = ~np.isin(a1, a2)
        mask = np.reshape(np.stack([mask, mask], axis=-1), (-1, 2))
        return np.reshape(a[mask], (-1, 2))

      original = np.concatenate([np.reshape(contour, (-1, 2)) for contour in contours],
                                axis=0)
      previous_contours = self.contour_history[self.step_index - 1]
      new = np.concatenate([np.reshape(contour, (-1, 2)) for contour in previous_contours],
                                axis=0)
      added_points = __set_difference(original, new)
      removed_points = __set_difference(new, original)

      for added_point in added_points:
        (x, y) = added_point
        cv.circle(display_image, (x, y), 1, (0, 255, 0), 6)
      for removed_point in removed_points:
        x, y = removed_point
        cv.circle(display_image, (x, y), 1, (255, 0, 0), 6)

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

  contours, _ = cv.findContours(borders_detected, cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)

  # contours2 = [np.copy(contour) for contour in contours]
  # contours2[0] = contours2[0][:-25]

  # box_contour = [np.array([[[4, 4]], [[20, 20]], [[4, 20]], [[20, 4]], [[20, 8]],
  #                          [[24, 8]], [[24, 4]]])]

  joinOperation = JoinContour(0, 1)
  contours2 = joinOperation.generate_new_contour(contours)
  joinOperation = JoinContour(0, 2)
  contours3 = joinOperation.generate_new_contour(contours2)

  visualizer = ContourViewer(borders_detected, [contours, contours2, contours3])
