'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

GUI for parameter experimenting
'''

from typing import Union
import os.path
from PIL import Image, ImageTk

import torch
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import ttk, filedialog

from src.image_filters.image_filter import ImageFilter
from src.image_filters.contrast_enhancement import ContrastEnhancement
from src.image_filters.attention_map_filter import AttentionMap
from src.image_filters.gaussian_filter import GaussianBlurFilter
from src.image_filters.border_detection_statistical_range import (
  BorderDetectionStatisticalRange
)

class NoFilter(ImageFilter):
   def process(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
       return image
   def get_name(self) -> str:
      return "None"
   def get_params(self) -> dict:
        return {}

available_filters = {
    "GaussianBlurFilter": {
      "class": GaussianBlurFilter,
      "params": {
        "kernel_size": {"type": "int", "min": 1, "max": 11, "default": 3, "step": 2},
        "sigma": {"type": "float", "min": 0.1, "max": 5.0, "default": 1, "step": 0.1},
      }
    },
    "ContrastEnhancement": {
        "class": ContrastEnhancement,
        "params": {}
    },
    "AttentionMap": {
        "class": AttentionMap,
        "params": {}
    },
    "BorderDetectionStatisticalRange": {
        "class": BorderDetectionStatisticalRange,
        "params": {
            "padding": {"type": "int", "min": 1, "max": 11, "default": 2, "step": 1},
         }
    },
}

class ImagePipeline:
    def __init__(self):
        self.steps = []  # list of ImageFilter objects

    def add_step(self, step: ImageFilter, index=None):
        if index is None:
            self.steps.append(step)
        else:
            self.steps.insert(index, step)

    def remove_step(self, index):
       if 0 <= index < len(self.steps):
           del self.steps[index]

    def process_image(self, image: np.ndarray) -> np.ndarray:
       processed_image = image.copy()
       index = 0
       while index < len(self.steps):
        try:
          processed_image = self.steps[index].process(processed_image)
          index += 1
        except Exception as e:
          print(f'Error processing image filter step {index}, step removed.')
          print('Original error:')
          print(e)
          del self.steps[index]
       return processed_image

    def get_step(self, index):
        if 0 <= index < len(self.steps):
            return self.steps[index]

        return None

    def get_steps_names(self):
        names = []
        for step in self.steps:
           names.append(step.get_name())
        return names

    def __str__(self):
         step_strings = [str(step) for step in self.steps]
         return "\n".join(step_strings)


class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filter App")

        self.pipeline = ImagePipeline()
        self.image_path = None
        self.original_image = None
        self.processed_image = None

        self.create_widgets()
        self.load_default_image()


    def load_default_image(self):
      default_image_path = "docs/radiography.jpg"
      if os.path.exists(default_image_path):
        self.image_path = default_image_path
        self.load_image_from_path()
        self.update_image_display()

    def create_widgets(self):
        # --- Image Frame ---
        image_frame = ttk.Frame(self.root)
        image_frame.pack(side=tk.LEFT, padx=10, pady=10)

        #Image load label
        load_image_button = ttk.Button(image_frame, text="Load Image", command=self.load_image_dialog)
        load_image_button.pack(pady=5)

        self.image_label = ttk.Label(image_frame)
        self.image_label.pack()

        # --- Filter Frame ---
        filter_frame = ttk.Frame(self.root)
        filter_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Filter sequence list
        filter_list_label = ttk.Label(filter_frame, text="Applied Filters:")
        filter_list_label.pack()

        self.filter_list = tk.Listbox(filter_frame, width=40)
        self.filter_list.pack(fill=tk.BOTH, expand=True)

        filter_list_scroll = ttk.Scrollbar(filter_frame, orient=tk.VERTICAL, command=self.filter_list.yview)
        filter_list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.filter_list.config(yscrollcommand=filter_list_scroll.set)

        # Add Filter button
        add_filter_button = ttk.Button(filter_frame, text="Add Filter", command=self.add_filter_dialog)
        add_filter_button.pack(pady=5)


    def load_image_dialog(self):
      file_path = filedialog.askopenfilename(title="Select Image", filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")))
      if file_path:
        self.image_path = file_path
        self.load_image_from_path()
        self.update_image_display()

    def load_image_from_path(self):
        self.original_image = cv.imread(self.image_path, cv.IMREAD_GRAYSCALE)
        if self.original_image is None:
            tk.messagebox.showerror("Error", "Failed to load image.")
            self.original_image = None
            self.processed_image = None
        else:
            self.processed_image = self.original_image.copy()

    def update_image_display(self):
        if self.processed_image is None:
            self.image_label.config(image = None)
            self.image_label.image = None
            return

        processed_image_copy = self.processed_image.copy()
        try:
          processed_image_copy = self.pipeline.process_image(processed_image_copy)
        except:
           print('An image filter step resulted in error, updating filter list.')
           self.update_filter_list()
        finally:
          img = Image.fromarray(processed_image_copy)
          img = ImageTk.PhotoImage(img)
          self.image_label.config(image=img)
          self.image_label.image = img  # keep a reference!
        


    def add_filter_dialog(self):
        add_dialog = tk.Toplevel(self.root)
        add_dialog.title("Add Filter")

        # --- Selection ---
        filter_type_label = ttk.Label(add_dialog, text="Filter Type:")
        filter_type_label.pack()

        filter_names = list(available_filters.keys())
        self.filter_selected = tk.StringVar(add_dialog)
        self.filter_selected.set(filter_names[0])

        filter_dropdown = ttk.OptionMenu(add_dialog, self.filter_selected, *filter_names)
        filter_dropdown.pack(pady=5)

         # --- Index selector ---
        index_label = ttk.Label(add_dialog, text="Index")
        index_label.pack()

        amount_of_applied_filters = max(len(self.pipeline.get_steps_names()) - 1, 0)
        index_spinbox = tk.Spinbox(add_dialog, from_=0, to=len(self.pipeline.steps),
                                   width=5,
                                   textvariable=tk.StringVar(value=amount_of_applied_filters))
        index_spinbox.pack()


        def on_add_button():
            selected_filter_name = self.filter_selected.get()
            selected_index = int(index_spinbox.get())

            if selected_filter_name != 'None':
              filter_class = available_filters[selected_filter_name]['class']
              filter_params = available_filters[selected_filter_name]['params']
              filter_dialog = tk.Toplevel(add_dialog)
              filter_dialog.title("Add Filter Parameters")

              params = {}
              param_widgets = {}

              def on_params_button():
                 for name, widget in param_widgets.items():
                   value = widget.get()
                   if available_filters[selected_filter_name]['params'][name]['type'] == "int":
                     value = int(value)
                   elif available_filters[selected_filter_name]['params'][name]['type'] == "float":
                       value = float(value)
                   params[name] = value
                 filter_instance = filter_class(**params)
                 self.pipeline.add_step(filter_instance, selected_index)
                 self.update_filter_list()
                 self.update_image_display()
                 filter_dialog.destroy()
                 add_dialog.destroy()

              for name, data in filter_params.items():
                 label = ttk.Label(filter_dialog, text=name)
                 label.pack()
                 if data['type'] == 'int':
                   widget = tk.Spinbox(filter_dialog, from_= data['min'], to=data['max'], increment=data.get('step',1))
                   widget.delete(0,"end")
                   widget.insert(0, data.get('default', 0))
                 elif data['type'] == 'float':
                   widget = tk.Spinbox(filter_dialog, from_= data['min'], to=data['max'], increment=data.get('step',0.1))
                   widget.delete(0,"end")
                   widget.insert(0, data.get('default', 0.0))
                 else:
                     widget = ttk.Entry(filter_dialog)
                 widget.pack()
                 param_widgets[name] = widget
                 param_widgets[name]

              ok_button = ttk.Button(filter_dialog, text="Add", command=on_params_button)
              ok_button.pack(pady=5)
              filter_dialog.wait_window()

            else:
                filter_instance = available_filters['None']['class']()
                self.pipeline.add_step(filter_instance, selected_index)
                self.update_filter_list()
                self.update_image_display()
                add_dialog.destroy()

        ok_button = ttk.Button(add_dialog, text="OK", command=on_add_button)
        ok_button.pack(pady=5)



    def update_filter_list(self):
       self.filter_list.delete(0, tk.END) # remove all elements
       for name in self.pipeline.get_steps_names():
         self.filter_list.insert(tk.END, name)

def execute_ui(filename: str) -> None:
  root = tk.Tk()
  app = ImageFilterApp(root)
  root.mainloop()
