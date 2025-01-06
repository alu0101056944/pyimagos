'''
Universidad de La Laguna
Máster en Ingeniería Informática
Trabajo de Final de Máster
Pyimagos development

Visualization of corner orientation calculation to inform a decision on how to
decide which corner is the top left corner of the bounding box of a contour.
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def get_bounding_box_corners(contour):
  rect = cv.minAreaRect(contour)
  box = cv.boxPoints(rect)
  box = np.int32(box)  # Convert to int
  return box

def get_top_left_corner(bounding_box_corners : list,
                        image_width: int, image_height: int):
  x_coords = bounding_box_corners[:, 0]
  y_coords = bounding_box_corners[:, 1]

  # Calculate distances from point i to next point in the box
  points_a = bounding_box_corners
  points_b = np.roll(bounding_box_corners, 1, axis=0)
  line_distances = np.sqrt(np.sum((points_a - points_b) ** 2, axis=1))
  
  if len(np.unique(line_distances)) == 1: # square
    x1, y1 = points_a[0]
    x2, y2 = points_b[0]
    if x1 == x2: # vertical line, so no slope and cannot use line equation
      top_left_index = np.argmin(np.sqrt((x_coords ** 2) + (y_coords ** 2)))
      top_left_corner = bounding_box_corners[top_left_index]
      return top_left_corner, top_left_index
    else:
      m = (y2 - y1) / (x2 - x1)
      if m == 0:
        top_left_index = np.argmin(np.sqrt((x_coords ** 2) + (y_coords ** 2)))
        top_left_corner = bounding_box_corners[top_left_index]
        return top_left_corner, top_left_index
      else:
        closest_distance_to_x_cero_index = np.argmin(x_coords)
        closest_distance_to_x_cero = x_coords[closest_distance_to_x_cero_index]
        closest_distance_to_x_max_index = np.argmax(image_width - x_coords)
        closest_distance_to_x_max = x_coords[closest_distance_to_x_max_index]

        # Here a problem becomes obvious: approximating top left corner based
        # on the position of the contour in the image is a bad idea because
        # the finger can be pointing towards the left but if it's position
        # is closest to the right then it will be assumed that the finger is 
        # ponting towards the right.
        if closest_distance_to_x_cero <= closest_distance_to_x_max:
          return (
            bounding_box_corners[closest_distance_to_x_cero_index],
            closest_distance_to_x_cero_index
          )
        else:
          closest_distance_to_y_cero_index = np.argmin(y_coords)
          closest_distance_to_y_cero = y_coords[closest_distance_to_y_cero_index]
          return (
            bounding_box_corners[closest_distance_to_y_cero_index],
            closest_distance_to_y_cero
          )
  else: # rectangle
    small_side_index = np.argmin(line_distances)

    x1, y1 = points_a[small_side_index]
    x2, y2 = points_b[small_side_index]

    if x1 == x2: # vertical line, so no slope and cannot use line equation
      closest_distance_to_x_cero_index = np.argmin(x_coords)
      closest_distance_to_x_cero = x_coords[closest_distance_to_x_cero_index]
      closest_distance_to_x_max_index = np.argmax(image_width - x_coords)
      closest_distance_to_x_max = x_coords[closest_distance_to_x_max_index]

      # Here a problem becomes obvious: approximating top left corner based
      # on the position of the contour in the image is a bad idea because
      # the finger can be pointing towards the left but if it's position
      # is closest to the right then it will be assumed that the finger is 
      # ponting towards the right.
      if closest_distance_to_x_cero <= closest_distance_to_x_max:
        closest_distance_to_x_cero_index = np.argmin(
          np.sqrt(
            np.sum((bounding_box_corners - [0, image_height]) ** 2, axis=1)
          )
        )
        return (
          bounding_box_corners[closest_distance_to_x_cero_index],
          closest_distance_to_x_cero_index
        )
      else:
        closest_distance_to_x_max_index = np.argmin(
          np.sqrt(
            np.sum((bounding_box_corners - [image_width, 0]) ** 2, axis=1)
          )
        )
        return (
          bounding_box_corners[closest_distance_to_x_max_index],
          closest_distance_to_x_max_index
        )
    else:
      m = (y2 - y1) / (x2 - x1)
      if m == 0:
        top_left_index = np.argmin(np.sqrt((x_coords ** 2) + (y_coords ** 2)))
        top_left_corner = bounding_box_corners[top_left_index]
        return top_left_corner, top_left_index
      elif m > 0:
        minimum_y_index = np.argmin(y_coords)
        return bounding_box_corners[minimum_y_index], minimum_y_index
      elif m < 0:
        minimum_x_index = np.argmin(x_coords)
        return bounding_box_corners[minimum_x_index], minimum_x_index

def generate_rotated_rectangle_image(angle, size=(300, 300)):
  center = (size[0] // 2, size[1] // 2)
  width = 100
  height = 50

  # Define four corner points for a non rotated rectangle
  rect_points = np.array(
    [
      (center[0] - (width // 2), center[1] - (height // 2)),  # top-left
      (center[0] + (width // 2), center[1] - (height // 2)),  # top-right
      (center[0] + (width // 2), center[1] + (height // 2)),  # bottom-right
      (center[0] - (width // 2), center[1] + (height // 2))   # bottom-left
    ], dtype=np.float32
  )

  # Create rotation matrix
  rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)

  # Apply rotation to the corner points
  rotated_rect_points = cv.transform(np.array([rect_points]), rotation_matrix)[0]
  rotated_rect_points = np.int32(rotated_rect_points)

  # Calculate bounding box. I want to test the method as I would implement it on
  # an expected contour class
  bounding_box_corners = get_bounding_box_corners(rotated_rect_points)
  top_left_corner, top_left_index = get_top_left_corner(
    bounding_box_corners,
    size[0],
    size[1]
  )

  image = np.full((size[0], size[1], 3), 0, dtype=np.uint8)

  for point in rotated_rect_points:
    x, y = point
    image[y, x] = (255, 0, 0)

  bounding_box_corners = np.reshape(bounding_box_corners, (-1, 1, 2))
  cv.drawContours(image, [bounding_box_corners], -1, (0, 255, 0), 5)

  cv.circle(image, top_left_corner, 5, (0, 0, 255), -1)

  cv.putText(
    image,
    f"Index: {top_left_index}",
    (20, 20),
    cv.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
  )
  cv.putText(
    image,
    f"Angle: {angle} degrees",
    (20, 40),
    cv.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
  )
  return image


def test_cut():
  generated_images = []
  angles = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    130,
    140,
    150,
    160,
    170,
    180,
  ]
  for angle in angles:
    image = generate_rotated_rectangle_image(angle)
    generated_images.append(image)

  num_images = len(generated_images)
  cols = 3
  rows = (num_images + cols - 1) // cols

  fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
  axes = axes.flatten()  # Flatten axes for easy indexing

  for i, image in enumerate(generated_images):
    axes[i].axis('off')
    axes[i].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

  # To not have any axes showing when there are blank axes
  for j in range(i + 1, len(axes)):
        axes[j].axis('off')

  # Taking into account that for the left hand radiograph I will move
  # from left to right, then I can calculate four different cases based
  # on the slope of the smallest 

  plt.tight_layout()
  plt.show()

def visualize_topleft_corner() -> None:
  test_cut()
