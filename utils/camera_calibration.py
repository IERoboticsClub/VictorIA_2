import cv2
import numpy as np


def crop_view(image, top_left, top_right, bottom_right, bottom_left):
    # coordinates of the corners of the distorted rectangle
    pts_src = np.array([
        [top_left[0], top_left[1]],  # top-left corner
        [top_right[0], top_right[1]],  # top-right corner
        [bottom_right[0], bottom_right[1]],  # bottom-right corner
        [bottom_left[0], bottom_left[1]]   # bottom-left corner
    ], dtype="float32")

    # define the target rectangle dimensions
    W, H = 700, 600
    pts_dst = np.array([
        [0, 0],      # rop-left corner
        [W, 0],      # rop-right corner
        [W, H],      # bottom-right corner
        [0, H]       # bottom-left corner
    ], dtype="float32")

    # compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # apply the perspective transformation
    warped_image = cv2.warpPerspective(image, matrix, (W, H))

    return warped_image
