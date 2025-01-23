import os
import random
import sys
import threading
import time

import cv2
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt

# For SAM model, Connect4 predictor, circle detection, etc.
from ImagePointSelection import ImageClick
from connect4 import predict
from utils.sam_model_handler import SamModelHandler
from utils.calibration_utils import show_mask, show_points
from utils.calculate_intersection import calculate_intersection
from utils.border_detection import BorderDetector
from utils.camera_calibration import crop_view
from utils.detection_utils_v2 import process_each_cell_single_core, process_each_cell_multithreaded
from utils.cv2_utils import display_mask_image, display_mask_image_with_intersections
from utils.useTeachableMachine import CircleRecognition

import tensorflow as tf


# NOTE: provisional comments done by chatGPT


try:
    import requests
except ImportError:
    requests = None

input_point = []
input_label = []
image_path = ""


def handle_points_labels(points, labels, path):
    global input_point, input_label, image_path
    input_point = points
    input_label = labels
    image_path = path
    print("Image path set to: ", image_path)


def main():
    """ DEFAULT CONFIGURATION """
    USE_WEBCAM = True
    DEBUG = False
    MULTY_TREAD = True

    # Specify which player is controlled by the robot:
    #   1 for "robot" as Player 1
    #   2 for "robot" as Player 2
    ROBOT_PLAYER = 1

    # Set this to True to enable sending the move to a local server
    CONNECT_ROBOT = True
    ROBOT_SERVER_URL = "http://192.168.202.243:5000/move"#"http://127.0.0.1:5000/move"  # Example endpoint
    payload = {"column": int(0)}
    resp = requests.post(ROBOT_SERVER_URL, json=payload)

    # Suppress TensorFlow debugging logs
    tf.keras.utils.disable_interactive_logging()

    # Default image path (or fallback if the user doesn't select one)
    image_path = "Images/connect4_6.jpeg"

    """ LOAD SAM MODEL """
    sam_handler = SamModelHandler("Models/SAM Model Checkpoint.pth")

    while True:
        # Initialize tkinter for image selection
        root = tk.Tk()
        print("Starting with default image path: ", image_path)
        app = ImageClick(root, handle_points_labels, image_path, USE_WEBCAM=USE_WEBCAM)
        root.mainloop()

        print("Image path:", image_path)
        if not image_path:
            print("No image selected. Exiting.")
            return

        # open image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set the image for the SAM model
        sam_handler.set_image(image)

        if DEBUG:
            # Show clicked points on the original image
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            show_points(input_point, input_label, plt.gca())
            plt.show()

        # Predict the mask(s) from user-clicked points
        masks, scores, _ = sam_handler.predict(input_point, input_label)
        # Use the best mask (highest score)
        best_mask = masks[np.argmax(scores)]

        # Border detection
        detector = BorderDetector(best_mask)
        left_m, left_b = detector.find_left_border()
        top_m, top_b = detector.find_top_border()
        right_m, right_b = detector.find_right_border()
        bottom_m, bottom_b = detector.find_bottom_border()

        # Calculate corner intersections
        try:
            top_left = calculate_intersection(left_m, left_b, top_m, top_b)
            top_right = calculate_intersection(right_m, right_b, top_m, top_b)
            bottom_left = calculate_intersection(left_m, left_b, bottom_m, bottom_b)
            bottom_right = calculate_intersection(right_m, right_b, bottom_m, bottom_b)
            intersections = {
                "top_left": top_left,
                "top_right": top_right,
                "bottom_left": bottom_left,
                "bottom_right": bottom_right
            }
        except ValueError as e:
            print(e)
            return

        # Show the mask and corners. If user closes the window, break out;
        # if the user wants to reselect, the loop continues.
        if display_mask_image_with_intersections(image, best_mask, intersections):
            break
        else:
            # If user wants to do it again, keep going, but do not use the webcam next time
            USE_WEBCAM = False

    if DEBUG:
        # Visualize final mask with corners
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        show_mask(best_mask, plt.gca())
        plt.scatter(*top_left, color='red', s=50, label="Top Left")
        plt.scatter(*top_right, color='green', s=50, label="Top Right")
        plt.scatter(*bottom_left, color='blue', s=50, label="Bottom Left")
        plt.scatter(*bottom_right, color='orange', s=50, label="Bottom Right")
        plt.legend()
        plt.show()

    # Capture from webcam
    webcam = cv2.VideoCapture(0)
    cd = CircleRecognition()

    fps = 0
    fps_start_time = time.time()
    frame_count = 0
    matrix = np.zeros((6, 7), dtype=int)
    last_matrix = np.zeros((6, 7), dtype=int)

    # By default, let's say Player 1 starts
    player_that_needs_to_play = 1

    # Initial prediction for Player 1
    move, score = predict(matrix)
    print("Predicted move:", move)

    while True:
        frame_count += 1
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        if time_diff >= 1:
            fps = frame_count / time_diff
            fps_start_time = time.time()
            frame_count = 0

        ret, frame = webcam.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break

        # "Crop" and transform the board area
        warped_image = crop_view(frame, top_left, top_right, bottom_right, bottom_left)

        # Process the Connect4 board state
        if MULTY_TREAD:
            matrix, overlay = process_each_cell_multithreaded(
                image_path="Images/rectified_image.jpg",
                columns=7,
                rows=6,
                circle_detector=cd,
                old_matrix=matrix,
                force_image=warped_image
            )
        else:
            matrix, overlay = process_each_cell_single_core(
                image_path="Images/rectified_image.jpg",
                columns=7,
                rows=6,
                circle_detector=cd,
                old_matrix=matrix,
                force_image=warped_image
            )

        # Detect any change in the matrix
        new_matrix = matrix - last_matrix

        # If a piece from Player 1 has appeared
        if new_matrix.max() == 1:
            player_that_needs_to_play = 2
            status_message = "Player 1 has played."
            print(f"\rFPS: {fps:.2f} | Matrix:\n{matrix}\nStatus: {status_message}", end='', flush=True)

        # If a piece from Player 2 has appeared
        elif new_matrix.max() == 2:
            player_that_needs_to_play = 1
            # If Player 2 just played, compute next move for Player 1
            move, best_score = predict(matrix)
            status_message = f"Player 2 has played. New predicted move: {move}"
            print(f"\rFPS: {fps:.2f} | Matrix:\n{matrix}\nStatus: {status_message}", end='', flush=True)

        # If nothing changed, let's see if it's the Robot's turn
        else:
            # If the robot is the player_that_needs_to_play
            if player_that_needs_to_play == ROBOT_PLAYER:
                # We already have `move` from the predict call above
                column = move
                # (Optional) If you want to highlight that move on the overlay
                row = np.argmin(matrix[:, column])  # find the lowest empty row in that column
                if row < 6:
                    cell_height = overlay.shape[0] // 6
                    cell_width = overlay.shape[1] // 7
                    center_x = int(column * cell_width + cell_width / 2)
                    center_y = int(row * cell_height + cell_height / 2)

                    # Draw the prospective circle on the overlay
                    cv2.circle(overlay, (center_x, center_y), 20, (0, 255, 0), 2)

                # If we want to notify the local server to make the move:
                if CONNECT_ROBOT and requests is not None:
                    payload = {"column": int(column)}
                    try:
                        resp = requests.post(ROBOT_SERVER_URL, json=payload)
                        if resp.status_code == 200:
                            print(f"\n[Robot] Move {column} sent successfully.")
                        else:
                            print(f"\n[Robot] Error sending move, status code: {resp.status_code}")
                    except Exception as e:
                        print(f"\n[Robot] Exception sending move: {e}")

        last_matrix = matrix

        # Show the overlay
        cv2.imshow("Rectified Image", overlay)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
