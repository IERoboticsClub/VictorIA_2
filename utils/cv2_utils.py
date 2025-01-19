import cv2
import numpy as np


def display_mask_image(image, best_mask):
    # Overlay the best mask on the original image
    overlay = image.copy()
    alpha = 0.5  # Transparency factor
    mask_overlay = (best_mask.astype(np.uint8) * 255)  # Convert mask to uint8

    # Convert the mask to a 3-channel image for blending
    mask_overlay = cv2.cvtColor(mask_overlay, cv2.COLOR_GRAY2BGR)

    # Blend the original image and mask
    overlay = cv2.addWeighted(mask_overlay, alpha, overlay, 1 - alpha, 0)

    # Add text prompts
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)  # White color

    # Right corner: "Click Enter to continue"
    text_enter = "Click Enter to continue"
    text_size_enter = cv2.getTextSize(text_enter, font, font_scale, font_thickness)[0]
    text_x_enter = overlay.shape[1] - text_size_enter[0] - 10
    text_y_enter = overlay.shape[0] - 20
    cv2.putText(overlay, text_enter, (text_x_enter, text_y_enter), font, font_scale, text_color, font_thickness)

    # Left corner: "Click R to restart"
    text_restart = "Click R to restart"
    text_x_restart = 10
    text_y_restart = overlay.shape[0] - 20
    cv2.putText(overlay, text_restart, (text_x_restart, text_y_restart), font, font_scale, text_color, font_thickness)

    # Show the result and block the execution until a key is pressed
    cv2.imshow("Image with Best Mask Overlay", overlay)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):  # 'R' key for restart
            cv2.destroyAllWindows()
            print("Retrying mask selection...")
            return False
        elif key == 13:  # Enter key
            cv2.destroyAllWindows()
            print("Continuing Process...")
            return True


