import math
import numpy as np
import cv2
import pathlib
import os


def safe_imshow(img, name):
    if img is not None:
        print(f"Displaying {name}:")
        cv2.imshow(name, img)
    else:
        print(f"Error: {name} could not be loaded. Check path: {globals()['file_path_' + name.split()[-1].lower()]}")



# def click_event(event, x, y, flags, params):
#     # This function triggers every time the mouse interacts with the window
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f"Recorded click at: x={x}, y={y}")

def click_event(event, x, y, flags, param):
    # Unpack our state from the param dictionary
    clicks = param["clicks"]
    image = param["image"]
    
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Ground Truth Annotator", image)

data_path = pathlib.Path(r"E:\\UMCP\\GS\\study_plan\\fall_2026\\CMSC818V\\data\\worm_up_data")

save_path = pathlib.Path(r"E:\\UMCP\\GS\\study_plan\\fall_2026\\CMSC818V\\data\\worm_up_data\\processed")

video_name = "20260915_082555.mp4"

video_path = data_path / video_name
print(f"Video path: {video_path}")
print(f"Video exists: {os.path.exists(video_path)}")


cap = cv2.VideoCapture(video_path)
success, frame = cap.read()

if success:
    print(f"Frame shape: {frame.shape}")
    cv2.imwrite(str(save_path / "original_frame.png"), frame)
    # safe_imshow(frame, "Original Frame")
    # cv2.waitKey(0)

# Isolate the arena, and measure its area in pixels.
window_name = "Select Arena (Press Enter when done)"

# 1. Create a resizable window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# 2. Set display dimensions that fit comfortably on your screen (e.g., 600x800)
cv2.resizeWindow(window_name, 600, 800)

# 3. Select ROI using the pre-configured window name
r = cv2.selectROI(window_name, frame)
cv2.destroyAllWindows()

x, y, w, h = r
print(f"Selected ROI: x={x}, y={y}, width={w}, height={h}")

if w > 0:
    # Crop the full-resolution image using the selected coordinates
    cropped_arena = frame[y:y+h, x:x+w]
    cv2.imwrite(str(save_path / "cropped_arena.jpg"), cropped_arena)
    print(f"Saved {save_path / 'cropped_arena.jpg'}")
    
    # Calculate conversion factor (Arena is 30.5 cm wide)
    cm_per_pixel = 30.5 / w
    print(f"Conversion factor: {cm_per_pixel:.4f} cm per pixel")
else:
    print("No ROI selected.")

arena_image = cv2.imread(str(save_path / "cropped_arena.jpg"))


print("Additional processing for evaluator")


if arena_image is not None:
    window_name = "Ground Truth Annotator"

    # 1. Display the image first so OpenCV creates the window
    cv2.imshow(window_name, arena_image)

    # 2. Initialize state dictionary and attach the callback
    state = {"clicks": [], "image": arena_image}
    cv2.setMouseCallback(window_name, click_event, param=state)

    print("Click 1: Robot Center")
    print("Click 2 & 3: Two points along the body axis")

    # 3. Check state["clicks"] instead of clicks
    while len(state["clicks"]) < 3:
        cv2.waitKey(10)

    cv2.destroyAllWindows()

    # Extract points from state["clicks"]
    center = state["clicks"][0]
    p1 = state["clicks"][1]
    p2 = state["clicks"][2]

    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    theta = math.degrees(math.atan2(dy, dx)) % 180

    print(f"Calculated Center: {center}")
    print(f"Calculated Orientation: {theta:.2f} degrees")
else:
    print(f"Error: Could not load {save_path / 'cropped_arena.jpg'}")