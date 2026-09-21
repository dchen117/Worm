import numpy as np
import cv2
import pathlib

def click_event(event, x, y, flags, param):
    # Only register clicks when paused and fewer than 3 points recorded
    if event == cv2.EVENT_LBUTTONDOWN and param["paused"]:
        if len(param["clicks"]) < 3:
            param["clicks"].append((x, y))
            # Draw point on the current image copy
            cv2.circle(param["display_frame"], (x, y), 3, (0, 0, 255), -1)
            cv2.imshow(param["window_name"], param["display_frame"])

def get_center_and_axis(clicks):
    if len(clicks) < 3:
        raise ValueError("At least 3 clicks required.")
    
    center = clicks[0]
    p1, p2 = clicks[1], clicks[2]
    axis_vector = np.array(p2) - np.array(p1)
    
    return center, axis_vector

# --- Setup Paths ---
data_path = pathlib.Path(r"E:\UMCP\GS\study_plan\fall_2026\CMSC818V\data\worm_up_data")
video_name = "20260915_082555.mp4"
video_path = data_path / video_name

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"Error: Could not open {video_path}")
    exit()

window_name = "Ground Truth Annotator"
cv2.namedWindow(window_name)

# State dictionary managed cleanly across frames
state = {
    "window_name": window_name,
    "clicks": [],
    "display_frame": None,
    "paused": False
}

cv2.setMouseCallback(window_name, click_event, param=state)
results = {"frames": [], "centers": [], "angles": []}
frame_idx = 0

while True:
    # Only read a new frame if NOT paused
    if not state["paused"]:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
            
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        state["display_frame"] = frame.copy()
        state["clicks"] = []  # Reset click buffer for new frame
        cv2.imshow(window_name, state["display_frame"])

    # Always process key inputs at 30ms intervals
    key = cv2.waitKey(30) & 0xFF
    
    # Toggle Pause with Spacebar
    if key == 32:  
        state["paused"] = not state["paused"]
        if state["paused"]:
            print(f"\n[PAUSED - Frame {frame_idx}]")
            print("Click 1: Robot Center | Click 2 & 3: Body Axis Points")
        else:
            print("Resuming video...")

    # Process clicks when exactly 3 have been made
    if state["paused"] and len(state["clicks"]) == 3:
        center, axis_vector = get_center_and_axis(state["clicks"])
        
        # Calculate angle relative to horizontal in [0, 180) degrees
        angle = np.degrees(np.arctan2(axis_vector[1], axis_vector[0])) % 180
        
        # Visual feedback: Draw axis line on frame
        p1, p2 = state["clicks"][1], state["clicks"][2]
        cv2.line(state["display_frame"], p1, p2, (0, 255, 0), 2)
        cv2.imshow(window_name, state["display_frame"])
        
        print(f"Saved Frame {frame_idx} -> Center: {center}, Angle: {angle:.2f}°")

        results["frames"].append(frame_idx)
        results["centers"].append(center)
        results["angles"].append(angle)

        # Append dummy point so length becomes 4, preventing repeated calculations on same frame
        state["clicks"].append("PROCESSED")

    elif key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
print("\nAnnotation Complete. Results:")
for i, (frame, center, angle) in enumerate(zip(results["frames"], results["centers"], results["angles"])):
    print(f"Frame {frame}: Center {center}, Angle {angle:.2f}°")