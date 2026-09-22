import cv2
import numpy as np
import pathlib
import csv

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and param["paused"]:
        if len(param["clicks"]) < 3:
            param["clicks"].append((x, y))
            cv2.circle(param["display_frame"], (x, y), 3, (0, 0, 255), -1)
            cv2.imshow(param["window_name"], param["display_frame"])

def get_center_and_axis(clicks):
    if len(clicks) < 3:
        raise ValueError("At least 3 clicks required.")
    center = clicks[0]
    p1, p2 = clicks[1], clicks[2]
    axis_vector = np.array(p2) - np.array(p1)
    return center, axis_vector

def pixel_to_cm(x, y, arena_width_px, arena_real_width_cm):
    scale = arena_real_width_cm / arena_width_px
    return x * scale, y * scale

def annotate_video(video_path, output_csv, arena_box, arena_real_width_cm=30.5):
    x0, y0, x1, y1 = arena_box
    arena_width_px = x1 - x0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    window_name = "Ground Truth Annotator"
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(window_name)
    # Lock the aspect ratio and allow resizing
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    
    # Set to exactly half-scale of the 1920x1080 source
    cv2.resizeWindow(window_name, 960, 540)


    state = {
        "window_name": window_name,
        "clicks": [],
        "display_frame": None,
        "paused": False
    }

    cv2.setMouseCallback(window_name, click_event, param=state)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x_cm", "y_cm", "theta_deg"])

        frame_idx = 0
        while True:
            if not state["paused"]:
                ret, frame = cap.read()
                if not ret:
                    print("End of video.")
                    break
                    
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                state["display_frame"] = frame.copy()
                state["clicks"] = []
                cv2.imshow(window_name, state["display_frame"])

            key = cv2.waitKey(30) & 0xFF
            
            if key == 32:  
                state["paused"] = not state["paused"]
                if state["paused"]:
                    print(f"\n[PAUSED - Frame {frame_idx}] - Awaiting 3 clicks...")
                else:
                    print("Resuming video...")

            if state["paused"] and len(state["clicks"]) == 3:
                center, axis_vector = get_center_and_axis(state["clicks"])
                
                # Math first
                angle = np.degrees(np.arctan2(axis_vector[1], axis_vector[0])) % 180
                rel_x = center[0] - x0
                rel_y = center[1] - y0
                x_cm, y_cm = pixel_to_cm(rel_x, rel_y, arena_width_px, arena_real_width_cm)
                
                # Write to CSV
                writer.writerow([frame_idx, round(x_cm, 3), round(y_cm, 3), round(angle, 2)])
                
                # Visual feedback
                p1, p2 = state["clicks"][1], state["clicks"][2]
                cv2.line(state["display_frame"], p1, p2, (0, 255, 0), 2)
                cv2.imshow(window_name, state["display_frame"])
                
                print(f"Saved Frame {frame_idx} -> X: {x_cm:.2f}cm, Y: {y_cm:.2f}cm, Angle: {angle:.2f}°")
                state["clicks"].append("PROCESSED")

            elif key == 27:  
                break

    cap.release()
    cv2.destroyAllWindows()

# --- Execution ---
if __name__ == "__main__":
    # Update these paths and coordinates for your held-out test videos
    video_file = pathlib.Path(r"E:\UMCP\GS\study_plan\fall_2026\CMSC818V\data\worm_up_data\evaluation_videos\20260915_084130.mp4")
    csv_file = pathlib.Path(r"E:\UMCP\GS\study_plan\fall_2026\CMSC818V\data\worm_up_data\evaluation_videos\20260915_084130_gt.csv")
    
    # Replace with the actual arena bounds for this specific video
    arena_coordinates = (100, 100, 1100, 1100) 
    
    annotate_video(video_file, csv_file, arena_coordinates)