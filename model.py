import cv2
import mediapipe as mp
import numpy as np

# ---------- USER HEIGHT (in cm) ----------
USER_HEIGHT_CM = 165   # change for user
USER_HEIGHT_M = USER_HEIGHT_CM / 100.0

# ---------- HORIZONTAL REFERENCE (in meters) ----------
REFERENCE_LENGTH_M = 1.0      # Known length of reference object
USE_HORIZONTAL_REFERENCE = False  # Set to False to use body height method
# -------------------------------------------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

video_path = r"C:\Users\Jasleen\Downloads\jils.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video not found:", video_path)
    exit()

# ---------- HORIZONTAL REFERENCE CALIBRATION ----------
reference_points = []         # Will store two (x, y) tuples
pixels_per_meter_horizontal = None
calibration_frame = None

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to select reference points"""
    global reference_points, calibration_frame
    if event == cv2.EVENT_LBUTTONDOWN and len(reference_points) < 2:
        reference_points.append((x, y))
        # Draw the point on calibration frame
        cv2.circle(calibration_frame, (x, y), 8, (0, 255, 0), -1)
        if len(reference_points) == 1:
            cv2.putText(calibration_frame, "Now click END of reference", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Calibration", calibration_frame)
# ------------------------------------------------------

# ---------- BASELINE VARIABLES ----------
baseline_head_y = []
baseline_heel_y = []
baseline_toe_x = []           # Track toe position during standing baseline
BASELINE_FRAMES = 25

height_pixels = None          # person height in pixels
baseline_heel_y_mean = None
baseline_toe_x_mean = None    # Average toe position when standing

state = "CALIBRATING"         # CALIBRATING, READY, IN_AIR, DONE
takeoff_x = None
landing_x = None
jump_distance_m = None
takeoff_frame = None          # Track when takeoff happened
MIN_AIRTIME_FRAMES = 15       # Minimum frames to be in air (increased to prevent early landing)

# For stability-based landing detection - track TOE position, not heel!
recent_toe_positions = []     # Track horizontal toe position
STABILITY_WINDOW = 12         # Check last 12 frames for stability (increased for more confidence)

# For perspective distortion detection
takeoff_body_height_px = None
landing_body_height_px = None

frame_idx = 0

# ---------- HORIZONTAL REFERENCE CALIBRATION (if enabled) ----------
if USE_HORIZONTAL_REFERENCE:
    print("\n" + "="*60)
    print("HORIZONTAL REFERENCE CALIBRATION")
    print("="*60)
    print(f"Please place a reference object of {REFERENCE_LENGTH_M}m on the ground")
    print("along the jump path (parallel to camera).\n")
    
    # Get first frame for calibration
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Cannot read video!")
        exit()
    
    calibration_frame = first_frame.copy()
    h, w, _ = calibration_frame.shape
    
    # Instructions on frame
    cv2.putText(calibration_frame, f"Click START of {REFERENCE_LENGTH_M}m reference", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(calibration_frame, "Press SPACE to skip (use body height instead)", (20, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    cv2.imshow("Calibration", calibration_frame)
    cv2.setMouseCallback("Calibration", mouse_callback)
    
    print("Click two points on the video window to mark the reference length.")
    print(f"The distance between points should be {REFERENCE_LENGTH_M}m.")
    print("Press SPACE to skip horizontal calibration.\n")
    
    # Wait for two clicks or SPACE to skip
    while len(reference_points) < 2:
        key = cv2.waitKey(10)
        if key == ord(' '):  # Skip calibration
            print("⊙ Skipped horizontal calibration - using body height method\n")
            USE_HORIZONTAL_REFERENCE = False
            break
    
    if USE_HORIZONTAL_REFERENCE and len(reference_points) == 2:
        # Calculate pixels per meter
        p1, p2 = reference_points
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        pixels_per_meter_horizontal = pixel_distance / REFERENCE_LENGTH_M
        
        # Draw line between points
        cv2.line(calibration_frame, p1, p2, (0, 255, 0), 3)
        cv2.putText(calibration_frame, f"{pixel_distance:.1f}px = {REFERENCE_LENGTH_M}m", 
                   ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(calibration_frame, "Calibration complete! Press any key to start...", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration", calibration_frame)
        cv2.waitKey(0)
        
        print(f"✓ Horizontal calibration: {pixel_distance:.1f} pixels = {REFERENCE_LENGTH_M}m")
        print(f"  Scale: {pixels_per_meter_horizontal:.2f} pixels/meter\n")
    
    cv2.destroyWindow("Calibration")
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# -------------------------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n⚠ Video finished.")
        if state == "IN_AIR":
            print(f"  Landing was NOT detected! Person was in air for {frame_idx - takeoff_frame} frames.")
            print(f"  Last heel position: {heel_y:.1f}, needed to reach: {baseline_heel_y_mean:.1f}")
        elif state == "DONE":
            print(f"  ✓ Jump successfully measured!")
        break

    frame_idx += 1

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    status_text = state

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        head = lm[mp_pose.PoseLandmark.NOSE]
        l_heel = lm[mp_pose.PoseLandmark.LEFT_HEEL]
        r_heel = lm[mp_pose.PoseLandmark.RIGHT_HEEL]
        l_foot_index = lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]  # Toe
        r_foot_index = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]  # Toe

        head_y = head.y * h
        
        # For vertical tracking (standing/jumping), use heel
        heel_y = ((l_heel.y + r_heel.y) / 2.0) * h
        
        # For horizontal takeoff position, use TOE (forward-most point)
        toe_x = max(l_foot_index.x, r_foot_index.x) * w  # Furthest forward toe
        
        # For horizontal landing position, also use TOE (forward-most point)  
        landing_toe_x = max(l_foot_index.x, r_foot_index.x) * w  # Furthest forward toe at landing
        
        # For vertical detection, keep using heel
        heel_x = min(l_heel.x, r_heel.x) * w  # For vertical stability check only

        head_pt = (int(head.x * w), int(head_y))
        L_heel = (int(l_heel.x * w), int(l_heel.y * h))
        R_heel = (int(r_heel.x * w), int(r_heel.y * h))
        L_toe = (int(l_foot_index.x * w), int(l_foot_index.y * h))
        R_toe = (int(r_foot_index.x * w), int(r_foot_index.y * h))

        # Draw head, heels, and toes
        cv2.circle(frame, head_pt, 6, (255, 0, 0), -1)
        cv2.circle(frame, L_heel, 6, (0, 255, 0), -1)
        cv2.circle(frame, R_heel, 6, (0, 255, 0), -1)
        cv2.circle(frame, L_toe, 5, (255, 255, 0), -1)  # Yellow for toes
        cv2.circle(frame, R_toe, 5, (255, 255, 0), -1)

        # ---------- 1) CALIBRATION PHASE ----------
        if frame_idx <= BASELINE_FRAMES:
            baseline_head_y.append(head_y)
            baseline_heel_y.append(heel_y)
            baseline_toe_x.append(toe_x)  # Track toe position while standing
            status_text = "Calibrating..."
        else:
            if height_pixels is None:
                # Compute baseline only once
                baseline_head_y_mean = np.mean(baseline_head_y)
                baseline_heel_y_mean = np.mean(baseline_heel_y)
                baseline_toe_x_mean = np.mean(baseline_toe_x)  # Average toe position when standing
                height_pixels = abs(baseline_head_y_mean - baseline_heel_y_mean)

                print(f"Baseline head_y: {baseline_head_y_mean:.2f}")
                print(f"Baseline heel_y: {baseline_heel_y_mean:.2f}")
                print(f"Baseline toe_x: {baseline_toe_x_mean:.2f}")
                print(f"Height in pixels: {height_pixels:.2f}")
                print(f"User height (m): {USER_HEIGHT_M:.2f}\n")

                state = "READY"

            # ---------- 2) STATE MACHINE ----------
            if state == "READY":
                status_text = "Ready"
                # Takeoff when heel clearly above baseline (smaller y = up)
                lift_threshold = 0.03 * height_pixels   # 12% of body height
                if baseline_heel_y_mean - heel_y > lift_threshold:
                    state = "IN_AIR"
                    takeoff_x = baseline_toe_x_mean  # Use baseline TOE position (when standing)
                    takeoff_frame = frame_idx
                    takeoff_body_height_px = abs(head_y - heel_y)
                    print(f"Takeoff at frame {frame_idx}, using baseline TOE X = {takeoff_x:.2f}")
                    print(f"  Body height at takeoff: {takeoff_body_height_px:.1f}px")

            elif state == "IN_AIR":
                frames_in_air = frame_idx - takeoff_frame
                status_text = f"In air ({frames_in_air} frames)"
                
                # Progressive landing detection based on heel returning to baseline
                if frames_in_air < 20:
                    land_margin = 0.03 * height_pixels      # strict: within 3%
                elif frames_in_air < 35:
                    land_margin = 0.05 * height_pixels      # within 5%
                elif frames_in_air < 50:
                    land_margin = 0.08 * height_pixels      # within 8%
                else:
                    land_margin = 0.12 * height_pixels      # very lenient: within 12%
                
                heel_diff = heel_y - baseline_heel_y_mean
                
                # Track TOE horizontal position for landing detection
                recent_toe_positions.append(landing_toe_x)
                if len(recent_toe_positions) > STABILITY_WINDOW:
                    recent_toe_positions.pop(0)
                
                # Check if TOES are stable horizontally (person has stopped moving forward)
                is_stable = False
                if len(recent_toe_positions) == STABILITY_WINDOW:
                    toe_variance = max(recent_toe_positions) - min(recent_toe_positions)
                    if toe_variance < 20:  # toes stayed within 20 pixels horizontally for STABILITY_WINDOW frames
                        is_stable = True
                
                # Debug output every 5 frames
                if frames_in_air % 5 == 0:
                    stable_str = " [TOE STABLE]" if is_stable else ""
                    print(f"  Frame {frame_idx}: heel_y={heel_y:.1f}, baseline={baseline_heel_y_mean:.1f}, diff={heel_diff:.1f}, toe_var={toe_variance if len(recent_toe_positions)==STABILITY_WINDOW else 'N/A'}{stable_str}")
                
                if frames_in_air >= MIN_AIRTIME_FRAMES:
                    # Landing: heels near baseline AND toes stable (stopped moving forward)
                    heels_at_baseline = abs(heel_diff) < land_margin
                    
                    if heels_at_baseline and is_stable:
                        state = "DONE"
                        landing_x = landing_toe_x  # Use TOE position at landing (for toe-to-toe measurement)
                        landing_body_height_px = abs(head_y - heel_y)
                        
                        # Calculate jump with baseline calibration
                        # Jump distance = TOE at takeoff to TOE at landing
                        pixel_jump = abs(landing_x - takeoff_x)
                        
                        # Use horizontal calibration if available, otherwise body height
                        if USE_HORIZONTAL_REFERENCE and pixels_per_meter_horizontal is not None:
                            jump_distance_m = pixel_jump / pixels_per_meter_horizontal
                            calibration_method = "horizontal reference"
                        else:
                            jump_distance_m = (pixel_jump * USER_HEIGHT_M) / height_pixels
                            calibration_method = "body height"
                        
                        # Diagnose perspective issues (for debugging only)
                        height_change_pct = abs(landing_body_height_px - takeoff_body_height_px) / takeoff_body_height_px * 100
                        
                        landing_type = "margin" if abs(heel_diff) < land_margin else "stability"
                        print(f"\n✓ Landing at frame {frame_idx}, TOE X = {landing_x:.2f} (detected via {landing_type})")
                        print(f"  Frames in air: {frames_in_air}")
                        print(f"  Pixel jump (toe→toe): {pixel_jump:.2f}px")
                        print(f"  Jump distance: {jump_distance_m:.3f} m ({jump_distance_m:.2f} m)")
                        print(f"  Calibration method: {calibration_method}")
                        
                        # Diagnostic info
                        print(f"\n  [Diagnostics]")
                        if USE_HORIZONTAL_REFERENCE and pixels_per_meter_horizontal is not None:
                            print(f"  Horizontal: {pixels_per_meter_horizontal:.1f} pixels/meter")
                        print(f"  Baseline body height: {height_pixels:.1f}px (vertical calibration)")
                        print(f"  Body at takeoff: {takeoff_body_height_px:.1f}px")
                        print(f"  Body at landing: {landing_body_height_px:.1f}px")
                        if height_change_pct > 15:
                            print(f"  ⚠ Body height changed {height_change_pct:.1f}% during jump (person not fully extended)")
                        print()

            elif state == "DONE":
                if jump_distance_m is not None:
                    status_text = f"Jump: {jump_distance_m:.2f} m"
                else:
                    status_text = "Done"

    # ---------- DISPLAY (scaled but logic stays same) ----------
    scale = min(900 / w, 700 / h, 1.0)  # fit nicely on screen
    disp_w, disp_h = int(w * scale), int(h * scale)
    display_frame = cv2.resize(frame, (disp_w, disp_h))

    # Draw baseline reference line (horizontal at heel baseline)
    if baseline_heel_y_mean is not None:
        baseline_y_scaled = int(baseline_heel_y_mean * scale)
        cv2.line(display_frame, (0, baseline_y_scaled), (disp_w, baseline_y_scaled), (0, 255, 255), 2)
    
    # Draw takeoff marker (vertical line at TOE position)
    if takeoff_x is not None:
        takeoff_x_scaled = int(takeoff_x * scale)
        cv2.line(display_frame, (takeoff_x_scaled, 0), (takeoff_x_scaled, disp_h), (0, 255, 0), 3)
        cv2.putText(display_frame, "TAKEOFF (TOE)", (takeoff_x_scaled + 5, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw landing marker (vertical line at TOE position)
    if landing_x is not None:
        landing_x_scaled = int(landing_x * scale)
        cv2.line(display_frame, (landing_x_scaled, 0), (landing_x_scaled, disp_h), (255, 0, 0), 3)
        cv2.putText(display_frame, "LANDING (TOE)", (landing_x_scaled + 5, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw current heel position (vertical line)
    if results.pose_landmarks and state in ["READY", "IN_AIR"]:
        heel_x_scaled = int(heel_x * scale)
        cv2.line(display_frame, (heel_x_scaled, 0), (heel_x_scaled, disp_h), (255, 255, 0), 1)

    cv2.putText(display_frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    
    # Show calibration info
    if height_pixels is not None:
        cv2.putText(display_frame, f"Calibration: {height_pixels:.1f}px = {USER_HEIGHT_M:.2f}m", 
                    (20, disp_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Broad Jump Test", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
