import cv2
import numpy as np
import time

# Define HSV color ranges for traffic lights
COLOR_RANGES = {
    "RED": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([160, 100, 100]), np.array([179, 255, 255]))
    ],
    "YELLOW": [(np.array([18, 100, 100]), np.array([30, 255, 255]))],
    "GREEN": [(np.array([40, 70, 70]), np.array([90, 255, 255]))]
}

# Define actions based on traffic signal color
DECISIONS = {
    "RED": "STOP",
    "YELLOW": "SLOW DOWN",
    "GREEN": "GO"
}

# Load transparent car image with alpha channel
car_img = cv2.imread("car.png", cv2.IMREAD_UNCHANGED)
car_img = cv2.resize(car_img, (60, 40))  # Resize to fit scene

def overlay_car(background, car_img, x, y):
    """Overlay transparent car image on the frame at position (x, y)."""
    h, w = car_img.shape[:2]
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return
    alpha_car = car_img[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            alpha_car * car_img[:, :, c] + 
            (1 - alpha_car) * background[y:y+h, x:x+w, c]
        )

def detect_traffic_light(hsv_frame):
    """Detect traffic light color in the top half of the frame."""
    roi = hsv_frame[0:240, :]
    for color, ranges in COLOR_RANGES.items():
        combined_mask = None
        for lower, upper in ranges:
            mask = cv2.inRange(roi, lower, upper)
            mask = cv2.GaussianBlur(mask, (9, 9), 2)
            combined_mask = mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)

        # Detect circular traffic lights
        circles = cv2.HoughCircles(combined_mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                   param1=50, param2=13, minRadius=5, maxRadius=50)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                return color, i  # Return color and circle info
    return None, None

# Setup webcam capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Car position and speed
car_x = 50
car_speed = 3
frame_width, frame_height = 640, 480

# For FPS calculation
prev_time = time.time()
fps = 0

print("[INFO] Starting traffic light detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect the current traffic light state
    signal_color, circle_data = detect_traffic_light(hsv)
    if signal_color:
        decision = DECISIONS[signal_color]
        cv2.putText(frame, f"DETECTED: {decision}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        if circle_data is not None:
            cv2.circle(frame, (circle_data[0], circle_data[1]), circle_data[2], (0, 255, 255), 3)
    else:
        decision = "NO SIGNAL"
        cv2.putText(frame, "NO SIGNAL DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

    # Set car speed based on traffic light
    if decision == "GO":
        car_speed = 5
    elif decision == "SLOW DOWN":
        car_speed = 2
    elif decision == "STOP":
        car_speed = 0
    else:
        car_speed = 0

    # Update car position
    car_x += car_speed
    if car_x > frame_width - 60:
        car_x = 50  # Reset position
        print("[INFO] Car reached end, restarting from left.")

    # Overlay the car on the frame
    overlay_car(frame, car_img, car_x, 400)

    # Display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # Show current traffic signal decision
    cv2.putText(frame, f"Signal: {decision}", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Traffic Light Simulation", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[INFO] Exiting program.")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
