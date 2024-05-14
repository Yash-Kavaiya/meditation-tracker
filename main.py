import cv2
import pandas as pd
from datetime import datetime

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize the first frame in the video stream
first_frame = None
movement_count = 0
movement_log = []

# Define the duration of the meditation in seconds (40 minutes * 60 seconds)
meditation_duration = 40 * 60

start_time = datetime.now()

while (datetime.now() - start_time).seconds < meditation_duration:
    # Capture the current frame
    check, frame = cap.read()

    # If we couldn't get a frame, break out of the loop
    if not check:
        break

    # Convert the frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # If the first frame is None, initialize it
    if first_frame is None:
        first_frame = gray
        continue

    # Compute the absolute difference between the current frame and the first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes, then find contours on the thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for contour in contours:
        # If the contour is too small, ignore it
        if cv2.contourArea(contour) < 1000:
            continue

        # There's movement
        movement_count += 1
        movement_log.append({'timestamp': datetime.now(), 'movement': 'detected'})

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # If the 'q' key is pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

# Convert the movement log to a DataFrame and save it as a CSV file
df = pd.DataFrame(movement_log)
df.to_csv('movement_log.csv', index=False)

print(f"Total movements detected: {movement_count}")