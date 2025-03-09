import cv2
import mediapipe as mp

def draw_body_outline(frame, landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
    )

def detect_pose_and_overlay_shirt():
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Load the shirt image with alpha channel (transparency)
    shirt_img = cv2.imread("Resources/Shirts/1.png", cv2.IMREAD_UNCHANGED)

    # Check if the shirt image is loaded properly
    if shirt_img is None:
        print("Error: Shirt image not loaded.")
        return

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make pose detection
        results = pose.process(frame_rgb)

        # Overlay shirt on detected pose keypoints
        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks

            # Draw body outline
            draw_body_outline(frame, landmarks)

            # Get coordinates of key points for overlay (both shoulders and the hips)
            left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

            # Calculate shirt position and size
            shirt_width = int(abs(left_shoulder.x - right_shoulder.x) * frame.shape[1] * 1.25)  # Increase width by 25%
            shirt_height = int(abs(left_shoulder.y - left_hip.y) * frame.shape[0] * 1.25)  # Increase height by 25%

            # Ensure calculated dimensions are valid
            if shirt_width > 0 and shirt_height > 0:
                shirt_resized = cv2.resize(shirt_img, (shirt_width, shirt_height))

                # Calculate the overlay position
                center_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1])
                center_y = int((left_shoulder.y + left_hip.y) / 2 * frame.shape[0])

                x_offset = center_x - shirt_width // 2
                y_offset = center_y - shirt_height // 2

                # Ensure the overlay position is within frame bounds
                y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + shirt_resized.shape[0])
                x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + shirt_resized.shape[1])

                # Calculate regions of interest
                shirt_resized_cropped = shirt_resized[0:(y2-y1), 0:(x2-x1)]

                # Ensure the dimensions match
                if shirt_resized_cropped.shape[0] == (y2-y1) and shirt_resized_cropped.shape[1] == (x2-x1):
                    alpha_s = shirt_resized_cropped[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s

                    # Overlay shirt image
                    for c in range(0, 3):
                        frame[y1:y2, x1:x2, c] = (alpha_s * shirt_resized_cropped[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])

        # Display the frame
        cv2.imshow('Pose Detection with Shirt Overlay', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Run the function to start pose detection with shirt overlay
detect_pose_and_overlay_shirt()
