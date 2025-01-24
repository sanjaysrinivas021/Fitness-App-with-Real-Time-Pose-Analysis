import cv2
import mediapipe as mp
import json
import os
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Folder containing correct posture JSON files
correct_pose_folder = "correct_poses"  # Replace with the path to your folder containing JSON files
correct_pose_files = [
    os.path.join(correct_pose_folder, file)
    for file in os.listdir(correct_pose_folder) if file.endswith(".json")
]

# Load the local image to compare
local_image_path = "pushupcomp1.jpg"  # Replace with your local image path
image = cv2.imread(local_image_path)

if image is None:
    print(f"Error: Cannot load image at {local_image_path}")
    exit()

# Convert the image to RGB for Mediapipe
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the local image to extract pose landmarks
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Extract landmarks from the local image
        local_pose_landmarks = [
            {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
            for lm in results.pose_landmarks.landmark
        ]

        # Compare with each correct posture JSON
        best_match_file = None
        best_mse = float('inf')
        best_correct_landmarks = None

        for idx, json_file in enumerate(correct_pose_files):
            with open(json_file, "r") as file:
                correct_pose_landmarks = json.load(file)

            # Calculate similarity
            local_pose_array = np.array(
                [[lm['x'], lm['y'], lm['z']] for lm in local_pose_landmarks]
            )
            correct_pose_array = np.array(
                [[lm['x'], lm['y'], lm['z']] for lm in correct_pose_landmarks]
            )

            # Compute the mean squared error (MSE) as a measure of similarity
            mse = np.mean((local_pose_array - correct_pose_array) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_match_file = json_file
                best_correct_landmarks = correct_pose_landmarks

        # Set threshold and decide if it matches
        threshold = 0.001
        if best_mse < threshold:
            print(f"Best match found: {os.path.basename(best_match_file)}")
            print("Yes, it is the correct posture!")

            # Display differences
            correct_image = np.zeros_like(image)  # Create a blank image to overlay correct posture
            for local_lm, correct_lm in zip(local_pose_landmarks, best_correct_landmarks):
                local_point = (int(local_lm['x'] * image.shape[1]), int(local_lm['y'] * image.shape[0]))
                correct_point = (int(correct_lm['x'] * image.shape[1]), int(correct_lm['y'] * image.shape[0]))

                # Draw local landmarks
                cv2.circle(image, local_point, 5, (0, 255, 0), -1)  # Green for local pose

                # Draw correct posture landmarks
                cv2.circle(correct_image, correct_point, 5, (0, 0, 255), -1)  # Red for correct pose

                # Draw differences
                cv2.line(image, local_point, correct_point, (255, 0, 0), 2)  # Blue for difference

            # Combine images side by side
            combined_image = cv2.hconcat([image, correct_image])

            # Display the result in a window
            cv2.imshow("Pose Comparison", combined_image)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()

            # Optionally, save the result
            output_path = "combined_pose_comparison.jpg"
            cv2.imwrite(output_path, combined_image)
            print(f"Comparison image saved as {output_path}.")
        else:
            print("No, it is not the correct posture.")
    else:
        print("No pose landmarks detected in the local image.")
