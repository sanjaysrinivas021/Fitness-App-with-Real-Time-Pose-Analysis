import cv2
import mediapipe as mp
import json
import os

print("Current working directory:", os.getcwd())


# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose

# Folder to save correct posture JSON files
correct_pose_folder = "correct_poses"
os.makedirs(correct_pose_folder, exist_ok=True)

# List of image paths for correct postures (replace these with actual image file paths)
image_paths = [
    "pushuppose_1.jpg",  # Replace with your image paths
    "pushuppose_2.jpg",
    "pushuppose_3.jpg"
]

# Process each image and save landmarks as JSON
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    for idx, image_path in enumerate(image_paths):
        # Load the image
        print(f"Processing image: {image_path}")  # Debugging print statement
        image = cv2.imread(image_path)

        # Check if image is loaded correctly
        if image is None:
            print(f"Error: Cannot load image at {image_path}")
            continue

        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Check if pose landmarks are detected
        if results.pose_landmarks:
            # Extract landmarks (x, y, z, visibility) and save them
            pose_landmarks = [
                {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                for lm in results.pose_landmarks.landmark
            ]

            # Save landmarks to a JSON file
            json_output_path = os.path.join(correct_pose_folder, f"correct_pose{idx + 1}.json")
            with open(json_output_path, "w") as json_file:
                json.dump(pose_landmarks, json_file, indent=4)
            print(f"Correct posture saved to {json_output_path}")
        else:
            print(f"No pose landmarks detected in image {image_path}.")
