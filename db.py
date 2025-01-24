import pandas as pd

# Function to read the dataset and calculate average/ideal angles for each joint
def calculate_ideal_angles(csv_file_path, output_csv_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Print column names to debug
    print("Columns in the CSV file:", df.columns)

    # List of angle columns for which we need to calculate averages
    angle_columns = [
        'Shoulder_Angle', 'Elbow_Angle', 'Hip_Angle', 'Knee_Angle', 'Ankle_Angle',
        'Shoulder_Ground_Angle', 'Elbow_Ground_Angle', 'Hip_Ground_Angle', 'Knee_Ground_Angle', 'Ankle_Ground_Angle'
    ]

    # Dictionary to store the average angles for each joint
    ideal_angles = {}

    # Calculate average angles for each of the listed columns
    for angle_column in angle_columns:
        avg_angle = df[angle_column].mean()  # Calculate the average angle
        ideal_angles[angle_column] = avg_angle
    
    # Convert the ideal angles into a DataFrame for saving
    ideal_angles_df = pd.DataFrame.from_dict(ideal_angles, orient='index', columns=['Avg_Angle'])

    # Save the new ideal angles to a new CSV file
    ideal_angles_df.to_csv(output_csv_path)
    print(f"New CSV with ideal angles has been saved to {output_csv_path}")

# Call the function with the path to your CSV dataset
input_csv_path = 'pushup2_dataset.csv'  # Path to your input CSV dataset
output_csv_path = 'pushup_ideal_angles.csv'  # Path to the output CSV with average angles

calculate_ideal_angles(input_csv_path, output_csv_path)
