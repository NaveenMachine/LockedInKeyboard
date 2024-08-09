import numpy as np


def parse_eye_file(eye_file_path):
    left_eye = [0, 0]
    right_eye = [0, 0]
    
    with open(eye_file_path, 'r') as file:
    # Read the entire file content
        next(file)
        content = file.read()

        # Split the content by spaces
        numbers = content.split()

        # Convert each split string into a number (int or float)
        numbers = [int(num) for num in numbers]  # or float(num) for float numbers
    left_eye = [numbers[0], numbers[1]]
    right_eye = [numbers[2], numbers[3]]
    
    return left_eye, right_eye

# Example usage
eye_file_path = './Eye_data/BioID_0000.eye'
left_eye, right_eye = parse_eye_file(eye_file_path)
print(f"Left Eye: {left_eye}, Right Eye: {right_eye}")

