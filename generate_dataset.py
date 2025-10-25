import pandas as pd
import numpy as np
import os

def generate_color_data(samples_per_color=500, file_path="colors.csv"):

    # Define the "center" HSV values for our primary colors.
    # These are the "perfect" versions of each color.
    color_centers = {
        'red':    [0, 220, 220],
        'orange': [15, 220, 220],
        'yellow': [30, 220, 220],
        'green':  [60, 220, 220],
        'blue':   [110, 220, 220],
        'purple': [140, 220, 220]
    }

    # Add a second range for red, as it wraps around the 0/180 mark in HSV
    color_centers_wrap = {
        'red_wrap': [175, 220, 220] 
    }

    all_colors = []

    print("Generating synthetic color data...")

    # Generate data for standard colors
    for color_name, center_hsv in color_centers.items():
        # Add random noise to the center values to create variations
        # Hue has a smaller deviation, Saturation and Value have larger deviations
        h_noise = np.random.normal(loc=center_hsv[0], scale=5, size=samples_per_color)
        s_noise = np.random.normal(loc=center_hsv[1], scale=20, size=samples_per_color)
        v_noise = np.random.normal(loc=center_hsv[2], scale=30, size=samples_per_color)

        # Clip the values to ensure they are within the valid HSV range
        h_clipped = np.clip(h_noise, 0, 179).astype(int)
        s_clipped = np.clip(s_noise, 40, 255).astype(int) # Min saturation to avoid grays
        v_clipped = np.clip(v_noise, 40, 255).astype(int) # Min value to avoid blacks

        for h, s, v in zip(h_clipped, s_clipped, v_clipped):
            all_colors.append([h, s, v, color_name])

    # Generate data for the wrapped red hue (near 180)
    for color_name, center_hsv in color_centers_wrap.items():
        h_noise = np.random.normal(loc=center_hsv[0], scale=5, size=samples_per_color)
        s_noise = np.random.normal(loc=center_hsv[1], scale=20, size=samples_per_color)
        v_noise = np.random.normal(loc=center_hsv[2], scale=30, size=samples_per_color)

        h_clipped = np.clip(h_noise, 0, 179).astype(int)
        s_clipped = np.clip(s_noise, 40, 255).astype(int)
        v_clipped = np.clip(v_noise, 40, 255).astype(int)

        for h, s, v in zip(h_clipped, s_clipped, v_clipped):
            # We label this as 'red', not 'red_wrap'
            all_colors.append([h, s, v, 'red'])

    # Create a pandas DataFrame
    df = pd.DataFrame(all_colors, columns=['hue', 'saturation', 'value', 'color_name'])

    # Save to CSV
    df.to_csv(file_path, index=False)
    
    print(f"Successfully generated '{file_path}' with {len(df)} samples.")
    print("\nDataset preview:")
    print(df.head())


if __name__ == "__main__":
    generate_color_data()