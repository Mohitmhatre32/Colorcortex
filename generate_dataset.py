import pandas as pd
import numpy as np

def generate_color_data(samples_per_color=1000, file_path="colors.csv"):
    
    all_colors = []
    print("Generating synthetic color data...")

    # --- 1. Standard Chromatic Colors ---
    # We keep the min saturation/value to 40 to ensure they look like "colors"
    color_centers = {
        'red':    [0, 220, 220],
        'orange': [15, 220, 220],
        'yellow': [30, 220, 220],
        'green':  [60, 220, 220],
        'blue':   [110, 220, 220],
        'purple': [140, 220, 220]
    }

    for color_name, center_hsv in color_centers.items():
        h_noise = np.random.normal(loc=center_hsv[0], scale=5, size=samples_per_color)
        s_noise = np.random.normal(loc=center_hsv[1], scale=20, size=samples_per_color)
        v_noise = np.random.normal(loc=center_hsv[2], scale=30, size=samples_per_color)

        h_clipped = np.clip(h_noise, 0, 179).astype(int)
        s_clipped = np.clip(s_noise, 50, 255).astype(int) # High Saturation
        v_clipped = np.clip(v_noise, 50, 255).astype(int) # High Value

        for h, s, v in zip(h_clipped, s_clipped, v_clipped):
            all_colors.append([h, s, v, color_name])
            
    # --- 2. Red Wrap (High Hue) ---
    h_noise = np.random.normal(loc=175, scale=5, size=samples_per_color)
    s_noise = np.random.normal(loc=220, scale=20, size=samples_per_color)
    v_noise = np.random.normal(loc=220, scale=30, size=samples_per_color)
    
    h_clipped = np.clip(h_noise, 0, 179).astype(int)
    s_clipped = np.clip(s_noise, 50, 255).astype(int)
    v_clipped = np.clip(v_noise, 50, 255).astype(int)
    
    for h, s, v in zip(h_clipped, s_clipped, v_clipped):
        all_colors.append([h, s, v, 'red'])

    # --- 3. BLACK Generation ---
    # Black is defined by Low Value (Brightness). Hue and Saturation can vary wildly.
    # We generate random Hue/Sat, but keep Value very low (0-50).
    for _ in range(samples_per_color):
        h = np.random.randint(0, 180)
        s = np.random.randint(0, 256)
        v = np.random.randint(0, 60) # Very dark
        all_colors.append([h, s, v, 'black'])

    # --- 4. WHITE Generation ---
    # White is defined by Low Saturation and High Value.
    for _ in range(samples_per_color):
        h = np.random.randint(0, 180)
        s = np.random.randint(0, 40)   # Very low saturation (pale)
        v = np.random.randint(200, 256) # Very bright
        all_colors.append([h, s, v, 'white'])

    # Save
    df = pd.DataFrame(all_colors, columns=['hue', 'saturation', 'value', 'color_name'])
    df.to_csv(file_path, index=False)
    
    print(f"Successfully generated '{file_path}' with {len(df)} samples.")
    print("Unique classes:", df['color_name'].unique())

if __name__ == "__main__":
    generate_color_data()