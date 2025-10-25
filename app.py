import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import joblib

class ColorClassifier:
    def __init__(self, model_path="color_model.pkl"):
        try:
            self.model = joblib.load(model_path)
            print("Machine learning model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found at '{model_path}'. Please run train_model.py.")
            self.model = None
    def predict_color(self, hsv_pixel):
        if self.model is None: return "Model not loaded"
        # The .values change in train_model.py fixed the warning, so we send a NumPy array
        pixel_to_predict = np.array(hsv_pixel).reshape(1, -1)
        return self.model.predict(pixel_to_predict)[0]

class VideoStream:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            return (ret, frame) if ret else (ret, None)
        return (False, None)
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class AutomaticColorProcessor:
    def __init__(self):
        self.classifier = ColorClassifier()
        self.color_ranges = {
            'red':    ([161, 155, 84], [179, 255, 255]),
            'red_wrap': ([0, 155, 84], [10, 255, 255]), # Red also exists near 0 hue
            'blue':   ([94, 80, 2], [126, 255, 255]),
            'green':  ([40, 40, 40], [90, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'orange': ([10, 100, 20], [25, 255, 255]),
            'purple': ([129, 50, 70], [158, 255, 255])
        }
        self.kernel = np.ones((5, 5), np.uint8)
        self.MIN_OBJECT_AREA = 500 # Minimum size of an object to be drawn

    def process_frame(self, frame):
        output_frame = frame.copy()
        
        try:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            
            for color_name, (lower, upper) in self.color_ranges.items():
                
                # Create a mask using the fast cv2.inRange function
                mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
                
                # Clean up the mask
                mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
                
                # Find objects within this specific color mask
                contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) > self.MIN_OBJECT_AREA:
                        
                        
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        
                        contour_mask = np.zeros_like(mask_cleaned)
                        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                        mean_hsv = cv2.mean(hsv_frame, mask=contour_mask)
                        predicted_name = self.classifier.predict_color(mean_hsv[:3])

                        # Draw one clean box per object
                        # If the color is red_wrap, we still want to label it as "Red"
                        display_name = color_name.replace('_wrap', '')
                        
                        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(output_frame, display_name.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        except Exception as e:
            print(f"ERROR in process_frame: {e}")
            return frame
            
        return output_frame


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_stream = VideoStream()
        self.processor = AutomaticColorProcessor()

        # A simple GUI with just one canvas and a quit button
        self.canvas = tk.Canvas(window, width=self.video_stream.width, height=self.video_stream.height)
        self.canvas.pack(padx=10, pady=10)
        
        self.btn_quit = ttk.Button(window, text="Quit", command=self.quit)
        self.btn_quit.pack(pady=10)
        
        self.delay = 30 
        self.update()
        self.window.mainloop()

    def update(self):
        ret, frame = self.video_stream.get_frame()
        if ret:
         
            processed_frame = self.processor.process_frame(frame)
            
            # result
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def quit(self):
        self.window.destroy()

if __name__ == "__main__":
    App(tk.Tk(), "Color Detector")