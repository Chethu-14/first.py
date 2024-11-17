import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from ubidots import ApiClient
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import uuid

# Ubidots configuration
api = ApiClient(token="BBUS-MIs0fBgBzDisKlas3T9ptmTIID4xlK")
Variable = api.get_variable("6612bc645e660d163be32eaf")

# Load the saved model
try:
    model = load_model("Fruitmodel.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Dictionary mapping class indices to fruit names
class_names = {
    1: 'Apple',
    2: 'Apple',
    3: 'Apple',
    4: 'Apple',
    5: 'Apple',
    6: 'Apple',
    7: 'Apple',
    8: 'Apple',
    9: 'Apple',
    16: 'Banana',
    17: 'Banana',
    18: 'Banana',
    14: 'Avocado',
    64: 'Mango',
    56: 'Kiwi',
    # Add more class names as needed based on your dataset
}

# Load the image
img_path = "C:/Users/Chethana BV/Downloads/Fruit_recognition/Fruit_recognition/fruits-360_dataset/fruits-360/Training/Apple Golden 1/r_29_100.jpg"
try:
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Make prediction using the loaded model
try:
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()

# Get the fruit name from class index
fruit_name = class_names.get(predicted_class, "Unknown")

print("Predicted class index:", predicted_class)
print("Predicted fruit name:", fruit_name)

# Update Ubidots variable based on prediction
try:
    if predicted_class in {1, 2, 3, 4, 5, 6, 7, 8, 9}:
        Variable.save_value({'value': 1})
        print("Round")
    elif predicted_class in {16, 17, 18}:
        Variable.save_value({'value': 2})
        print("Moon")
    elif predicted_class == 14:
        Variable.save_value({'value': 3})
    elif predicted_class == 64:
        Variable.save_value({'value': 4})
    elif predicted_class == 56:
        Variable.save_value({'value': 5})
except Exception as e:
    print(f"Error updating Ubidots: {e}")

# Function to find closest match using ORB
def find_closest_match(kp1, des1):
    dataset_dir = "C:/Users/Chethana BV/Downloads/Fruit_recognition/Fruit_recognition/fruits-360_dataset/fruits-360/Training"
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    closest_match = None
    max_matches = 0

    if des1 is None:
        return closest_match, max_matches

    for class_id, class_name in class_names.items():
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            kp2, des2 = cv2.ORB_create().detectAndCompute(img, None)

            if des2 is None:
                continue

            if des1.shape[1] != des2.shape[1]:
                continue

            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            num_matches = len(matches)

            if num_matches > max_matches:
                max_matches = num_matches
                closest_match = class_name

    return closest_match, max_matches

# Function to display image in a new window
def display_image(image_path):
    img_window = tk.Toplevel()
    img_window.attributes('-fullscreen', True)

    img = Image.open(image_path)
    photo = ImageTk.PhotoImage(img)

    img_label = tk.Label(img_window, image=photo)
    img_label.image = photo
    img_label.pack(expand=True)

    close_button = tk.Button(img_window, text="Close", command=img_window.destroy, font=('Helvetica', 20))
    close_button.pack()

# Function to capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera")
        return

    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Unable to capture frame")
        cap.release()
        return

    # Define the directory paths for storing images
    dataset_dir = "C:/Users/Chethana BV/Downloads/Fruit_recognition/Fruit_recognition/fruits-360_dataset/fruits-360/Training"
    current_images_dir = os.path.join(dataset_dir, "current_images")
    processed_images_dir = os.path.join(dataset_dir, "processed_images")

    # Create the directories if they don't exist
    os.makedirs(current_images_dir, exist_ok=True)
    os.makedirs(processed_images_dir, exist_ok=True)

    # Save the captured image with a unique filename
    image_filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(current_images_dir, image_filename)
    cv2.imwrite(image_path, frame)

    cap.release()
    messagebox.showinfo("Success", f"Image captured successfully and saved to {image_path}!")

    # Display the captured image in full-screen
    display_image(image_path)

    # Classify the captured image
    from fi24 import classify_image
    classify_image(image_path)

    # Save processed image
    processed_image_path = os.path.join(processed_images_dir, image_filename)
    try:
        # Assuming some processing is done here; for now, just save the original frame
        cv2.imwrite(processed_image_path, frame)
        print(f"Processed image saved to: {processed_image_path}")
    except Exception as e:
        print(f"Error saving processed image: {e}")

# Function to create gradient background
def create_gradient(canvas, width, height, color1, color2):
    r1, g1, b1 = canvas.winfo_rgb(color1)
    r2, g2, b2 = canvas.winfo_rgb(color2)
    r_ratio = float(r2 - r1) / height
    g_ratio = float(g2 - g1) / height
    b_ratio = float(b2 - b1) / height

    for i in range(height):
        nr = int(r1 + (r_ratio * i))
        ng = int(g1 + (g_ratio * i))
        nb = int(b1 + (b_ratio * i))
        color = f'#{nr:04x}{ng:04x}{nb:04x}'
        canvas.create_line(0, i, width, i, fill=color, tags=("gradient",))

# Main interface function
def main_interface():
    window = tk.Tk()
    window.title("Image Capture and Classification")
    window.attributes('-fullscreen', True)

    canvas = tk.Canvas(window, width=window.winfo_screenwidth(), height=window.winfo_screenheight())
    canvas.pack(fill="both", expand=True)

    create_gradient(canvas, window.winfo_screenwidth(), window.winfo_screenheight(), "#ffcc66", "#ff9966")

    label = tk.Label(window, text="Welcome! Choose an option below:", font=('Helvetica', 24), bg='#ffcc66')
    canvas.create_window(window.winfo_screenwidth() // 2, 100, window=label)

    capture_button = tk.Button(window, text="Capture Image", command=capture_image, font=('Helvetica', 20), bg='#ff9966', fg='white')
    canvas.create_window(window.winfo_screenwidth() // 2, 200, window=capture_button)

    window.mainloop()

# Run program interface
def run_program():
    app_window = tk.Tk()
    app_window.title("Run Program")
    app_window.attributes('-fullscreen', True)

    canvas = tk.Canvas(app_window, width=app_window.winfo_screenwidth(), height=app_window.winfo_screenheight())
    canvas.pack(fill="both", expand=True)

    create_gradient(canvas, app_window.winfo_screenwidth(), app_window.winfo_screenheight(), "#ffcc66", "#ff9966")

    label = tk.Label(app_window, text="Welcome To Our Project: Color-Based Fruit Sorting Machine\nPresented By\nChethana BV\nAbhishek M\nHarshitha S K\nDarshan", font=('Helvetica', 15), bg='#ffcc66', justify='center')
    canvas.create_window(app_window.winfo_screenwidth() // 2, 100, window=label)

    run_button = tk.Button(app_window, text="Run", command=lambda: [app_window.destroy(), main_interface()], font=('Helvetica', 20), bg='#ff9966', fg='white')
    canvas.create_window(app_window.winfo_screenwidth() // 2, app_window.winfo_screenheight() // 2, window=run_button)

    app_window.mainloop()

# Start the program
run_program()
