import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import json
from PIL import Image, ImageTk

# Initialize Tkinter window
root = tk.Tk()
root.title("EEG Movement Analyzer")
root.attributes('-fullscreen', True)

# Canvas for gradient background
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack(fill="both", expand=True)
for i, color in enumerate(('#87CEEB', '#FFFFFF')):  # Light blue to white gradient
    canvas.create_rectangle(0, i * root.winfo_screenheight() / 2, root.winfo_screenwidth(), (i + 1) * root.winfo_screenheight() / 2, fill=color, outline='')

# Model loading
model = tf.keras.models.load_model("motion_detection_model.keras")

# Score storage
SCORE_FILE = "game_progress.json"
progress = {"highest_score": 0, "highest_medal": "None"}
if os.path.exists(SCORE_FILE):
    with open(SCORE_FILE, 'r') as f:
        old_progress = json.load(f)
        if "highest_level" in old_progress:
            progress["highest_score"] = 100 + (30520 - 100) * (old_progress.get("highest_level", 1) - 1) / 9  # Convert level to score
            progress["highest_medal"] = old_progress.get("highest_medal", "None")
        else:
            progress = old_progress

# Global variables
selected_file = None
current_score = 0
current_medal = "None"
fig, ax = plt.subplots()
canvas_mat = None
ani = None

# Image cycling variables
current_image = 0
image_files = ["1.png", "2.png", "3.png"]
image_label = None
image_tk = None
close_button_analysis = None
quote_label = None

# Motivation goals
motivation_goals = {
    range(0, 10174): "Keep moving!",
    range(10174, 20347): "Great effort!",
    range(20347, 30521): "Champion achieved!"
}

# Function to close the app
def close_app():
    global ani, close_button_analysis, quote_label
    if ani and hasattr(ani, 'event_source') and ani.event_source:
        ani.event_source.stop()
    if close_button_analysis:
        close_button_analysis.destroy()
    if quote_label:
        quote_label.destroy()
    if image_label:
        image_label.destroy()
    with open(SCORE_FILE, 'w') as f:
        json.dump(progress, f)
    root.destroy()

# Function to close analysis view
def close_analysis():
    global ani, image_label, close_button_analysis, quote_label
    if ani and hasattr(ani, 'event_source') and ani.event_source:
        ani.event_source.stop()
    if close_button_analysis:
        close_button_analysis.destroy()
    if quote_label:
        quote_label.destroy()
    if image_label:
        image_label.destroy()
    ani = None
    image_label = None
    close_button_analysis = None
    quote_label = None

# Function to reset achievements
def reset_achievements():
    global progress
    progress = {"highest_score": 0, "highest_medal": "None"}
    with open(SCORE_FILE, 'w') as f:
        json.dump(progress, f)
    score_label.config(text=f"Score: {current_score}\nMedal: {current_medal}\nHighest Score: {progress['highest_score']}\nHighest Medal: {progress['highest_medal']}")
    messagebox.showinfo("Success", "Achievements reset to initial values!")

# Function to select file
def select_file():
    global selected_file, canvas_mat, ani, image_label, image_tk, close_button_analysis, quote_label
    if canvas_mat:
        canvas_mat.get_tk_widget().pack_forget()
        if ani and hasattr(ani, 'event_source') and ani.event_source:
            ani.event_source.stop()
    if image_label:
        image_label.pack_forget()
    if close_button_analysis:
        close_button_analysis.pack_forget()
    if quote_label:
        quote_label.pack_forget()
    file_path = filedialog.askopenfilename(
        initialdir="testfiles",
        title="Select EEG File",
        filetypes=[("NumPy files", "*.npy")]
    )
    if file_path:
        selected_file = file_path
        next_button.config(state='normal')  # Enable Next button
        status_label.config(text=f"Selected: {os.path.basename(file_path)}")
    else:
        status_label.config(text="No file selected")

# Function to cycle images
def cycle_images():
    global current_image, image_label, image_tk
    if image_label and current_score > 0:  # Cycle if score above minimum
        current_image = (current_image + 1) % len(image_files)
        image_path = image_files[current_image]
        pil_image = Image.open(image_path)
        # Fit image to frame size, maintaining aspect ratio
        frame = image_label.master
        frame_width = frame.winfo_width()
        frame_height = frame.winfo_height()
        pil_image.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
        image = ImageTk.PhotoImage(pil_image)
        image_label.config(image=image)
        image_label.image = image  # Keep a reference
        # Adjust speed: 2000ms at 0, 200ms at 30520, linear interpolation
        speed_factor = max(200, 2000 - ((current_score) * (1800 / 30520)))
        root.after(int(speed_factor), cycle_images)

# Function to animate quote
def animate_quote():
    global quote_label
    if quote_label:
        alpha = 0
        def fade_in():
            nonlocal alpha
            alpha += 0.1
            if alpha <= 1:
                quote_label.config(text=motivation_goals[[r for r in motivation_goals.keys() if current_score in r][0]], font=("Times New Roman", 24, "bold"), fg=f'#{int(255*alpha):02x}0000')
                root.after(50, fade_in)
        fade_in()

# Function to process file and update game
def process_file():
    global current_score, current_medal, canvas_mat, ani, image_label, image_tk, close_button_analysis, quote_label, progress
    if not selected_file:
        messagebox.showerror("Error", "No file selected")
        return

    try:
        # Load and preprocess data
        data = np.load(selected_file)[:, :, np.newaxis]
        predictions = model.predict(data, batch_size=16, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)
        normalized_score = np.mean(predicted_class) / 10  # Normalize to 0-1 scale
        current_score = int(100 + (30520 - 100) * normalized_score)  # Map to 100-30520
        if normalized_score == 0:  # Special case for zero prediction
            current_score = 100
        if current_score <= 10173:
            current_medal = "Bronze"
        elif current_score <= 20346:
            current_medal = "Silver"
        else:
            current_medal = "Gold"
        if current_score > progress["highest_score"]:
            progress["highest_score"] = current_score
        if current_medal == "Gold" and progress["highest_medal"] != "Gold":
            progress["highest_medal"] = "Gold"
        elif current_medal == "Silver" and progress["highest_medal"] in ["None", "Bronze"]:
            progress["highest_medal"] = "Silver"
        elif current_medal == "Bronze" and progress["highest_medal"] == "None":
            progress["highest_medal"] = "Bronze"

        # Update scores
        score_label.config(text=f"Score: {current_score}\nMedal: {current_medal}\nHighest Score: {progress['highest_score']}\nHighest Medal: {progress['highest_medal']}")
        with open(SCORE_FILE, 'w') as f:
            json.dump(progress, f)

        # Create white background frame for image
        if not image_label:
            image_frame = tk.Frame(root, bg="white")
            image_frame.place(relx=0.5, rely=0.5, anchor="center", width=root.winfo_screenwidth(), height=root.winfo_screenheight() // 2)
            image_label = tk.Label(image_frame, bg="white")
            image_label.place(relx=0.5, rely=0.5, anchor="center")
        if current_score == 100:
            image_path = "1.png"
            pil_image = Image.open(image_path)
            frame_width = image_frame.winfo_width()
            frame_height = image_frame.winfo_height()
            pil_image.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
            image = ImageTk.PhotoImage(pil_image)
            image_label.config(image=image)
            image_label.image = image
        else:
            cycle_images()

        # Add animated quote
        if not quote_label:
            quote_label = tk.Label(root, font=("Times New Roman", 24, "bold"), fg="#000000")
            quote_label.place(relx=0.5, rely=0.3, anchor="center")
        animate_quote()

        # Add close button for analysis
        if not close_button_analysis:
            close_button_analysis = tk.Button(root, text="✖", command=close_analysis, font=("Arial", 20), bg="red", fg="white")
        close_button_analysis.place(relx=0.05, rely=0.9, anchor="center")

        # Mathematical model: Correlate movement (EEG amplitude) to model output
        eeg_amplitude = np.std(data, axis=1)  # Simplified movement proxy
        movement_factor = (current_score) / (30520) * 10  # Scale factor based on score
        max_amplitude = np.max(eeg_amplitude)
        normalized_amplitude = eeg_amplitude / max_amplitude if max_amplitude > 0 else eeg_amplitude

        # Setup or update graphic
        ax.clear()
        line, = ax.plot([], [], 'g-', lw=2)  # Green line for game feel
        ax.set_xlim(0, len(normalized_amplitude))
        ax.set_ylim(0, 1.5 * movement_factor / 10)
        ax.set_title("Movement Intensity", color="#FFFFFF", fontsize=12)
        ax.set_xlabel("Sample", color="#FFFFFF")
        ax.set_ylabel("Intensity", color="#FFFFFF")
        ax.set_facecolor("#2E2E2E")
        for spine in ax.spines.values():
            spine.set_color('#FFFFFF')

        def update(frame):
            x = range(len(normalized_amplitude))
            y = normalized_amplitude * (movement_factor / 10)
            line.set_data(x, y)
            return line,

        if canvas_mat:
            canvas_mat.get_tk_widget().pack_forget()
        canvas_mat = FigureCanvasTkAgg(fig, master=root)
        canvas_mat.draw()
        canvas_mat.get_tk_widget().place(relx=0.5, rely=0.8, anchor="center", width=root.winfo_screenwidth() * 0.8, height=root.winfo_screenheight() * 0.2)
        ani = FuncAnimation(fig, update, frames=range(len(normalized_amplitude)), interval=50, blit=True)
    except Exception as e:
        messagebox.showerror("Error", f"Processing failed: {str(e)}")

# GUI Elements
status_label = tk.Label(root, text="Select an EEG file", font=("Arial", 14))
status_label.place(relx=0.5, rely=0.2, anchor="center")

select_button = tk.Button(root, text="Select File", command=select_file, font=("Arial", 12))
select_button.place(relx=0.5, rely=0.3, anchor="center")

reset_button = tk.Button(root, text="Reset Achievements", command=reset_achievements, font=("Arial", 12))
reset_button.place(relx=0.5, rely=0.35, anchor="center")

next_button = tk.Button(root, text="Next", command=process_file, font=("Arial", 12), state='disabled')
next_button.place(relx=0.5, rely=0.4, anchor="center")

score_label = tk.Label(root, text=f"Score: {current_score}\nMedal: {current_medal}\nHighest Score: {progress['highest_score']}\nHighest Medal: {progress['highest_medal']}", font=("Arial", 12))
score_label.place(relx=0.5, rely=0.1, anchor="center")

close_button = tk.Button(root, text="✖", command=close_app, font=("Arial", 20), bg="red", fg="white")
close_button.place(relx=0.05, rely=0.9, anchor="center")

# Bind Enter key to appropriate action
root.bind('<Return>', lambda event: process_file() if selected_file else select_file())

root.mainloop()