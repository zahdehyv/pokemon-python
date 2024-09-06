import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os

# Function to handle button clicks
def on_button_click(action):
    messagebox.showinfo("Action", f"You selected: {action}")

# Function to change button color on mouse enter
def on_enter(event):
    event.widget.config(bg="yellow")  # Change background to yellow

# Function to change button color on mouse leave
def on_leave(event):
    event.widget.config(bg="black")  # Change background back to black

# Create the main application window
root = tk.Tk()
root.title("Pokémon Simulation and AI")
root.geometry("800x600")
root.configure(bg="black")  # Set the global background to black

# Load the Pokémon logo image
logo_path = "pokemon_logo.png"  # Replace with your logo image path
if os.path.exists(logo_path):
    logo_image = Image.open(logo_path)
    # logo_image = logo_image.resize((400, 200), Image.ANTIALIAS)
    logo_photo = ImageTk.PhotoImage(logo_image)

    # Create the logo label with transparent background
    logo_label = tk.Label(root, image=logo_photo, bg="black")
    logo_label.pack(pady=20)
else:
    print(f"Error: The image file '{logo_path}' does not exist.")

# Create a frame for the buttons with transparent background
button_frame = tk.Frame(root, bg="black")
button_frame.pack(pady=20)

# Create buttons for various actions with 8-bit font, white font color, and transparent background
button_font = ("Press Start 2P", 14)  # Use the installed 8-bit font

training_room_button = tk.Button(
    button_frame, 
    text="Training Room", 
    font=button_font, 
    fg="white", 
    bg="black",  # Set initial background to black
    bd=0,  # Remove border
    highlightthickness=0,  # Remove highlight border
    command=lambda: on_button_click("Training Room")
)
training_room_button.bind("<Enter>", on_enter)  # Bind mouse enter event
training_room_button.bind("<Leave>", on_leave)  # Bind mouse leave event
training_room_button.pack(pady=10)

simulation_button = tk.Button(
    button_frame, 
    text="Simulation Box", 
    font=button_font, 
    fg="white", 
    bg="black",  # Set initial background to black
    bd=0,  # Remove border
    highlightthickness=0,  # Remove highlight border
    command=lambda: on_button_click("Simulation Box")
)
simulation_button.bind("<Enter>", on_enter)  # Bind mouse enter event
simulation_button.bind("<Leave>", on_leave)  # Bind mouse leave event
simulation_button.pack(pady=10)

battle_coliseum_button = tk.Button(
    button_frame, 
    text="Battle Coliseum", 
    font=button_font, 
    fg="white", 
    bg="black",  # Set initial background to black
    bd=0,  # Remove border
    highlightthickness=0,  # Remove highlight border
    command=lambda: on_button_click("Battle Coliseum")
)
battle_coliseum_button.bind("<Enter>", on_enter)  # Bind mouse enter event
battle_coliseum_button.bind("<Leave>", on_leave)  # Bind mouse leave event
battle_coliseum_button.pack(pady=10)

pokedex_button = tk.Button(
    button_frame, 
    text="Pokedex", 
    font=button_font, 
    fg="white", 
    bg="black",  # Set initial background to black
    bd=0,  # Remove border
    highlightthickness=0,  # Remove highlight border
    command=lambda: on_button_click("Pokedex")
)
pokedex_button.bind("<Enter>", on_enter)  # Bind mouse enter event
pokedex_button.bind("<Leave>", on_leave)  # Bind mouse leave event
pokedex_button.pack(pady=10)

exit_button = tk.Button(
    root, 
    text="Exit", 
    font=button_font, 
    fg="white", 
    bg="black",  # Set initial background to black
    bd=0,  # Remove border
    highlightthickness=0,  # Remove highlight border
    command=root.quit
)
exit_button.bind("<Enter>", on_enter)  # Bind mouse enter event
exit_button.bind("<Leave>", on_leave)  # Bind mouse leave event
exit_button.pack(pady=20)

# Run the application
root.mainloop()  # This line starts the Tkinter event loop and creates the window
