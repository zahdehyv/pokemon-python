import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os
import pandas as pd


# Function to change button color on mouse enter
def on_enter(event):
    event.widget.config(bg="yellow")  # Change background to yellow

# Function to change button color on mouse leave
def on_leave(event):
    event.widget.config(bg="black")  # Change background back to black

# Function to switch frames
def show_frame(frame_to_show, frame_to_hide):
    frame_to_hide.pack_forget()  # Hide the frame that is currently visible
    frame_to_show.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)  # Show the desired frame

# Create the main application window
root = tk.Tk()
root.title("Pokémon Simulation and AI")
root.geometry("800x600")
root.configure(bg="black")  # Set the global background to black

# Load the Pokémon logo image
logo_path = "pokemon_logo.png"  # Replace with your logo image path
if os.path.exists(logo_path):
    logo_image = Image.open(logo_path)
    logo_photo = ImageTk.PhotoImage(logo_image)

# Create a frame for the main menu buttons
button_frame = tk.Frame(root, bg="black")
button_frame.pack(pady=20)

# Create the logo label with transparent background only in the main menu
if os.path.exists(logo_path):
    logo_label = tk.Label(button_frame, image=logo_photo, bg="black")
    logo_label.pack(pady=20)

# Create buttons for various actions with 8-bit font, white font color, and transparent background
button_font = ("Press Start 2P", 14)  # Use the installed 8-bit font

# Create frames for each section
training_room_frame = tk.Frame(root, bg="black")
simulation_box_frame = tk.Frame(root, bg="black")
battle_coliseum_frame = tk.Frame(root, bg="black")
pokedex_frame = tk.Frame(root, bg="black")

# Function to create a back button
def create_back_button(frame):
    back_button = tk.Button(
        frame,
        text="Back to Menu",
        font=button_font,
        fg="white",
        bg="black",
        command=lambda: show_frame(button_frame, frame)  # Show the button frame
    )
    back_button.pack(pady=10)  # Pack the back button below the title

# Create buttons for the main menu
def create_main_menu_button(text, target_frame):
    button = tk.Button(
        button_frame,
        text=text,
        font=button_font,
        fg="white",
        bg="black",
        bd=0,
        highlightthickness=0,
        command=lambda: show_frame(target_frame, button_frame)  # Show the target frame
    )
    button.bind("<Enter>", on_enter)  # Bind mouse enter event
    button.bind("<Leave>", on_leave)  # Bind mouse leave event
    button.pack(pady=10)

# Create main menu buttons
create_main_menu_button("Training Room", training_room_frame)
create_main_menu_button("Simulation Box", simulation_box_frame)
create_main_menu_button("Battle Coliseum", battle_coliseum_frame)
create_main_menu_button("Pokedex", pokedex_frame)

# Add content to each frame
for frame, name in zip(
    [training_room_frame, simulation_box_frame, battle_coliseum_frame, pokedex_frame],
    ["Training Room", "Simulation Box", "Battle Coliseum", "Pokedex"]
):
    # Reduce the font size of the title
    label = tk.Label(frame, text=name, font=("Press Start 2P", 20), fg="white", bg="black")  # Reduced font size
    label.place(relx=0.5, rely=0.4, anchor="center")  # Center the title label
    
    # Create the back button for each frame
    create_back_button(frame)

# Add a table to the Pokedex frame
def create_pokedex_table():
    # Load the CSV file
    csv_file_path = "data/pokemon_view_pokedex.csv"  # Replace with your CSV file path
    if os.path.exists(csv_file_path):
        df = (pd.read_csv(csv_file_path)[['No', 'Name', 'Type 1', 'Type 2']])[:807]

        # Create a Treeview widget for the table
        tree = ttk.Treeview(pokedex_frame, columns=list(df.columns), show='headings')
        tree.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Set the style for the table
        style = ttk.Style()
        style.configure("Custom.Treeview", background="black", foreground="white", fieldbackground="black", borderwidth=0)
        tree.configure(style="Custom.Treeview")

        # Define the column headings
        for column in df.columns:
            tree.heading(column, text=column)
            tree.column(column, anchor="center", width=100)  # Set column width

        # Insert the data into the Treeview

        # Configure the type color tags
        tree.tag_configure('Normal', background='#A8A77A')  # Keep background black for default
        tree.tag_configure('Fire', background='#EE8130')  # Keep background black for default
        tree.tag_configure('Water', background='#6390F0')  # Keep background black for default
        tree.tag_configure('Electric', background='#F7D02C')  # Keep background black for default
        tree.tag_configure('Grass', background='#7AC74C')  # Keep background black for default
        tree.tag_configure('Ice', background='#96D9D6')  # Keep background black for default
        tree.tag_configure('Fighting', background='#C22E28')  # Keep background black for default
        tree.tag_configure('Poison', background='#A33EA1')  # Keep background black for default
        tree.tag_configure('Ground', background='#E2BF65')  # Keep background black for default
        tree.tag_configure('Flying', background='#A98FF3')  # Keep background black for default
        tree.tag_configure('Psychic', background='#F95587')  # Keep background black for default
        tree.tag_configure('Bug', background='#A6B91A')  # Keep background black for default
        tree.tag_configure('Rock', background='#B6A136')  # Keep background black for default
        tree.tag_configure('Ghost', background='#735797')  # Keep background black for default
        tree.tag_configure('Dragon', background='#6F35FC')  # Keep background black for default
        tree.tag_configure('Dark', background='#705746')  # Keep background black for default
        tree.tag_configure('Steel', background='#B7B7CE')  # Keep background black for default
        tree.tag_configure('Fairy', background='#D685AD')  # Keep background black for default
        
        for index, row in df.iterrows():
            type1_color = row['Type 1']
            tree.insert("", "end", values=(row['No'], row['Name'], row['Type 1'], row['Type 2']), tags=(type1_color))
                

# Function to get the type color
def get_type_color(type_name):
    type_colors = {
        "Normal": "#A8A77A",
        "Fire": "#EE8130",
        "Water": "#6390F0",
        "Electric": "#F7D02C",
        "Grass": "#7AC74C",
        "Ice": "#96D9D6",
        "Fighting": "#C22E28",
        "Poison": "#000000",
        "Ground": "#E2BF65",
        "Flying": "#A98FF3",
        "Psychic": "#F95587",
        "Bug": "#A6B91A",
        "Rock": "#B6A136",
        "Ghost": "#735797",
        "Dragon": "#6F35FC",
        "Dark": "#705746",
        "Steel": "#B7B7CE",
        "Fairy": "#D685AD"
    }
    return type_colors.get(type_name, "#FFFFFF")  # Default color if type not found

# Call the function to create the Pokedex table
create_pokedex_table()

# Show the button frame by default
show_frame(button_frame, training_room_frame)  # Initially show the button frame

# Run the application
root.mainloop()  # This line starts the Tkinter event loop and creates the window
