import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os
from time import sleep
import pandas as pd
import random
from misc_utils import PokemonEntity, load_pokemon, save_pokemon, BasicPkmnLogic, GeneticEvolution
from nlp_utils import call_gemini_api
from pathlib import Path
from tkinter import filedialog
import threading

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
root.pokedex_data = None
csv_file_path = "data/pokemon_view_pokedex.csv"  # Replace with your CSV file path
if os.path.exists(csv_file_path):
    root.pokedex_data = (pd.read_csv(csv_file_path)[['No', 'Name', 'Type 1', 'Type 2']])[:807]


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

def create_nav_button(frame, next_frame, caption):
    back_button = tk.Button(
        frame,
        text=caption,
        font=button_font,
        fg="white",
        bg="black",
        bd=0,
        highlightthickness=0,
        command=lambda: show_frame(next_frame, frame)  # Show the button frame
    )
    back_button.pack(pady=10)  # Pack the back button below the title

# Function to create a back button
def create_back_button(frame):
    back_button = tk.Button(
        frame,
        text="Back to Menu",
        font=button_font,
        fg="white",
        bg="black",
        bd=0,
        highlightthickness=0,
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
        command=lambda: show_frame(generate_frame(target_frame), button_frame)  # Show the target frame
    )
    button.bind("<Enter>", on_enter)  # Bind mouse enter event
    button.bind("<Leave>", on_leave)  # Bind mouse leave event
    button.pack(pady=10)

def generate_frame(x) -> tk.Frame:
    if x == 1:
        def train(ind):
            lbl_trn =tk.Label(frame, text="Training "+str(root.pokedex_data["Name"].iloc[int(ind)-1])+" -> "+str(0.0),fg="white", bg="black")
            lbl_trn.pack()
            
            genetic = GeneticEvolution(lambda: int(ind))
            def set_fitness(value):
                lbl_trn.config(text="Training "+str(root.pokedex_data["Name"].iloc[int(ind)-1])+" -> "+str(value)+" / "+str(genetic.m_fitness))
            genetic.ax_method = set_fitness
            for pat in Path("./pokemons/to_train/").iterdir():
                genetic.available_pokemons.append(load_pokemon(pat))
            
            print("\ntraining",PokemonEntity(int(ind),None, None, False).entity[0].species, "with:")
            for pokemn in genetic.available_pokemons:
                print(" -",pokemn.entity[0].species)
        
            genetic.genome_id_to_pokemon = {}
            
            top_genome = genetic.run_i_gen_training(int(generations.get()), brain_size.get())[0]
            winner: PokemonEntity = genetic.genome_id_to_pokemon[top_genome.key]
            
            print()
            print("Results:")
            print("Specie:", winner.entity[0].species)
            print("Won:",winner.won_battles, "from", winner.total_battles)
            print("Level:",winner.lvl)
            print("Fitness:", winner.genome.fitness)
            print("Ability:", winner.entity[0].ability)
            print("Nature:", winner.entity[0].nature)
            print("Moves:", *winner.entity[0].moves, sep = "\n - ")
            print()
            # print("press a key")
            # input()
            if top_genome.fitness > 0.9:
                save_pokemon(winner)
                print("pokemon_saved")
            genetic.available_pokemons.append(winner)
            
            lbl_trn.pack_forget()
            
        def thread_train(ind):
            train_thrd = threading.Thread(target=train, args=(ind,))
            train_thrd.start()
            
        frame = tk.Frame(root, bg="black")
        create_back_button(frame)
        # training room
        pok_sel = tk.Label(frame, text="Select Pokemon ID:",fg="white", bg="black")
        pok_sel.pack()
        # pok_sel.pack_forget()
        pokemon_number = tk.StringVar()
        number_selector = ttk.Combobox(frame, textvariable=pokemon_number, width=5, state='readonly',background="black")
        number_selector['values'] = tuple(range(1, 808))
        number_selector.current(0)
        number_selector.pack()
        
        gens_sel = tk.Label(frame, text="Select Train Gens:",fg="white", bg="black")
        gens_sel.pack()
        generations = tk.StringVar()
        gens_selector = ttk.Combobox(frame, textvariable=generations, width=5, state='readonly',background="black")
        gens_selector['values'] = tuple([None]+list(range(1, 1000)))
        gens_selector.current(0)
        gens_selector.pack()
        
        brain_sel = tk.Label(frame, text="Select Brain Size:",fg="white", bg="black")
        brain_sel.pack()
        brain_size = tk.StringVar()
        brain_selector = ttk.Combobox(frame, textvariable=brain_size, width=5, state='readonly',background="black")
        brain_selector['values'] = ("S", "M", "L")
        brain_selector.current(0)
        brain_selector.pack()
        
        train_button = tk.Button(
            frame,
            text="+",
            font=button_font,
            fg="white",
            bg="black",
            bd=0,
            highlightthickness=0,
            command=lambda:thread_train(pokemon_number.get()))
        train_button.pack(pady=20)
        return frame
    if x == 2:
        frame = tk.Frame(root, bg="black")
        create_back_button(frame)
        return frame
    if x == 3:
        pkmns = [None, None]
        print("a")
        frame = tk.Frame(root, bg="black")
        create_back_button(frame)
        # Create a frame for the battle buttons
        battle_buttons_frame = tk.Frame(frame, bg="black")
        battle_buttons_frame.place(relx=0.5, rely=0.5, anchor="center")


        select_pokemon1 = tk.Button(
            battle_buttons_frame,
            text="Select Pokemon",
            font=button_font,
            fg="white",
            bg="black",
            bd=0,
            highlightthickness=0,
            command=lambda:select_pkmn_s_battle(0)
        )
        select_pokemon1.bind("<Enter>", on_enter)  # Bind mouse enter event
        select_pokemon1.bind("<Leave>", on_leave)  # Bind mouse leave event
        select_pokemon1.pack(pady=10)
        
        select_pokemon2 = tk.Button(
            battle_buttons_frame,
            text="Select Pokemon",
            font=button_font,
            fg="white",
            bg="black",
            bd=0,
            highlightthickness=0,
            command=lambda:select_pkmn_s_battle(1)
        )
        select_pokemon2.bind("<Enter>", on_enter)  # Bind mouse enter event
        select_pokemon2.bind("<Leave>", on_leave)  # Bind mouse leave event
        select_pokemon2.pack(pady=10)
        
        def select_pkmn_s_battle(x):
            # Open a file selection dialog and store the selected file path
            file_path = filedialog.askopenfilename(initialdir="./pokemons")
            if file_path:
                    try:
                        pkmns[x] = load_pokemon(file_path)
                        pkmns[x].entity[0].level = 100
                        pkmns[x].lvl = 100
                        print("loaded",x)
                    except:
                        print("no se pudo cargar el pokemon")
                    if pkmns[x]:
                        if type(pkmns[x]) is PokemonEntity:
                            if x == 0:
                                select_pokemon1.config(text=pkmns[x].entity[0].species)
                                pkmns[x].lvl = 100
                                pkmns[x].entity[0].level = 100
                            if x == 1:
                                select_pokemon2.config(text=pkmns[x].entity[0].species)
                                pkmns[x].lvl = 100
                                pkmns[x].entity[0].level = 100
                
            
            
        def do_battle_and_narrate():
            pkmn_logic = BasicPkmnLogic()
            print(type(pkmns[0]))
            print(type(pkmns[1]))
            if pkmns[0] and pkmns[1]:
                pkmn_logic.battle(pkmns[0], pkmns[1])
                print("thinking")
                answer = narrate_battle_logs(pkmn_logic.lb_log, pkmns[0], pkmns[1])
                print("finished")
                
                window = tk.Tk()
                window.title("Battle Narrative")
                window.configure(bg='black')  # Set the background color to black

                # Create a label in the window with white text
                label = ttk.Label(window, text=answer, background='black', foreground='white', wraplength=750, font=("Arial", 11, "bold"))
                label.pack(pady=20)

                # Create an "Accept" button in the window with white text
                button = ttk.Button(window, text="Accept", command=window.destroy, style='TButton')
                button.pack(pady=10)

                # Configure the button style to have a black background and white text
                style = ttk.Style()
                style.configure('TButton', background='black', foreground='white')

                # Run the main event loop
                window.mainloop()
                
        def narrative_open_popup():
            # Create a new thread to handle the popup creation
            popup_thread = threading.Thread(target=do_battle_and_narrate)
            popup_thread.start()
            
        # Create the automatic AI battle button
        auto_battle_button = tk.Button(
            battle_buttons_frame,
            text="Automatic AI Battle",
            font=button_font,
            fg="white",
            bg="black",
            bd=0,
            highlightthickness=0,
            command=narrative_open_popup
        )
        auto_battle_button.bind("<Enter>", on_enter)  # Bind mouse enter event
        auto_battle_button.bind("<Leave>", on_leave)  # Bind mouse leave event
        auto_battle_button.pack(pady=10)

        def run_manual_battle():
            def set_hp(hp1_value, hp2_value):
                """Set the health points for both bars and update their colors."""
                hp1['value'] = hp1_value
                hp2['value'] = hp2_value
            
            def calculate_hp_p():
                a = pkmn_logic.manual_battle.p1.active_pokemon[0].hp/pkmn_logic.manual_battle.p1.active_pokemon[0].maxhp
                a = a*100
                b = pkmn_logic.manual_battle.p2.active_pokemon[0].hp/pkmn_logic.manual_battle.p2.active_pokemon[0].maxhp
                b = b*100
                return a,b
                
            if pkmns[0] and pkmns[1]:
                pkmn_logic = BasicPkmnLogic()
                pkmn_logic.manual_battle_create(pkmns[0], pkmns[1])
                battle_frame = tk.Frame(root, bg='black')
                create_back_button(battle_frame)
                style = ttk.Style()
                style.configure("Custom.Horizontal.TProgressbar", troughcolor='black', background='green')
                # log_labels = []
                # for _ in range(4):
                #     ll = tk.Label(battle_frame,text="",fg="white", bg="black")
                #     ll.pack()
                #     log_labels.append(ll)
                turn_moves = tk.Label(battle_frame,text="",fg="white", bg="black", wraplength=500)
                turn_moves.pack()
                def caption_change(log):
                    logs = []
                    for lg in reversed(log):
                        lg: str
                        logs.insert(0, lg)
                        if lg.startswith("Turn"):
                            break
                    turn_moves.config(text=get_caption(logs))
                name1 = tk.Label(battle_frame,text="your's "+pkmns[0].entity[0].species,fg="white", bg="black")
                name1.pack()
                hp1 = ttk.Progressbar(battle_frame, length=150, mode='determinate', maximum=100, style="Custom.Horizontal.TProgressbar")
                hp1.pack()
                name2 = tk.Label(battle_frame,text="foe's "+pkmns[1].entity[0].species,fg="white", bg="black")
                name2.pack()
                hp2 = ttk.Progressbar(battle_frame, length=150, mode='determinate', maximum=100, style="Custom.Horizontal.TProgressbar")
                hp2.pack()
                set_hp(*calculate_hp_p())
                    
                show_frame(battle_frame, frame)

                def make_turn(selection):
                    print("selected", selection)
                    ended = pkmn_logic.manual_battle_do_turn(pkmns[0], pkmns[1], selection)
                    set_hp(*calculate_hp_p())
                    print(calculate_hp_p())
                    # mn = min(len(log_labels), len(pkmn_logic.manual_battle.log))
                    # for i in range(mn):
                    #     log_labels[i-mn].config(text = pkmn_logic.manual_battle.log[i-mn])
                    #     print(pkmn_logic.manual_battle.log[i-mn])
                    cpt_thread = threading.Thread(target=caption_change, args=(pkmn_logic.manual_battle.log,))
                    cpt_thread.start()
                    if ended:
                        result = tk.Label(battle_frame,text="YOU WIN" if pkmn_logic.manual_battle.winner =="p1" else "YOU LOSE",fg="white", bg="black", font=("Arial", 50, "bold"))
                        # for pkc in [name1,hp1,name2,hp2,a0_button,a1_button,a2_button,a3_button,pass_button,run_button]:
                        #     pkc.pack_forget()
                        result.pack()
                        create_nav_button(battle_frame, frame, "Back to Colisseum")
                        a0_button.pack_forget()
                        a1_button.pack_forget()
                        a2_button.pack_forget()
                        a3_button.pack_forget()
                        
                        pass_button.pack_forget()
                        run_button.pack_forget()
                        
                    
                a0_button = tk.Button(
                battle_frame,
                text=pkmns[0].entity[0].moves[0],
                font=button_font,
                fg="white",
                bg="black",
                bd=0,
                highlightthickness=0,
                command=lambda: make_turn(0)
                )
                a0_button.bind("<Enter>", on_enter)  # Bind mouse enter event
                a0_button.bind("<Leave>", on_leave)  # Bind mouse leave event
                a0_button.pack(pady=0)
                
                a1_button = tk.Button(
                battle_frame,
                text=pkmns[0].entity[0].moves[1],
                font=button_font,
                fg="white",
                bg="black",
                bd=0,
                highlightthickness=0,
                command=lambda: make_turn(1)
                )
                a1_button.bind("<Enter>", on_enter)  # Bind mouse enter event
                a1_button.bind("<Leave>", on_leave)  # Bind mouse leave event
                a1_button.pack(pady=0)
                
                a2_button = tk.Button(
                battle_frame,
                text=pkmns[0].entity[0].moves[2],
                font=button_font,
                fg="white",
                bg="black",
                bd=0,
                highlightthickness=0,
                command=lambda: make_turn(2)
                )
                a2_button.bind("<Enter>", on_enter)  # Bind mouse enter event
                a2_button.bind("<Leave>", on_leave)  # Bind mouse leave event
                a2_button.pack(pady=0)
                
                a3_button = tk.Button(
                battle_frame,
                text=pkmns[0].entity[0].moves[3],
                font=button_font,
                fg="white",
                bg="black",
                bd=0,
                highlightthickness=0,
                command=lambda: make_turn(3)
                )
                a3_button.bind("<Enter>", on_enter)  # Bind mouse enter event
                a3_button.bind("<Leave>", on_leave)  # Bind mouse leave event
                a3_button.pack(pady=0)
                
                
                
                pass_button = tk.Button(
                battle_frame,
                text="pass",
                font=button_font,
                fg="white",
                bg="black",
                command=lambda: make_turn(-1)
                )
                pass_button.bind("<Enter>", on_enter)  # Bind mouse enter event
                pass_button.bind("<Leave>", on_leave)  # Bind mouse leave event
                pass_button.pack(pady=20)
                
                run_button = tk.Button(
                battle_frame,
                text="run",
                font=button_font,
                fg="white",
                bg="black",
                command=lambda: make_turn(4)
                )
                run_button.bind("<Enter>", on_enter)  # Bind mouse enter event
                run_button.bind("<Leave>", on_leave)  # Bind mouse leave event
                run_button.pack(pady=10)
                
            
                def update_advice():
                    
                    answer = give_advice(pkmn_logic.manual_battle)
                    print("finished")
                    
                    window = tk.Tk()
                    window.title("Advice")
                    window.configure(bg='black')  # Set the background color to black

                    # Create a label in the window with white text
                    label = ttk.Label(window, text=answer, background='black', foreground='white', wraplength=750, font=("Arial", 11, "bold"))
                    label.pack(pady=20)

                    # Create an "Accept" button in the window with white text
                    button = ttk.Button(window, text="Accept", command=window.destroy, style='TButton')
                    button.pack(pady=10)

                    # Configure the button style to have a black background and white text
                    style = ttk.Style()
                    style.configure('TButton', background='black', foreground='white')

                    # Run the main event loop
                    window.mainloop()
                
                def thread_adv():
                    train_thrd = threading.Thread(target=update_advice)
                    train_thrd.start()
                
                adv_button = tk.Button(
                battle_frame,
                text="?",
                font=button_font,
                fg="white",
                bg="black",
                command=thread_adv
                )
                adv_button.bind("<Enter>", on_enter)  # Bind mouse enter event
                adv_button.bind("<Leave>", on_leave)  # Bind mouse leave event
                adv_button.pack(pady=10)

                
        
        # Create the battle against a Pokémon button
        battle_pokemon_button = tk.Button(
            battle_buttons_frame,
            text="Start Battle",
            font=button_font,
            fg="white",
            bg="black",
            bd=0,
            highlightthickness=0,
            command=run_manual_battle
        )
        battle_pokemon_button.bind("<Enter>", on_enter)  # Bind mouse enter event
        battle_pokemon_button.bind("<Leave>", on_leave)  # Bind mouse leave event
        battle_pokemon_button.pack(pady=10)
        return frame
    if x == 4:
        frame = tk.Frame(root, bg="black")
        create_back_button(frame)
        
        # Create a Treeview widget for the table
        tree = ttk.Treeview(frame, columns=list(root.pokedex_data.columns), show='headings')
        tree.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Set the style for the table
        style = ttk.Style()
        style.configure("Custom.Treeview", background="black", foreground="white", fieldbackground="black", borderwidth=0)
        tree.configure(style="Custom.Treeview")

        # Define the column headings
        for column in root.pokedex_data.columns:
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
        
        for index, row in root.pokedex_data.iterrows():
            type1_color = row['Type 1']
            tree.insert("", "end", values=(row['No'], row['Name'], row['Type 1'], row['Type 2']), tags=(type1_color))
        return frame        
    
        # Create buttons
        
# Create main menu buttons
create_main_menu_button("Training Room", 1)
# create_main_menu_button("Simulation Box", 2)
create_main_menu_button("Battle Coliseum", 3)
create_main_menu_button("Pokedex", 4)

show_frame(button_frame, tk.Frame())  # Initially show the button frame

# Run the application
root.mainloop()  # This line starts the Tkinter event loop and creates the window
