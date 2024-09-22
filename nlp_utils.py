import google.generativeai as genai

# genai.configure(api_key=os.environ["API_KEY"])

def call_gemini_api(sys_inst, prmpt):
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=sys_inst)
    response = model.generate_content(
        prmpt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            stop_sequences=["x"],
            max_output_tokens=1025,
            temperature=0.2,
        ),
    )
    return response.text


def give_advice(B: Battle):
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction="You are a Pokemon Battle Strategies expert. Whenever you receive a battle state, you give an tip considering types, advantages, etc. You know how to keep it short, funny, and correct. Pay special attention to the HP")
    prompt = f"""Current state of the battle is:
My Pokemon HP:{B.p1.active_pokemon[0].hp}/{B.p1.active_pokemon[0].maxhp}
My Pokemon Types: {B.p1.active_pokemon[0].types}
My Pokemon Moves: {B.p1.active_pokemon[0].moves}

Rival's Pokemon HP:{B.p2.active_pokemon[0].hp}/{B.p2.active_pokemon[0].maxhp}
Rival's Pokemon Types: {B.p2.active_pokemon[0].types}
Rival's Pokemon Moves: {B.p2.active_pokemon[0].moves}"""

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            stop_sequences=["x"],
            max_output_tokens=1025,
            temperature=0.2,
        ),
    )
    print(response.text)
    return response.text

def narrate_battle_logs(logs, pkmn1: PokemonEntity, pkmn2: PokemonEntity):
    prompt = f"Piad's {pkmn1.entity[0].species} vs Daniel's {pkmn2.entity[0].species}"
    for log in logs:
        prompt = prompt + log + "\n"
    prompt = prompt + "\n combate finished"
    
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction="You are an humorous pokemon battle results informer, which receives a list of events and tell the sad news. Keep it as short as you can.")
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            # Only one candidate for now.
            candidate_count=1,
            stop_sequences=["x"],
            max_output_tokens=1025,
            temperature=0.7,
        ),
    )
    print(response.text)
    return response.text
    
def get_caption(logs):
    prompt = ""
    for log in logs:
        prompt = prompt + log + "\n"
    
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction="""You are a Pokémon battle narrator. For each turn, provide a detailed narration about the attack performed, its effect, and what it provoked, paying special attention to the life of the affected Pokémon. Make sure your response is brief and concise, respecting the size the battle system allows for its captions. Remember to add a touch of humor to make the narration more entertaining. For example, if a Pokémon uses a powerful attack, mention how the opponent staggers as if hit by a Snorlax. Here's the information for the last turn:
""")
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            # Only one candidate for now.
            candidate_count=1,
            stop_sequences=["x"],
            max_output_tokens=151,
            temperature=0.7,
        ),
    )
    print(response.text)
    return response.text
    
if __name__ == '__main__':
    pass