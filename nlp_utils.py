from I_PERPLEXITY.perplexity import Perplexity
import google.generativeai as genai
import os
from misc_utils import PokemonEntity
from sim.structs import Battle

genai.configure(api_key=os.environ["API_KEY"])

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
            # Only one candidate for now.
            candidate_count=1,
            stop_sequences=["x"],
            max_output_tokens=1025,
            temperature=0.7,
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
    
def narrate_battle_logs_perp(logs):
    prompt = """Please give a super short summary of the combat, without titles, just analyzing the strategies and the result:
    """
    for log in logs:
        prompt = prompt + log + "\n"
    prompt = prompt + "\n Combate terminado"
    # perplexity = Perplexity("kuutaiyuu@gmail.com")
    perplexity = Perplexity()
    answer = perplexity.search(prompt)
    false = False
    null = None
    ans = ""
    # print("PROMPT")
    # print(prompt)
    # print("thinking...")
    # i = 0
    for a in answer:
        # i = i +1
        ans = a
        # print(i)
    # print("search end!")
    # print("ANSWER\n", eval(ans['text'])['answer'])
    # print(ans['text'])
    # print(eval(ans['text']))
    # print()
    perplexity.close()
    return eval(ans['text'])['answer']
    
if __name__ == '__main__':
    perplexity = Perplexity()
    pre_prompt = "Please give a super short answer, like a person which just give the most important details. Don't write down agreement phrases. "
    answer = perplexity.search(pre_prompt+"summarize the type weaknesses against each type in pokemon")
    false = False
    null = None
    ans = ""
    # i=0
    print("thinking...")
    for a in answer:
        # i = i +1
        ans = a
        # print(i)
    print("search end!")
    # print(ans['text'])
    # print(eval(ans['text']))
    print()
    print(eval(ans['text'])['answer'])
    perplexity.close()