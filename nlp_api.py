import google.generativeai as genai
import os
import timeit
import time
genai.configure(api_key=os.environ["API_KEY"])

def call_gemini_api(sys_inst, prmpt):
    model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=sys_inst)
    response = model.generate_content(
        prmpt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            # stop_sequences=["x"],
            max_output_tokens=1025,
            temperature=0.2,
        ),
    )
    return response.text

if __name__ == "__main__":
    resp = call_gemini_api("no escribas nunca un '*', eres un entrenador pokemon, cuando te digan algo generaras una partida o situacion aleatoria", "haras un analisis de los posibles movimientos de una supuesta partida q te inventaras y al final pondras: INSTRUCCION:<instruccion a efectuar>")
    print(resp)
    
