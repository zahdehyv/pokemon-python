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
