from I_PERPLEXITY.perplexity import Perplexity

def narrate_battle_logs(logs):
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