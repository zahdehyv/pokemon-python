from I_PERPLEXITY.perplexity import Perplexity

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