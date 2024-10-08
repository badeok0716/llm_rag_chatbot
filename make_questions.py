from tools.prompt import ko_similar_q, ko_trans_similar_q
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
import re
def remove_numbers(text_list):
    result = []
    for item in text_list:
        lines = item.strip().split('\n')
        for line in lines:
            # Remove leading numbers and periods
            sentence = re.sub(r'^\d+\.\s*', '', line).strip()
            if sentence:
                result.append(sentence)
    return result

os.environ['OPENAI_API_KEY'] = "..."

# duplicate questions
paths = [
    "data/questions/edu.txt", # 100 -> 500
    "data/questions/greeting.txt", # 250 -> 1000
    "data/questions/relation.txt", # 100 -> 500
]
for path, n in zip(paths,[4,3,4]):
    with open(path, "r", encoding="utf-8") as f:
        questions = f.readlines()
    prompt = ChatPromptTemplate.from_template(ko_similar_q(number=n))
    LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=1.0)
    chatbot = prompt | LLM | StrOutputParser()
    outputs = remove_numbers(chatbot.batch(questions))
    # save text
    with open(path.replace('.txt','_dup.txt'), "w", encoding="utf-8") as f:
        f.write("\n".join(outputs))
