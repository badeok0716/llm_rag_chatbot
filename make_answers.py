from math import e
import os

from sqlalchemy import exists 

from tools.corpus import (
    load_docs,
)

from tools.chain import (
    RetrievalChatBot,
)
from tools.write_op import write_pkl

os.environ['OPENAI_API_KEY'] = "..."


def create_chatbot(corpus='data/20.txt', modelname='gpt-4o-mini', load_retriever_path='data/scene_20'):
    # construct retrieval documents
    documents = [
        doc.page_content
        for doc in load_docs(corpus_path=corpus, chunk_size=256, chunk_overlap=16)
    ]
    
    print(f"Loaded {len(documents)} documents.")
    # initialize chatbot
    chatbot = RetrievalChatBot(
            documents=documents, modelname=modelname, load_retriever_path=load_retriever_path, verbose=False, single_turn=True
        )
    return chatbot

if __name__ == "__main__":
    chatbot = create_chatbot()
    dirname = 'data/questions/'
    os.makedirs(dirname.replace('questions','answers'), exist_ok=True)
    os.makedirs(dirname, exist_ok=True)
    for path in [dirname + f for f in ["toy.txt","edu_dup.txt", "edu.txt", "greeting_dup.txt", "greeting.txt", "relation_dup.txt", "relation.txt"]]:
        print(path)
        with open(path, "r", encoding="utf-8") as f:
            questions = f.readlines()
        questions = [q.strip() for q in questions]
        answers = chatbot.chain.batch(questions)
        answers = [a['response'].strip() for a in answers]
        write_pkl([questions, answers], path.replace('.txt','_answers4.pkl').replace('questions','answers'))

