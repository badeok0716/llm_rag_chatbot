import os 
import argparse
import streamlit as st

from tools.corpus import (
    load_docs,
)

from tools.chain import (
    RetrievalChatBot,
)
from tools.interfaces import CommandLine, Streamlit

os.environ['OPENAI_API_KEY'] = "..."
OUTPUT_ROOT = "output"


def create_chatbot(corpus, modelname, load_retriever_path):
    # construct retrieval documents
    documents = [
        doc.page_content
        for doc in load_docs(corpus_path=corpus, chunk_size=256, chunk_overlap=16)
    ]
    
    print(f"Loaded {len(documents)} documents.")
    # initialize chatbot
    chatbot = RetrievalChatBot(
            documents=documents, modelname=modelname, load_retriever_path=load_retriever_path
        )
    return chatbot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", type=str, default="data/20.txt"
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default="gpt-4o",
        choices=["gpt-4o-mini", "gpt-4o", "gpt-4", "TheBloke/Llama-2-7b-Chat-AWQ"],
    )
    parser.add_argument(
        "--interface", type=str, default="cli", choices=["cli", "streamlit"]
    )
    parser.add_argument(
        "--load_retriever_path", type=str, default="data/scene_20",
    )
    args = parser.parse_args()

    if args.interface == "cli":
        chatbot = create_chatbot(
            args.corpus,
            args.modelname,
            args.load_retriever_path,
        )
        app = CommandLine(chatbot=chatbot)
    elif args.interface == "streamlit":
        chatbot = st.cache_resource(create_chatbot)(
            args.corpus,
            args.modelname,
            args.load_retriever_path,
        )
        st.title("성균관 스캔들 <이선준> 챗봇")
        st.write("이선준 캐릭터와 대화해보세요!")
        st.divider()
        st.markdown(f"**model name**: *{args.modelname}*")
        st.markdown(f"**corpus path**: *{args.corpus}*")
        st.markdown(f"**retriever path**: *{args.load_retriever_path}*")
        app = Streamlit(chatbot=chatbot)
    else:
        raise ValueError(f"Unknown interface: {args.interface}")
    app.run()


if __name__ == "__main__":
    main()