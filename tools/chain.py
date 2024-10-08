
import faiss
import os
from tqdm import tqdm

from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import (
    ConversationBufferMemory,
    CombinedMemory,
)
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from tools.prompt import ko_scene_intro, ko_scene_query, ko_scene_retrieval
from tools.memory import ConversationVectorStoreRetrieverMemory, ConversationVectorStoreRetrieverMemoryNoUpdate


class RetrievalChatBot:
    def __init__(self, documents, modelname="gpt-4o-mini", load_retriever_path="", verbose=True, single_turn=False):
        self.documents = documents
        self.modelname = modelname
        self.single_turn = single_turn
        self.load_retriever_path = load_retriever_path

        self.num_context_memories = 3

        self.chat_history_key = "chat_history"
        self.context_key = "context"
        self.input_key = "input"
        if self.single_turn:
            self.chat_history_template =  f"""[현재 대화]
사용자: {{{self.input_key}}}
선준:
"""
        else:
            self.chat_history_template =  f"""[기존 대화 내역]
---
{{{self.chat_history_key}}}
---
[현재 대화]
사용자: {{{self.input_key}}}
선준:
"""            
        self.chain = self.create_chain(verbose=verbose)
        self.name = "선준"

    def prepare_memory(self):
        conv_memory = ConversationBufferMemory(
            memory_key=self.chat_history_key, input_key=self.input_key, human_prefix="사용자", ai_prefix="선준"
        )

        if self.single_turn:
            context_memory_ = ConversationVectorStoreRetrieverMemoryNoUpdate
        else:
            context_memory_ = ConversationVectorStoreRetrieverMemory
        context_memory = context_memory_(
            retriever=FAISS(
                OpenAIEmbeddings().embed_query,
                faiss.IndexFlatL2(1536),  # Dimensions of the OpenAIEmbeddings
                InMemoryDocstore({}),
                {},
            ).as_retriever(search_kwargs=dict(k=self.num_context_memories)),
            memory_key=self.context_key,
            output_prefix="선준",
            blacklist=[self.chat_history_key],
        )
        if self.load_retriever_path and os.path.exists(self.load_retriever_path):
            context_memory.retriever.vectorstore = context_memory.retriever.vectorstore.load_local(
                                                            folder_path=self.load_retriever_path, 
                                                            embeddings=OpenAIEmbeddings().embed_query,
                                                            allow_dangerous_deserialization=True,
                                                        )
        else:
            # add the documents to the context memory
            for i, summary in tqdm(enumerate(self.documents)):
                context_memory.save_context(inputs={}, outputs={f"[{i}]": summary})
            if self.load_retriever_path:
                context_memory.retriever.vectorstore.save_local(self.load_retriever_path)
        # Combined
        if self.single_turn:
            return context_memory
        else:
            memory = CombinedMemory(memories=[conv_memory, context_memory])
            return memory
    
    def create_chain(self, verbose=True):
        memory = self.prepare_memory()

        template = ko_scene_intro + ko_scene_retrieval + ko_scene_query + self.chat_history_template
        
        prompt = PromptTemplate.from_template(template)
        if 'gpt' in self.modelname:
            LLM = ChatOpenAI(model_name=self.modelname, temperature=0.4)
        else:
            from langchain_community.llms import VLLM
            LLM = VLLM(
                model=self.modelname,
                trust_remote_code=True,
                max_new_tokens=512,
                vllm_kwargs={"quantization": "awq"},
            )
        chatbot = ConversationChain(
            llm=LLM, verbose=verbose, memory=memory, prompt=prompt
        )
        return chatbot

    def step(self, input):
        return self.chain.run(input=input)