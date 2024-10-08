# A Tutorial for LLM Chat Streaming via RAG 

## 환경 설치
'''
    source install_env.sh
'''

## test vllm
'''
    conda activate vllm
    python test_vllm.py
'''

## RAG chatbot via langchain
'''
    python chat.py --modelname gpt-4o
'''


## Distillation 
### 질문 생성
'''
    python make_questions.py
'''

### 답변 생성
'''
    python make_answers.py
'''

### finetuning
'''
    python finetune_chat.py
'''

## TODO for efficiency
- Implement `.batch()` in langchain ChatOpenAI to support batch inference in OpenAI API.
- vLLM inference of unsloth finetuned model.

## 참고
- 학습된 Unsloth 체크포인트를 vLLM으로 구동하기. https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm
- vLLM으로 개인 모델을 OpenAI API 형태의 inference api로 만들기. https://docs.vllm.ai/en/v0.6.1/getting_started/quickstart.html
- RAG 심화 tutorial. https://github.com/langchain-ai/rag-from-scratch/tree/main
- Unsloth chatbot fine-tuning 튜토리얼. https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing#scrollTo=_rD6fl8EUxnG
