from vllm import LLM, SamplingParams
from langchain_community.llms import VLLM
from argparse import ArgumentParser
# parser
parser = ArgumentParser()
parser.add_argument("--temperature", type=float, default=0.0) # 0.0 for greedy
parser.add_argument("--vllm", type=str, default='langchain', choices=['vllm', 'langchain'])
args = parser.parse_args()
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
#1. vllm.LLM 
if args.vllm == 'vllm':
    sampling_params = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=128)
    llm = LLM(model="facebook/opt-125m")
    outputs = llm.generate(prompts, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]

#2. langchain_community.llms.VLLM
elif args.vllm == 'langchain':
    llm = VLLM(
        model="facebook/opt-125m",
        trust_remote_code=True,  # mandatory for hf models
        max_new_tokens=128,
        top_k=10,
        top_p=0.95,
        temperature=args.temperature,
    )
    outputs = llm.batch(prompts)
    
# Print the outputs.
for output, prompt in zip(outputs, prompts):
    print(f"Prompt: {prompt}, Generated text: {output}")

