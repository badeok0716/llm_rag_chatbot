import torch
from collections import defaultdict
from tools.read_op import read_pkl

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only

from trl import SFTTrainer
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import TrainingArguments, DataCollatorForSeq2Seq

system_prompt = """[인물 소개] 이선준은 드라마 <성균관 스캔들>의 이선준은 주인공 중 한 명으로, 조선시대를 배경으로 한 이 작품에서 매우 중요한 역할을 담당하오. 이선준은 성균관 유생으로서 높은 도덕성과 학문적인 재능을 가진 인물이오.
자네는 드라마 <성균관 스캔들>의 이선준이오. 그러니, 상대방과 조선어로 대화해야 하오.
자네는 유저의 질문 언어와 상관없이 대화 내내 이선준, 자네의 입장에서 조선어로 대답해야 하네.
자네가 성균관 유생이라는 점에 유념하여 대화를 진행해주시오.
"""
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
PATHS = [
    "edu_dup_answers.pkl",
    "edu_answers.pkl",
    "greeting_dup_answers.pkl",
    "greeting_answers.pkl",
    "relation_dup_answers.pkl",
    "relation_answers.pkl",
]
PATHS = [p.replace('.pkl',f'{i}.pkl') for i in ['', '2', '3', '4'] for p in PATHS]

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-3B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model, tokenizer

def load_private(paths=['data/answers/' + p for p in PATHS]):
    data = defaultdict(list)
    prompt = system_prompt
    for p in paths:
        questions, answers = read_pkl(p)
        for q, a in zip(questions, answers):
            conversations = [{"from": "system", "value": prompt}] if prompt else []
            conversations.extend([
                {'from': 'human', 'value': q},
                {'from': 'gpt', 'value': a},
            ])
            data["conversations"].append(conversations)
            data["source"].append("private")
            data["score"].append(5.0)
                    
    dataset = Dataset.from_dict(data)
    return dataset.shuffle(seed = 3407)
    

def prepare_dataset(tokenizer):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    dataset = load_dataset("mlabonne/FineTome-100k", split = "train").shuffle(seed = 3407)
    private_dataset = load_private()
    dataset = concatenate_datasets([private_dataset, dataset.select(range(6000))])
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    return dataset

def mem_stat_before():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return start_gpu_memory, max_memory

def mem_stat(trainer_stats, start_gpu_memory, max_memory):
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

def infer_one_shot(model, tokenizer, query="Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,", use_system=True):
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    if use_system:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
    else:
        messages = [
            {"role": "user", "content": query},
        ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
                            temperature = 1.5, min_p = 0.1)
    response = tokenizer.batch_decode(outputs)
    print("====="*10)
    if use_system:
        print("System:", system_prompt)
    print("Query:", query)
    print("Response:", response)
    print("====="*10)
    print("\n")
    return response

def infer_stream(model, tokenizer, query="Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,", use_system=True):
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    if use_system:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
    else:
        messages = [
            {"role": "user", "content": query},
        ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                    use_cache = True, temperature = 1.5, min_p = 0.1)

def prepare_trainer(model, tokenizer, dataset):
    FastLanguageModel.for_training(model) 
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 40,
            # per_device_train_batch_size = 40,
            gradient_accumulation_steps = 1,
            warmup_steps = 5,
            num_train_epochs = 1, # Set this for 1 full training run.
            # max_steps = 100,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    return trainer

def main():
    model, tokenizer = load_model()
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )
    dataset = prepare_dataset(tokenizer)
    
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    infer_one_shot(model, tokenizer, use_system=False)
    infer_one_shot(model, tokenizer, query="안녕!", use_system=False)
    infer_one_shot(model, tokenizer, use_system=True)
    infer_one_shot(model, tokenizer, query="안녕!", use_system=True)

    trainer = prepare_trainer(model, tokenizer, dataset)
    start_gpu_memory, max_memory = mem_stat_before()
    trainer_stats = trainer.train()
    mem_stat(trainer_stats, start_gpu_memory, max_memory)

    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    infer_one_shot(model, tokenizer)
    infer_one_shot(model, tokenizer, query="안녕!")
    infer_one_shot(model, tokenizer, use_system=True)
    infer_one_shot(model, tokenizer, query="안녕!", use_system=True)

    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()