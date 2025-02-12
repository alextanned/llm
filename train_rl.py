import os
import re

from datasets import load_dataset
from huggingface_hub import login
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig
from unsloth import FastLanguageModel

from config.algo import AlgoConfig
from cutom_trainer import GRPOTrainer

import wandb

# TODO it seems that trainer allocates a separate reference model instead of disabling unsloth adapter

def train(cfg: AlgoConfig):
    assert cfg.name != "", "Please provide a name for the experiment"
    # Login to Hugging Face H
    with open("hf_token.txt", "r") as f:
        token = f.read().strip()
    login(token=token)

    model_name = cfg.model_name

    model_config = {
        "model_name": model_name,
        "max_seq_length": cfg.max_prompt_length + cfg.max_response_length,
        "load_in_4bit": True,
    }
    peft_config = {
        "r": cfg.lora_r,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": 0, # Supports any, but = 0 is optimized
        "bias": "none",# Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        "use_gradient_checkpointing": True,#"unsloth", # True or "unsloth" for very long context
        "random_state": cfg.seed,
        "use_rslora": False,
        "loftq_config": None,
    }

    full_model, tokenizer = FastLanguageModel.from_pretrained(**model_config)


    model = FastLanguageModel.get_peft_model(
        full_model,
        **peft_config,
    )

    # Load dataset from Hugging Face Hub
    dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset = load_dataset(dataset_id, split="train")
    # select a random subset of 50k samples
    dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.samples))


    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(numbers, target):
        r1_prefix = [{
            "role": "system",
            "content": cfg.system_prompt
        },
            {
                "role": "user",
                "content": cfg.user_prompt.format(numbers=numbers, target=target)
                # Use <think> and <answer> tags exactly once."
            },
            {
                "role": "assistant",
                "content": cfg.assistant_prompt
            }]
        return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
                "target": target}


    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))

    # split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]


    def format_reward_func(completions, target, **kwargs):
        """
        Format: <think>...</think><answer>...</answer>
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers

          Returns:
              list[float]: Reward scores
        """
        rewards = []

        for completion, gt in zip(completions, target):

            try:
                # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
                completion = "<think>" + completion
                # Check if the format is correct
                regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

                match = re.search(regex, completion, re.DOTALL)
                # if the format is not correct, reward is 0
                if match is None or len(match.groups()) != 2:
                    rewards.append(0.0)
                else:
                    rewards.append(1.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    def equation_reward_func(completions, target, nums, **kwargs):
        """
        Evaluates completions based on:
        2. Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
                completion = "<think>" + completion
                # Check if the format is correct
                match = re.search(r"<answer>(.*?)<\/answer>", completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builtins__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards

    checkpoint_path = f"out/{cfg.name}"
    if os.path.exists(checkpoint_path):
        checkpoint_path = get_last_checkpoint(checkpoint_path)
        print(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None

    # Hyperparameters
    training_args = GRPOConfig(
        output_dir=f"out/{cfg.name}",
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=1,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=0.03,
        bf16=True,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        optim = cfg.optimizer,
        # GRPO specific parameters
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_response_length,  # max length of the generated output for our solution
        num_generations=cfg.num_generations,
        beta=cfg.beta,
        report_to="wandb"
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, equation_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    if not cfg.test:

        run = wandb.init(
            project=cfg.project,
            config=cfg.__dict__ | model_config | peft_config | training_args.__dict__,
            group=cfg.group,
            name=cfg.name,
            id=cfg.name,
            resume="allow"
        )
        print(run.get_url())
        trainer.train(resume_from_checkpoint=checkpoint_path)
        trainer.save_model(training_args.output_dir)
    else:
        trainer._load_from_checkpoint(checkpoint_path)
        FastLanguageModel.for_inference(trainer.model)
        test_case = test_dataset[0]
        prompt = test_case["prompt"]
        print("Prompt:", prompt)
        prompt_tensor = tokenizer(prompt, return_tensors="pt")
        input_ids = prompt_tensor["input_ids"].cuda()
        generation = trainer.model.generate(
            input_ids,
            max_length=cfg.max_response_length,
            num_return_sequences=8,
            temperature=0.9,
            do_sample=True,
        )
        prompt_length = input_ids.shape[1]
        generated_text = tokenizer.batch_decode(generation[:, prompt_length:], skip_special_tokens=True)

        with open(f"out/{cfg.name}/generations.md", "wb") as f:
            for text in generated_text:
                f.write(("\n# Generation\n" + text + "\n\n").encode())
                format_reward = format_reward_func([text], [test_case["target"]])
                equation_reward = equation_reward_func([text], [test_case["target"]], [test_case["nums"]])
                f.write(f"Format Reward: {format_reward}\n\n".encode())
                f.write(f"Equation Reward: {equation_reward}\n\n".encode())
                print(f"Format Reward: {format_reward}, Equation Reward: {equation_reward}")

