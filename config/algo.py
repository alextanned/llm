from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class AlgoConfig:
    model_name: str
    dataset: str
    max_response_length: int
    max_prompt_length: int
    lora_r: int
    lora_alpha: int
    gradient_checkpointing: bool
    gradient_accumulation_steps: int
    per_device_train_batch_size: int
    samples: int
    system_prompt: str
    user_prompt: str
    assistant_prompt: str
    max_steps: int
    save_steps: int
    num_generations: int
    beta: float
    learning_rate: float

    name: str
    project: str
    group: str
    seed: int

    test: bool


cs = ConfigStore.instance()
cs.store(name="algo_base", node=AlgoConfig)
