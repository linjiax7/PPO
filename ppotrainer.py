from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from datasets import load_dataset
import torch

def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name)
    return model, tokenizer

def load_reward_model(reward_model_name: str):
    reward_pipeline = pipeline("sentiment-analysis", model=reward_model_name)
    return reward_pipeline

def load_dataset_for_ppo(dataset_name: str, split: str = "train"):
    return load_dataset(dataset_name, split=split)

def train_ppo_model(model, tokenizer, reward_model, dataset, output_dir="ppo_output", batch_size=2, epochs=1):

    ppo_config = PPOConfig(
        model_name_or_path=None,
        output_dir=output_dir,
        learning_rate=1.41e-5,
        log_with="tensorboard",
        mini_batch_size=batch_size,
        batch_size=batch_size * 8,
        optimize_cuda_cache=True
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset
    )

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch in dataset:
            query_tensors = tokenizer(batch["text"], return_tensors="pt", padding=True).input_ids

            response_tensors = model.generate(query_tensors)

            responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

            rewards = []
            for response in responses:
                reward_score = reward_model(response)[0]["score"]
                rewards.append(torch.tensor(reward_score))

            trainer.step(query_tensors, response_tensors, rewards)

def main():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    reward_model_name = "OpenRLHF/Llama-3-8b-rm-mixture"
    dataset_name = "RLHFlow/prompt-collection-v0.1"

    model, tokenizer = load_model_and_tokenizer(model_name)
    reward_model = load_reward_model(reward_model_name)
    dataset = load_dataset_for_ppo(dataset_name)

    train_ppo_model(model, tokenizer, reward_model, dataset, output_dir="ppo_Qwen2.5", batch_size=2, epochs=3)

if __name__ == "__main__":
    main()
