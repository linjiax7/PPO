from trl import RewardConfig, RewardTrainer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

def load_data(dataset_name: str, split: str = "train"):
    return load_dataset(dataset_name, split=split)

def train_reward_model(model, tokenizer, dataset, output_dir="Qwen2.5-0.5B-Reward", batch_size=2):
    training_args = RewardConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size
    )
    trainer = RewardTrainer(
        args=training_args,
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name = "trl-lib/ultrafeedback_binarized"

    model, tokenizer = load_model(model_name)
    dataset = load_data(dataset_name)

    train_reward_model(model, tokenizer, dataset)

if __name__ == "__main__":
    main()
