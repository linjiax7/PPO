# PPO
Apply Proximal Policy Optimization (PPO) RLHF to the open-source model Qwen2.5-1.5B-Instruction, leveraging reinforcement learning techniques to improve the model’s response quality and alignment with human preferences.
Used prompt-collection-v0.1 dataset from RLHFlow, which is a text dataset designed for natural language processing tasks. It contains a collection of prompts designed to test and train
AI systems, particularly those handling conversational and question-answering tasks. The dataset includes around 179,000 rows of prompt content, offering a rich set of scenarios where
AI models can interact with human-like inputs.The prompts cover a wide range of topics,allowing models to better understand user intent and generate coherent responses.
Improved BLEU and ROUGE scores indicate that the model outperforms the base model in text generation quality, demonstrating that it has effectively learned the expression style of the test dataset.

OpenAI's RLHF Overview:https://openai.com/research/instruction-following
Hugging Face TRL (RLHF Library):https://huggingface.co/docs/trl/index
Qwen Model from Alibaba: https://huggingface.co/Qwen
Proximal Policy Optimization (PPO) Paper: https://arxiv.org/abs/1707.06347
