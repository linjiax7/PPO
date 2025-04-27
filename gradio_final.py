import os
import gradio as gr
from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def chat_with_qwen(message, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    for entry in history:
        if isinstance(entry, dict) and "role" in entry and "content" in entry:
            if entry["role"] == "user":
                messages.append({"role": "user", "content": entry["content"]})
            elif entry["role"] == "assistant":
                messages.append({"role": "assistant", "content": entry["content"]})


    messages.append({"role": "user", "content": message})

    try:
        completion = client.chat.completions.create(
            model="qwen2.5-1.5b-instruct",
            messages=messages,
            stream=True
        )

        full_content = ""
        for chunk in completion:
            if chunk.choices:
                partial_response = chunk.choices[0].delta.content
                full_content += partial_response
                yield full_content
        
    except Exception as e:
        yield f"Error: {e}"

gr.ChatInterface(
    fn=chat_with_qwen,
    type="messages",
).launch(share=True)
