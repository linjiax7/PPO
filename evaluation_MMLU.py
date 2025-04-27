from evalscope.run import run_task
import os


task_cfg = {
    'model': 'qwen2.5',
    'api_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
    'api_key': os.getenv("DASHSCOPE_API_KEY"),
    'eval_type': 'service',
    'datasets': ['gsm8k'],
    'limit': 10
}

run_task(task_cfg=task_cfg)