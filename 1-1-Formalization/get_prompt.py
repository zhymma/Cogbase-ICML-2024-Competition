import json
import openai
import re
import textwrap
import random
def get_prompt():
    training_data = r"1-1-Formalization/new_training_data/training_data.json"
    training_label = r"1-1-Formalization/new_training_data/training_label.json"
    with open(training_data, 'r', encoding="utf8") as f:
        training_data = json.load(f)
    with open(training_label, 'r', encoding="utf8") as f:
        training_label = json.load(f)
    examples_idx = []
    # 按照题目长度排序
    texts = [data['informal_statement']+data["informal_proof"] for data in training_data]
    for idx in sorted(range(len(texts)), key=lambda x: len(texts[x]),reverse=True)[:520]:
        examples_idx.append(idx)
    # examples_idx随机选择8个
    random.shuffle(examples_idx)
    examples_idx = examples_idx[:10]
    prompt = """You are a math expert and familar with Lean 3 formal language. Now please translate the following statement and solution of a math word problem into Lean 3 formal solution. Given a informal problem and its informal solution, analyze the mathematical problem and gain an in-depth understanding of the informal solution, then generate the corresponding formal solution in Lean 3. You should output the code in the ```lean xxx ``` tag. Please note that the informal solution and the formal solution need to be identical and the formal solution should be able to pass the Lean 3 compiler.
"""
    examples = """Here are some examples you can refer to:\n"""
    for idx in examples_idx:
        problem = training_data[idx]['informal_statement']
        solution = training_data[idx]['informal_proof']
        lean_code = training_label[idx]["formal_proof"]
        examples += f"# Problem:\n{problem}\n# Solution:\n{solution}\n"
        examples += f"# Formal solution in Lean 3:\n```lean\n{lean_code}\n```\n---\n"
    
    return prompt + examples

prompt = get_prompt()
# 生成prompt本身，保留换行符等格式
prompt = repr(prompt)
with open(r"1-1-Formalization/new_training_data/prompt.py", 'w', encoding="utf8") as f:
    f.write(prompt)
print(prompt)