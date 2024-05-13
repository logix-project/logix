import json

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

prompts = [
    "How can we reduce air pollution?",
    "Discuss the causes of the Great Depression",
    "Propose an ethical solution to the problem of data privacy",
    "Generate a poem that expresses joy.",
    "Design an app for a delivery company.",
    "Generate a pitch for a new and original product.",
    "Write a short review for the novel 'The Catcher in the Rye'.",
    "What is the process of photosynthesis and why is it important?",
    "Explain the difference between HTML and CSS.",
    "What is the difference between machine learning and deep learning?",
    "Brainstorm creative ideas for designing a conference room.",
]



prompt_list = []
output_list = []
for p in prompts:
    messages = [
        {"role": "user", "content": p},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    print(outputs[0]["generated_text"])
    # print(outputs[0]["generated_text"][len(prompt):])
    prompt_list.append(prompt)
    output_list.append(outputs[0]["generated_text"])

model_name = model_id.split("/")[-1]
filename = f"custom_data/generated/{model_name}/data.json"
data = [{"prompt": p, "text": o} for p, o in zip(prompt_list, output_list)]
with open(filename, "w") as f:
    json.dump(data, f, indent=4)
