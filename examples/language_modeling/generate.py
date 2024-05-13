import argparse
import csv
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_prompt(prompt=None):
    if prompt is None:
        prompt = [
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
            "Generate a list of business ideas for a food delivery service.",
            "Compose a tweet that addresses the issue of environmental degradation.",
            "What is the process of photosynthesis and why is it important?",
            "Explain why computational models can be used in analysing text.",
            "Follow the law of supply and demand, describe what would happen to the price of a good if the demand increased.",
            "Generate a possible future for humankind if artificial intelligence (AI) becomes more advanced and prevalent.",
            "Suppose you are managing a marketing campaign. What are some methods you can use to measure the success of the campaign?",
            "Now that you know the different ways to say hello in French, which one would you use if you were greeting a friend?",
            "Write a short story summarizing the following events: (events) An engineer discovers a new form of energy, but it requires a large amount of money to develop.",
            "Generate a thesis statement based on the following description. Description: The key principles behind effective physical exercise routines including warming up, cooling down, and rest recovery.",
            "Evaluate the quality of the following sentence: 'To get a better understanding of the material we have been studying, going through examples is a good choice.'",
            "Identify the main idea of the following paragraph: The driving force behind the success of companies today is their ability to be innovative and adaptive in a constantly changing environment.",
        ]

    return [prompt] if isinstance(prompt, str) else prompt


# the model generates \n after the prompt when batch generating
def batch_generate(
    model,
    tokenizer,
    prompts=None,
    device=None,
    top_k=1,
    top_p=0.95,
    temperature=0.9,
    repetition_penalty=1.5,
):
    if prompts is None:
        prompts = get_prompt()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        return_token_type_ids=False,
        padding=True,
        truncation=True,
    )
    inputs = inputs.to(device)
    response = model.generate(
        **inputs,
        max_new_tokens=args.maxlen,
        do_sample=True,
        top_k=top_k,
top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.batch_decode(response, skip_special_tokens=True)
    return prompts, generated


def iterative_generate(
    model,
    tokenizer,
    prompts=None,
    device=None,
    top_k=1,
    top_p=0.95,
    temperature=0.9,
    repetition_penalty=1.5,
):
    if prompts is None:
        prompts = get_prompt()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generated = []
    for p in prompts:
        _, g = batch_generate(
            model,
            tokenizer,
            p,
            device,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
        )
        generated.append(g[0])
    return prompts, generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gpt2 Generation")
    parser.add_argument("--model_name", type=str, default="gpt2-xl")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--topp", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--maxlen", type=float, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.random.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )
    model.eval()
    model.to(device)
    # print(tokenizer.eos_token_id, tokenizer.pad_token_id)

    prompts = get_prompt(args.prompt)
    prompts, generated = iterative_generate(
        model,
        tokenizer,
        prompts,
        top_k=args.topk,
        top_p=args.topp,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )

    # write prompt and generation into a csv file
    model_name = args.model_name.split("/")[-1]
    out_file = f"./custom_data/generated/{model_name}/data.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    data_json = [
        {
            "prompt": p,
            "text": g,
        }
        for p, g in zip(prompts, generated)
    ]
    with open(out_file, "w") as f:
        json.dump(data_json, f, indent=4)
    print(f"Generated data saved to {out_file}")

