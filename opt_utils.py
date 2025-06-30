import torch
from models.models import build_opt
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from config import DEVICE
from utils.confidences import StatsTestConfidence, ThresholdConfidence
from models.linear_module import MyLinear
import optuna

def eval_piqa(model, tokenizer, confidences, split="validation", prune_checkpoint=None, acc_lower_bound=None):
    num_layers = model.get_num_dynamic_prune_layers()
    print(f"Number of dynamic prune layers: {num_layers}")
    assert len(confidences) == num_layers
    model.my_eval(MyLinear, confidences)
    # Load PIQA dataset
    if split == "train":
        split = "train[:10%]"
    dataset = load_dataset("piqa", split=split)

    # Evaluate GPT-2 using log-likelihood
    def score_answer(prompt, answer):
        input_text = prompt + answer
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
        return -loss  # higher is better

    correct = 0
    total = 0

    for item in tqdm(dataset):
        goal = item["goal"]
        choices = [item["sol1"], item["sol2"]]
        gold = item["label"]  # 0 or 1 indicating the correct solution

        prompt = goal.strip() + "\n"
        scores = []

        for choice in choices:
            text = f"{prompt}{choice.strip()}"
            scores.append(score_answer(prompt, choice.strip()))

        pred = int(np.argmax(scores))
        if pred == gold:
            correct += 1
        total += 1
        if split != "validation":
            if prune_checkpoint is not None and total == prune_checkpoint:
                if correct / total < acc_lower_bound:
                    print(f"pruned, partial accuracy: {correct}/{total} = {correct / total}")
                    raise optuna.TrialPruned()

    acc = correct / total if total else 0
    flops = model.gather_flops() / total
    return acc, flops

if __name__ == "__main__":
    model_name, model, tokenizer = build_opt()
    model.to(DEVICE)
    model.eval()
    confidences = [ThresholdConfidence(-5)] * model.get_num_dynamic_prune_layers()
    acc, flops = eval_piqa(model, tokenizer, confidences, split="validation")
    print(f"acc: {acc}, flops: {flops}")