import argparse
import os
from data_utils import load_glue_dataset, split_non_iid
from client import Client
from server import Server
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)
import wandb
from datasets import load_dataset
import torch
from torch.optim import AdamW
import copy
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

def run_single_experiment(args, seed=None):
    """Run a single federated learning experiment and return metrics"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        set_seed(seed)
    
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    if "mnli" in args.dataset:
        client_data_splits, eval_dataloader, mismatched_eval_dataloader, metric = load_glue_dataset(dataset_name=args.dataset, num_clients=args.num_clients, batch_size=32)
    else:
        client_data_splits, eval_dataloader, metric = load_glue_dataset(dataset_name=args.dataset, num_clients=args.num_clients, batch_size=32)
    # client_data = split_non_iid(train_data, args.num_clients)
    peft_config = LoraConfig(use_dora=False, task_type="SEQ_CLS", inference_mode=False, r=4, lora_alpha=16, lora_dropout=0.1)
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, return_dict=True)
    model = get_peft_model(base_model, peft_config)

    server = Server(global_model=model, device=device)
    clients = [Client(client_id=i, model=copy.deepcopy(model), data=client_data_splits[i], device=device, train_method=args.method) for i in range(args.num_clients)]

    max_acc = 0
    mismatched_max_acc = 0
    final_acc = 0
    final_mismatched_acc = 0
    
    for round_num in range(args.num_rounds):
        print(f"\n--- Federated Learning Round {round_num + 1} ---")

        client_models = []
        if "muon" in args.method:
            processes = []
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
        # Train each client locally
        for client in clients:
            print(f"\nTraining Client {client.client_id}")
            if "kd" in args.method:
                client.train(epochs=args.num_epochs, train_method=args.method, base_model=base_model)
            else:
                client.train(epochs=args.num_epochs, train_method=args.method)
            client_models.append(client.get_parameters())

        # Aggregate client models on the server
        print("\nAggregating client models on the server...")
        server.aggregate(client_models)
        eval_metric = server.evaluate(eval_dataloader, metric=metric)
        if "mnli" in args.dataset:
            mismatched_eval_metric = server.evaluate(mismatched_eval_dataloader, metric=metric)
        print(eval_metric)

        # Update each client's model with the new global model
        if max_acc < eval_metric['accuracy']:
            max_acc = eval_metric['accuracy']

        if "mnli" in args.dataset and mismatched_max_acc < mismatched_eval_metric["accuracy"]:
            mismatched_max_acc = mismatched_eval_metric["accuracy"]

        for client in clients:
            client.set_parameters(server.global_model.state_dict())

    final_acc = eval_metric['accuracy']
    if "mnli" in args.dataset:
        final_mismatched_acc = mismatched_eval_metric['accuracy']
        return {
            'final_acc': final_acc,
            'max_acc': max_acc,
            'final_mismatched_acc': final_mismatched_acc,
            'max_mismatched_acc': mismatched_max_acc
        }
    else:
        return {
            'final_acc': final_acc,
            'max_acc': max_acc
        }

def main(args):
    """Run experiments multiple times and log statistics"""
    num_runs = args.num_runs
    results = []
    
    for run_idx in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Running Experiment {run_idx + 1}/{num_runs}")
        print(f"{'='*60}\n")
        result = run_single_experiment(args, seed=run_idx)
        results.append(result)
        print(f"\nExperiment {run_idx + 1} Result: {result}")
    
    # Calculate statistics
    final_accs = [r['final_acc'] for r in results]
    max_accs = [r['max_acc'] for r in results]
    
    stats = {
        'final_acc_mean': np.mean(final_accs),
        'final_acc_std': np.std(final_accs),
        'max_acc_mean': np.mean(max_accs),
        'max_acc_std': np.std(max_accs),
    }
    
    if "mnli" in args.dataset:
        final_mismatched_accs = [r['final_mismatched_acc'] for r in results]
        max_mismatched_accs = [r['max_mismatched_acc'] for r in results]
        stats.update({
            'final_mismatched_acc_mean': np.mean(final_mismatched_accs),
            'final_mismatched_acc_std': np.std(final_mismatched_accs),
            'max_mismatched_acc_mean': np.mean(max_mismatched_accs),
            'max_mismatched_acc_std': np.std(max_mismatched_accs),
        })
    
    # Write results to file
    with open(args.output_file, "w") as f:
        f.write(f"Method: {args.method}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Number of Clients: {args.num_clients}\n")
        f.write(f"Number of Rounds: {args.num_rounds}\n")
        f.write(f"Number of Epochs per Round: {args.num_epochs}\n")
        f.write(f"Number of Runs: {num_runs}\n\n")
        
        f.write("="*60 + "\n")
        f.write("DETAILED RESULTS FOR EACH RUN\n")
        f.write("="*60 + "\n\n")
        
        for run_idx, result in enumerate(results):
            f.write(f"Run {run_idx + 1}:\n")
            f.write(f"  Final Accuracy: {result['final_acc']:.4f}\n")
            f.write(f"  Max Accuracy: {result['max_acc']:.4f}\n")
            if "mnli" in args.dataset:
                f.write(f"  Final Mismatched Accuracy: {result['final_mismatched_acc']:.4f}\n")
                f.write(f"  Max Mismatched Accuracy: {result['max_mismatched_acc']:.4f}\n")
            f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("STATISTICS (MEAN ± STD)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Final Accuracy: {stats['final_acc_mean']:.4f} ± {stats['final_acc_std']:.4f}\n")
        f.write(f"Max Accuracy: {stats['max_acc_mean']:.4f} ± {stats['max_acc_std']:.4f}\n")
        if "mnli" in args.dataset:
            f.write(f"Final Mismatched Accuracy: {stats['final_mismatched_acc_mean']:.4f} ± {stats['final_mismatched_acc_std']:.4f}\n")
            f.write(f"Max Mismatched Accuracy: {stats['max_mismatched_acc_mean']:.4f} ± {stats['max_mismatched_acc_std']:.4f}\n")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Final Accuracy: {stats['final_acc_mean']:.4f} ± {stats['final_acc_std']:.4f}")
    print(f"Max Accuracy: {stats['max_acc_mean']:.4f} ± {stats['max_acc_std']:.4f}")
    if "mnli" in args.dataset:
        print(f"Final Mismatched Accuracy: {stats['final_mismatched_acc_mean']:.4f} ± {stats['final_mismatched_acc_std']:.4f}")
        print(f"Max Mismatched Accuracy: {stats['max_mismatched_acc_mean']:.4f} ± {stats['max_mismatched_acc_std']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning with DoRA/LoRA")
    parser.add_argument("--method", type=str, required=True, choices=["base", "fedora", "kd", "muon", "ns", "ns_manifold", "fedora+kd", "fedora+muon",
                        "fedora+kd+muon", "kd+muon", "muon+ns", "muon+ns_manifold"])
    parser.add_argument("--dataset", type=str, required=True, choices=["sst2", "qqp", "qnli", "mnli_matched", "mnli_mismatched"])
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Model name from Hugging Face")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--num_runs", type=int, default=3, help="Number of times to run the experiment")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--cuda_device", type=int, default=0)
    args = parser.parse_args()
    main(args)