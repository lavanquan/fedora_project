import os
import argparse
import torch
import copy
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)
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

from data_utils import load_glue_dataset, split_non_iid
from client import Client
from server import Server

def main(args):
    # Set up device
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

    # Load dataset and create dataloaders
    if "mnli" in args.dataset:
        client_data_splits, eval_dataloader, mismatched_eval_dataloader, metric = load_glue_dataset(
            dataset_name=args.dataset,
            num_clients=args.num_clients,
            batch_size=args.batch_size  # Using batch_size parameter
        )
    else:
        client_data_splits, eval_dataloader, metric = load_glue_dataset(
            dataset_name=args.dataset,
            num_clients=args.num_clients,
            batch_size=args.batch_size  # Using batch_size parameter
        )

    # Initialize model with correct number of labels
    num_labels = 3 if "mnli" in args.dataset else 2

    # Initialize model with LoRA configuration
    peft_config = LoraConfig(
        use_dora=True,
        task_type="SEQ_CLS",
        inference_mode=False,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=num_labels,
        return_dict=True
    )
    model = get_peft_model(base_model, peft_config)

    # Initialize server and clients
    server = Server(global_model=model, device=device)
    clients = [
        Client(
            client_id=i,
            model=copy.deepcopy(model),
            data=client_data_splits[i],
            device=device,
            train_method=args.method
        ) for i in range(args.num_clients)
    ]

    # Training loop
    with open(args.output_file, "w") as f:
        # Write experiment configuration
        f.write(f"Method: {args.method}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Number of Clients: {args.num_clients}\n")
        f.write(f"Number of Rounds: {args.num_rounds}\n")
        f.write(f"Number of Epochs per Round: {args.num_epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        if args.gradient_accumulation_steps > 1:
            f.write(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}\n")
        if args.fp16:
            f.write(f"Mixed Precision Training: Enabled\n")
        f.write("\n--- Training Start ---\n")

        max_acc = 0
        mismatched_max_acc = 0

        # Federated learning rounds
        rounds_iter = tqdm(range(args.num_rounds), desc="Federated Learning Rounds", leave=True, ncols=100, mininterval=1.0)
        for round_num in rounds_iter:
            print(f"\n--- Round {round_num + 1}/{args.num_rounds} Starting ---")
            
            client_models = []
            if "muon" in args.method:
                processes = []
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = str(args.master_port)

            # Train each client
            print("  Starting client training for this round...")
            for client in clients:
                if "kd" in args.method:
                    client.train(
                        epochs=args.num_epochs, 
                        train_method=args.method, 
                        base_model=base_model,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        use_fp16=args.fp16
                    )
                else:
                    client.train(
                        epochs=args.num_epochs, 
                        train_method=args.method,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        use_fp16=args.fp16
                    )
                client_models.append(client.get_parameters())
            print("  Client training finished for this round.")

            # Server-side aggregation and evaluation
            print("\n  Aggregating client models...")
            server.aggregate(client_models)
            print("  Server aggregation complete.")
            eval_metric = server.evaluate(eval_dataloader, metric=metric)
            
            if "mnli" in args.dataset:
                mismatched_eval_metric = server.evaluate(mismatched_eval_dataloader, metric=metric)
            
            # Update best accuracy
            if max_acc < eval_metric['accuracy']:
                max_acc = eval_metric['accuracy']

            if "mnli" in args.dataset and mismatched_max_acc < mismatched_eval_metric["accuracy"]:
                mismatched_max_acc = mismatched_eval_metric["accuracy"]
                
            # Update progress bar with current metrics
            round_metrics = {"accuracy": f"{eval_metric['accuracy']:.4f}"}
            if "mnli" in args.dataset:
                round_metrics["mismatched"] = f"{mismatched_eval_metric['accuracy']:.4f}"
            round_metrics["best_acc"] = f"{max_acc:.4f}"
            rounds_iter.set_postfix(round_metrics)

            print("\n  Updating clients with new global model...")
            # Update clients with new global model
            for client in clients:
                client.set_parameters(server.global_model.state_dict())

            # Log results
            if "mnli" in args.dataset:
                f.write(f"  Global Accuracy after Round {round_num + 1}: {eval_metric['accuracy']:.4f}, {mismatched_eval_metric['accuracy']:.4f}%\n\n")
            else:
                f.write(f"  Global Accuracy after Round {round_num + 1}: {eval_metric['accuracy']:.4f}%\n\n")
        
        f.write("\n--- Training End ---\n")
        # Write final results
        if "mnli" in args.dataset:
            f.write(f"  Global Accuracy matched final: {eval_metric['accuracy']:.4f} and max_acc: {max_acc:.4f}%\n\n")
            f.write(f"  Global Accuracy mismatched final: {mismatched_eval_metric['accuracy']:.4f} and max_acc: {mismatched_max_acc:.4f}%\n\n")
        else:
            f.write(f"  Global Accuracy final: {eval_metric['accuracy']:.4f} and max_acc: {max_acc:.4f}%\n\n")

    print(f"\nCompleted {args.method} on {args.dataset} with {args.model_name}. Final Accuracy: {eval_metric['accuracy']:.4f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning with DoRA/LoRA")
    parser.add_argument("--method", type=str, required=True, 
                    choices=["base", "fedora", "kd", "muon", "ns", "ns_manifold", 
                            "fedora+kd", "fedora+muon", "fedora+kd+muon", "kd+muon",
                            "muon+ns", "muon+ns_manifold"])
    parser.add_argument("--dataset", type=str, required=True,
                    choices=["sst2", "qqp", "qnli", "mnli_matched", "mnli_mismatched"])
    parser.add_argument("--model_name", type=str, default="roberta-base",
                    help="Model name from Hugging Face")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    # Memory optimization parameters
    parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for training and evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of steps to accumulate gradients (1 = no accumulation)")
    parser.add_argument("--fp16", action="store_true",
                    help="Use mixed precision training to reduce memory usage")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--cuda_device", type=int, default=0)
    # Dynamic port for multi-GPU training with muon
    parser.add_argument("--master_port", type=int, default=int(os.environ.get("MASTER_PORT", "12355")))
    args = parser.parse_args()
    main(args)