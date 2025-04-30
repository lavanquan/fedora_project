# data_utils.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate

def tokenize_function(examples, tokenizer, dataset_name):
    """
    Tokenize examples based on the dataset type.
    
    Args:
        examples: The examples to tokenize
        tokenizer: The tokenizer to use
        dataset_name: Name of the dataset to determine tokenization approach
    
    Returns:
        Tokenized outputs
    """
    # Different tokenization based on dataset type
    if dataset_name == "sst2":
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=None)
    elif dataset_name == "qnli":
        outputs = tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=None)
    elif dataset_name == "qqp":
        outputs = tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=None)
    else:  # MNLI datasets
        outputs = tokenizer(examples['premise'], examples['hypothesis'], truncation=True, max_length=None)
    return outputs

def collate_fn(examples, tokenizer):
    """
    Collate function for DataLoader to pad examples to the same length.
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer for padding
        
    Returns:
        Padded batch
    """
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

def load_glue_dataset(dataset_name, model_name_or_path="roberta-large", num_clients=3, batch_size=32):
    """
    Load and prepare GLUE dataset for federated learning.
    
    Args:
        dataset_name: Name of the GLUE dataset
        model_name_or_path: Model name for tokenizer
        num_clients: Number of clients for federated learning
        batch_size: Batch size for dataloaders
        
    Returns:
        Client data splits and evaluation dataloader(s)
    """
    # Set padding side based on model type
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset and metric
    if dataset_name == "mnli_matched" or dataset_name == "mnli_mismatched":
        # Always load the full MNLI dataset when either matched or mismatched is specified
        datasets = load_dataset("glue", "mnli")
        metric = evaluate.load("glue", "mnli")
    else:
        datasets = load_dataset("glue", dataset_name)
        metric = evaluate.load("glue", dataset_name)
    
    # Determine columns to remove based on dataset
    if dataset_name == "sst2":
        remove_columns = ["idx", "sentence"]
    elif dataset_name == "qqp":
        remove_columns = ["idx", "question1", "question2"]
    elif dataset_name == "qnli":
        remove_columns = ["idx", "question", "sentence"]
    else:  # MNLI datasets
        remove_columns = ["idx", "premise", "hypothesis"]
    
    # Tokenize datasets
    tokenized_datasets = datasets.map(
        lambda examples: tokenize_function(examples, tokenizer=tokenizer, dataset_name=dataset_name),
        batched=True,
        remove_columns=remove_columns,
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Create collate function for DataLoader
    def dataloader_collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Create evaluation dataloaders
    if dataset_name == "mnli_matched" or dataset_name == "mnli_mismatched":
        eval_dataloader = DataLoader(
            tokenized_datasets["validation_matched"], 
            shuffle=False, 
            collate_fn=dataloader_collate_fn, 
            batch_size=batch_size
        )
        mismatched_eval_dataloader = DataLoader(
            tokenized_datasets["validation_mismatched"], 
            shuffle=False, 
            collate_fn=dataloader_collate_fn, 
            batch_size=batch_size
        )
    else:
        eval_dataloader = DataLoader(
            tokenized_datasets["validation"], 
            shuffle=False, 
            collate_fn=dataloader_collate_fn, 
            batch_size=batch_size
        )

    # Split training data for clients
    total_len = len(tokenized_datasets["train"])
    split_size = total_len // num_clients
    remainder = total_len % num_clients
    
    # Distribute remainder evenly
    split_lengths = [split_size] * num_clients
    for i in range(remainder):
        split_lengths[i] += 1
        
    # Create client data splits
    train_dataloader = torch.utils.data.random_split(tokenized_datasets["train"], split_lengths)
    client_data_splits = [
        DataLoader(
            train_dataloader[i], 
            shuffle=True, 
            collate_fn=dataloader_collate_fn, 
            batch_size=batch_size
        ) for i in range(num_clients)
    ]

    # Return appropriate data based on dataset type
    if dataset_name == "mnli_matched" or dataset_name == "mnli_mismatched":
        return client_data_splits, eval_dataloader, mismatched_eval_dataloader, metric
    else:
        return client_data_splits, eval_dataloader, metric

def split_non_iid(dataset, num_clients=3):
    """
    Split dataset into non-IID partitions for federated learning clients.
    
    Args:
        dataset: The dataset to split
        num_clients: Number of clients
        
    Returns:
        List of dataset subsets for each client
    """
    data_size = len(dataset)
    indices = np.random.permutation(data_size)
    client_size = data_size // num_clients
    client_data = []
    
    for i in range(num_clients):
        start = i * client_size
        end = (i + 1) * client_size
        subset = dataset.select(indices[start:end])
        client_data.append(subset)
        
    return client_data
