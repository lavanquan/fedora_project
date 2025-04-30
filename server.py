import copy
import torch
from tqdm import tqdm

class Server:
    def __init__(self, global_model, device):
        """
        Initialize the server with a global model and device.
        
        Args:
            global_model: The initial model (typically with PEFT/LoRA configuration)
            device: The device to run computations on (CPU or GPU)
        """
        self.global_model = global_model
        self.device = device

    def aggregate(self, client_models):
        """
        Aggregate models using Federated Averaging (FedAvg)
        
        This is the core of federated learning - combining multiple locally trained
        models into a single global model by averaging their parameters.
        
        Args:
            client_models: List of model state dictionaries from clients
        """
        global_params = copy.deepcopy(client_models[0])

        # Average the parameters of all client models
        for key in global_params.keys():
            for i in range(1, len(client_models)):
                global_params[key] += client_models[i][key]
            global_params[key] = torch.div(global_params[key], len(client_models))

        # Update global model with averaged parameters
        self.global_model.load_state_dict(global_params)

    def evaluate(self, eval_dataloader, metric):
        """
        Evaluate the global model on the validation dataset
        
        Runs inference on the evaluation dataset and computes metrics
        (e.g., accuracy for GLUE tasks) to assess model performance.
        
        Args:
            eval_dataloader: DataLoader containing validation data
            metric: The evaluation metric object (from HuggingFace evaluate)
            
        Returns:
            eval_metric: Dictionary containing evaluation results
        """
        # dataloader = DataLoader(val_dataset, batch_size=16)
        # torch.cuda.current_device()
        self.global_model.to(self.device)
        self.global_model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch.to(self.device)
            with torch.no_grad():
                outputs = self.global_model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        
        eval_metric = metric.compute()
        return eval_metric
