from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import copy
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
from muon import Muon, zeropower_via_newtonschulz5
from torch.nn import CosineSimilarity
# New imports for mixed precision training
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """Soft cross entropy loss for knowledge distillation"""
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

class Client():
    def __init__(self, client_id, model, data, device, train_method="base", privacy_engine=None, num_epochs=2):
        """Initialize client with model, data, and training configuration"""
        self.client_id = client_id
        self.model = model
        self.device = device
        self.data = data
        self.optimizers = []
        self.privacy_engine = privacy_engine
        self.num_epochs = num_epochs
        
        # Memory optimization settings (default values)
        self.gradient_accumulation_steps = 1
        self.use_fp16 = False
        
        # Set up optimizers based on training method
        self.set_optimizers(train_method)

        # Privacy setup using make_private_with_epsilon (setting epsilon = 6)
        if self.privacy_engine is not None:
            self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=DataLoader(self.data, batch_size=16, shuffle=True),
                epochs=5,
                target_epsilon=6.0,  # Desired privacy budget
                target_delta=1e-5,
                max_grad_norm=1.0
            )

    def set_optimizers(self, train_method):
        """Configure optimizers based on parameter dimensions and training method"""
        # Separate parameters for different optimizers
        muon_params = []
        adamw_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.ndim >= 2:
                    muon_params.append(param)  # Matrix parameters for Muon
                else:
                    adamw_params.append(param)  # Vector parameters for AdamW

        self.optimizers = []
        # Add Muon optimizer for matrix parameters if available
        if muon_params:
            if "muon" in train_method:
                self.optimizers.append(Muon(muon_params, lr=3e-4, momentum=0.95))
            else:
                self.optimizers.append(torch.optim.AdamW(muon_params, lr=3e-4))
        else:
            print("Warning: No parameters for Muon optimizer")
        
        # Add AdamW optimizer for vector parameters if available
        if adamw_params:
            self.optimizers.append(torch.optim.AdamW(adamw_params, lr=3e-4))
        else:
            print("Warning: No parameters for AdamW optimizer")

        if not self.optimizers:
            raise ValueError("No trainable parameters found for any optimizer!")        

    def train(self, epochs=2, regularization_strength=1, train_method="base", base_model=None, 
              gradient_accumulation_steps=1, use_fp16=False):
        """Main training method supporting multiple training approaches with memory optimizations"""
        # Initialize distributed training environment if using Muon
        if "muon" in train_method:
            dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)

        # Set memory optimization parameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_fp16 = use_fp16
        
        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler() if self.use_fp16 else None

        # Prepare models
        self.model.to(self.device)
        if "kd" in train_method:  # Knowledge distillation requires a base model
            base_model.to(self.device)
            base_model.eval()

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            for step, batch in enumerate(self.data):
                batch.to(self.device)
                
                # Use autocast for mixed precision if enabled
                with torch.amp.autocast('cuda') if self.use_fp16 else nullcontext():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Get base model outputs for knowledge distillation if needed
                    if "kd" in train_method:
                        with torch.no_grad():
                            base_outputs = base_model(**batch, output_hidden_states=True)

                    # Calculate additional losses based on training method
                    loss_kd = 0
                    loss_orth = 0
                    if "kd" in train_method:
                        loss_kd = self.kd_loss(outputs, base_outputs)
                    if "fedora" in train_method:
                        loss_orth = self.orthogonal_loss()
                        
                    # Combine all losses
                    loss = loss + loss_kd + loss_orth
                    
                    # Scale loss for gradient accumulation
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps

                # Handle backpropagation with or without mixed precision
                if self.use_fp16:
                    # Mixed precision backward pass
                    scaler.scale(loss).backward()
                    
                    # Only update weights after accumulating enough gradients
                    if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.data):
                        for opt in self.optimizers:
                            scaler.step(opt)
                        scaler.update()
                        for opt in self.optimizers:
                            opt.zero_grad()
                else:
                    # Standard backward pass
                    loss.backward()
                    
                    # Only update weights after accumulating enough gradients
                    if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.data):
                        for opt in self.optimizers:
                            opt.step()
                        for opt in self.optimizers:    
                            opt.zero_grad()

            # Apply orthogonal repair after each epoch if using Newton-Schulz method
            if "ns" in train_method:
                self.orthogonal_repair(train_method)    

        # Clean up distributed training environment
        if "muon" in train_method:
            dist.destroy_process_group()


    def kd_loss(self, outputs, base_outputs, weighted=0.5):
        """Calculate knowledge distillation loss between student and teacher models"""
        # Distance from output logits
        dis_loss = distillation_loss(outputs.logits, base_outputs.logits)
        return dis_loss

    def orthogonal_loss(self, lora_alpha=16):
        """Calculate orthogonality regularization loss for LoRA parameters"""
        regu_loss = 0
        with torch.no_grad():
            for name, param in list(self.model.named_parameters()):
                if 'lora_B' in name:
                    # Get related parameters
                    B = param
                    A_name = name.replace('lora_B', 'lora_A')
                    W_name = name.replace('lora_B', 'default')
                    magnitude_name = name.replace('lora_B', 'lora_magnitude_vector')
                    A = dict(self.model.named_parameters())[A_name]
                    W = dict(self.model.named_parameters())[W_name] if W_name in dict(self.model.named_parameters()) else torch.zeros_like(B @ A)
                    magnitude = dict(self.model.named_parameters())[magnitude_name]

                    # Calculate orthogonality loss using Frobenius norm
                    sqr_matrix = (W+lora_alpha*B@A).T@(W+lora_alpha*B@A)
                    temp = torch.norm(torch.eye(sqr_matrix.shape[0]).cuda()-sqr_matrix, p='fro')/sqr_matrix.shape[0]**1.5
                    regu_loss += temp
        return regu_loss

    def orthogonal_repair(self, train_method="base", lora_alpha=16, lr_AB=0.01, ns_step=5):
        """Repair orthogonality of LoRA parameters using Newton-Schulz or QR decomposition"""
        with torch.no_grad():
            for name, param in list(self.model.named_parameters()):
                if 'lora_B' in name:
                    # Get related parameters
                    B = param
                    A_name = name.replace('lora_B', 'lora_A')
                    W_name = name.replace('lora_B', 'default')
                    magnitude_name = name.replace('lora_B', 'lora_magnitude_vector')
                    A = dict(self.model.named_parameters())[A_name]
                    W = dict(self.model.named_parameters())[W_name] if W_name in dict(self.model.named_parameters()) else torch.zeros_like(B @ A)
                    magnitude = dict(self.model.named_parameters())[magnitude_name]
                
                    # Calculate combined weight matrix
                    combined = W + lora_alpha * B @ A
                    
                    # Orthogonalize using QR decomposition or Newton-Schulz algorithm
                    if "manifold" in train_method:
                        Q, _ = torch.qr(combined)
                    else:
                        Q = zeropower_via_newtonschulz5(combined, steps=ns_step)
                    
                    # Calculate delta for gradient descent
                    delta = (Q - W)/lora_alpha
                
                    # Update B and A using local gradient descent
                    B_grad = delta @ A.T  # Gradient for B
                    A_grad = B.T @ delta  # Gradient for A
                    B.add_(-lr_AB * B_grad)
                    A.add_(-lr_AB * A_grad)

                    # Update magnitude to preserve scale
                    new_combined = W + lora_alpha * B @ A
                    target_magnitude = magnitude * combined @ new_combined.T.to(combined.dtype)
                    magnitude = target_magnitude

    def evaluate(self, eval_dataloader):
        """Evaluate model performance on evaluation dataset"""
        self.model.to(self.device)
        self.model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch.to(self.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        return eval_metric

    def get_parameters(self):
        """Return model parameters for federated averaging"""
        return copy.deepcopy(self.model.state_dict())

    def set_parameters(self, new_params):
        """Set model parameters received from the server"""
        self.model.load_state_dict(new_params)

    def train_kd(self, epochs=5, batch_size=16, regularization_strength=1, base_model=None, use_cosine=False):
        """Legacy knowledge distillation training method"""
        self.model.to(self.device)
        base_model.to(self.device)

        self.model.train()
        base_model.eval()
        for epoch in range(epochs):
            for step, batch in enumerate(tqdm(self.data)):
                batch.to(self.device)

                # Get teacher model outputs
                with torch.no_grad():
                    base_outputs = base_model(**batch, output_hidden_states=True)

                # Get student model outputs
                outputs = self.model(**batch, output_hidden_states=True)
                loss = outputs.loss

                # Calculate orthogonality regularization
                is_done = False
                regu_loss = 0
                num_param = 0
                for n, p in self.model.named_parameters():
                    if 'base_layer.weight' in n:
                        base = p
                    if 'lora_A' in n:
                        lora_A = p
                    if 'lora_B' in n:
                        lora_B = p
                        is_done = True

                    if is_done:
                        sqr_matrix = (base+lora_B@lora_A).T@(base+lora_B@lora_A)
                        temp = torch.norm(torch.eye(sqr_matrix.shape[0]).cuda()-sqr_matrix)/sqr_matrix.shape[0]**1.5
                        regu_loss += temp

                        num_param += 1
                        is_done = False
                    
                # Calculate distillation loss
                dis_loss = distillation_loss(outputs.logits, base_outputs.logits)

                # Combine losses
                loss += regularization_strength*regu_loss
                loss = 0.5*loss + 0.5*dis_loss

                # Add cosine similarity loss if enabled
                if use_cosine==True:
                    last_hidden_state_base = base_outputs.hidden_states[-1]
                    cls_embedding_base = last_hidden_state_base[:, 0, :]
                    last_hidden_state = outputs.hidden_states[-1]
                    cls_embedding = last_hidden_state[:, 0, :]
                    
                    cos = nn.CosineSimilarity(eps=1e-6)
                    loss = 0.9*loss+0.1*(1-cos(cls_embedding, cls_embedding_base).mean())

                # Backpropagation and optimization
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
