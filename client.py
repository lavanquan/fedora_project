from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import copy
# from .client import *
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
from muon import Muon, zeropower_via_newtonschulz5
from torch.nn import CosineSimilarity

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """Soft cross entropy loss for knowledge distillation"""
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

class Client():
    def __init__(self, client_id, model, data, device, train_method="base", privacy_engine=None, num_epochs=2):
        self.client_id = client_id
        self.model = model
        self.device = device
        self.data = data
        self.optimizers = []
        self.set_optimizers(train_method)
        self.privacy_engine = privacy_engine
        self.num_epochs = num_epochs
        # self.training_name = None

        # Privacy setup using make_private_with_epsilon (setting epsilon = 6)
        if self.privacy_engine is not None:
            self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=DataLoader(self.data, batch_size=16, shuffle=True),
                epochs=5,  # Number of epochs for which privacy accounting is done
                target_epsilon=6.0,  # Desired privacy budget
                target_delta=1e-5,
                max_grad_norm=1.0
            )

    def train(self, epochs=2, batch_size=16,  regularization_strength=1, train_method="base", base_model=None):
        # define training env
        if "muon" in train_method:
            dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)

        # define model and base model
        self.model.to(self.device)
        if "kd" in train_method:
            base_model.to(self.device)
            base_model.eval()

        # define optimizer
        # self.optimizers = self.set_optimizers(train_method)

        # training
        self.model.train()
        for epoch in range(epochs):
            for step, batch in enumerate(self.data):
                batch.to(self.device)
                outputs = self.model(**batch)
                loss = outputs.loss
                if "kd" in train_method:
                    with torch.no_grad():
                        base_outputs = base_model(**batch, output_hidden_states=True)

                loss_kd = 0
                loss_orth = 0
                if "kd" in train_method:
                    loss_kd = self.kd_loss(outputs, base_outputs)
                if "fedora" in train_method:
                    loss_orth = self.orthogonal_loss()
                loss = loss + loss_kd + loss_orth

                loss.backward()
                for opt in self.optimizers:
                    opt.step()
                for opt in self.optimizers:    
                    opt.zero_grad()

            if "ns" in train_method:
                self.orthogonal_repair()    

        if "muon" in train_method:
            dist.destroy_process_group()

    def set_optimizers(self, train_method):
        # define optimizer
        muon_params = []
        adamw_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.ndim >= 2:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

        self.optimizers = []
        if muon_params:
            if "muon" in train_method:
                self.optimizers.append(Muon(muon_params, lr=2e-5, momentum=0.95))
            else:
                self.optimizers.append(torch.optim.AdamW(muon_params, lr=2e-5))
        else:
            print("Warning: No parameters for Muon optimizer")
        
        if adamw_params:
            self.optimizers.append(torch.optim.AdamW(adamw_params, lr=2e-5))
        else:
            print("Warning: No parameters for AdamW optimizer")

        if not self.optimizers:
            raise ValueError("No trainable parameters found for any optimizer!")        

    def kd_loss(self, outputs, base_outputs, weighted=0.5):
        # distance from output logits
        dis_loss = distillation_loss(outputs.logits, base_outputs.logits)

        # distance from hidden space 
        # last_hidden_state_base = base_outputs.hidden_states[-1]
        # cls_embedding_base = last_hidden_state_base[:, 0, :]
        # last_hidden_state = outputs.hidden_states[-1]
        # cls_embedding = last_hidden_state[:, 0, :]            
        # cos = CosineSimilarity(eps=1e-6)

        # combine loss
        # regu_loss = weighted*dis_loss+(1-weighted)*(1-cos(cls_embedding, cls_embedding_base).mean())
        return dis_loss

    def orthogonal_loss(self, lora_alpha=16):
        regu_loss = 0
        with torch.no_grad():
            for name, param in list(self.model.named_parameters()):
                if 'lora_B' in name:
                    B = param
                    A_name = name.replace('lora_B', 'lora_A')
                    W_name = name.replace('lora_B', 'default')
                    magnitude_name = name.replace('lora_B', 'lora_magnitude_vector')
                    A = dict(self.model.named_parameters())[A_name]
                    W = dict(self.model.named_parameters())[W_name] if W_name in dict(self.model.named_parameters()) else torch.zeros_like(B @ A)
                    magnitude = dict(self.model.named_parameters())[magnitude_name]

                    sqr_matrix = (W+lora_alpha*B@A).T@(W+lora_alpha*B@A)
                    temp = torch.norm(torch.eye(sqr_matrix.shape[0]).cuda()-sqr_matrix, p='fro')/sqr_matrix.shape[0]**1.5
                    regu_loss += temp
        return regu_loss

    def orthogonal_repair(self, train_method="base", lora_alpha=16, lr_AB=0.01, ns_step=5):
        with torch.no_grad():
            for name, param in list(self.model.named_parameters()):
                if 'lora_B' in name:
                    B = param
                    A_name = name.replace('lora_B', 'lora_A')
                    W_name = name.replace('lora_B', 'default')
                    magnitude_name = name.replace('lora_B', 'lora_magnitude_vector')
                    A = dict(self.model.named_parameters())[A_name]
                    W = dict(self.model.named_parameters())[W_name] if W_name in dict(self.model.named_parameters()) else torch.zeros_like(B @ A)
                    magnitude = dict(self.model.named_parameters())[magnitude_name]
                
                    # Tính W + BA và trực giao hóa bằng Newton-Schulz
                    combined = W + lora_alpha * B @ A
                    if "manifold" in train_method:
                        Q, _ = torch.qr(combined)
                    else:
                        Q = zeropower_via_newtonschulz5(combined, steps=ns_step)
                    delta = (Q - W)/lora_alpha
                
                    # Gradient descent cục bộ để cập nhật B và A
                    B_grad = delta @ A.T  # Gradient đối với B
                    A_grad = B.T @ delta  # Gradient đối với A
                    B.add_(-lr_AB * B_grad)
                    A.add_(-lr_AB * A_grad)

                    # magnitude = magnitude.to(combined.dtype)
                    new_combined = W + lora_alpha * B @ A
                    target_magnitude = magnitude * combined @ new_combined.T.to(combined.dtype)
                    # magnitude_delta = target_magnitude - magnitude
                    # magnitude.add_(magnitude_delta)
                    magnitude = target_magnitude

    # def train(self, epochs=2, batch_size=16, train_method=None, base_model=None):
    #     """Local training on client's data"""
    #     if train_method=="fedora":
    #         self.train_fedora(epochs=epochs, batch_size=batch_size)
    #     elif train_method=="ns":
    #         self.train_ns(epochs=epochs, batch_size=batch_size)
    #     elif train_method=="muon":
    #         self.train_muon(epochs=epochs, batch_size=batch_size)
    #     elif train_method=="kd":
    #         self.train_kd(epochs=epochs, batch_size=batch_size, base_model=base_model)
    #     else:
    #         self.train_base(epochs=epochs, batch_size=batch_size)

    def evaluate(self, eval_dataloader):
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

    # def train_fedora(self, epochs=5, batch_size=16, regularization_strength=1, lora_alpha=16):
    #     """Local training on client's data"""
    #     self.model.to(self.device)

    #     self.model.train()
    #     for epoch in range(epochs):
    #         for step, batch in enumerate(tqdm(self.data)):
    #         # for batch in tqdm(self.data):
    #             batch.to(self.device)
    #             outputs = self.model(**batch)
    #             loss = outputs.loss
    #             # print("----------loss = ", loss)
    #             is_done = False
    #             regu_loss = 0
    #             num_param = 0
    #             # cos = nn.CosineSimilarity(eps=1e-6)
    #             for n, p in self.model.named_parameters():
    #                 if 'base_layer.weight' in n:
    #                     base = p
    #                 if 'lora_A' in n:
    #                     lora_A = p
    #                 if 'lora_B' in n:
    #                     lora_B = p
    #                     is_done = True

    #                 if is_done:
    #                     # cos_loss = cos(base, lora_B@lora_A)
    #                     # regu_loss += cos_loss.norm()
    #                     sqr_matrix = (base+lora_alpha*lora_B@lora_A).T@(base+lora_alpha*lora_B@lora_A)
    #                     # temp = torch.norm(torch.eye(sqr_matrix.shape[0]).cuda()-sqr_matrix, p='fro')/sqr_matrix.shape[0]**1.5
    #                     temp = torch.norm(torch.eye(sqr_matrix.shape[0]).cuda()-sqr_matrix, p='fro')/sqr_matrix.shape[0]**1.5
                        
    #                     regu_loss += temp

    #                     num_param += 1
    #                     is_done = False
                    
    #             loss += regularization_strength*regu_loss

    #             loss.backward()
    #             self.optimizer.step()
    #             # self.lr_scheduler.step()
    #             self.optimizer.zero_grad()

    # def train_ns(self, epochs=5, batch_size=16, regularization_strength=1, lora_alpha=16, lr_AB=0.01, ns_step=5):
    #     """Local training on client's data"""
    #     self.model.to(self.device)
    #     # dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)

    #     self.model.train()
    #     for epoch in range(epochs):
    #         for step, batch in enumerate(tqdm(self.data)):
    #         # for batch in tqdm(self.data):
    #             batch.to(self.device)
    #             outputs = self.model(**batch)
    #             loss = outputs.loss

    #             loss.backward()
    #             self.optimizer.step()
    #             # self.lr_scheduler.step()
    #             self.optimizer.zero_grad()

    #         with torch.no_grad():
    #             for name, param in list(self.model.named_parameters()):
    #                 if 'lora_B' in name:
    #                     B = param
    #                     A_name = name.replace('lora_B', 'lora_A')
    #                     W_name = name.replace('lora_B', 'default')
    #                     magnitude_name = name.replace('lora_B', 'lora_magnitude_vector')
    #                     A = dict(self.model.named_parameters())[A_name]
    #                     W = dict(self.model.named_parameters())[W_name] if W_name in dict(self.model.named_parameters()) else torch.zeros_like(B @ A)
    #                     magnitude = dict(self.model.named_parameters())[magnitude_name]
                
    #                     # Tính W + BA và trực giao hóa bằng Newton-Schulz
    #                     combined = W + lora_alpha * B @ A
    #                     # Q = zeropower_via_newtonschulz5(combined, steps=ns_step)
    #                     Q, _ = torch.qr(combined)
    #                     delta = (Q - W)/lora_alpha
                
    #                     # Gradient descent cục bộ để cập nhật B và A
    #                     B_grad = delta @ A.T  # Gradient đối với B
    #                     A_grad = B.T @ delta  # Gradient đối với A
    #                     B.add_(-lr_AB * B_grad)
    #                     A.add_(-lr_AB * A_grad)

    #                     # magnitude = magnitude.to(combined.dtype)
    #                     new_combined = W + lora_alpha * B @ A
    #                     target_magnitude = magnitude * combined @ new_combined.T.to(combined.dtype)
    #                     # magnitude_delta = target_magnitude - magnitude
    #                     # magnitude.add_(magnitude_delta)
    #                     magnitude = target_magnitude

    # def train_muon(self, epochs=5, batch_size=16, regularization_strength=1):
    #     """Local training on client's data"""
    #     dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
    #     self.model.to(self.device)
    #     muon_params = []
    #     adamw_params = []
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             if param.ndim >= 2:
    #                 muon_params.append(param)
    #             else:
    #                 adamw_params.append(param)
    #     # print("-----------------------------", muon_params, "-------------------")
    #     optimizers = []
    #     if muon_params:
    #         optimizers.append(Muon(muon_params, lr=3e-4, momentum=0.95))
    #         # optimizers.append(torch.optim.AdamW(muon_params, lr=3e-4))
    #     else:
    #         print("Warning: No parameters for Muon optimizer")
        
    #     if adamw_params:
    #         optimizers.append(torch.optim.AdamW(adamw_params, lr=3e-4))
    #     else:
    #         print("Warning: No parameters for AdamW optimizer")

    #     if not optimizers:
    #         raise ValueError("No trainable parameters found for any optimizer!")

    #     self.model.train()
    #     for epoch in range(epochs):
    #         for step, batch in enumerate(tqdm(self.data)):
    #         # for batch in tqdm(self.data):
    #             batch.to(self.device)
    #             outputs = self.model(**batch)
    #             loss = outputs.loss

    #             is_done = False
    #             regu_loss = 0
    #             num_param = 0
    #             # cos = nn.CosineSimilarity(eps=1e-6)
    #             for n, p in self.model.named_parameters():
    #                 if 'base_layer.weight' in n:
    #                     base = p
    #                 if 'lora_A' in n:
    #                     lora_A = p
    #                 if 'lora_B' in n:
    #                     lora_B = p
    #                     is_done = True

    #                 if is_done:
    #                     # cos_loss = cos(base, lora_B@lora_A)
    #                     # regu_loss += cos_loss.norm()
    #                     sqr_matrix = (base+lora_B@lora_A).T@(base+lora_B@lora_A)
    #                     temp = torch.norm(torch.eye(sqr_matrix.shape[0]).cuda()-sqr_matrix)/sqr_matrix.shape[0]**1.5
    #                     regu_loss += temp

    #                     num_param += 1
    #                     is_done = False
                    
    #             loss += regularization_strength*regu_loss

    #             loss.backward()
    #             for opt in optimizers:
    #                 opt.step()
    #             for opt in optimizers:    
    #                 opt.zero_grad()
    #     dist.destroy_process_group()

    def train_kd(self, epochs=5, batch_size=16, regularization_strength=1, base_model=None, use_cosine=False):
        """Local training on client's data"""
        self.model.to(self.device)
        base_model.to(self.device)
        # dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)

        self.model.train()
        base_model.eval()
        for epoch in range(epochs):
            for step, batch in enumerate(tqdm(self.data)):
            # for batch in tqdm(self.data):
                batch.to(self.device)

                with torch.no_grad():
                    base_outputs = base_model(**batch, output_hidden_states=True)

                outputs = self.model(**batch, output_hidden_states=True)
                loss = outputs.loss

                is_done = False
                regu_loss = 0
                num_param = 0
                # cos = nn.CosineSimilarity(eps=1e-6)
                for n, p in self.model.named_parameters():
                    if 'base_layer.weight' in n:
                        base = p
                    if 'lora_A' in n:
                        lora_A = p
                    if 'lora_B' in n:
                        lora_B = p
                        is_done = True

                    if is_done:
                        # cos_loss = cos(base, lora_B@lora_A)
                        # regu_loss += cos_loss.norm()
                        sqr_matrix = (base+lora_B@lora_A).T@(base+lora_B@lora_A)
                        temp = torch.norm(torch.eye(sqr_matrix.shape[0]).cuda()-sqr_matrix)/sqr_matrix.shape[0]**1.5
                        regu_loss += temp

                        num_param += 1
                        is_done = False
                    
                # if num_param > 0:
                #     regu_loss = regu_loss / num_param
                # else:
                #     regu_loss = 0
                # print('loss:', loss, "regu loss:", regu_loss)
                dis_loss = distillation_loss(outputs.logits, base_outputs.logits)

                loss += regularization_strength*regu_loss

                loss = 0.5*loss + 0.5*dis_loss

                if use_cosine==True:
                    last_hidden_state_base = base_outputs.hidden_states[-1]
                    cls_embedding_base = last_hidden_state_base[:, 0, :]
                    last_hidden_state = outputs.hidden_states[-1]
                    cls_embedding = last_hidden_state[:, 0, :]
                    
                    cos = nn.CosineSimilarity(eps=1e-6)
                    loss = 0.9*loss+0.1*(1-cos(cls_embedding, cls_embedding_base).mean())



                loss.backward()
                self.optimizer.step()
                # self.lr_scheduler.step()
                self.optimizer.zero_grad()

    # def train_base(self, epochs=5, batch_size=16):
    #     """Local training on client's data"""
    #     self.model.to(self.device)
    #     # dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)

    #     self.model.train()
    #     for epoch in range(epochs):
    #         for step, batch in enumerate(tqdm(self.data)):
    #         # for batch in tqdm(self.data):
    #             batch.to(self.device)
    #             outputs = self.model(**batch)
    #             loss = outputs.loss
    #             loss.backward()
    #             self.optimizer.step()
    #             # self.lr_scheduler.step()
    #             self.optimizer.zero_grad()