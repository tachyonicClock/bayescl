Continual LoRA (c-LoRA)
-----------------------

Smith, J. S., Hsu, Y.-C., Zhang, L., Hua, T., Kira, Z., Shen, Y., & Jin, H. (2024).
Continual diffusion: Continual customization of text-to-image diffusion with c-LoRA.
2024. https://openreview.net/forum?id=TZdEgwZ6f3

Prompt:

> https://arxiv.org/html/2304.06027v2
>
> Extract the technical details necessary to produce a faithful implementation of
> the attached research. You should provide python style psudo-code, equation, and
> make reference to the equations in the text.

# Gemini 3.1 Pro Preview:

Based on the provided research paper, **Continual Diffusion: Continual Customization of Text-to-Image Diffusion with C-LoRA (2304.06027)**, here are the technical details, equations, and Python-style pseudocode necessary to produce a faithful implementation.

### 1. Architectural Targets 
The method specifically targets the cross-attention blocks within the Stable Diffusion U-Net, as these layers are most sensitive to learning new concepts. Specifically, only the Key ($W^K$) and Value ($W^V$) projection matrices responsible for projecting the text features are modified using a continually self-regularized Low-Rank Adaptation (C-LoRA). 

### 2. Equations & Mathematical Formulation
The formulation prevents catastrophic forgetting by applying a novel self-regularization penalty that prevents current LoRA weights from altering parameters that have been heavily edited by prior concepts. 

When learning customization task $t$, we minimize the overall objective detailed in **Equation (1)**:
$$ \min_{\bm{W}^{K,V}_{t}\in\theta} \mathcal{L}_{diff}(x,\bm{\theta}) + \lambda \mathcal{L}_{forget}(\bm{W}^{K,V}_{t-1},\bm{W}^{K,V}_{t}) \quad \text{--- (Eq. 1)} $$
*Where $\mathcal{L}_{diff}$ is the standard diffusion denoising loss, $\lambda$ is a hyperparameter scaling the forgetting penalty, and $x$ is the input data for the new concept.*

To achieve this, the weights of the Key and Value matrices at task $t$ are decomposed as defined in **Equation (2)**:
$$ \bm{W}^{K,V}_{t} = \bm{W}^{K,V}_{init} + \left[\sum_{t^{\prime}=1}^{t-1}\bm{A}^{K,V}_{t^{\prime}}\bm{B}^{K,V}_{t^{\prime}}\right] + \bm{A}^{K,V}_{t}\bm{B}^{K,V}_{t} \quad \text{--- (Eq. 2)} $$
*Where $W_{init}$ contains the frozen pretrained Stable Diffusion weights, the summation brackets contain the accumulated locked weight-deltas from all previous tasks, and $A_t \in \mathbb{R}^{D_1 \times r}$, $B_t \in \mathbb{R}^{r \times D_2}$ are the new low-rank parameters being optimized for the current task.*

To prevent forgetting, the regularizer explicitly penalizes modifications to regions of the weight matrix that were heavily updated in the past, as shown in **Equation (3)**:
$$ \mathcal{L}_{forget} = \left|\left| \left| \sum_{t^{\prime}=1}^{t-1}\bm{A}^{K,V}_{t^{\prime}}\bm{B}^{K,V}_{t^{\prime}} \right| \odot \bm{A}^{K,V}_{t}\bm{B}^{K,V}_{t} \right|\right|^{2}_{F} \quad \text{--- (Eq. 3)} $$
*Where $\odot$ is the element-wise (Hadamard) product, $|\cdot|$ denotes the element-wise absolute value, and $||\cdot||_F$ is the Frobenius norm. This forces the new LoRA weights to adapt using parameters less likely to interfere with past tasks.*

### 3. Customized Token Strategy
To separate concept embeddings and drastically reduce task interference, C-LoRA introduces two tokenization rules:
1. **Random Initialization:** Instead of initializing the new custom token $V_t^*$ with a lesser-used existing word, it is initialized as a completely random vector in the embedding space.
2. **Concept Name Omission:** The class noun is removed from the customization prompt. Instead of using *"A photo of $V_t^*$ person"*, the prompt is strictly *"A photo of $V_t^*$"*.

### 4. Python-Style Pseudocode

The following PyTorch-style pseudocode demonstrates how to faithfully implement C-LoRA, integrating the mathematics from **(Eq. 1)**, **(Eq. 2)**, and **(Eq. 3)**.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLoraLinear(nn.Module):
    """
    C-LoRA wrapper for Key and Value Cross-Attention projection matrices.
    Implements Eq. 2 and Eq. 3.
    """
    def __init__(self, in_features, out_features, rank=16):
        super().__init__()
        self.rank = rank
        
        # W_{init}: Frozen pre-trained Stable Diffusion weights
        self.W_init = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        
        # Accumulated historical LoRA weight deltas: \sum A_{t'} B_{t'}
        self.W_past = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        
        # Current task specific parameters: A_t and B_t
        self.A = nn.Parameter(torch.zeros(out_features, rank), requires_grad=True)
        self.B = nn.Parameter(torch.randn(rank, in_features), requires_grad=True)
        
    def forward(self, x):
        # Implementation of Eq. 2: 
        # W_t = W_{init} + \sum_{t'} A_{t'}B_{t'} + A_t B_t
        current_lora = self.A @ self.B
        W_t = self.W_init + self.W_past + current_lora
        return F.linear(x, W_t)
        
    def get_forgetting_loss(self):
        # Implementation of Eq. 3: 
        # L_{forget} = || |W_{past}| \odot (A_t B_t) ||^2_F
        current_lora = self.A @ self.B
        # Element-wise absolute of past weights multiplied by current delta
        penalty = torch.abs(self.W_past) * current_lora 
        return torch.sum(penalty ** 2) # Frobenius norm squared
        
    def end_of_task_update(self):
        # After training a concept, fold the learned A and B into W_past
        with torch.no_grad():
            self.W_past += (self.A @ self.B)
            # Re-initialize A_t and B_t for the next incoming task
            self.A.zero_()
            nn.init.normal_(self.B)


def train_continual_diffusion(task_dataloaders, unet, text_encoder, tokenizer, num_tasks, lambda_reg=0.1):
    """
    Main training loop demonstrating the Customized Token Strategy and Total Loss (Eq. 1)
    """
    # Replace U-Net Cross-Attention W^K and W^V layers with CLoraLinear prior to this step
    
    for t in range(num_tasks):
        # 1. Custom Tokenization Strategy
        new_token = f"<V*_{t}>"
        tokenizer.add_tokens([new_token])
        new_token_id = tokenizer.convert_tokens_to_ids(new_token)
        text_encoder.resize_token_embeddings(len(tokenizer))
        
        # Initialize token as completely random embedding (diverging from CustomDiffusion)
        with torch.no_grad():
            random_embed = torch.randn_like(text_encoder.get_input_embeddings().weight[new_token_id])
            text_encoder.get_input_embeddings().weight[new_token_id] = random_embed
             
        # Optimizer targets only the C-LoRA layer weights (A, B) and the newly added token embedding
        optimizer = torch.optim.AdamW(
             [p for p in unet.parameters() if p.requires_grad] +
             [text_encoder.get_input_embeddings().weight], lr=1e-5
        )

        # Iterate over images and prompts (Note: Prompt omits the class name, e.g., "A photo of <V*_t>")
        for batch in task_dataloaders[t]:
            images, prompts = batch
            
            # L_{diff}: Standard latent diffusion loss
            loss_diff = compute_diffusion_loss(unet, text_encoder, images, prompts)
            
            # Aggregate L_{forget} (Eq. 3) across all C-LoRA layers in the U-Net
            loss_forget = 0.0
            for module in unet.modules():
                if isinstance(module, CLoraLinear):
                    loss_forget += module.get_forgetting_loss()
                    
            # Total Loss (Eq. 1)
            total_loss = loss_diff + (lambda_reg * loss_forget)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
        # 2. Continual Step: lock weights into the past task state
        for module in unet.modules():
            if isinstance(module, CLoraLinear):
                module.end_of_task_update()
```
