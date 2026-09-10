# InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning

> https://arxiv.org/html/2404.00228v3
> https://github.com/liangyanshuo/InfLoRA
>
> Extract the technical details necessary to produce a faithful implementation of
> the attached research. You should provide python style psudo-code, equation, and
> make reference to the equations in the text.

## Gemini Pro 3.1

Based on the technical details from the provided arXiv paper (2404.00228) and the `liangyanshuo/InfLoRA` GitHub repository, **InfLoRA (Interference-Free Low-Rank Adaptation)** addresses the stability-plasticity dilemma in continual learning by injecting a sequence of task-specific, low-rank adapters. 

By freezing a cleverly initialized projection matrix and only fine-tuning an expansion matrix, InfLoRA ensures that the weight updates for a new task lie strictly within a subspace orthogonal to the representation spaces of previous tasks. 

Here is the extraction of the necessary equations and python-style pseudo-code to reproduce the methodology.

### 1. Mathematical Formulation

For a pre-trained linear layer with weight $W \in \mathbb{R}^{d_{O} \times d_{I}}$ and a sequence of tasks $t=1 \dots T$, InfLoRA reparameterizes the weights for the $t$-th task as:
$$W_t = W_{t-1} + A_t B_t$$

Where $B_t \in \mathbb{R}^{r \times d_{I}}$ is a dimension-reducing matrix (frozen during training) and $A_t \in \mathbb{R}^{d_{O} \times r}$ is a dimension-increasing matrix (trainable). 

To make learning **interference-free**, $B_t$ is carefully constructed to act as an orthogonal projector against old tasks. 
1. Let $H_t \in \mathbb{R}^{d_{I} \times N}$ be the activation matrix (input features) collected from the training data of the current task $t$.
2. Let $M_{t-1} \in \mathbb{R}^{d_{I} \times k_{t-1}}$ be the matrix containing the orthonormal bases of the past tasks' gradient spaces (maintained via DualGPM [INDEX]).
3. InfLoRA projects $H_t$ onto the orthogonal complement of the past task space ($\mathcal{M}_{t-1}^\perp$):
   $$\hat{H}_t = H_t - M_{t-1} M_{t-1}^T H_t$$
4. SVD is performed on the transpose of this projected matrix:
   $$(\hat{H}_t)^T = V_t \Sigma_t U_t$$
5. $B_t$ is then instantiated using the principal components (top-$r$ right singular vectors). As referenced in **Equation (8)** in the text:
   $$B_t = (U_t)_r$$
   Because $B_t$ is extracted from the null space of $M_{t-1}$, $B_t M_{t-1} \approx 0$. When past task data passes through the layer, the added adapter ignores it ($A_t B_t x_{old} \approx 0$), preserving performance.

During training, the objective focuses locally on the current task's classes. As defined in **Equation (9)** in the text:
$$\mathcal{L}(\mathcal{D}_t) = \frac{1}{|\mathcal{D}_t|} \sum_{(x,y) \in \mathcal{D}_t} \mathcal{L}_{ce}(\text{mask}(h_\Phi(f_\Theta(x))), y)$$
Where the `mask` function zeros out logits belonging to previously learned and unseen future classes.

### 2. Python Pseudo-Code

Below is a faithful PyTorch-style implementation of the InfLoRA core layer and task loop.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False) # W_{t-1}
        self.rank = rank
        
        # Module lists to store adapters for sequentially arriving tasks
        self.A = nn.ParameterList()
        self.B = nn.ParameterList()
        
    def forward(self, x):
        # Base forward pass
        out = self.linear(x)
        
        # Accumulate task-specific LoRA branches
        for A_t, B_t in zip(self.A, self.B):
            # x shape: (batch, d_in), B_t shape: (r, d_in), A_t shape: (d_out, r)
            out += (x @ B_t.T) @ A_t.T
        return out

    def expand_task_branch(self, H_t, M_prev):
        """
        Initializes A_t and B_t for a new task.
        H_t: Activation matrix of current task, shape (d_in, N)
        M_prev: Orthonormal basis of old tasks, shape (d_in, k)
        """
        # 1. Project H_t to the orthogonal complement of M_prev
        if M_prev is not None:
            # \hat{H}_t = H_t - M_{t-1} M_{t-1}^T H_t
            H_hat = H_t - M_prev @ (M_prev.T @ H_t)
        else:
            H_hat = H_t
            
        # 2. SVD on (\hat{H}_t)^T
        # In PyTorch SVD, Vh corresponds to U_t in the paper
        U, S, Vh = torch.linalg.svd(H_hat.T, full_matrices=False)
        
        # 3. Equation (8): B_t = (U_t)_r
        B_t_val = Vh[:self.rank, :] # Take top-r rows, shape (r, d_in)
        
        # 4. Initialize parameters
        B_t = nn.Parameter(B_t_val, requires_grad=False) # B_t is FROZEN
        
        A_t_val = torch.zeros(self.linear.out_features, self.rank)
        A_t = nn.Parameter(A_t_val, requires_grad=True)  # A_t is TRAINABLE
        
        self.B.append(B_t)
        self.A.append(A_t)


def local_cross_entropy(logits, targets, current_task_classes):
    """Equation (9): Local Cross-Entropy loss with masking."""
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[:, current_task_classes] = False
    
    # Mask out classes belonging to old or future tasks
    logits = logits.masked_fill(mask, float('-inf'))
    return F.cross_entropy(logits, targets)


# --- Example Task Loop ---
# memory_basis is updated using DualGPM after each task
memory_basis = {layer_name: None for layer_name in model.inflora_layers.keys()}

for task_id, (task_data, task_classes) in enumerate(continual_learning_stream):
    
    # 1. Collect Activation Matrices (H_t) using a forward hook over a sample of task_data
    activations_Ht = extract_activations(model, task_data) 
    
    # 2. Expand branches & construct B_t across all InfLoRA layers
    for layer_name, layer in model.inflora_layers.items():
        layer.expand_task_branch(
            H_t=activations_Ht[layer_name], 
            M_prev=memory_basis[layer_name]
        )
        
    # 3. Train on current task (only A_t parameters will receive gradients)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for x, y in task_data:
        optimizer.zero_grad()
        logits = model(x)
        
        # Loss using Eq. 9
        loss = local_cross_entropy(logits, y, task_classes)
        loss.backward()
        optimizer.step()
        
    # 4. Update Memory Subspace (M_t) using DualGPM algorithms
    # This recalculates the orthonormal bases from \hat{H}_t and updates M_prev
    memory_basis = update_dual_gpm_memory(memory_basis, activations_Ht)
```

### 3. Implementation Details for Best Results
* **DualGPM Thresholds**: The threshold bounds (controlling the expansion tolerance of $M_t$ after each task) critically dictate the stability/plasticity trade-off. Start with an explained variance retention parameter like $\epsilon = 0.97 \sim 0.99$ as advised in the research implementation.
* **Trainable Head**: Remember that in classification applications of InfLoRA, in addition to $A_t$, the classifier head for the newly introduced classes should also be registered to `requires_grad=True` during the learning of task $t$.
