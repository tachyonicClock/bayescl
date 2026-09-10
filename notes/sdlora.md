# SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning

Wu, Y., Piao, H., Huang, L.-K., Wang, R., Li, W., Pfister, H., Meng, D., Ma, K., & Wei,
Y. (2025). SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental
Learning (arXiv:2501.13198). arXiv. https://doi.org/10.48550/arXiv.2501.13198

> https://arxiv.org/html/2501.13198v3
> https://github.com/WuYichen-97/SD-Lora-CL
>
> Extract the technical details necessary to produce a faithful implementation of
> the attached research. You should provide python style psudo-code, equation, and
> make reference to the equations in the text.

# Gemini 3.1 Pro Preview:

Based on the provided research paper and its official repository, **SD-LoRA** (Scalable Decoupled Low-Rank Adaptation) is an architecture designed for Class Incremental Learning (CL). It avoids catastrophic forgetting without requiring rehearsal of past task data by decoupling the *magnitude* and *direction* of the LoRA weight matrices.

Below is a faithful extraction of the technical details, equations, and Python pseudo-code necessary to reproduce the methodology.

### 1. Mathematical Formulation & Equations

Vanilla LoRA updates the pre-trained weight matrix $\mathbf{W}_0$ by a low-rank product $\mathbf{AB}$. The core idea of SD-LoRA is decomposing this update into magnitude (Frobenius norm) and direction (normalized matrix):

**Equation (3): Decoupling Magnitude and Direction**
$$ \Delta\mathbf{W} = \|\mathbf{A}\mathbf{B}\|_F \cdot \overline{\mathbf{A}\mathbf{B}} = \|\mathbf{A}\mathbf{B}\|_F \cdot \frac{\mathbf{A}\mathbf{B}}{\|\mathbf{A}\mathbf{B}\|_F} $$

For sequential learning over $t$ tasks, SD-LoRA assigns an independent scalar magnitude $\alpha_k$ and a direction $\overline{\mathbf{A}_k\mathbf{B}_k}$ to each task $k$.

**Equation (4): SD-LoRA Forward Pass**
At current task $t$, the output of the layer $\mathbf{h}'$ is computed as:
$$ \mathbf{h}' = \left(\mathbf{W}_0 + \alpha_1 \overline{\mathbf{A}_1\mathbf{B}_1} + \dots + \alpha_t \overline{\mathbf{A}_t\mathbf{B}_t}\right)\mathbf{x} $$

**Key Training Mechanics:**
* **Frozen Weights:** The base model $\mathbf{W}_0$ is frozen permanently.
* **Frozen Past Directions:** For all previous tasks ($k < t$), the normalized direction matrices $\overline{\mathbf{A}_k\mathbf{B}_k}$ are fixed. 
* **Learnable Parameters:** Only the new direction matrices $\mathbf{A}_t, \mathbf{B}_t$ and **all** past and present magnitudes $\{\alpha_1, \dots, \alpha_t\}$ are continuously optimized during task $t$.

### 2. Parameter-Efficient Variants
To avoid unbounded parameter expansion as the number of tasks grows, the paper introduces two variants:

**SD-LoRA-RR (Rank Reduction):**
The empirical analysis notes that later tasks contribute less to the global low-loss region. Thus, SD-LoRA-RR enforces stepwise rank reduction for $\mathbf{A}_t$ and $\mathbf{B}_t$:
**Equation (6):**
$$ r_1 = r_2 = \ldots > r_\mu = r_{\mu+1} = \ldots > r_\nu = r_{\nu+1} = \ldots = r_N $$
Where $\mu$ and $\nu$ are predefined task transition indices (e.g., at task $\mu=4$, rank decays from $r_1$ to $r_\mu$).

**SD-LoRA-KD (Knowledge Distillation):**
To completely halt parameter growth for later tasks, SD-LoRA-KD employs a least squares based distillation. It evaluates if a newly introduced direction $\overline{\mathbf{A}_t\mathbf{B}_t}$ can be approximated as a linear combination of frozen previous directions:
$$ \overline{\mathbf{A}_t\mathbf{B}_t} \approx \sum_{k=1}^{t-1} \beta_k \overline{\mathbf{A}_k\mathbf{B}_k} $$
Solving this via least squares produces coefficients $\boldsymbol{\beta}$, which are used to distill or absorb the current task's parameter variations into the magnitudes $\alpha$ of previous tasks, thereby discarding the need to store $\mathbf{A}_t, \mathbf{B}_t$. 

---

### 3. Faithful Python Pseudo-Code (PyTorch)
The following is a PyTorch-style implementation of an `SDLoRALinear` module reflecting the equations and parameter handling described above.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SDLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, ranks: list):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # W_0: Frozen Pre-trained Weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        
        # SD-LoRA Learnable Components
        self.alphas = nn.ParameterList()      # The magnitudes {\alpha_k}
        self.A_matrices = nn.ParameterList()  # The LoRA down-projections
        self.B_matrices = nn.ParameterList()  # The LoRA up-projections
        
        # Cache for frozen normalized directions \overline{A_k B_k} (not in state_dict parameters)
        self.frozen_directions = [] 
        
        self.ranks = ranks # ranks[t] handles Eq. (6) SD-LoRA-RR (Rank Decay)
        self.current_task = 0

    def add_task(self):
        """Prepare the layer before training on incremental task `t`."""
        t = self.current_task
        r_t = self.ranks[t] # e.g. [16, 16, 16, 8, 8, 4, 4] for SD-LoRA-RR
        
        # 1. Initialize new A_t, B_t for the current task
        A_t = nn.Parameter(torch.randn(self.out_features, r_t) * 0.01)
        B_t = nn.Parameter(torch.zeros(r_t, self.in_features))
        self.A_matrices.append(A_t)
        self.B_matrices.append(B_t)
        
        # 2. Initialize new task magnitude \alpha_t (initialized to 1.0)
        self.alphas.append(nn.Parameter(torch.tensor(1.0)))
        
        # 3. Freeze the previous task's direction
        if t > 0:
            self.A_matrices[t-1].requires_grad = False
            self.B_matrices[t-1].requires_grad = False
            
            # Precompute and cache the normalized direction (Eq. 3) for inference efficiency
            A_prev, B_prev = self.A_matrices[t-1], self.B_matrices[t-1]
            dir_prev = A_prev @ B_prev
            dir_prev_norm = dir_prev / (torch.norm(dir_prev, p='fro') + 1e-8)
            self.frozen_directions.append(dir_prev_norm.detach())

        self.current_task += 1

    def forward(self, x):
        """Implements SD-LoRA Forward Pass - Equation (4)"""
        # Base forward pass: W_0 * x
        h = F.linear(x, self.weight)
        
        # Add contributions from all tasks up to current_task
        for k in range(self.current_task):
            alpha_k = self.alphas[k]
            
            if k < self.current_task - 1:
                # Use cached frozen normalized direction
                direction_k = self.frozen_directions[k]
            else:
                # Dynamically compute normalized direction for current actively training task
                A_k, B_k = self.A_matrices[k], self.B_matrices[k]
                prod = A_k @ B_k
                direction_k = prod / (torch.norm(prod, p='fro') + 1e-8)
                
            # h' = W_0 * x + \sum (alpha_k * \overline{A_k B_k} * x)
            h = h + alpha_k * F.linear(x, direction_k)
            
        return h

    def get_sd_lora_kd_loss(self):
        """
        SD-LoRA-KD: Knowledge Distillation step solving Least Squares.
        Projects the newest active direction onto the subspace of fixed previous directions.
        """
        t = self.current_task - 1
        if t == 0:
            return torch.tensor(0.0, device=self.weight.device)
            
        # Get active normalized direction
        A_t, B_t = self.A_matrices[t], self.B_matrices[t]
        prod_t = A_t @ B_t
        dir_t = prod_t / (torch.norm(prod_t, p='fro') + 1e-8)
        
        # Flatten previous directions to form the dictionary D
        D = torch.stack([d.flatten() for d in self.frozen_directions], dim=1) # Shape: (out*in, t)
        y = dir_t.flatten() # Shape: (out*in)
        
        # Solve least squares: argmin_\beta || D \beta - y ||_2^2
        beta = torch.linalg.lstsq(D, y).solution
        
        # Reconstruction and penalty/distillation loss
        reconstruction = D @ beta
        kd_loss = F.mse_loss(reconstruction, y)
        
        return kd_loss
```