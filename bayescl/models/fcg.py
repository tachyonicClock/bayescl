from bnn.nn.modules import FCGLinear, FFGLinear
from torch import Tensor, nn


class SimpleFCGMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
    ):
        super().__init__()
        self.fc1 = FFGLinear(in_features, hidden_features)
        self.fc2 = FCGLinear(hidden_features, out_features)
        self.head = nn.Linear(hidden_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.head(x)
        return x
