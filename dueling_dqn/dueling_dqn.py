import math
import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    # state_dim: size of observation vector
    # dueling_type: 'average', 'max', or 'naive'

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        dueling_type: str = "average",
    ):
        super().__init__()
        assert dueling_type in {"average", "max", "naive"}

        layers = []
        last = state_dim
        for h in hidden_dims:
            lin = nn.Linear(last, h)
            nn.init.orthogonal_(lin.weight, gain=math.sqrt(2))
            nn.init.constant_(lin.bias, 0.0)
            layers += [lin, nn.ReLU(inplace=True)]
            last = h
        self.feature = nn.Sequential(*layers)

        self.value = nn.Sequential(
            nn.Linear(last, last),
            nn.ReLU(inplace=True),
            nn.Linear(last, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(last, last),
            nn.ReLU(inplace=True),
            nn.Linear(last, num_actions),
        )
        # Small init on final adv head (helps stability)
        nn.init.orthogonal_(self.value[-1].weight, 1.0)
        nn.init.constant_(self.value[-1].bias, 0.0)
        nn.init.orthogonal_(self.adv[-1].weight, 0.01)
        nn.init.constant_(self.adv[-1].bias, 0.0)

        self.dueling_type = dueling_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature(x)
        v = self.value(feats)          # (B, 1)
        a = self.adv(feats)            # (B, A)

        if self.dueling_type == "naive":
            q = v + a
        else:
            if self.dueling_type == "average":
                a_centred = a - a.mean(dim=1, keepdim=True)
            else:  # 'max'
                a_centred = a - a.max(dim=1, keepdim=True).values
            q = v + a_centred
        return q
