import torch.nn as nn
import torch.nn.functional as F

from diffuser.utils.debug import debug

class LevelClassifier(nn.Module):
    def __init__(self, observation_dim, action_dim, num_classes, num_layers=3, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.observation_dim = observation_dim #- 2 ###### no more velocity!
        self.action_dim = action_dim
        layers = []
        input_dim = 2 * self.observation_dim
        for i in range(num_layers):
            output_dim = self.num_classes if i == num_layers-1 else hidden_dim
            layers.extend([nn.Linear(input_dim, output_dim)])
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
            input_dim = hidden_dim
        self.classifier = nn.Sequential(*layers)
        print(self.classifier)
        
    def forward(self, x):
        # x: [batch_size, 2, (action_dim+observation_dim)]
        B = x.shape[0]
        # x = x[:, :, self.action_dim:-2] ###### no more velocity!
        x = x.reshape(B, -1)
        return self.classifier(x)
    
    def loss(self, x, y):
        loss = F.cross_entropy(self(x), y)
        info = {"loss": loss}
        return loss, info