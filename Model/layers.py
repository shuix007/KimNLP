import torch.nn as nn

class DenseLayer(nn.Module):
    '''Linear layer with normal initialization'''

    def __init__(self, in_dims, out_dims, activation=None, bias=False):
        super(DenseLayer, self).__init__()
        self.fc = nn.Linear(in_dims, out_dims, bias=bias)
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias.data)

    def forward(self, x):
        if self.activation is None:
            return self.fc(x)
        else:
            return self.activation(self.fc(x))