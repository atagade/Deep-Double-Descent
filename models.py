import torch.nn as nn

class FCNN(nn.Module):

  def __init__(self, input_size, num_hidden, num_classes):
    super(FCNN, self).__init__()
    self.fc1 = nn.Linear(input_size, num_hidden)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(num_hidden, num_classes)

  def forward(self, x):

    y = self.fc1(x)
    y = self.relu(y)
    y = self.fc2(y)

    return y

def get_model(model, num_hidden_units, dataset):

    if model == 'FCNN':
        
        if dataset == 'MNIST':
            input_size = 784
            num_classes = 10

        return FCNN(input_size, num_hidden_units, num_classes)
