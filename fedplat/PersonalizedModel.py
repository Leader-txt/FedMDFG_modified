import torch.nn as nn
 
class PersonalizedModel(nn.Module):
    def __init__(self, shared_model, personal_layer_size):
        super(PersonalizedModel, self).__init__()
        self.shared_model = shared_model
        self.personal_layer =nn.Sequential(
            nn.Linear(personal_layer_size, personal_layer_size)
            ,nn.ReLU()
            ,nn.Linear(personal_layer_size,10))

    def forward(self, x):
        shared_output = nn.functional.relu(self.shared_model(x))
        personal_output = self.personal_layer(shared_output)
        return personal_output