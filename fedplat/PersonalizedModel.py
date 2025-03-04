import torch.nn as nn

class PersonalizedModel(nn.Module):
    def __init__(self, shared_model, personal_layer_size):
        super(PersonalizedModel, self).__init__()
        self.shared_model = shared_model
        self.personal_layer = nn.Linear(personal_layer_size, personal_layer_size)
        self.output_layer = nn.Linear(personal_layer_size, 10)  # Assuming 10 classes, adjust as needed

    def forward(self, x):
        shared_output = self.shared_model(x)
        personal_output = self.personal_layer(shared_output)
        return self.output_layer(personal_output)