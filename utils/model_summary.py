import torch
from torchsummary import summary


def print_model_summary(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    # model = torch.load('model\\history\\best_policy_value_net_4400.pth')
    summary(model, input_size=(6, 9, 9))
