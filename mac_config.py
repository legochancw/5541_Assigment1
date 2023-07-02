# Specific Setting for Mac M1 chip to replace .cuda() function
import torch

device = torch.device("mps")
