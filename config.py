import torch
import os
# Set device
DEVICE = (
    torch.device("mps") 
    if torch.backends.mps.is_available() 
    else torch.device("cuda:0") 
    if torch.cuda.is_available() 
    else torch.device("cpu")
)
# DEVICE = torch.device("cpu")  # Force CPU for now

# Other configurations
ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts") 

# number of samples to use for early termination
early_terminate_it = 32