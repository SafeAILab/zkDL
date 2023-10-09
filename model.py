import torch
import torch.nn as nn

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a Sequential model without bias
model = nn.Sequential(
    nn.Linear(3, 5, bias=False),
    nn.ReLU(),
    nn.Linear(5, 3, bias=False),
    nn.ReLU(),
    nn.Linear(3, 2, bias=False)
).to(device)

# For demo purposes, using dummy weights
torch.save(model.state_dict(), "model.pth")

model.load_state_dict(torch.load("model.pth"))
model.eval()

# Using tracing
sample_input = torch.randn(1, 3).to(device)
traced_model = torch.jit.trace(model, sample_input)
print(sample_input)

# Save traced model
traced_model.save("traced_model.pt")
