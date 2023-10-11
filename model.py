import torch
import torch.nn as nn

def save_tensor(t, fn):
    m = nn.Module()
    par = nn.Parameter(t)
    m.register_parameter("0",par)
    torch.jit.script(m).save(fn)

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a Sequential model without bias
model = nn.Sequential(
    nn.Linear(784, 1000, bias=False),
    nn.ReLU(),
    nn.Linear(1000, 1773, bias=False),
    nn.ReLU(),
    nn.Linear(1773, 1773, bias=False),
    nn.ReLU(),
    nn.Linear(1773, 1773, bias=False),
    nn.ReLU(),
    nn.Linear(1773, 1773, bias=False),
    nn.ReLU(),
    nn.Linear(1773, 1773, bias=False),
    nn.ReLU(),
    nn.Linear(1773, 1124, bias=False),
    nn.ReLU(),
    nn.Linear(1124, 1000, bias=False)
).to(device)

# For demo purposes, using dummy weights
torch.save(model.state_dict(), "model.pth")

model.load_state_dict(torch.load("model.pth"))
model.eval()

# Using tracing
sample_input = torch.randn(256, 784).to(device)
sample_output = model(sample_input)
# Save sample_input
save_tensor(sample_input, "sample_input.pt")
save_tensor(sample_output, "sample_output.pt")

traced_model = torch.jit.trace(model, sample_input[:1])

print(sample_input)
print(sample_output)

# Save traced model
traced_model.save("traced_model.pt")

# _ = torch.load("sample_input.pt")
# _ = torch.load("sample_output.pt")


# x = torch.randn(3, 3)
# traced_script_module = torch.jit.trace_module(x, ())
# traced_script_module.save("model.pt")