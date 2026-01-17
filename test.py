import torch
import torch.nn as nn
import torch.optim as optim
import time

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Simple MLP
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNet().to(device)

# Fake data (large enough to stress GPU)
batch_size = 8192
x = torch.randn(batch_size, 1024, device=device)
y = torch.randn(batch_size, 1024, device=device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Warm-up
for _ in range(10):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

# Timed run
torch.cuda.synchronize()
start = time.time()

for step in range(10000):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))


torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Final loss: {loss.item():.6f}")
print(f"Time for 100 steps: {elapsed:.2f} seconds")
