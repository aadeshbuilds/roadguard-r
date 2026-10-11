import sys
import torch
import ultralytics

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("ultralytics:", ultralytics.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

# Try a real op on the GPU, not just the flag
if torch.backends.mps.is_available():
    x = torch.randn(1000, 1000, device="mps")
    y = x @ x
    print("MPS matmul OK:", tuple(y.shape))
