from model import MSRSEA
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch

input_tensor = torch.randn(1, 3, 256, 256) 
model = MSRSEA()

print(model)
with open('model.txt', 'w') as f:
    f.write(repr(model))
    
flops = FlopCountAnalysis(model, input_tensor)
print(flop_count_table(flops))

total_flops = flops.total()
print("Total FLOPs:", total_flops)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total params:", total_params)
print("Trainable params:", trainable_params)

with open('flops.txt', 'w') as f:
    f.write(f"Total FLOPs: {total_flops}\n")
    f.write(f"Total No. of Parameters: {total_params}\n")
    f.write(f"Total No. of Trainable Parameters: {trainable_params}\n")
    f.write(flop_count_table(flops))
