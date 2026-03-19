import sys
import torch

sys.path.append('/Users/michico/Documents/大和先輩修論/Yamacode')

import vxm_torch.networks as networks
import vxm_torch.layers as layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

nb_features = [
    [32, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 32, 16, 16]
]

model = networks.VxmDense_128_256_256((128, 256, 256), nb_features, int_steps=0)
model.to(device)
model.eval()

print("model ready")