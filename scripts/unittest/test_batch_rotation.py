import numpy as np
import torch
from flemme.utils import batch_transform, batch_normalize_vector, batch_rotations_from_vectors

a = torch.randn(10, 3)
a = batch_normalize_vector(a)
b = torch.randn(10, 3)
b = batch_normalize_vector(b)
M = batch_rotations_from_vectors(a, b)
print(batch_transform(a, rotation = M) - b)