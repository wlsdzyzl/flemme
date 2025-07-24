import torch
from flemme.utils import batch_transform, batch_normalize_vector, batch_rotations_from_vectors
from flemme.logger import get_logger
logger = get_logger('unittest::test_batch_rotation')
a = torch.randn(10, 3)
a = batch_normalize_vector(a)
b = torch.randn(10, 3)
b = batch_normalize_vector(b)
M = batch_rotations_from_vectors(a, b)
logger.info("Sum of error after rotation: {}".format((batch_transform(a, rotation = M) - b).sum()))