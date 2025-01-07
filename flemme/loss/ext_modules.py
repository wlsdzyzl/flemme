# EMD approximation module (based on auction algorithm)
# memory complexity: O(n)
# time complexity: O(n^2 * iter) 
# author: Minghua Liu

# Input:
# xyz1, xyz2: [#batch, #points, 3]
# where xyz1 is the predicted point cloud and xyz2 is the ground truth point cloud 
# two point clouds should have same size and be normalized to [0, 1]
# #points should be a multiple of 1024
# #batch should be no greater than 512
# eps is a parameter which balances the error rate and the speed of convergence
# iters is the number of iteration
# we only calculate gradient for xyz1

# Output:
# dist: [#batch, #points],  sqrt(dist) -> L2 distance 
# assignment: [#batch, #points], index of the matched point in the ground truth point cloud
# the result is an approximation and the assignment is not guranteed to be a bijection
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import os
import cpp_extension.emd as emd
import cpp_extension.cd as cd

class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2
    ## gradidx1, gradidx2 should be zero
    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2


class ChamferDistance(torch.nn.Module):
    def forward(self, xyz1, xyz2, return_idx = False):
        dist1, dist2, idx1, idx2 = ChamferDistanceFunction.apply(xyz1, xyz2)
        if return_idx:
            return dist1, dist2, idx1, idx2
        return dist1, dist2



class emdFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters):

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        assert(n == m), f"point clouds should have the same dimensions, get {n} and {m}"
        assert(xyz1.size()[0] == xyz2.size()[0]), "{} and {} should be the same.".format(xyz1.size()[0], xyz2.size()[0])
        assert(n % 1024 == 0), "number of points should be divided by 1024"
        assert(batchsize <= 512)

        xyz1 = xyz1.contiguous().float().cuda()
        xyz2 = xyz2.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batchsize, m, device='cuda', dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batchsize, m, device='cuda').contiguous()
        bid = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device='cuda').contiguous()
        max_increments = torch.zeros(batchsize, m, device='cuda').contiguous()
        unass_idx = torch.zeros(batchsize * n, device='cuda', dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batchsize * m, device='cuda', dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()

        emd.forward(xyz1, xyz2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx, unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters)

        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx):
        xyz1, xyz2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()

        gradxyz1 = torch.zeros(xyz1.size(), device='cuda').contiguous()
        gradxyz2 = torch.zeros(xyz2.size(), device='cuda').contiguous()

        emd.backward(xyz1, xyz2, gradxyz1, graddist, assignment)
        return gradxyz1, gradxyz2, None, None

class emdModule(nn.Module):
    def __init__(self, eps, iters):
        super(emdModule, self).__init__()
        self.eps = eps
        self.iters = iters
    def forward(self, input1, input2):
        return emdFunction.apply(input1, input2, self.eps, self.iters)


def test_cd():
    import time
    x1 = torch.rand(20, 8192, 3).cuda() # please normalize your point cloud to [0, 1]
    x2 = torch.rand(20, 8192, 3).cuda()
    cd = ChamferDistance()
    start_time = time.perf_counter()
    d1, d2 = cd(x1, x2) # 0.005, 50 for training 
    print("Input_size: ", x1.shape)
    print('distance size:', d1.shape, d2.shape)


def test_emd():
    import time
    x1 = torch.rand(20, 8192, 3).cuda() # please normalize your point cloud to [0, 1]
    x2 = torch.rand(20, 8192, 3).cuda()
    emd = emdModule(0.002, 10000)
    start_time = time.perf_counter()
    dis, assigment = emd(x1, x2) # 0.005, 50 for training 
    print("Input_size: ", x1.shape)
    print('distance size:', dis.shape)
    print("Runtime: %lfs" % (time.perf_counter() - start_time))
    print("EMD: %lf" % np.sqrt(dis.cpu()).mean())
    print("|set(assignment)|: %d" % assigment.unique().numel())
    assigment = assigment.cpu().numpy()
    assigment = np.expand_dims(assigment, -1)
    x2 = np.take_along_axis(x2, assigment, axis = 1)
    d = (x1 - x2) * (x1 - x2)
    print("Verified EMD: %lf" % np.sqrt(d.cpu().sum(-1)).mean())

# test_emd()
        