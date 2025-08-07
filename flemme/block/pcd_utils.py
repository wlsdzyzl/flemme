### adopted from pointnet2: https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py
import torch
import torch.nn as nn
import warnings
from torch.autograd import Function
from typing import *
import cpp_extension.pcd_ops as pcd_ops
try:
    from knn_cuda import KNN
except:
    KNN = None

def knn_with_topk(x, k):
    # x: B*N*D, inner: B*N*N
    inner = 2*torch.matmul(x, x.transpose(1, 2))
    # x^2: B*N*1
    xx = torch.sum(x**2, dim=-1, keepdim=True)
    # negative distance
    pairwise_distance = -(xx.transpose(1, 2) - inner + xx)
    # return the closest k indices, idx: B*N*K
    idx = pairwise_distance.topk(k=k, dim=-1)[1] 
    return idx

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        out = pcd_ops.furthest_point_sampling(xyz, npoint)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


furthest_point_sample = FurthestPointSampling.apply



# class GatherOperation(Function):
#     @staticmethod
#     def forward(ctx, features, idx):
#         # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
#         r"""

#         Parameters
#         ----------
#         features : torch.Tensor
#             (B, C, N) tensor

#         idx : torch.Tensor
#             (B, npoint) tensor of the features to gather

#         Returns
#         -------
#         torch.Tensor
#             (B, C, npoint) tensor
#         """

#         ctx.save_for_backward(idx, features)

#         return pcd_ops.gather_points(features, idx)

#     @staticmethod
#     def backward(ctx, grad_out):
#         idx, features = ctx.saved_tensors
#         N = features.size(2)

#         grad_features = pcd_ops.gather_points_grad(grad_out.contiguous(), idx, N)
#         return grad_features, None


# gather_features = GatherOperation.apply
### reimplemented with pytorch
def gather_features(features, index, channel_dim, gather_dim):
    index = index.unsqueeze(channel_dim)
    expend_shape = [-1,] * features.ndim
    expend_shape[channel_dim] = features.shape[channel_dim]
    index = index.expand(*expend_shape) 
    return torch.gather(features, dim = gather_dim, index = index.long())


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = pcd_ops.three_nn(unknown, known)
        dist = torch.sqrt(dist2)

        ctx.mark_non_differentiable(dist, idx)

        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        ctx.save_for_backward(idx, weight, features)

        return pcd_ops.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, features = ctx.saved_tensors
        m = features.size(2)

        grad_features = pcd_ops.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, torch.zeros_like(idx), torch.zeros_like(weight)


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        ctx.save_for_backward(idx, features)

        return pcd_ops.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = pcd_ops.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, torch.zeros_like(idx)


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        output, dist = pcd_ops.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(output)
        ctx.mark_non_differentiable(dist)
        return output, dist

    @staticmethod
    def backward(ctx, grad_out, grad_dist):
        return ()


ball_query = BallQuery.apply


class BatchBallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : (B, N)
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        output, dist = pcd_ops.batch_ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(output)
        ctx.mark_non_differentiable(dist)
        return output, dist

    @staticmethod
    def backward(ctx, grad_out, grad_dist):
        return ()

batch_ball_query = BatchBallQuery.apply

class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, nsample, radius = 0.1, use_xyz=True, sorted_query=False, knn_query = None):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample = radius, nsample,
        self.use_xyz = use_xyz
        self.sorted_query = sorted_query
        self.knn = None
        if knn_query and KNN is not None:
            self.knn = KNN(k = self.nsample, transpose_mode=True)
            self.knn_query = knn_query

    def forward(self, xyz, xyz_embed, new_xyz, new_xyz_embed, features=None, new_features = None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, pos_emb + C, npoint, nsample) tensor
        """
        if self.knn is not None:
            ## indices are all ready sorted.
            if self.knn_query == 'xyz':
                _, idx = self.knn(xyz, new_xyz)
            elif self.knn_query == 'feature' and \
                features is not None and \
                new_features is not None:
                _, idx = self.knn(features.transpose(1, 2), new_features.transpose(1, 2))
            else:
                _, idx = self.knn(xyz_embed, new_xyz_embed)
            ## index is long
            idx = idx.int()
        else:
            idx, dist = ball_query(self.radius, self.nsample, xyz, new_xyz)
            if self.sorted_query:
                _,  sorted_id = torch.sort(dist, dim = -1)
                idx = torch.gather(idx, dim=-1,index = sorted_id)
        xyz_embed_trans = xyz_embed.transpose(1, 2).contiguous()
        grouped_xyz_emb = grouping_operation(xyz_embed_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz_emb -= new_xyz_embed.transpose(1, 2).contiguous().unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz_emb, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Point position has to be used as a feature when feature is none!"
            new_features = grouped_xyz_emb

        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, xyz_embed, new_xyz, new_xyz_embed, features=None, new_features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz_embed.unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features


class TrilinearDevoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution, is_training=True):
        """
        :param ctx:
        :param coords: the coordinates of points, FloatTensor[B, 3, N]
        :param features: FloatTensor[B, C, R, R, R]
        :param resolution: int, the voxel resolution
        :param is_training: bool, training mode
        :return:
            FloatTensor[B, C, N]
        """
        B, C = features.shape[:2]
        features = features.contiguous().view(B, C, -1)
        coords = coords.contiguous()
        outs, inds, wgts = pcd_ops.trilinear_devoxelize_forward(resolution, is_training, coords, features)
        if is_training:
            ctx.save_for_backward(inds, wgts)
            ctx.r = resolution
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: 
        :param grad_output: gradient of outputs, FloatTensor[B, C, N]
        :return:
            gradient of inputs, FloatTensor[B, C, R, R, R]
        """
        inds, wgts = ctx.saved_tensors
        grad_inputs = pcd_ops.trilinear_devoxelize_backward(grad_output.contiguous(), inds, wgts, ctx.r)
        return grad_inputs.view(grad_output.size(0), grad_output.size(1), ctx.r, ctx.r, ctx.r), None, None, None


trilinear_devoxelize = TrilinearDevoxelization.apply


class AvgVoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution):
        """
        :param ctx:
        :param features: Features of the point cloud, FloatTensor[B, C, N]
        :param coords: Voxelized Coordinates of each point, IntTensor[B, 3, N]
        :param resolution: Voxel resolution
        :return:
            Voxelized Features, FloatTensor[B, C, R, R, R]
        """
        features = features.contiguous()
        coords = coords.int().contiguous()
        b, c, _ = features.shape
        out, indices, counts = pcd_ops.avg_voxelize_forward(features, coords, resolution)
        ctx.save_for_backward(indices, counts)
        return out.view(b, c, resolution, resolution, resolution)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of output, FloatTensor[B, C, R, R, R]
        :return:
            gradient of inputs, FloatTensor[B, C, N]
        """
        b, c = grad_output.shape[:2]
        indices, counts = ctx.saved_tensors
        grad_features = pcd_ops.avg_voxelize_backward(grad_output.contiguous().view(b, c, -1), indices, counts)
        return grad_features, None, None


avg_voxelize = AvgVoxelization.apply

# class LayerScale(nn.Module):
#     def __init__(self, in_channel, init_values=1e-5, inplace=False):
#         super().__init__()
#         self.inplace = inplace
#         self.gamma = nn.Parameter(init_values * torch.ones(in_channel))

#     def forward(self, x):
#         return x.mul_(self.gamma) if self.inplace else x * self.gamma

