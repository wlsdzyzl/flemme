from .common import *
from .pcd_utils import *
from functools import partial
from flemme.config import module_config
if module_config['mamba']:
    from .pmamba import get_psmamba_block, get_scanners,\
            PointMambaBlock, PointMambaNonFFNBlock



if module_config['transformer']:
    ## insight from PCT: https://arxiv.org/pdf/2012.09688
    class OffSetAttentionBlock(nn.Module):
        """
        ### Attention block

        This is similar to [transformer multi-head attention](../../transformers/mha.html).
        """

        def __init__(self, in_channel, num_heads = 3, d_k = None, 
            qkv_bias = True, qk_scale = None, atten_dropout = None, 
            dropout = None, skip_connection = False):
            """
            * `in_channel` is the number of channel in the input
            * `num_heads` is the number of heads in multi-head attention
            * `d_k` is the number of dimensions in each head
            * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
            """
            super().__init__()
            self.d_k = d_k or int(in_channel / num_heads)
            self.scale = qk_scale or 1
            if atten_dropout is None or atten_dropout <= 0:
                self.atten_dropout = nn.Identity()
            else:
                self.atten_dropout = nn.Dropout(p=atten_dropout)

            # Project x to query, key and values
            self.qkv = nn.Linear(in_channel, num_heads * self.d_k * 3, bias = qkv_bias)
            # project to original space
            self.proj = nn.Linear(num_heads * self.d_k, in_channel)
            # offset transformation
            self.offset_trans = nn.Linear(in_channel, in_channel)
            self.num_heads = num_heads
            self.skip_connection = skip_connection
            if dropout is None or dropout <= 0:
                self.dropout = nn.Identity()
            else:
                self.dropout = nn.Dropout(p=dropout)
        
        def attention(self, q, k, v):
            # [batch_size, num_heads, seq, d_k]
            # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
            atten = torch.einsum('bhik,bhjk->bhij', q, k) * self.scale
            atten = atten.softmax(dim=-1)
            atten = atten / (1e-8 + atten.sum(dim=1, keepdims = True))
            atten = self.atten_dropout(atten)
            res = torch.einsum('bhik,bhkj->bhij', atten, v)
            return res
        
        def forward(self, x: torch.Tensor):
            # Get shape
            
            batch_size, in_channel = x.shape[0], x.shape[-1]
            # Change `x` to shape `[batch_size, seq, in_channel]`
            x = x.reshape(batch_size, -1, in_channel)
            # Get query, key, and values (concatenated) and shape it to `[3, batch_size, num_heads, seq, d_k]`
            qkv = self.qkv(x).reshape(batch_size, -1, 3, self.num_heads, self.d_k).permute(2, 0, 3, 1, 4)
            # Split query, key, and values.
            q, k, v = qkv[0], qkv[1], qkv[2]
            # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
            res = self.attention(q, k, v)

            # Reshape to `[batch_size, seq, num_heads * d_k]`
            res = res.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_k)
            # Transform to `[batch_size, seq, in_channel]`
            res = self.proj(res)
            res = self.offset_trans(x - res)
            res = self.dropout(res)
            # Add skip connection
            if self.skip_connection:
                res = res + x
            return res

    class PointTransformerBlock(nn.Module):
        def __init__(self, in_channel, out_channel = None, 
            time_channel = 0, num_heads = 3, d_k = None, 
            qkv_bias = True, qk_scale = None, atten_dropout = None, 
            mlp_hidden_ratios=[4.0,], 
            dropout = None, attention = 'SA', 
            activation = 'relu', norm='batch', num_norm_groups = -1, 
            time_injection = 'gate_bias', 
            condition_channel = 0, 
            condition_injection = 'gate_bias',
            condition_first = False,
            **kwargs):
            
            super().__init__()
            self.in_channel = in_channel
            self.out_channel = out_channel or in_channel
            if attention == 'SA':
                self.atten = SelfAttentionBlock(in_channel=in_channel, num_heads=num_heads, d_k = d_k, 
                qkv_bias = qkv_bias, qk_scale = qk_scale, atten_dropout = atten_dropout, 
                dropout = dropout, skip_connection = False, channel_dim = -1)
            elif attention == 'OA':
                self.atten = OffSetAttentionBlock(in_channel=in_channel, num_heads=num_heads, d_k = d_k, 
                qkv_bias = qkv_bias, qk_scale = qk_scale, atten_dropout = atten_dropout, 
                dropout = dropout, skip_connection = False)

            self.norm1 = NormBlock(*(get_norm(norm, in_channel, 1, num_norm_groups) + (-1,)))
            self.norm2 = NormBlock(*(get_norm(norm, in_channel, 1, num_norm_groups) + (-1,)))

            # self.act = get_act(activation)
            mlp_hidden_channels = [int(in_channel * r) for r in mlp_hidden_ratios]
            self.mlp = MultiLayerPerceptionBlock(in_channel=in_channel, 
                            out_channel=self.out_channel, 
                            hidden_channels=mlp_hidden_channels,
                            activation=activation, dropout=dropout)
            self.dense = nn.Linear(self.in_channel, self.out_channel) if self.in_channel != self.out_channel else nn.Identity()

            self.cinj = None
            if time_channel > 0 or condition_channel > 0:
                self.cinj = ContextInjectionBlock(time_channel = time_channel,
                    condition_channel = condition_channel,
                    out_channel = in_channel,
                    time_injection=time_injection,
                    condition_injection=condition_injection,
                    channel_dim = -1,
                    condition_first = condition_first)
        def forward(self, x, t = None, c = None):
            x = self.atten(self.norm1(x)) + x
            if self.cinj:
                x = self.cinj(x, t, c)
            x = self.dense(x) + self.mlp(self.norm2(x))
            return x
        @staticmethod
        def is_sequence_modeling():
            return True
    ## point transformer without feed-forward network (nlp)
    class PointTransformerNonFFNBlock(NormBlock):
        def __init__(self, in_channel, out_channel = None, 
            time_channel = 0, num_heads = 3, d_k = None, 
            qkv_bias = True, qk_scale = None, atten_dropout = None, 
            dropout = None, residual_attention = False, 
            skip_connection = True, attention = 'SA', 
            activation = 'relu', norm='batch', num_norm_groups = -1,
            time_injection = 'gate_bias',  
            condition_channel = 0, 
            condition_injection = 'gate_bias',
            condition_first = False,
            **kwargs):
            
            super().__init__(_channel_dim = -1)
            self.in_channel = in_channel
            self.out_channel = out_channel or in_channel
            if attention == 'SA':
                self.atten = SelfAttentionBlock(in_channel=in_channel, num_heads=num_heads, d_k = d_k, 
                qkv_bias = qkv_bias, qk_scale = qk_scale, atten_dropout = atten_dropout, 
                dropout = dropout, skip_connection = residual_attention, channel_dim = -1)
            elif attention == 'OA':
                self.atten = OffSetAttentionBlock(in_channel=in_channel, num_heads=num_heads, d_k = d_k, 
                qkv_bias = qkv_bias, qk_scale = qk_scale, atten_dropout = atten_dropout, 
                dropout = dropout, skip_connection = residual_attention)
            self.norm, self.norm_type = get_norm(norm, in_channel, 1, num_norm_groups)
            self.act = get_act(activation)
            self.time_channel = time_channel
            self.skip_connection = skip_connection
            self.dense = nn.Linear(self.in_channel, self.out_channel) if self.in_channel != self.out_channel else nn.Identity()

            self.cinj = None
            if time_channel > 0 or condition_channel > 0:
                self.cinj = ContextInjectionBlock(time_channel = time_channel,
                    condition_channel = condition_channel,
                    out_channel = out_channel,
                    time_injection=time_injection,
                    condition_injection=condition_injection,
                    channel_dim = -1,
                    condition_first = condition_first)
                
        def forward(self, x, t = None, c = None):
            res = self.atten(x)
            res = x + self.act(self.normalize(res))
            x = self.dense(res)
            if self.cinj:
                x = self.cinj(x, t, c)
            return x
        @staticmethod
        def is_sequence_modeling():
            return True

        

## part of this code is adopted from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
def get_graph_feature(x, k=20, knn = None, idx=None):
    # B * N * D
    batch_size, num_points, num_dims = x.shape
    if idx is None:
        if knn is not None:
            _, idx = knn(x, x)
        else:
            idx = knn_with_topk(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    # feature: B*N*K*D 
    feature = feature.view(batch_size, num_points, k, num_dims) 
    # x: B*N*K*D
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # feature: B*N*K*2D
    feature = torch.cat((feature-x, x), dim=3)
    return feature

class GroupSeqModelingLayer(nn.Module):
    def __init__(self, in_channel, out_channel,
        BuildingBlock):
        super().__init__()
        self.bb = BuildingBlock(in_channel = in_channel,
                                out_channel = out_channel) 
    ## input: grouped points (B, N, K, C)
    def forward(self, x, t, c):
        B, N, K, _ = x.shape
        # B * N * K * 2D -> (B * N) * K * 2D
        x = x.reshape(B * N, K, -1)
        # B * D -> B * N * D
        if t is not None:
            t = t.unsqueeze(1).expand(-1, N, -1)
            t = t.reshape(B * N, -1)
        x = self.bb(x, t = t, c = c)
        # (B * N) * K * D -> B * N * K * D
        x = x.reshape(B, N, K, -1)
        return x

class LocalGraphLayer(nn.Module):
    def __init__(self, k, 
                in_channel, 
                out_channel, 
                BuildingBlock, 
                num_blocks = 2,
                hidden_channels = None,
                **kwargs):
        super().__init__()
        self.k = k
        self.knn = None
        if KNN is not None:
            self.knn = KNN(k=self.k, transpose_mode=True)
            
        is_seq = BuildingBlock.func.is_sequence_modeling()
        if not is_seq:
            self.bb = MultipleBuildingBlocks(n = num_blocks, 
                                    hidden_channels = hidden_channels,
                                    in_channel = 2*in_channel,
                                    out_channel = out_channel,
                                    BuildingBlock = BuildingBlock)
        else:
            self.bb = MultipleBuildingBlocks(in_channel = 2*in_channel,
                                    out_channel = out_channel,
                                    n = num_blocks, 
                                    hidden_channels = hidden_channels,
                                    BuildingBlock = partial(
                                        GroupSeqModelingLayer,
                                        BuildingBlock = BuildingBlock))
    def forward(self, x, t=None, c = None):
        ## x: B * N * D -> B * N * K * 2D
        x = get_graph_feature(x, k = self.k, knn = self.knn)
        x = self.bb(x, t, c)
        ## x: B * N * D
        x, _ = x.max(dim=2)
        return x
        

class FoldingLayer(MultiLayerPerceptionBlock):
    """
    The folding operation of FoldingNet
    """

    def __init__(self, in_channel, 
                out_channel, 
                n = 1,
                hidden_channels = [],
                time_channel = 0, 
                norm = None, 
                num_norm_groups = 0, 
                activation = 'relu', 
                dropout=None, 
                time_injection = 'gate_bias',
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
                
        super().__init__(in_channel=in_channel,
                            out_channel=out_channel,
                            n = n,
                            hidden_channels=hidden_channels,
                            time_channel=time_channel,
                            time_injection=time_injection,
                            norm=norm,
                            num_norm_groups=num_norm_groups,
                            activation=activation,
                            dropout=dropout,
                            condition_channel = condition_channel,
                            condition_injection = condition_injection,
                            condition_first = condition_first,
                            final_activation=False)
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        self.in_channel = in_channel
    ## codewords
    def forward(self, shapes, codewords, t = None, c = None):
        """
        Parameters
        ----------
            codewords = B * N * D
            shapes: shapes or intermediate reconstructed point clouds
        """
        assert shapes.shape[-1] + codewords.shape[-1] == self.in_channel,\
            f"channel of grid + channel of codewords should be equal to the channel of input, we get: {shapes.shape[-1]} {codewords.shape[-1]} and {self.in_channel}."
        # concatenate
        x = torch.cat([shapes, codewords], dim=-1)
        return self.mlp(x, t, c)
        
### sampling and multi-scale grouping
### furthest point sampling
### group features based on some centers with radius.
### point cloud "down-sampling"
class SamplingAndGroupingLayer(nn.Module):
    # in_channel: in_channel of input features
    # out_channel: out_channel of output featuress
    def __init__(self, in_channel, out_channels, 
            num_fps_points, k, 
            BuildingBlock, radius = 0.1, num_blocks = 2, 
            hidden_channels = None, use_xyz = True, 
            sorted_query = False,
            knn_query = False,
            pos_embedding_channel = 3):
        super().__init__()
        self.in_channel = in_channel
        self.out_channels = out_channels or self.in_channel
        self.num_fps_points = num_fps_points
        self.k = k
        self.radius = radius

        if not isinstance(self.out_channels, list):
            self.out_channels = [self.out_channels, ]
        if not isinstance(self.k, list):
            self.k = [self.k, ] * len(self.out_channels)
        if not isinstance(self.radius, list):
            self.radius = [self.radius, ]* len(self.out_channels)
        if not isinstance(num_blocks, list):
            num_blocks = [num_blocks, ]* len(self.out_channels)
        if not (isinstance(hidden_channels, list) and \
            len(hidden_channels) > 0 and  \
            isinstance(hidden_channels[0], list)):
            hidden_channels = [hidden_channels, ] * len(self.out_channels)

        assert len(self.out_channels) == len(self.k) and \
            len(self.out_channels) == len(self.radius) and \
            len(self.out_channels) == len(num_blocks) and \
            len(self.out_channels) == len(hidden_channels), 'The numbers of scales inferred from different parameters are not identical.'
        
        if use_xyz:
                self.in_channel += pos_embedding_channel
        ### real out_channel
        self.out_channel = sum(self.out_channels)
        is_seq = BuildingBlock.func.is_sequence_modeling()        
        
        self.groupers = []

        self.bb = nn.ModuleList()
        for sid in range(len(self.out_channels)):
            self.groupers.append(QueryAndGroup(self.k[sid], 
                        radius = self.radius[sid], 
                        use_xyz=use_xyz, 
                        sorted_query = sorted_query,
                        knn_query = knn_query) if self.num_fps_points > 0 
                        else GroupAll(use_xyz))
            if not is_seq:
                self.bb.append(MultipleBuildingBlocks(in_channel = self.in_channel, 
                        out_channel = self.out_channels[sid], 
                        hidden_channels = hidden_channels[sid],
                        n = num_blocks[sid],
                        BuildingBlock = BuildingBlock))
            else:
                self.bb.append(MultipleBuildingBlocks(in_channel = self.in_channel,
                        out_channel = self.out_channels[sid],
                        n = num_blocks[sid],
                        hidden_channels = hidden_channels[sid],
                        BuildingBlock = partial(
                            GroupSeqModelingLayer,
                            BuildingBlock = BuildingBlock),
                        ))
    def forward(self, xyz, xyz_embed, features = None, t = None, c = None):
        r"""
        Parameters
        ----------
        xyz : (B, N, 3) tensor of the xyz coordinates of the features
        features: (B, N, C_in) tensor of the descriptors of the the features

        Returns
        -------
        centers: # (B, M, 3)
        center_features : torch.Tensor
            (B, M, C_out) tensor of the center_features descriptors
        """
        centers = None
        center_embed = None
        features_trans = channel_recover(features) if features is not None else None
        center_features_trans = None
        sample_ids = None
        if self.num_fps_points > 0:
            sample_ids = furthest_point_sample(xyz, self.num_fps_points)
            centers = gather_features(xyz, index = sample_ids, 
                channel_dim = -1, gather_dim = 1)
            center_embed = gather_features(xyz_embed, index = sample_ids, 
                channel_dim = -1, gather_dim = 1)
            
            if features_trans is not None:
                center_features_trans = gather_features(features_trans, 
                    index = sample_ids, channel_dim = 1,
                    gather_dim = -1)
        
        center_feature_list = []
        for gid in range(len(self.groupers)):
            ## (B, N, 3), (B, M, 3), (B, N, C_in) -> (B, C_in (+3), M, k) 
            grouped_features_trans = self.groupers[gid](
                xyz, xyz_embed, centers, center_embed, features_trans, center_features_trans
            ) 
            ## (B, C (+3), M, k) -> (B, M, k, C (+3))
            grouped_features = channel_transfer(grouped_features_trans)
            ## (B, M, k, C (+3)) -> (B, M, k, C_out)
            center_features = self.bb[gid](grouped_features, t, c)  
            # (B, M, out_channel)
            ## optional improvements: add average pooling here
            center_features = center_features.max(dim=2)[0]
            center_feature_list.append(center_features)
        
        return centers, center_embed, torch.cat(center_feature_list, dim = -1), sample_ids


### Propagates the features of one set to another
### point cloud "up-sampling"
class FeaturePropogatingLayer(nn.Module):
    r"""Propigates the features of one set to another

    """
    def __init__(self, in_channel_known, in_channel_unknown, 
        out_channel, BuildingBlock, num_blocks = 2, hidden_channels = None):
        super().__init__()
        self.bb = MultipleBuildingBlocks(in_channel = in_channel_known + in_channel_unknown, 
                out_channel = out_channel, 
                n = num_blocks,
                hidden_channels = hidden_channels,
                BuildingBlock = BuildingBlock)

    def forward(self, unknown, known, unknown_feats, known_feats, t = None, c = None):
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknown_feats : torch.Tensor
            (B, n, c1) tensor of the features to be propagated to
        known_feats : torch.Tensor
            (B, m, c2) tensor of features to be propagated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """
        
        
        if known is not None:
            known_feats_trans = channel_recover(known_feats)
            dist, idx = three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = channel_transfer(three_interpolate(
                known_feats_trans, idx, weight
            ))
        else:
            ## known_feats is global feature
            interpolated_feats = known_feats.unsqueeze(1).expand(
                # B, n, C2
                *(known_feats.shape[0], unknown.shape[1], known_feats.shape[-1])
            )   

        if unknown_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknown_feats], dim=-1
            )  # (B, n, C1 + C2)
        else:
            new_features = interpolated_feats
        # print(new_features.shape, unknown_feats.shape, known_feats.shape)
        new_features = self.bb(new_features, t, c)

        return new_features


class SampledFeatureCatLayer(nn.Module):
    def __init__(self, in_channels,
                out_channel, 
                num_blocks = 2,
                hidden_channels = None,
                time_channel = 0, 
                time_injection = 'gate_bias',
                norm = None, 
                num_norm_groups = 0, 
                activation = 'relu', 
                dropout=None, 
                condition_channel = 0, 
                condition_injection = 'gate_bias',
                condition_first = False,):
        super().__init__()
        self.in_channels = in_channels
        self.mlp = MultiLayerPerceptionBlock(in_channel = sum(in_channels), 
                            out_channel=out_channel,
                            n = num_blocks,
                            hidden_channels=hidden_channels,
                            time_channel=time_channel,
                            time_injection=time_injection,
                            norm=norm,
                            num_norm_groups=num_norm_groups,
                            activation=activation,
                            dropout=dropout,
                            condition_channel = condition_channel,
                            condition_injection = condition_injection,
                            condition_first = condition_first,
                            final_activation=False)
    def forward(self, feature_list, sample_id_list, t = None, c = None):
        assert len(feature_list) - len(sample_id_list) == 1, \
            'Unmatched lengths of feature_list and sample_id_list.'
        assert len(feature_list) == len(self.in_channels) and sum( [ ic == tensor.shape[-1] for ic, tensor in zip(self.in_channels, feature_list)]),\
            f'input features are not consistent with in_channels: {self.in_channels}'
        feature_list = feature_list[::-1]
        sample_id_list = sample_id_list[::-1]
        gathered_feature_list = [feature_list[0]]

        sample_ids = None
        for i in range(len(sample_id_list)):
            if i > 0:
                sample_ids = torch.gather(sample_id_list[i], dim = -1, index=sample_ids.long())
            else:
                sample_ids = sample_id_list[0]
            features = gather_features(
                    feature_list[i+1], index = sample_ids, 
                    channel_dim = -1, gather_dim = 1)
            gathered_feature_list.append(features)
        gathered_feature = torch.concat(gathered_feature_list[::-1], dim = -1)
        return self.mlp(gathered_feature, t = t, c = c)
        

### point-voxel
class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=1e-8):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        return avg_voxelize(features, vox_coords, self.r), norm_coords

class SE3d(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return inputs * self.fc(inputs.mean(-1).mean(-1).mean(-1)).view(inputs.shape[0], inputs.shape[1], 1, 1, 1)

class VoxelLayer(nn.Module):
    def __init__(self,  
                in_channel, 
                out_channel, 
                BuildingBlock,
                num_blocks = 2,
                hidden_channels = None, 
                resolution = 16, 
                with_se=False, 
                coordinate_normalize=True, 
                eps=1e-8):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.resolution = resolution
        if resolution > 0:
            self.voxelization = Voxelization(resolution, normalize=coordinate_normalize, eps=eps)
            self.vbb = MultipleBuildingBlocks(BuildingBlock=BuildingBlock,
                                                in_channel=in_channel, 
                                                out_channel=out_channel, 
                                                n=num_blocks,
                                                hidden_channels=hidden_channels)
            if with_se:
                self.se3d = SE3d(out_channel)
            else:
                self.se3d = nn.Identity()

    def forward(self, features, coords, t = None, c = None):
        if self.resolution > 0:
            feature_trans, coord_trans = channel_recover(features), channel_recover(coords)
            voxel_features, voxel_coords = self.voxelization(feature_trans, coord_trans)
            voxel_features = self.se3d(self.vbb(voxel_features, t, c))
            voxel_features = trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
            return channel_transfer(voxel_features)
        return 0.0