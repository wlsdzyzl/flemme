from .common import *
from .pcd_utils import *
from functools import partial
from flemme.config import module_config

## insight from PCT: https://arxiv.org/pdf/2012.09688
if module_config['transformer']:
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
            atten = torch.einsum('bhik,bhjk->bhij', q, k)
            # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
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

    class PointTransformerBlock(NormBlock):
        def __init__(self, in_channel, out_channel = None, 
            time_channel = 0, num_heads = 3, d_k = None, 
            qkv_bias = True, qk_scale = None, atten_dropout = None, 
            dropout = None, residual_attention = False, 
            skip_connection = True, attention = 'SA', 
            activation = 'relu', norm='batch', num_norm_groups = -1, **kwargs):
            
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

            if self.time_channel > 0:
                self.hyper_bias = nn.Linear(self.time_channel, out_channel, bias=False)
                self.hyper_gate = nn.Linear(self.time_channel, out_channel)
        def forward(self, x, t = None):
            res = self.atten(x)
            res = self.act(self.normalize(res))
            if self.skip_connection:
                res = x + res
            x = self.dense(res)
            if t is not None:
                assert self.time_channel == t.shape[-1], \
                    f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
                gate = expand_as(torch.sigmoid(self.hyper_gate(t)), x, channel_dim=-1)
                bias = expand_as(self.hyper_bias(t), x, channel_dim = -1)
                x = x * gate + bias
            return x
        @staticmethod
        def is_sequence_modeling():
            return True
if module_config['mamba']:
    from mamba_ssm import Mamba, Mamba2
    class PointMambaBlock(NormBlock):
        def __init__(self, in_channel, out_channel = None, 
            time_channel = 0, state_channel = 64, 
            conv_kernel_size = 4, inner_factor = 2.0,  
            head_channel = 64,
            conv_bias=True, bias=False,
            learnable_init_states = True, chunk_size=256,
            dt_min=0.001, A_init_range=(1, 16),
            dt_max=0.1, dt_init_floor=1e-4, 
            dt_rank = None, dt_scale = 1.0,
            dropout = None, skip_connection = True, mamba = 'Mamba', 
            activation = 'relu', norm='batch', num_norm_groups = -1, **kwargs):
            
            super().__init__(_channel_dim = -1)
            self.in_channel = in_channel
            self.out_channel = out_channel or in_channel
            if mamba == 'Mamba':
                dt_rank = dt_rank or math.ceil(in_channel / 16)
                self.mamba = Mamba(d_model = in_channel, d_state = state_channel, 
                            d_conv = conv_kernel_size, expand = inner_factor, 
                            bias = bias, conv_bias = conv_bias,                         
                            dt_max = dt_max, dt_min = dt_min, dt_rank = dt_rank,
                            dt_init_floor = dt_init_floor, dt_scale = dt_scale)
            elif mamba == 'Mamba2':
                self.mamba = Mamba2(d_model = in_channel, d_state = state_channel, 
                            d_conv = conv_kernel_size, expand = inner_factor, 
                            headdim = head_channel,
                            bias = bias, conv_bias = conv_bias, chunk_size = chunk_size,
                            learnable_init_states = learnable_init_states, 
                            dt_max = dt_max, dt_min = dt_min, 
                            dt_init_floor = dt_init_floor,
                            A_init_range = A_init_range)

            self.norm, self.norm_type = get_norm(norm, in_channel, 1, num_norm_groups)
            self.act = get_act(activation)
            self.time_channel = time_channel
            self.skip_connection = skip_connection
            self.dense = nn.Linear(self.in_channel, self.out_channel) if self.in_channel != self.out_channel else nn.Identity()

            if self.time_channel > 0:
                self.hyper_bias = nn.Linear(self.time_channel, out_channel, bias=False)
                self.hyper_gate = nn.Linear(self.time_channel, out_channel)

        def forward(self, x, t = None):
            res = self.mamba(x)
            res = self.act(self.normalize(res))
            if self.skip_connection:
                res = x + res
            x = self.dense(res)
            if t is not None:
                assert self.time_channel == t.shape[-1], \
                    f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
                gate = expand_as(torch.sigmoid(self.hyper_gate(t)), x, channel_dim=-1)
                bias = expand_as(self.hyper_bias(t), x, channel_dim = -1)
                x = x * gate + bias
            return x
        @staticmethod
        def is_sequence_modeling():
            return True
## part of this code is adopted from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
def knn(x, k):
    # x: B*N*D, inner: B*N*N
    inner = 2*torch.matmul(x, x.transpose(1, 2))
    # x^2: B*N*1
    xx = torch.sum(x**2, dim=-1, keepdim=True)
    # negative distance
    pairwise_distance = -(xx.transpose(1, 2) - inner + xx)
    # return the closest k indices, idx: B*N*K
    idx = pairwise_distance.topk(k=k, dim=-1)[1] 
    return idx

def get_graph_feature(x, k=20, idx=None):
    # B * N * D
    batch_size, num_points, num_dims = x.shape
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

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
    def __init__(self, in_channel, out_channel, k,
        BuildingBlock, use_local = True, use_global = True):
        super().__init__()
        assert use_local or use_global, 'At least one of []'
        self.bb_local = None
        self.bb_global = None
        self.k = k
        if use_local and self.k > 1:
            self.bb_local = BuildingBlock(in_channel = in_channel,
                                    out_channel = in_channel if use_global else out_channel) 
        if use_global:
            self.bb_global = BuildingBlock(in_channel = in_channel * self.k,
                                    out_channel = out_channel * self.k) 
    ## input: grouped points (B, N, K, C)
    def forward(self, x, t):
        B, N, K, _ = x.shape
        assert K == self.k, f'Unmatched group size: {K} and {self.k}'
        if self.bb_local:
            # B * N * K * 2D -> (B * N) * K * 2D
            x = x.reshape(B * N, K, -1)
            # (B * N) * K * 2D -> (B * N) * K * D
            x = self.bb_local(x, t)
            # (B * N) * K * D -> B * N * K * D
            x = x.reshape(B, N, K, -1)
        if self.bb_global:
            # B * N * K * D -> B * N * (K * D) 
            x = x.reshape(B, N, -1)
            # B * N * (K * D) -> B * N * (K * D) 
            x = self.bb_global(x, t)
            # B * N * (K * D) -> B * N * K * D
            x = x.reshape(B, N, K, -1)
        return x

class LocalGraphLayer(nn.Module):
    def __init__(self, k, 
                in_channel, 
                out_channel, 
                BuildingBlock, 
                num_blocks = 1,
                hidden_channels = None,
                use_local = True,
                use_global = True,
                **kwargs):
        super().__init__()
        self.k = k
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
                                        k = self.k,
                                        BuildingBlock = BuildingBlock,
                                        use_local = use_local, 
                                        use_global = use_global))
    def forward(self, x, t=None):
        ## x: B * N * D -> B * N * K * 2D
        x = get_graph_feature(x, k = self.k)
        x = self.bb(x, t)
        ## x: B * N * D
        x, _ = x.max(dim=2)
        return x
        

class FoldingLayer(nn.Module):
    """
    The folding operation of FoldingNet
    """

    def __init__(self, in_channel, 
                out_channel, 
                BuildingBlock,
                hidden_channels = None, 
                num_blocks = 2,
                **kwargs):
                
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        
        self.in_channel = in_channel
        self.layers = MultipleBuildingBlocks(n = num_blocks, in_channel = in_channel, 
                    out_channel = out_channel, hidden_channels = hidden_channels,
                    BuildingBlock = BuildingBlock)        
        self.final = nn.Linear(out_channel, out_channel)
    ## codewords
    def forward(self, shapes, codewords, t = None):
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
        x = self.layers(x, t)
        return self.final(x)

        
### sampling and multi-scale grouping
### furthest point sampling
### group features based on some centers with radius.
### point cloud "down-sampling"
class SamplingAndGroupingBlock(nn.Module):
    # in_channel: in_channel of input features
    # out_channel: out_channel of output featuress
    def __init__(self, in_channel, out_channels, 
            num_fps_points, k, radius, 
            BuildingBlock, num_blocks = 1, 
            hidden_channels = None, use_xyz = True, 
            use_local = True, use_global = True):
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
            self.in_channel += 3
        ### real out_channel
        self.out_channel = sum(self.out_channels)
        is_seq = BuildingBlock.func.is_sequence_modeling()        
        
        self.groupers = []

        self.bb = nn.ModuleList()
        for sid in range(len(self.out_channels)):
            self.groupers.append(QueryAndGroup(self.radius[sid], 
                        self.k[sid], use_xyz=use_xyz) if self.num_fps_points > 0 
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
                                k = self.k[sid], 
                                BuildingBlock = BuildingBlock,
                                use_local = use_local, 
                                use_global = use_global)))

    def forward(self, xyz, features = None, t = None):
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
        if self.num_fps_points > 0:
            xyz_trans = channel_recover(xyz)
            centers = channel_transfer(gather_operation(
                    xyz_trans, furthest_point_sample(xyz, self.num_fps_points)
                ))
            
        ## None center indicates we will group all the points.
        features_trans = channel_recover(features) if features is not None else None
        center_feature_list = []
        for gid in range(len(self.groupers)):
            ## (B, N, 3), (B, M, 3), (B, N, C_in) -> (B, C_in (+3), M, k) 
            grouped_features_trans = self.groupers[gid](
                xyz, centers, features_trans
            ) 
            ## (B, C (+3), M, k) -> (B, M, k, C (+3))
            grouped_features = channel_transfer(grouped_features_trans)
            ## (B, M, k, C (+3)) -> (B, M, k, C_out)
            center_features = self.bb[gid](grouped_features, t)  
            # (B, M, out_channel)
            center_feature_list.append(center_features.max(dim=2)[0])
        return centers, torch.cat(center_feature_list, dim = -1)


### Propagates the features of one set to another
### point cloud "up-sampling"
class FeaturePropogatingBlock(nn.Module):
    r"""Propigates the features of one set to another

    """
    def __init__(self, in_channel_known, in_channel_unknown, 
        out_channel, BuildingBlock, num_blocks = 1, hidden_channels = None):
        # type: (PointnetFPModule, List[int], bool) -> None
        super().__init__()
        self.bb = MultipleBuildingBlocks(in_channel = in_channel_known + in_channel_unknown, 
                out_channel = out_channel, 
                n = num_blocks,
                hidden_channels = hidden_channels,
                BuildingBlock = BuildingBlock)

    def forward(self, unknown, known, unknown_feats, known_feats, t = None):
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
        new_features = self.bb(new_features, t)

        return new_features

