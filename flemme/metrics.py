### define evaluation metrics
from flemme.config import module_config
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim, \
                            hausdorff_distance as hd, peak_signal_noise_ratio as psnr
from sklearn.metrics.cluster import rand_score as ri, adjusted_rand_score as ari, \
    mutual_info_score as mis, adjusted_mutual_info_score as amis, \
    normalized_mutual_info_score as nmis
from sklearn.neighbors import NearestNeighbors
import scipy.ndimage as ndimage
from scipy.special import softmax, expit as sigmoid
import numpy as np
import math
from functools import partial
from flemme.logger import get_logger
from scipy.optimize import linear_sum_assignment
from flemme.utils import label_to_onehot, DataForm, topk


logger = get_logger('metrics')
#### image similarity
### multi channel case
class SSIM:
    def __init__(self, data_range = None, channel_dim = 0):
        logger.info('using structural similarity')
        self.data_range = data_range    
        self.channel_dim = channel_dim
        self.ssim = partial(ssim, data_range = self.data_range, channel_axis = self.channel_dim)
    def __call__(self, x, y):
        return self.ssim(x, y)
class PSNR:
    def __init__(self, data_range = None):
        logger.info('using peak signal noise ratio')
        self.data_range = data_range  
        self.psnr = partial(psnr, data_range = self.data_range)
    def __call__(self, x, y):
        return self.psnr(x, y)
        
### segmentation accuracy
class ACC:
    def __init__(self, channel_dim = None):
        self.channel_dim = channel_dim
        logger.info('using Accuracy')
    def __call__(self, x, y):
        ### at least one of [x, y] should not be one hot
        assert x.shape == y.shape, "Inputs should have the same shape!"
        ### transfer one-hot embedding to label
        if self.channel_dim is not None and x.shape[self.channel_dim] > 1 and max(y.max(), x.max()) == 1:
            x, y = x.argmax(axis = self.channel_dim), y.argmax(axis = self.channel_dim)
        return (x == y).sum() / (math.prod(x.shape))   
class TopKACC:
    def __init__(self, channel_dim = None, k = 5):
        self.channel_dim = channel_dim
        self.k = 5
        logger.info('using Top-k Accuracy')
    def __call__(self, x, y):
        ### at least one of [x, y] should not be one hot
        assert x.shape == y.shape, "Inputs should have the same shape!"
        ### transfer one-hot embedding to label
        assert self.channel_dim is not None and x.shape[self.channel_dim] > self.k, \
            f"Not suitable to use Top-{self.k} Accuracy, please make sure the number of class is larger than {self.k}."
        y = y.argmax(axis = self.channel_dim)
        _, x = topk(x, k = self.k, axis = self.channel_dim)
        res = np.zeros_like(y, dtype=np.bool_)
        for kid in range(self.k):
            curr_x = np.take(x, axis = self.channel_dim, indices = kid)
            res = np.logical_or(res, curr_x == y)
        return res.sum() / (math.prod(res.shape))  

##### binary segmentation error
class SegMetrics:
    def __init__(self, channel_dim = None):
        self.channel_dim = channel_dim
    
    def __channel_rearrange__(self, x, y, soft = False):
        # print(x.shape, y.shape)
        assert x.shape == y.shape, "Inputs should have the same shape!"
        multi_channel = self.channel_dim is not None and x.shape[self.channel_dim] > 1
        channel_dim = self.channel_dim
        if not multi_channel and max(y.max(), x.max()) > 1:
            if channel_dim is not None:
                x = x.squeeze(channel_dim)
                y = y.squeeze(channel_dim)
            channel_dim = 0
            ### transfer to one hot embedding
            num_classes = int(max(y.max(), x.max())) + 1
            x = label_to_onehot(x, 
                num_classes = num_classes, channel_dim = 0)
            y = label_to_onehot(y, 
                num_classes = num_classes, channel_dim = 0)
            multi_channel = True
        y = y > 0
        if not soft:
            x = x > 0
        if multi_channel:
            if not channel_dim == 0:
                permute = list(range(x.ndim))
                permute[channel_dim], permute[0] = permute[0], permute[channel_dim]
                x, y = x.transpose(*permute), y.transpose(*permute)
        elif channel_dim is not None:
            x, y = x[channel_dim], y[channel_dim]
        return x, y, multi_channel

###  compute per label accuracy
class mIoU(SegMetrics):
    def __init__(self, channel_dim = None):
        super().__init__(channel_dim)
        logger.info('using mean Intersection over Units')
    def __call__(self, x, y):
        x, y, multi_channel = self.__channel_rearrange__(x, y)
        if multi_channel:
            c = x.shape[0]
            x, y = x.reshape((c, -1)), y.reshape((c, -1))
            numerator = (x & y).sum(axis=1) + 1e-8
            denominator = (x | y).sum(axis=1) + 1e-8
            ### per-channel IoU
            return (numerator / denominator).mean()
        else:
            ### binary segmentation
            numerator = (x & y).sum() + 1e-8
            denominator = (x | y).sum() + 1e-8
            return numerator / denominator

class Dice(SegMetrics):
    def __init__(self, channel_dim = None):
        super().__init__(channel_dim)
        logger.info('using Dice score')
    def __call__(self, x, y):       
        x, y, multi_channel = self.__channel_rearrange__(x, y)
        if multi_channel:
            c = x.shape[0]
            x, y = x.reshape((c, -1)), y.reshape((c, -1))
            numerator = 2 * (x & y).sum(axis=1) + 1e-8
            denominator = x.sum(axis=1) + y.sum(axis=1) + 1e-8
            ### per-channel dice
            return (numerator / denominator).mean()
        else:
            return (2 * (x & y).sum() + 1e-8) / (x.sum() + y.sum() + 1e-8) 
class SoftDice(SegMetrics):
    def __init__(self, channel_dim = None):
        super().__init__(channel_dim)
        logger.info('using Soft Dice score')
    def __call__(self, x, y):       
        x, y, multi_channel = self.__channel_rearrange__(x, y, soft = True)
        if multi_channel:
            x = softmax(x, axis = 0)
            c = x.shape[0]
            x, y = x.reshape((c, -1)), y.reshape((c, -1))
            numerator = 2 * (x * y).sum(axis=1) + 1e-8
            denominator = (x**2).sum(axis=1) + (y**2).sum(axis=1) + 1e-8
            ### per-channel dice
            return (numerator / denominator).mean()
        else:
            x = sigmoid(x)
            return (2 * (x * y).sum() + 1e-8) / (x.sum() + y.sum() + 1e-8) 
class HD(SegMetrics):
    def __init__(self, channel_dim = None, method='standard'):
        super().__init__(channel_dim)
        self.hd = partial(hd, method = method)
        logger.info('using Hausdorff distance')
    def __call__(self, x, y):
        x, y, multi_channel = self.__channel_rearrange__(x, y)
        if multi_channel:
            return sum([self.hd(cx, cy) for cx, cy in zip(x, y)]) / x.shape[0]
        else:
            return self.hd(x, y) 

class SegARI(SegMetrics):
    def __init__(self, boundary = True, dim = 2, channel_dim=None):
        super().__init__(channel_dim)
        self.boundary = boundary
        if dim == 2:
            self.structure = np.ones((3,3), int)
        elif dim == 3:
            self.structure = np.ones((3,3,3), int)
        else:
            raise NotImplementedError
        logger.info('using Adjusted Rand Index for binary segmentation')    
    def __call__(self, x, y):
        x, y, multi_channel = self.__channel_rearrange__(x, y)
        if self.boundary:
            x, y = 1-x, 1-y
        if multi_channel:
            c = x.shape[0]
            res = [ ari(ndimage.label(x[i], structure = self.structure)[0].flatten(), 
                        ndimage.label(y[i], structure = self.structure)[0].flatten()) for i in range(c)]
            ### per-channel ari
            return sum(res) / c
        else:
            x_cluster, _ = ndimage.label(x, structure = self.structure)
            y_cluster, _ = ndimage.label(y, structure = self.structure)
            return ari(x_cluster.flatten(), y_cluster.flatten())    

### clustering
class RI:
    def __init__(self):
        logger.info('using Rand Index for clustering')
    def __call__(self, x, y):
        return ri(x.flatten(), y.flatten())
class ARI:
    def __init__(self):
        logger.info('using Adjusted Rand Index for clustering')
    def __call__(self, x, y):
        return ari(x.flatten(), y.flatten())
class MIS:
    def __init__(self):
        logger.info('using Mutual Information Score for clustering')
    def __call__(self, x, y):
        return mis(x.flatten(), y.flatten())
class NMIS:
    def __init__(self):
        logger.info('using Normalized Mutual Information Score for clustering')
    def __call__(self, x, y):
        return nmis(x.flatten(), y.flatten())

class AMIS:
    def __init__(self):
        logger.info('using Adjusted Mutual Information Score for clustering')
    def __call__(self, x, y):
        return amis(x.flatten(), y.flatten())
    
class CACC:
    def __init__(self):
        logger.info('using Unsupervised Clustering Accuracy for clustering')
  # Code taken from the work 
  # VaDE (Variational Deep Embedding:A Generative Approach to Clustering)
    def __call__(self, x, y):
        x, y = x.flatten(), y.flatten()
        assert x.size == y.size
        D = max(x.max(), y.max())+1
        w = np.zeros((D,D), dtype=np.int64)
        for i in range(x.size):
            w[x[i], y[i]] += 1
        row, col = linear_sum_assignment(w.max()-w)
        return sum([w[row[i],col[i]] for i in range(row.shape[0])]) * 1.0/x.size

if module_config['point-cloud']:
    import ot
    #### point cloud similarity
    ## Earth mover's distance
    class EMD:
        def __init__(self, **kwargs):
            logger.info('using Earth Mover\'s distance')
            self.kwargs = kwargs
            self.ot = partial(ot.sinkhorn, **kwargs)
        def __call__(self, x, y):
            M = ot.dist(x, y)
            a = np.ones((x.shape[0], )) / x.shape[0]    
            b = np.ones((y.shape[0], )) / y.shape[0]
            gamma = self.ot(a, b, M)
            return (M * gamma).sum()

    ## chamfer distance
    class CD:
        def __init__(self, metric = 'l2', direction = 'bi'):
            self.metric = metric
            self.direction = direction
            logger.info('using Chamfer distance')
        def __call__(self, x, y):    
            if self.direction=='y_to_x':
                x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=self.metric).fit(x)
                min_y_to_x = x_nn.kneighbors(y)[0]
                chamfer_dist = np.mean(min_y_to_x)
            elif self.direction=='x_to_y':
                y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=self.metric).fit(y)
                min_x_to_y = y_nn.kneighbors(x)[0]
                chamfer_dist = np.mean(min_x_to_y)
            elif self.direction=='bi':
                x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=self.metric).fit(x)
                min_y_to_x = x_nn.kneighbors(y)[0]
                y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=self.metric).fit(y)
                min_x_to_y = y_nn.kneighbors(x)[0]
                chamfer_dist = (np.mean(min_y_to_x) + np.mean(min_x_to_y))/2
            else:
                logger.error("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
                raise ValueError
            return chamfer_dist
if module_config['graph']: 
    from flemme.loss import GraphNodeLoss
    ## graph node distance
    class GND(GraphNodeLoss):
        def __init__(self,
                    lambda_pos = 1.0, lambda_feature = 1.0):
            super().__init__(reduction = 'sum', 
                lambda_pos = lambda_pos, 
                lambda_feature = lambda_feature)
        def __call__(self, x, y):
            assert type(pred) == tuple, 'the prediction of graph should be a tuple.'
            x = ( _x.detach() if _x is not None else _x for _x in x) 
            y = y.detach()
            return super().forward(x, y).item()
    ## graph edge accuracy
    class GEA:
        def __call__(self, pred, data):
            assert type(pred) == tuple, 'the prediction of graph should be a tuple.'
            pred = ( _x.detach() if _x is not None else _x for _x in pred) 
            batch_size = data.batch_size
            recon_edge = pred[-1]
            gt_edge_index = data.edge_index

            if recon_edge is not None and edge_index is not None:
                valid_count = len(recon_edge.values())
                recon_edge = torch.sparse_coo_tensor(indices = recon_edge.indices(), 
                        values = (recon_edge.values() > 0 ).float(), size = recon_edge.shape)
                gt_edge = torch.sparse_coo_tensor(indices = gt_edge_index, 
                        values = torch.ones(gt_edge_index.shape[1], dtype=torch.float32, 
                        device = recon_edge.device), size = recon_edge.shape)
               
                incorrect_count = (recon_edge - gt_edge).abs().sum() 
                return (1.0 - incorrect_count / valid_count) * batch_size
            else:
                logger.warning('There is no edge in graph or model doesn\'t predict edges.')
            return 0.0

    

def get_metrics(metric_config, data_form = None):
    channel_dim = None
    if data_form == DataForm.IMG:
        channel_dim = 0
    elif data_form == DataForm.PCD:
        channel_dim = -1
    name = metric_config.pop('name')
    ### SSIM and MSE doesn't have any configuration.
    if name == 'SSIM':
        return SSIM(channel_dim=channel_dim, **metric_config)
    ### psnr doesn't need channel axis
    if name == 'PSNR':
        return PSNR(**metric_config)
    if name == 'MSE':
        return mse
    ## segmentation metrics
    if name == 'ACC':
        return ACC(channel_dim=channel_dim, **metric_config)
    if name == 'TopKACC':
        return TopKACC(channel_dim=channel_dim, **metric_config)
    if name == 'Dice' or name == 'F1':
        return Dice(channel_dim=channel_dim, **metric_config)
    if name == 'SoftDice':
        return SoftDice(channel_dim=channel_dim, **metric_config)
    if name == 'mIoU':
        return mIoU(channel_dim=channel_dim, **metric_config)
    if name == 'HD':
        return HD(channel_dim=channel_dim, **metric_config)
    if name == 'SegARI':
        return SegARI(channel_dim=channel_dim, **metric_config)
    ### clustering metrics
    if name == 'ARI':
        return ARI(**metric_config)
    if name == 'RI':
        return RI(**metric_config)
    if name == 'MIS':
        return MIS(**metric_config)
    if name == 'NMIS':
        return NMIS(**metric_config)
    if name == 'AMIS':
        return AMIS(**metric_config)
    if name == 'CACC':
        return CACC(**metric_config)
    ### point cloud
    if name == 'EMD':
        return EMD(**metric_config)
    if name == 'CD' or name == 'Chamfer':
        return CD(**metric_config)
    if name == 'GND' or name == 'GraphNodeDistance':
        return GND(**metric_config)
    if name == 'GEA' or name == 'GraphNodeAccuracy': 
        return GEA(**metric_config)
    return None
