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
from scipy.optimize import linear_sum_assignment
import numpy as np
import scipy
import torch
import math
from functools import partial
from flemme.logger import get_logger
from flemme.utils import label_to_onehot, DataForm, topk
from flemme.model import create_model
from flemme.loss import get_loss
from tqdm import tqdm
logger = get_logger('metrics')
device = "cuda" if torch.cuda.is_available() else "cpu"
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
    from flemme.utils import remove_small_components, remove_small_holes
    #### point cloud similarity
    ## Earth mover's distance
    class EMD:
        def __init__(self, reg = 1.0, **kwargs):
            logger.info('using Earth Mover\'s distance')
            self.kwargs = kwargs
            self.ot = partial(ot.sinkhorn, 
                reg = reg, 
                **kwargs)
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
    def euler_betti_numbers_from_mesh(mesh):
        components = mesh.split(only_watertight=False)
        beta0 = len(components)
        # β2：number of closed surfaces
        beta2 = sum(c.is_watertight for c in components)
        # β1：compute using Euler characteristic
        # χ = V - E - F
        # for 2-manifold: β1 = β0 - β2 - χ
        beta1 = 0
        for c in components:
            v = len(c.vertices)
            f = len(c.faces)
            # edges can be extracted:
            e = len(c.edges_unique)
            chi = v - e + f
            # For each component:
            b0_c = 1
            b2_c = 1 if c.is_watertight else 0
            b1_c = b0_c - b2_c - chi
            beta1 += b1_c
        return beta0, beta1, beta2
    def simplex_betti_numbers_from_mesh(mesh):
        import gudhi
        st = gudhi.SimplexTree()
        for tri in mesh.faces:
            st.insert([int(tri[0]), int(tri[1]), int(tri[2])])

        st.compute_persistence()

        bettis = st.betti_numbers()
        betti0 = bettis[0] if len(bettis)>0 else 0
        betti1 = bettis[1] if len(bettis)>1 else 0
        betti2 = bettis[2] if len(bettis)>2 else 0

        return betti0, betti1, betti2
    ### compute betti-number error for triangle mesh pair
    class BettiError:
        def __init__(self, order = [0, 1, 2], method = 'euler',
                min_num_faces = 10, min_hole_size = 10):
            self.order = order
            assert method in ['euler', 'simplex'], 'method should be one of [euler, simplex].'
            assert max(order) <=2 and min(order) >=0, 'order should be in [0, 1, 2].'
            logger.info(f'using Betti Number Error of order {order}')
            self.compute_betti = euler_betti_numbers_from_mesh if method == 'euler' else simplex_betti_numbers_from_mesh
            self.remove_small_components = partial(remove_small_components, threshold_faces=min_num_faces)
            self.remove_small_holes = partial(remove_small_holes, threshold_holes=min_hole_size)
        def __call__(self, x, y):
            x = self.remove_small_components(x)
            x = self.remove_small_holes(x)
            y = self.remove_small_components(y)
            y = self.remove_small_holes(y)
            x_betti = self.compute_betti(x)
            y_betti = self.compute_betti(y)
            return (np.abs(np.array(x_betti) - np.array(y_betti))).astype(float)[self.order]
    ### compute mean betti-number error for triangle mesh set
    class meanBettiError:
        def __init__(self, order = [0, 1, 2], method = 'euler',
                    min_num_faces = 10, min_hole_size = 10):
            self.order = order
            assert method in ['euler', 'simplex'], 'method should be one of [euler, simplex].'
            assert max(order) <=2 and min(order) >=0, 'order should be in [0, 1, 2].'
            logger.info(f'using Betti Number Error of order {order}')
            self.compute_betti = euler_betti_numbers_from_mesh if method == 'euler' else simplex_betti_numbers_from_mesh
            self.remove_small_components = partial(remove_small_components, threshold_faces=min_num_faces)
            self.remove_small_holes = partial(remove_small_holes, threshold_holes=min_hole_size)
        def __call__(self, x, y):
            x_betti_mean = sum([np.array(self.compute_betti(self.remove_small_holes(
                                        self.remove_small_components(tx)))) 
                                for tx in x]) / len(x)
            y_betti_mean = sum([np.array(self.compute_betti(self.remove_small_holes(
                                        self.remove_small_components(ty)))) 
                                for ty in y]) / len(y)
            return np.abs(x_betti_mean - y_betti_mean)[self.order]
    ### f_score and hausdorff distance
    def compute_f_score_and_hd(x, y, tau, metric):
        kdtree_x = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        kdtree_y = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        dist_pred_to_gt = kdtree_y.kneighbors(x)[0]
        precision = np.mean(dist_pred_to_gt < tau)
        dist_gt_to_pred = kdtree_x.kneighbors(y)[0]
        recall = np.mean(dist_gt_to_pred < tau)
        f_score = (2 * precision * recall + 1e-8) / (precision + recall + 1e-8)
        hd = np.max([dist_gt_to_pred.max(), dist_pred_to_gt.max()])
        return np.array([f_score, hd])
    class F1AndHDPoint:
        def __init__(self, dist_threshold = 0.05, metric = 'l2'):
            self.dist_threshold = dist_threshold
            self.metric = metric
        def __call__(self, x, y):
            return compute_f_score_and_hd(x, y, self.dist_threshold, self.metric)
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
            assert type(x) == tuple, 'the prediction of graph should be a tuple.'
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

            if recon_edge is not None and gt_edge_index is not None:
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

# evaluation for generative models
## use pre-trained model to compute Frechet Inception Distance    
class FID:
    def __init__(self, embedding_net,
                    embedding_net_path,
                    batch_size = 16):
        logger.info('using Frechet Inception Distance')
        self.embedding_net = create_model(embedding_net)
        self.embedding_net.load_state_dict(
            torch.load(embedding_net_path, map_location='cpu', weights_only=False)['trained_model'])
        self.embedding_net = self.embedding_net.to(device)
        self.embedding_net.eval()
        self.batch_size = batch_size
    def __call__(self, x, y):
        x = torch.split(torch.tensor(x).float(), self.batch_size, dim = 0)
        y = torch.split(torch.tensor(y).float(), self.batch_size, dim = 0)
        z_gen = np.concatenate([self.embedding_net(_x.to(device))['latent'].detach().cpu().numpy() for _x in x], axis = 0)
        z_real = np.concatenate([self.embedding_net(_y.to(device))['latent'].detach().cpu().numpy() for _y in y], axis = 0)        
        mu1 = np.mean(z_real, axis=0)
        mu2 = np.mean(z_gen, axis=0)
        sigma1 = np.cov(z_real, rowvar=False)
        sigma2 = np.cov(z_gen, rowvar=False)
        diff = mu1 - mu2
        # product might not be positive semi-definite numeric issues
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
        # print(fid)
        return fid    

## use pre-trained model to compute Kernel Inception Distance (based on Maximum Mean Discrepancy)
class KID:
    def __init__(self, embedding_net,
                    embedding_net_path, 
                    degree = 3, 
                    num_subsets = 100, 
                    subset_size = 100,
                    batch_size = 16):
        logger.info('using Kernel Inception Distance')
        self.embedding_net = create_model(embedding_net)
        self.embedding_net.load_state_dict(
            torch.load(embedding_net_path, map_location='cpu', weights_only=False)['trained_model'])
        self.degree = degree
        self.num_subsets = num_subsets
        self.subset_size = subset_size
        self.kernel = lambda X, Y: (np.dot(X, Y.T) / X.shape[1] + 1) ** self.degree
        self.embedding_net = self.embedding_net.to(device)
        self.embedding_net.eval()
        self.batch_size = batch_size
    def __call__(self, x, y):
        """Unbiased KID estimator"""
        x = torch.split(torch.tensor(x).float(), self.batch_size, dim = 0)
        y = torch.split(torch.tensor(y).float(), self.batch_size, dim = 0)
        z_gen = np.concatenate([self.embedding_net(_x.to(device))['latent'].detach().cpu().numpy() for _x in x], axis = 0)
        z_real = np.concatenate([self.embedding_net(_y.to(device))['latent'].detach().cpu().numpy() for _y in y], axis = 0)
        assert len(z_real) >= self.subset_size and len(z_gen) >= self.subset_size, \
            'Not enough samples to compute KID, please specify a smaller subset_size.'
        kid_scores = []
        for _ in range(self.num_subsets):
            idx_r = np.random.choice(len(z_real), self.subset_size, replace=False)
            idx_f = np.random.choice(len(z_gen), self.subset_size, replace=False)
            X = z_real[idx_r]
            Y = z_gen[idx_f]
            K_XX = self.kernel(X, X)
            K_YY = self.kernel(Y, Y)
            K_XY = self.kernel(X, Y)
            mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
            kid_scores.append(mmd)
        # print(np.mean(kid_scores))
        return np.mean(kid_scores)
## compute mean minimum distance
class MMDAndCov:
    def __init__(self, distance = {'name':'CD'}, 
                 package = 'loss', 
                 batch_size = 16,
                 data_form = DataForm.IMG):
        logger.info(f'using Mean Minimum Distance and Coverage with {distance}')
        assert package in ['loss', 'metric'], 'package should be one of [loss, metric].'
        self.package = package
        self.batch_size = batch_size
        if self.package == 'loss':
            distance['reduction'] = None
            self.dist_fn = get_loss(distance, data_form=data_form)
        else:
            self.dist_fn = get_metrics(distance, data_form=data_form)
    def __call__(self, x, y):
        N_real = len(y)
        N_fake = len(x)
        dist_mat = np.zeros((N_real, N_fake))
        for i in tqdm(range(N_real), desc="DistMat"):
            if self.package == 'loss':
                fake = torch.from_numpy(x).float().to(device)
                real = torch.from_numpy(y[i]).unsqueeze(0).float().to(device)
                real = real.expand(fake.shape)

                real = torch.split(real, self.batch_size, dim = 0)
                fake = torch.split(fake, self.batch_size, dim = 0)
                
                res_dist = []
                for r, f in zip(real, fake):
                    res_dist.append(self.dist_fn(r, f))
                res_dist = torch.cat(res_dist, dim = 0).cpu().detach().numpy()
                dist_mat[i] = res_dist
            else:
                for j in range(N_fake):
                    dist_mat[i, j] = self.dist_fn(y[i], x[j])
        mmd = np.mean(np.min(dist_mat, axis=1))

        matched_real = np.argmin(dist_mat, axis=0)
        cov = len(np.unique(matched_real)) / N_real
        return np.array([mmd, cov])

def get_metrics(metric_config, data_form = None, classification = False):
    channel_dim = None
    if classification:
        ### B * C
        channel_dim = 1
    elif data_form == DataForm.IMG:
        ### if classification, the channel dim = 1
        channel_dim = 0
    ## point cloud
    else:
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
    if name == 'Dice&HD-Point' or name == 'F1&HD-Point':
        return F1AndHDPoint(**metric_config)
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
    if name == 'BettiError':
        return BettiError(**metric_config)
    if name == 'meanBettiError':
        return meanBettiError(**metric_config)
    if name == 'GND' or name == 'GraphNodeDistance':
        return GND(**metric_config)
    if name == 'GEA' or name == 'GraphNodeAccuracy': 
        return GEA(**metric_config)
    ### generative model
    if name == 'FID':
        return FID(**metric_config)
    if name == 'KID':
        return KID(**metric_config) 
    if name == 'MMD&Cov':
        return MMDAndCov(data_form = data_form, **metric_config)
    logger.error(f'Unsupported metric: {name}')
    return None
