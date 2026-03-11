from flemme.model.ddpm import linear_schedule, cosine_schedule, gather, Gaussian
from flemme.utils import *
import sys, getopt
import os
from flemme.logger import get_logger
from tqdm import tqdm


def main(argv):
    input_file = None
    output_dir = ''
    num_steps = 1000
    interval = 10
    schedule = 'cos'
    opts, args = getopt.getopt(argv, "hi:o:s:", ['help', 'input_file=', 'output_dir=', 'num_steps=', 'interval=', 'schedule='])

    if len(opts) == 0:
        logger.error('unknow options, usage: diffuse_pcd.py -i <input_file> -o <output_dir> --num_steps <num_steps=1000> --interval <interval=10> --schedule <schedule=cos>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: diffuse_pcd.py -i <input_file> -o <output_dir> --num_steps <num_steps=1000> --interval <interval=10> --schedule <schedule=cos>')
            sys.exit()
        elif opt in ("-i", '--input_file'):
            input_file = arg
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        elif opt in ('--num_steps', ):
            num_steps = int(arg)
        elif opt in ('--interval', ):
            interval = int(arg)
        elif opt in ('-s', '--schedule'):
            schedule = arg
        else:
            logger.error('unknow option, usage: diffuse_pcd.py -i <input_file> -o <output_dir> --num_steps <num_steps=1000> --interval <interval=10> --schedule <schedule=cos>')
            sys.exit()
    if input_file is None:
        logger.error('Input pcd is not specified.')
        exit(1)
    mkdirs(output_dir)
    t = np.arange(0, num_steps, interval)
    if not t[-1] == num_steps - 1:
        t = np.append(t, num_steps-1)
    pcd = load_pcd(input_file)
    if schedule == 'cos':
        beta = cosine_schedule(num_steps)
    elif schedule == 'lin':
        beta = linear_schedule(num_steps)
    else:
        logger.error('Unsupported schedule for diffusion models, should be one of [cos, lin].')
        exit(1)
    t = torch.from_numpy(t)
    sqrt_alpha_bar = torch.cumprod(1 - beta, dim=0) ** .5
    alpha_bar = torch.cumprod(1 - beta, dim=0)
    x0 = torch.from_numpy(pcd[None,:]).expand(len(t), -1, -1)
    ## add noise
    # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
    mean = gather(sqrt_alpha_bar, t, dim=1) * x0
    # $(1-\bar\alpha_t) \mathbf{I}$
    var = 1 - gather(alpha_bar, t, dim=1)
    
    noisy_pcd = Gaussian(mean, var = var).sample()
    filename = os.path.splitext(os.path.basename(input_file))[0]
    for pid, p in zip(t, noisy_pcd):
        ofile = os.path.join(output_dir, filename+f'_{pid}.ply')
        save_pcd(ofile, p.cpu().numpy())
if __name__ == "__main__":
    main(sys.argv[1:])