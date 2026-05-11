from flemme.utils import draw_hists, mkdirs
import sys, getopt
import os 
from glob import glob
from flemme.logger import get_logger
from flemme.metrics import compute_stats
logger = get_logger('scripts.draw_mesh_stats_histograms')
def main(argv):

    data_path = None
    sub_dirs = ['']
    label_names = None
    subsub_dirs = ['']
    output_dir = ''
    suffix = ['.ply']
    stats = ['volume', 'area', 'compactness', 'curvature_mean']
    overlapped = False
    opts, _ = getopt.getopt(argv, "p:o:", ['help', 'data_path=', 'output_dir=', 'sub_dirs=', 'label_names=', 'subsub_dirs=', 'suffix=', 'stats=', 'overlapped'])
    if len(opts) == 0:
        logger.info('unknow options, usage: draw_mesh_stats_histograms.py -p <data_path> -o <output_dir=.> --sub_dirs <sub_dirs=.> --label_names <label_names=$sub_dirs> --subsub_dirs <subsub_dirs=.> --suffix <suffix=.ply> --stats <stats=volume,area,compactness,curvature_mean> --overlapped')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: draw_mesh_stats_histograms.py -p <data_path> -o <output_dir=.> --sub_dirs <sub_dirs=.> --label_names <label_names=$sub_dirs> --subsub_dirs <subsub_dirs=.> --suffix <suffix=.ply> --stats <stats=volume,area,compactness,curvature_mean> --overlapped')
            sys.exit()
        if opt in ('-p', '--data_path'):
            data_path = arg
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        elif opt in ('--sub_dirs',):
            sub_dirs = arg.split(',')
        elif opt in ('--label_names',):
            label_names = arg.split(',')
        elif opt in ('--subsub_dirs',):
            subsub_dirs = arg.split(',')
        elif opt in ('--suffix',):
            suffix = arg.split(',')
        elif opt in ('--stats',):
            stats = arg.split(',')
        elif opt in ('--overlapped',):
            overlapped = True
        else:
            logger.error('unknow option, usage: draw_mesh_stats_histograms.py -p <data_path> -o <output_dir=.> --sub_dirs <sub_dirs=.> --label_names <label_names=$sub_dirs> --subsub_dirs <subsub_dirs=.> --suffix <suffix=.ply> --stats <stats=volume,area,compactness,curvature_mean> --overlapped')
            sys.exit()
    if data_path is None:
        logger.error('data_path is required.')
        sys.exit()
    if len(suffix) == 1:
        suffix = suffix * len(sub_dirs)
    if label_names is None:
        label_names = sub_dirs
    assert len(suffix) == len(sub_dirs), 'sub_dirs and suffix should have the same length.'
    assert len(label_names) == len(sub_dirs), 'sub_dirs and label_names should have the same length.'
    if len(sub_dirs) > 1:
        assert sum([s == '' or s == '.' for s in sub_dirs]) == 0, 'sub_dirs contains empty folder name.'
    if not (output_dir == '' or output_dir == '.'):
        mkdirs(output_dir)
    ## class level
    for ssd in subsub_dirs:
        prefix = '' if not ssd else ssd + '_'
        all_stats = {k: {} for k in stats}
        ## method level
        for sd, s, l in zip(sub_dirs, suffix, label_names):
            tmp_file_path = os.path.join(data_path, sd, ssd)
            files = sorted(glob(os.path.join(tmp_file_path,  "*" + s)))
            logger.info(f"Calculate stats for {len(files)} files in {tmp_file_path}")
            tmp_stats = compute_stats(stats, files, normalized = True)
            for k in stats:
                all_stats[k][l] = tmp_stats[k] 
        for k in stats:
            draw_hists(os.path.join(output_dir, prefix + k + '.png'), 
                all_stats[k], xlabel=k, overlapped = overlapped, bins=20, 
                color_map='Set1', alpha = 0.8, 
                color_reverse = True)

if __name__ == "__main__":
    main(sys.argv[1:])