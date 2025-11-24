import sys, getopt
import os
import shutil
from flemme.logger import get_logger
from tqdm import tqdm
logger = get_logger('scripts.syncronize_splited_folds')

def copy_directory_structure(src, dst):
    for root, _, _ in os.walk(src):
        # relative_path
        rel_path = os.path.relpath(root, src)
        # target path
        target_dir = os.path.join(dst, rel_path)
        os.makedirs(target_dir, exist_ok=True)
def get_file_paths(root):
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            paths.append(os.path.abspath(os.path.join(r, f)))
    return paths
def get_basename_and_folder(path, folder_base_name):
    splited_paths = path.strip().split('/')
    fold = None
    for s in splited_paths:
        if folder_base_name in s:
            fold = s
            break
    basename = splited_paths[-2] +'/'+ splited_paths[-1].split('_')[0]
    return basename, fold
## tmp file
## python syncronize_splited_folds.py -s /media/wlsdzyzl/DATA/datasets/pcd/MedSDF/ --target_dir /media/wlsdzyzl/DATA/datasets/pcd/MedSDF_sycronized/ --reference ./syncronization_ref.txt --keyword raw 
def main(argv):
    source_dir = None
    target_dir = None
    reference = None
    folder_base_name = 'fold'
    ### ignore file path without keyword, to reduce duplicate operation
    keyword = 'raw'
    opts, _ = getopt.getopt(argv, "hs:t:r:f:k:", ['help', 'source_dir=', 'target_dir=', 'reference=', 'folder_base_name=', 'keyword=', 'method='])
    method = shutil.copy
    mn = 'copy'
    ### move is faster, but with higher risk of losing data.
    if len(opts) == 0:
        logger.info('unknow options, usage: syncronize_splited_folds.py -s <source_dir> --target_dir <target_dir> --reference <reference> --folder_base_name <folder_base_name=folder> --keyword <keyword=raw> --method <method=copy>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: syncronize_splited_folds.py -s <source_dir> --target_dir <target_dir> --reference <reference> --folder_base_name <folder_base_name=folder> --keyword <keyword=raw> --method <method=copy>')
            sys.exit()
        if opt in ('-s', '--source_dir'):
            source_dir = arg
        elif opt in ('-t','--target_dir'):
            target_dir = arg
        elif opt in ('-r', '--reference',):
            reference = arg
        elif opt in ('-f', '--folder_base_name',):
            folder_base_name = arg
        elif opt in ('-k', '--keyword'):
            keyword = arg
        elif opt in ('--method',):
            mn = arg
            if mn == 'move':
                method = shutil.move
            elif not mn == 'copy':
                logger.info('Unknow operation.')
                exit(1)
        else:
            logger.error('unknow option, usage: syncronize_splited_folds.py -s <source_dir> --target_dir <target_dir> --reference <reference> --folder_base_name <folder_base_name=folder> --keyword <keyword=raw> --method <method=copy>')
            sys.exit()

    if source_dir is None:
        logger.error('source_dir is required.')
        sys.exit()
    if target_dir is None:
        logger.error('target_dir is required.')
        sys.exit()
    if reference is None:
        logger.error('reference is required.')
        sys.exit()
    copy_directory_structure(source_dir, target_dir)
    file_to_fold = {}
    with open(reference, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if keyword in l:
                basename, fold = get_basename_and_folder(l, folder_base_name=folder_base_name)
                if fold:
                    assert not basename in file_to_fold, f'Conflict folds ({file_to_fold[basename]}) for file {l}.'
                    file_to_fold[basename] = fold
    # print('001313_bladder', file_to_fold['bladder/001313_bladder'])
    origin_paths = get_file_paths(source_dir)
    for op in tqdm(origin_paths, desc="Sycronizing"):
        tp = op.replace(source_dir, target_dir)
        basename, old_fold = get_basename_and_folder(op, folder_base_name=folder_base_name)
        if basename in file_to_fold:
            tp = tp.replace(old_fold, file_to_fold[basename])
        if not os.path.exists(tp) or not os.path.getsize(tp) == os.path.getsize(op):
            method(op, tp)
    
if __name__ == "__main__":
    main(sys.argv[1:])