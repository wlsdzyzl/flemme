import sys, getopt
import shutil
from pathlib import Path
from flemme.logger import get_logger
logger = get_logger('scripts.reorganize_cpl_cls_datasets')

def reorganize_dataset_structure(dataset_path):
    """
    Reorganize directory structure from /datasets/$label_names/$folds to /datasets/$folds/$label_names
    Uses copy-then-delete approach to prevent data loss in case of interruption
    """
    dataset_path = Path(dataset_path)
    
    # Check if source directory exists
    if not dataset_path.exists():
        logger.error(f"Dataset directory '{dataset_path}' does not exist")
        return False
    
    # Get all label_names directories
    label_names = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not label_names:
        logger.info("Error: No label_names directories found")
        return False
    
    # Get all folds names from the first label_name directory
    first_label = label_names[0]
    folds = [d.name for d in first_label.iterdir() if d.is_dir()]
    
    if not folds:
        logger.info("Error: No folds directories found")
        return False
    
    logger.info(f"Found {len(label_names)} label_names: {[ln.name for ln in label_names]}")
    logger.info(f"Found {len(folds)} folds: {folds}")
    
    # Create new fold directories in root
    for fold in folds:
        fold_path = dataset_path / fold
        fold_path.mkdir(exist_ok=True)
        logger.info(f"Created directory: {fold_path}")
    
    # Copy files to new structure first
    copied_items = []
    for label_dir in label_names:
        label_name = label_dir.name
        logger.info(f"Processing label: {label_name}")
        
        for fold in folds:
            old_fold_path = label_dir / fold
            new_fold_path = dataset_path / fold / label_name
            
            if old_fold_path.exists() and old_fold_path.is_dir():
                # Copy entire directory recursively
                try:
                    shutil.copytree(str(old_fold_path), str(new_fold_path))
                    copied_items.append((old_fold_path, new_fold_path))
                    logger.info(f"Copied: {old_fold_path} -> {new_fold_path}")
                except Exception as e:
                    logger.info(f"Error copying {old_fold_path}: {e}")
                    return False
    
    # Verify all copies were successful before deleting originals
    logger.info("Verifying all copies were successful...")
    all_copies_valid = True
    
    for old_path, new_path in copied_items:
        if not new_path.exists():
            logger.info(f"Error: Copy verification failed for {new_path}")
            all_copies_valid = False
        elif not any(new_path.iterdir()):
            logger.info(f"Warning: Target directory is empty: {new_path}")
    
    if not all_copies_valid:
        logger.info("Error: Some copies failed. Aborting deletion of original files.")
        return False
    
    # Delete original files only after successful copy and verification
    logger.info("Deleting original files...")
    for old_path, new_path in copied_items:
        try:
            shutil.rmtree(str(old_path))
            logger.info(f"Deleted: {old_path}")
        except Exception as e:
            logger.info(f"Error deleting {old_path}: {e}")
    
    # Remove empty label_names directories
    for label_dir in label_names:
        if not any(label_dir.iterdir()):
            try:
                label_dir.rmdir()
                logger.info(f"Removed empty directory: {label_dir}")
            except Exception as e:
                logger.info(f"Error removing directory {label_dir}: {e}")
    
    logger.info("Directory structure reorganization completed successfully!")
    return True

def main(argv):
    dataset_path = None 
    opts, _ = getopt.getopt(argv, "hd:", ['help', 'dataset_path='])
    if len(opts) == 0:
        logger.info('unknow options, usage: reorganize_cpl_cls_datasets.py -d <dataset_path>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: reorganize_cpl_cls_datasets.py -d <dataset_path>')
            sys.exit()
        if opt in ('-d', '--dataset_path'):
            dataset_path = arg
    if dataset_path is None:
        logger.info("Dataset path is required.")
        sys.exit(1)
    logger.info("Dataset reorganization script")
    logger.info("=" * 50)
    logger.info("This script will:")
    logger.info("1. Copy all files from /datasets/label_name/fold/ to /datasets/fold/label_name/")
    logger.info("2. Verify all copies are successful")
    logger.info("3. Delete original files only after successful verification")
    logger.info("4. Clean up empty directories")
    logger.info("=" * 50)
    
    # Confirm operation
    confirm = input(f"About to reorganize: {dataset_path}\nProceed? (y/n): ")
    if confirm.lower() == 'y':
        success = reorganize_dataset_structure(dataset_path)
        if success:
            logger.info("Operation completed successfully!")
        else:
            logger.info("Operation failed or was aborted due to errors.")
    else:
        logger.info("Operation cancelled")
if __name__ == "__main__":
    main(sys.argv[1:])