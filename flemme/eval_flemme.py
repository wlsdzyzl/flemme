from flemme.dataset import create_loader
from flemme.trainer import eval_run, evaluate
from flemme.utils import load_config

def main():
    eval_config = load_config()
    evaluate(eval_config,         
        create_loader_fn = create_loader,
        run_fn = eval_run)
    
