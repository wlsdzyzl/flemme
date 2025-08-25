from flemme.model import create_model
from flemme.dataset import create_loader
from flemme.sampler import create_sampler
from flemme.trainer import test_run, test, save_data
from flemme.utils import load_config

def main():
    test_config = load_config()
    test(test_config,         
        create_model_fn = create_model,
        create_loader_fn = create_loader,
        create_sampler_fn = create_sampler,
        run_fn = test_run,
        save_data_fn = save_data)
    
