
from flemme.model import create_model
from flemme.dataset import create_loader
from flemme.sampler import create_sampler
from flemme.trainer import train_run, train
from flemme.utils import load_config
def main():
    train_config = load_config()
    train(train_config, 
        create_model_fn = create_model,
        create_loader_fn = create_loader,
        create_sampler_fn = create_sampler,
        run_fn = train_run)