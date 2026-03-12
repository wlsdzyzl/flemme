from flemme.model import create_model
from flemme.trainer.trainer_utils import *
from flemme.logger import get_logger
import time

logger = get_logger('unittest.test_time_and_space')
model_configs = load_config().get('models')
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 2
num_iteration = 500
logger.info(f'training for {num_iteration} iters')
for model_config in model_configs:
    building_block = model_config['encoder'].get('building_block', 'single')
    model = create_model(model_config)
    model.to(device)
    num_layers = (len(model.encoder.down_path) + len(model.encoder.middle_path) + \
                len(model.encoder.dense_path) - 3) * model.encoder.num_blocks + \
                (len(model.decoder.dense_path) + len(model.decoder.up_path) +\
                 len(model.decoder.final_path) - 3) * model.decoder.num_blocks
    if 'res' in building_block or 'double' in building_block:
        num_layers *= 2
    x = torch.randn( (batch_size, ) + tuple(model.get_input_shape()) )
    y = torch.randn((batch_size, ) + tuple(model.get_output_shape()))
    logger.info("model info:\n{}".format(model))
    logger.info(f'Number of layers: {num_layers}')
    logger.info("Total number of model parameters: {}".format(sum(p.numel() for p in model.parameters())))
    logger.info(f"Input shape: {x.shape}")
    allocated_memory = 0 
    try:
        start_time = time.perf_counter()
        for iter_id in range(num_iteration):
            x, y = x.to(device), y.to(device)
            loss, _ = compute_loss(model, x, y, c = None)
            loss = sum(loss)
            loss.backward()
            allocated_memory += torch.cuda.memory_reserved() / 1024.0 ** 3

        end_time = time.perf_counter()
        train_time = end_time - start_time

        start_time = time.perf_counter()
        with torch.no_grad():
            model.eval()
            for iter_id in range(num_iteration):
                x, y = x.to(device), y.to(device)
                _ = forward_pass(model, x, y, c = None)
        end_time = time.perf_counter()
        infer_time = end_time - start_time
        logger.info('Average allocated memory for each iteration: {}GB'.format(allocated_memory / num_iteration))
        logger.info('Time of training for {} iterations: {}s'.format(num_iteration, train_time))
        logger.info('Time of inference for {} iterations: {}s'.format(num_iteration, infer_time))
        
    except Exception as e:
        logger.error(e)
    del model
    torch.cuda.empty_cache()
    logger.info('------------------------Finish------------------------\n\n\n')