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
    ### encoder
    en_num_layers = ((len(model.encoder.lf_path)  if hasattr(model.encoder, 'lf_path') else ## pointnet
                ## pointnet 2
                len(model.encoder.msg_path) if hasattr(model.encoder, 'msg_path') else 
                ## seqnet
                len(model.encoder.seq_path))  - 1) * model.encoder.num_blocks + \
                (len(model.encoder.dense_path) if hasattr(model.encoder, 'dense_path') else 1) - 1  ## pointnet and pointnet2        
                 
    ### decoder
    de_num_layers = ((len(model.decoder.dense_path) - 1) if hasattr(model.decoder, 'dense_path') else 0) + \
                (((model.decoder.folding_times + 1) if hasattr(model.decoder, 'folding_times') else ## pointnet with foldingnet
                len(model.decoder.fp_path) if hasattr(model.decoder, 'fp_path') else ## pontnet2
                ## seqnet
                len(model.decoder.seq_path)) - 1) * model.decoder.num_blocks + \
                ((len(model.decoder.final_path) - 2) if hasattr(model.decoder, 'final_path') else 0)  ## pointnet (without folding) and pointnet2
    num_layers = en_num_layers + de_num_layers
    if 'res' in building_block or 'double' in building_block:
        num_layers *= 2
    x = torch.randn( (batch_size, ) + tuple(model.get_input_shape()) )
    y = torch.randn((batch_size, ) + tuple(model.get_output_shape()))
    logger.info("model info:\n{}".format(model))
    logger.info(f'Number of layers: {num_layers}')
    logger.info("Total number of model parameters: {}".format(numel(model)))
    logger.info("Number of encoder parameters: {}".format(numel(model.encoder)))
    for name, layer in model.encoder.named_children():
        logger.info(f'{name}: {numel(layer)}')
    logger.info("Number of decoder parameters: {}".format(numel(model.decoder)))
    for name, layer in model.decoder.named_children():
        logger.info(f'{name}: {numel(layer)}')
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