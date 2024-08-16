import sys
import logging
from .config import module_config
## log information
loggers = {}

def get_logger(name, level=module_config['logger_level']):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        loggers[name] = logger
        return logger
    
def get_logger_to_file(name, logfile, level=module_config['logger_level']):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        file_handler = logging.FileHandler(logfile)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        loggers[name] = logger
        return logger