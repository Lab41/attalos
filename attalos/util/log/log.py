import logging

DEFAULT_LEVEL = logging.DEBUG
DEFAULT_CONFIG = [
    (logging.StreamHandler(), logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")),
]

DEFAULT_HANDLERS = []
loggers = {}

for handler, formatter in DEFAULT_CONFIG:
    handler.setFormatter(formatter)
    DEFAULT_HANDLERS.append(handler)

def configure(logger, level):
    logger.setLevel(level)
    for handler in DEFAULT_HANDLERS:
        logger.addHandler(handler)
    return logger
        
def getLogger(name, level=DEFAULT_LEVEL):
    if name in loggers:
        return loggers[name]
    else:
        logger = configure(logging.getLogger(name), level)
        loggers[name] = logger
        return logger