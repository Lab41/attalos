import attalos.util.log.log as l


logger = l.getLogger(__name__)


class Transformer(object):
    def transform(self, multihot_arr, *args, **kwargs):
        logger.error("Invalid call to transform(). Must override in child class!")
        raise NotImplementedError("Invalid call to transform(). Must override in child class!")
        
    def save_to_file(self, f):
        logger.error("Invalid call to save_to_file(). Must override in child class!")
        raise NotImplementedError("Invalid call to save_to_file(). Must override in child class!")
        
    @classmethod
    def load_from_file(cls, f):
        logger.error("Invalid call to load_from_file(). Must override in child class!")
        raise NotImplementedError("Invalid call to load_from_file(). Must override in child class!")
