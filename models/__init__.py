"""Makes any model accessible from the namespace `models`. Taken from
    https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3
"""
from models.basic_model import BasicModel
from models.sklearn_model import SklearnModel

__all__ = [
    "BasicModel",
    "SklearnModel"
]


def make_model(config):
    if config['model_name'] in __all__:
        return globals()[config['model_name']](config)
    else:
        raise Exception('The model name %s does not exist'
                        % config['model_name'])


def get_model_class(config):
    if config['model_name'] in __all__:
        return globals()[config['model_name']]
    else:
        raise Exception('The model name %s does not exist'
                        % config['model_name'])
