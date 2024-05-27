from motionnet.models.ptr.ptr import PTR
from motionnet.models.mmTransformer.stacked import STACKED

__all__ = {
    'ptr': PTR,
    'stacked':STACKED,
}


def build_model(config):

    model = __all__[config.method.model_name](
        config=config
    )

    return model
