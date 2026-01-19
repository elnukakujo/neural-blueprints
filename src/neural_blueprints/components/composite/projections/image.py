from .base import BaseProjection

import logging
logger = logging.getLogger(__name__)

class ImageProjection(BaseProjection):
    def __init__(self, config):
        pass

    def forward(self, x):
        raise NotImplementedError("ImageProjection is not yet implemented.")