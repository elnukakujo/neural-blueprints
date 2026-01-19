from .base import BaseProjection

import logging
logger = logging.getLogger(__name__)

class TextProjection(BaseProjection):
    def __init__(self, config):
        pass

    def forward(self, x):
        raise NotImplementedError("TextProjection is not yet implemented.")