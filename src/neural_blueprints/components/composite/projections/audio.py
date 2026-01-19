from .base import BaseProjection

import logging
logger = logging.getLogger(__name__)

class AudioProjection(BaseProjection):
    def __init__(self, config):
        pass

    def forward(self, x):
        raise NotImplementedError("AudioProjection is not yet implemented.")