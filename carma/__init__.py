# -*- coding: utf-8 -*-

__version__ = "0.0.0"

try:
    __CARMA_SETUP__
except NameError:
    __CARMA_SETUP__ = False

if not __CARMA_SETUP__:
    __all__ = []
