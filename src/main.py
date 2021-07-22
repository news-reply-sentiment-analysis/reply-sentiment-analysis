import sys

from loguru import logger

from model.nscm_test import test

if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    test()
