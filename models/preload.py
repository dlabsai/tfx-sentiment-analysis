import logging

import tensorflow_hub as hub
import tensorflow_text  # noqa

from . import constants

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Preloading BERT preprocessor")
    hub.KerasLayer(constants.PREPROCESSOR)
    logger.info("Preloading BERT encoder")
    hub.KerasLayer(constants.ENCODER)
    logger.info("Preloading done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
