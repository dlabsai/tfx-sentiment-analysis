from typing import Dict

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # noqa

from . import constants, features


# TFX Transform will call this function.
def preprocessing_fn(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    # We want to preprocess the input text by a ready-made BERT preprocessor
    preprocessor = hub.KerasLayer(constants.PREPROCESSOR)

    # Reviews are (None, 1) shape, but need to be (None, ) shape
    reviews = tf.squeeze(inputs[features.INPUT_TEXT_FEATURE])
    # Remove any HTML tags from input
    cleaned_reviews = tf.strings.regex_replace(reviews, "<[^>]+>", " ")

    # Run the cleaned texts through BERT preprocessor
    encoder_inputs = preprocessor(cleaned_reviews)

    # This prepares for us 3 tensors (tokens and masks) that we can feed into
    # BERT layer directly later.
    # Bert tokenizer outputs int32 tensors, but for pipeline storage in TFRecord
    # file we need int64, so we convert them here and we will have to convert them
    # back into int32 before feeding BERT model.
    int64_encoder_inputs = {
        key: tf.cast(value, tf.int64) for key, value in encoder_inputs.items()
    }
    # Encode labels from string "pos" / "neg" into 64bit integers 1 / 0
    int64_labels = tf.where(
        inputs[features.LABEL_KEY] == features.POSITIVE_LABEL,
        tf.constant(1, dtype=tf.int64),
        tf.constant(0, dtype=tf.int64),
    )
    # Return both BERT inputs and label
    return {**int64_encoder_inputs, features.LABEL_KEY: int64_labels}
