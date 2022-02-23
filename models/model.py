from typing import Any, Dict, List

import keras_tuner
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_transform as tft
from absl import logging
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx import v1 as tfx
from tfx_bsl.public import tfxio

from . import constants, features


def _get_tf_examples_serving_signature(
    model: tf.keras.Model,
    schema: schema_pb2.Schema,
    tf_transform_output: tft.TFTransformOutput,
) -> Any:
    """
    Returns a serving signature that accepts `tensorflow.Example`.

    This signature will be used for evaluation or bulk inference.
    """

    @tf.function(
        # Receive examples packed into bytes (unparsed)
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")]
    )
    def serve_tf_examples_fn(
        serialized_tf_example: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Returns the output to be used in the serving signature."""
        # Load the schema of raw examples.
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        # Remove label feature since these will not be present at serving time.
        raw_feature_spec.pop(features.LABEL_KEY)
        # Parse the examples using schema into raw features
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)

        # Preprocess the raw features
        transformed_features = model.tft_layer(raw_features)

        # Run preprocessed inputs through the model to get the prediction
        outputs = model(transformed_features)

        return {features.LABEL_KEY: outputs}

    return serve_tf_examples_fn


def _get_text_serving_signature(
    model: tf.keras.Model,
    schema: schema_pb2.Schema,
    tf_transform_output: tft.TFTransformOutput,
) -> Any:
    """
    Returns a serving signature that accepts flat text inputs.

    This signature will be used for online predictions.
    """

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                # Receive texts to analyze as plain string
                shape=[None],
                dtype=tf.string,
                name=features.INPUT_TEXT_FEATURE,
            )
        ]
    )
    def serve_text_fn(text_tensor: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Returns the output to be used in the serving signature."""

        # The transform layer expects (batch, 1) shape tensor, but our flat inputs
        # are (batch, ) shape, so we need to add a dimension.
        # We also simulate the same features structure as in the training set.
        input_features = {features.INPUT_TEXT_FEATURE: tf.expand_dims(text_tensor, 1)}

        # Preprocess the raw features
        transformed_features = model.tft_layer(input_features)

        # Run preprocessed inputs through the model to get the prediction
        outputs = model(transformed_features)

        # the outputs are of (batch, 1) shape, but for maximum simplicity we want
        # to return (batch, ) shape, so we eliminate the extra dimension.
        return {features.LABEL_KEY: tf.squeeze(outputs)}

    return serve_text_fn


def _get_transform_features_signature(
    model: Any, schema: schema_pb2.Schema, tf_transform_output: tft.TFTransformOutput
) -> Any:
    """
    Returns a serving signature that applies tf.Transform to features.

    This signature can be used to pre-process the inputs.
    """

    @tf.function(
        # Receive examples packed into bytes (unparsed)
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")]
    )
    def transform_features_fn(serialized_tf_example: tf.TensorSpec) -> Any:
        """Returns the transformed features."""

        # Load the schema of raw examples.
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        # Parse the examples using schema into raw features
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)

        # Preprocess the raw features
        transformed_features = model.tft_layer(raw_features)

        return transformed_features

    return transform_features_fn


def _input_fn(
    file_pattern: List[str],  # TFRecord file names
    data_accessor: tfx.components.DataAccessor,  # TFX dataset loading helper
    schema: schema_pb2.Schema,  # Dataset schema
    batch_size: int,  # batch size
) -> tf.data.Dataset:
    """Generates features and label for tuning/training."""
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size,
            # automatically pop the label so the dataset contains tuples (features, labels)
            label_key=features.LABEL_KEY,
            # We say that the whole dataset is a single epoch. After the first epoch
            # of training Tensorflow will remember the size of the dataset
            # (number of batches in epoch) and will provide correct ETA times
            num_epochs=1,
        ),
        schema,  # parse the examples into schema generated by SchemaGen component
    )


def _get_hyperparameters() -> keras_tuner.HyperParameters:
    """Returns hyperparameters for building Keras model."""
    hp = keras_tuner.HyperParameters()
    # Defines search space.
    hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4], default=1e-3)
    hp.Choice("hidden_size", [64, 128, 256], default=128)
    return hp


def _build_keras_model(hparams: keras_tuner.HyperParameters) -> tf.keras.Model:
    """Creates a DNN Keras model for classifying text."""
    # Input are matching outputs of BERT preprocessor.
    # Since our training examples are stored in TFRecord format
    # the inputs are coming as int64.
    inputs = {
        "input_type_ids": tf.keras.layers.Input(
            shape=(constants.SEQ_LEN,), name="input_type_ids", dtype=tf.int64
        ),
        "input_mask": tf.keras.layers.Input(
            shape=(constants.SEQ_LEN,), name="input_mask", dtype=tf.int64
        ),
        "input_word_ids": tf.keras.layers.Input(
            shape=(constants.SEQ_LEN,), name="input_word_ids", dtype=tf.int64
        ),
    }

    # BERT requires int32 inputs. We were explicitely converting them to int64
    # in preprocessing, so converting back does not loose any information.
    encoder_inputs = {
        key: tf.cast(value, dtype=tf.int32) for key, value in inputs.items()
    }

    # Run the inputs through BERT and get only the pooled (summary) output.
    encoder = hub.KerasLayer(
        constants.ENCODER,
        trainable=False,  # we don't allow BERT fine-tuning
    )
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]

    # Then we run the BERT outputs through a simple neural net
    # with one hidden layer and ReLU activation
    hidden = tf.keras.layers.Dense(hparams.get("hidden_size"), activation="relu")(
        pooled_output
    )

    # And we expect a single output in range <0, 1>
    predictions = tf.keras.layers.Dense(1, activation="sigmoid")(hidden)

    # Compile the model
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(
        loss="binary_crossentropy",  # loss appropriate for binary classification
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams.get("learning_rate")),
        metrics=["accuracy"],  # additional metrics to compute during training
    )
    model.summary(print_fn=logging.info)
    return model


def _build_keras_model_with_strategy(
    hparams: keras_tuner.HyperParameters,
) -> tf.keras.Model:
    gpus = tf.config.list_logical_devices("GPU")
    logging.info("Available GPU devices %r", gpus)
    cpus = tf.config.list_logical_devices("CPU")
    logging.info("Available CPU devices %r", cpus)

    # The strategy should pick all GPUs if available, otherwise all CPUs automatically
    with tf.distribute.MirroredStrategy().scope():
        return _build_keras_model(hparams)


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs) -> None:
    """Train the model based on given args."""
    # Load the Transform component output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    # The output contains among other attributs the schema of transformed examples
    schema = tf_transform_output.transformed_metadata.schema

    # Choose batch sizes
    train_batch_size = constants.TRAIN_BATCH_SIZE
    eval_batch_size = constants.EVAL_BATCH_SIZE

    # Load transformed examples as tf.data.Dataset
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        batch_size=train_batch_size,
    )
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        schema,
        batch_size=eval_batch_size,
    )

    # Load best set of hyperparameters from Tuner component
    assert fn_args.hyperparameters, "Expected hyperparameters from Tuner"
    hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
    logging.info("Hyper parameters for training: %s", hparams.get_config())

    # Build the model using the best hyperparameters
    model = _build_keras_model_with_strategy(hparams)

    # Write logs together with model, so that we can later view the training
    # curves in Tensorboard.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq="batch"
    )

    # Configure early stopping on validation loss increase
    epochs: int = fn_args.custom_config["epochs"]
    patience: int = fn_args.custom_config["patience"]
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, verbose=1, restore_best_weights=True
    )

    # Train the model in the usual Keras fashion
    model.fit(
        train_dataset,
        # steps_per_epoch=fn_args.train_steps,
        epochs=epochs,
        validation_data=eval_dataset,
        # validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, early_stopping_callback],
    )

    # We need to manually add the transformation layer to the model instance
    # in order for it to be tracked by the model and included in the saved model format.
    model.tft_layer = tf_transform_output.transform_features_layer()

    # Create signature (endpoints) for the model.
    signatures = {
        # What to do when serving from Tensorflow Serving
        "serving_default": _get_text_serving_signature(
            model, schema, tf_transform_output
        ),
        # What do do when processing serialized Examples in TFRecord files
        "from_examples": _get_tf_examples_serving_signature(
            model, schema, tf_transform_output
        ),
        # How to perform only preprocessing.
        "transform_features": _get_transform_features_signature(
            model, schema, tf_transform_output
        ),
    }

    # Save the model in SavedModel format together with the above signatures.
    # This saved model will be used by all other pipeline components that require
    # a model (for example Evaluator or Pusher).
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)


# TFX Tuner will call this function.
def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
    """Build the tuner using the KerasTuner API."""

    # Number of different hyperparameter combinations to check
    number_of_trials: int = fn_args.custom_config["number_of_trials"]

    # RandomSearch is a subclass of keras_tuner.Tuner which inherits from
    # BaseTuner.
    tuner = keras_tuner.RandomSearch(
        _build_keras_model_with_strategy,  # model building callback
        max_trials=number_of_trials,  # number of trials to perform
        hyperparameters=_get_hyperparameters(),  # hyperparameter search space
        allow_new_entries=False,  # don't allow requesting parameters outside the search space
        # We want to choose the set of hyperparms that causes fastest convergence
        # so we will select validation loss minimalization as objective.
        objective=keras_tuner.Objective("val_loss", "min"),
        directory=fn_args.working_dir,  # operating directory
        project_name="sentiment_analysis_tuning",  # will be used to add prefix to artifacts
    )

    # Load the Transform component output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    # Get the preprocessed inputs schema
    schema = tf_transform_output.transformed_metadata.schema

    # Load datasets
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        constants.TRAIN_BATCH_SIZE,
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, fn_args.data_accessor, schema, constants.EVAL_BATCH_SIZE
    )

    # Configure early stopping
    epochs: int = fn_args.custom_config["epochs"]
    patience: int = fn_args.custom_config["patience"]
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, verbose=1, restore_best_weights=True
    )

    # Request hyperparameter tuning
    return tfx.components.TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "epochs": epochs,
            "callbacks": [early_stopping_callback],
        },
    )
