from typing import Iterable

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from tfx.dsl.components.base.base_component import BaseComponent
from tfx_helper.interface import PipelineHelperInterface


def create_pipeline(
    pipeline_helper: PipelineHelperInterface,
    *,
    data_path: str,  # the directory with training CSV
    number_of_trials: int,  # number of hyperparam tuning trials
    eval_accuracy_threshold: float = 0.7,  # minimal accuracy required to bless the model
    # the proportions of train/validation/test split
    train_hash_buckets: int = 4,
    validation_hash_buckets: int = 1,
    evaluation_hash_buckets: int = 1,
    train_patience: int,  # early stopping patience (in epochs) in trainer
    tune_patience: int,  # early stopping patience (in epochs) in tuner
    train_epochs: int,  # maximum number of training epochs in trainer
    tune_epochs: int,  # maximum number of training epochs in tuner
    # set to `True` to skip tuning in this run and use hyperparams from previous run
    use_previous_hparams: bool,
) -> Iterable[BaseComponent]:
    """Pipeline definition."""

    # Import and split training data from CSV into TFRecord files
    splits = [
        tfx.proto.SplitConfig.Split(name="train", hash_buckets=train_hash_buckets),
        tfx.proto.SplitConfig.Split(name="valid", hash_buckets=validation_hash_buckets),
        tfx.proto.SplitConfig.Split(name="eval", hash_buckets=evaluation_hash_buckets),
    ]
    output_config = tfx.proto.Output(
        split_config=tfx.proto.SplitConfig(splits=splits),
    )
    example_gen = tfx.components.CsvExampleGen(  # type:ignore[attr-defined]
        input_base=data_path, output_config=output_config
    )
    yield example_gen

    # Computes statistics over data for visualization and example validation.
    statistics_gen = tfx.components.StatisticsGen(  # type:ignore[attr-defined]
        examples=example_gen.outputs["examples"]
    )
    yield statistics_gen

    # Generates schema based on statistics files.
    # Since we have a very straightforward text dataset, we can depend
    # on the auto-generated schema.
    # Otherwise one can use `ImportSchemaGen` to import customized schema
    # and `ExampleValidator` to check if examples are conforming to the schema.
    schema_gen = tfx.components.SchemaGen(  # type:ignore[attr-defined]
        statistics=statistics_gen.outputs["statistics"], infer_feature_shape=True
    )
    yield schema_gen

    # Performs data preprocessing and feature engineering
    transform = tfx.components.Transform(  # type:ignore[attr-defined]
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        # the preprocessing callback function can be provided as importable name
        # <package>.<module>.<preprocessing_function_name>
        preprocessing_fn="models.preprocessing.preprocessing_fn",
        splits_config=tfx.proto.SplitsConfig(
            analyze=["train", "valid"],  # fit transformations on part of training data
            transform=["train", "valid", "eval"],  # transform all splits
        ),
    )
    yield transform

    if use_previous_hparams:
        # Find latest best hyperparameters computed in a previous run
        hparams_resolver = tfx.dsl.Resolver(  # type:ignore[attr-defined]
            strategy_class=tfx.dsl.experimental.LatestArtifactStrategy,  # type:ignore[attr-defined]
            hyperparameters=tfx.dsl.Channel(  # type:ignore[attr-defined]
                type=tfx.types.standard_artifacts.HyperParameters
            ),
        ).with_id("latest_hyperparams_resolver")
        yield hparams_resolver
        hparams = hparams_resolver.outputs["hyperparameters"]
    else:
        # Launch hyperparamter tuning to find the best set of hyperparameters.
        tuner = pipeline_helper.construct_tuner(
            tuner_fn="models.model.tuner_fn",
            examples=transform.outputs["transformed_examples"],
            transform_graph=transform.outputs["transform_graph"],
            train_args=tfx.proto.TrainArgs(splits=["train"]),
            eval_args=tfx.proto.EvalArgs(splits=["valid"]),
            custom_config={
                "number_of_trials": number_of_trials,
                "epochs": tune_epochs,
                "patience": tune_patience,
            },
        )
        yield tuner
        hparams = tuner.outputs["best_hyperparameters"]

    # Train a Tensorflow model
    trainer = pipeline_helper.construct_trainer(
        # the training callback function provided as importable name
        run_fn="models.model.run_fn",
        # training will operate on examples already preprocessed
        examples=transform.outputs["transformed_examples"],
        # a Tensorflow graph of preprocessing function is exposed so that it
        # can be embedded into the trained model and used when serving.
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        # use hyperparameters from tuning
        hyperparameters=hparams,
        train_args=tfx.proto.TrainArgs(splits=["train"]),  # split to use for training
        eval_args=tfx.proto.EvalArgs(splits=["valid"]),  # split to use for validation
        # custom parameters to the training callback
        custom_config={"epochs": train_epochs, "patience": train_patience},
    )
    yield trainer

    # Get the latest blessed model for model validation comparison.
    model_resolver = tfx.dsl.Resolver(  # type:ignore[attr-defined]
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,  # type:ignore[attr-defined]
        model=tfx.dsl.Channel(  # type:ignore[attr-defined]
            type=tfx.types.standard_artifacts.Model
        ),
        model_blessing=tfx.dsl.Channel(  # type:ignore[attr-defined]
            type=tfx.types.standard_artifacts.ModelBlessing
        ),
    ).with_id("latest_blessed_model_resolver")
    yield model_resolver

    # Uses TFMA to compute evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compared to a baseline).
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name="from_examples",
                label_key="sentiment",
                preprocessing_function_names=["transform_features"],
            )
        ],
        # Can be used for fairness calculation. Our dataset is pure text,
        # so we run the evaluation only on full dataset.
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        # Metric to use
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            # Require an absolute value of metric to exceed threshold
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": eval_accuracy_threshold}
                            ),
                            # Require the candidate model to be better than
                            # previous (baseline) model by given margin
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-10},
                            ),
                        ),
                    )
                ]
            )
        ],
    )
    evaluator = tfx.components.Evaluator(  # type:ignore[attr-defined]
        examples=example_gen.outputs["examples"],
        example_splits=["eval"],  # split of examples to use for evaluation
        model=trainer.outputs["model"],  # candidate model
        baseline_model=model_resolver.outputs["model"],  # baseline model
        # Change threshold will be ignored if there is no baseline (first run).
        eval_config=eval_config,
    )
    yield evaluator

    # Pushes the model to a file/endpoint destination if checks passed.
    pusher = pipeline_helper.construct_pusher(
        model=trainer.outputs["model"],  # model to push
        model_blessing=evaluator.outputs["blessing"],  # required blessing
    )
    yield pusher
