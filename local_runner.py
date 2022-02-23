import os

import tfx.v1 as tfx
from absl import logging
from tfx_helper.local import LocalPipelineHelper

from pipeline import create_pipeline


def run() -> None:
    """Create and run a pipeline locally."""
    # Read the pipeline artifact directory from environment variable
    output_dir = os.environ["PIPELINE_OUTPUT"]
    # Directory for exporting trained model will be a sub-directory
    # of the pipeline artifact directory.
    serving_model_dir = os.path.join(output_dir, "serving_model")
    # Create pipeline helper instance of local flavour.
    pipeline_helper = LocalPipelineHelper(
        pipeline_name="sentimentanalysis",
        output_dir=output_dir,
        # Where should the model be pushed to
        model_push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    # Read the input CSV files directory from environment variable
    input_dir = os.environ["INPUT_DIR"]
    components = create_pipeline(
        # Pass our pipeline helper instance
        pipeline_helper,
        # The rest of the parameters are pipeline-specific.
        data_path=input_dir,
        # We want to run only a few hyperparameter optimization trials
        number_of_trials=2,
        # We don't aim for superior accuracy
        eval_accuracy_threshold=0.6,
        # Fast tuning runs
        tune_epochs=2,
        tune_patience=1,
        # A bit longer training run
        train_epochs=10,
        train_patience=3,
        use_previous_hparams=False,
    )
    pipeline_helper.create_and_run_pipeline(components)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
