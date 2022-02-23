from absl import logging
from tfx_helper.interface import Resources
from tfx_helper.vertex_ai import VertexAIPipelineHelper

from pipeline import create_pipeline


def run() -> None:
    # save pipeline artifacts to Cloud Storage
    output_dir = "gs://tfx-article/pipeline_artifacts"
    # minimal (less than the standard `e2-standard-4`) resource for components
    # that won't execute computations
    minimal_resources = Resources(cpu=1, memory=4)
    # create a helper instance of cloud flavour
    pipeline_helper = VertexAIPipelineHelper(
        pipeline_name="sentimentanalysis",
        output_dir=output_dir,
        google_cloud_project="tfx-article",
        google_cloud_region="europe-west4",
        # all the components will use our custom image for running
        docker_image="europe-west4-docker.pkg.dev/tfx-article/sentiment-analysis-repo/sentiment-analysis-image:latest",
        service_account="pipelines-service-account@tfx-article.iam.gserviceaccount.com",
        # name of the Vertex AI Endpoint
        serving_endpoint_name="sentimentanalysis",
        # NUmber of parallel hyperparameter tuning trails
        num_parallel_trials=2,
        # GPU for Trainer and Tuner components
        trainer_accelerator_type="NVIDIA_TESLA_T4",
        # Machine type for Trainer and Tuner components
        trainer_machine_type="n1-standard-4",
        # GPU for serving endpoint
        serving_accelerator_type="NVIDIA_TESLA_T4",
        # Machine type for serving endpoint
        serving_machine_type="n1-standard-4",
        # Override resource requirements of components. The dictionary key is the ID
        # of the component (usually class name, unless changed with `with_id` method).
        resource_overrides={
            # evaluator needs more RAM than standard machine can provide
            "Evaluator": Resources(cpu=16, memory=32),
            # training is done as Vertex job on a separate machine
            "Trainer": minimal_resources,
            # tuning is done as Vertex job on a separate set of machines
            "Tuner": minimal_resources,
            # pusher is just submitting a job
            "Pusher": minimal_resources,
        },
    )
    components = create_pipeline(
        pipeline_helper,
        # Input data in Cloud Storage
        data_path="gs://tfx-article/input_data/",
        # Run a few hyperparameter tuning trails
        number_of_trials=4,
        # Make the trails short
        tune_epochs=1,
        tune_patience=1,
        # Make the final training long (aim for best accuracy)
        train_epochs=50,
        train_patience=10,
        # After the first successful run you can change this to `True` to skip
        # tuning in subsequent runs.
        use_previous_hparams=False,
    )
    pipeline_helper.create_and_run_pipeline(components, enable_cache=False)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
