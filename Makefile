# The repository address for the docker image
IMAGE="europe-west4-docker.pkg.dev/tfx-article/sentiment-analysis-repo/sentiment-analysis-image:latest"
#  The location of JSON service account credentials file
GOOGLE_APPLICATION_CREDENTIALS="$(shell pwd)/service_account_key.json"

# Serve locally trained model using Tensorflow Serving
serve:
	docker run \
		-it --rm \
		-p 8501:8501 \
		-v "$(shell pwd)/tfx_pipeline_output/serving_model:/models/sentiment_analysis" \
		-e MODEL_NAME=sentiment_analysis \
		tensorflow/serving

# Query local Tensorflow Serving for predictions
query:
	curl \
		-d '{"instances": ["I loved it!", "The movie was bad", "I hated the whole thing", "Best flick ever"]}' \
		-X POST \
		http://localhost:8501/v1/models/sentiment_analysis:predict

# Build the docker image locally
build:
	docker build -t "$(IMAGE)" .

# Push the built docker image to the repository
push:
	docker push "$(IMAGE)"

# Pull the docker image from repository
pull:
	docker pull "$(IMAGE)"

# Run bash in a container with our image
# Expose credentials
bash:
	docker run \
		-it --rm \
		--entrypoint bash \
		--volume "$(GOOGLE_APPLICATION_CREDENTIALS):/creds.json:ro" \
		--env "GOOGLE_APPLICATION_CREDENTIALS=/creds.json" \
		"$(IMAGE)"

# Run the pipeline locally
# Expose pipeline output directory
# Expose credentials
local_pipeline:
	docker run \
		-it --rm \
		--entrypoint python \
		--volume "$(shell pwd)/tfx_pipeline_output:/tfx_pipeline_output:rw" \
		--env "PIPELINE_OUTPUT=/tfx_pipeline_output" \
		--env "INPUT_DIR=gs://tfx-article/input_data/" \
		--volume "$(GOOGLE_APPLICATION_CREDENTIALS):/creds.json:ro" \
		--env "GOOGLE_APPLICATION_CREDENTIALS=/creds.json" \
		"$(IMAGE)" \
		-m local_runner

# Schedule pipeline execution on Vertex AI
# Expose credentials
vertex_ai_pipeline:
	docker run \
		-it --rm \
		--entrypoint python \
		--volume "$(GOOGLE_APPLICATION_CREDENTIALS):/creds.json:ro" \
		--env "GOOGLE_APPLICATION_CREDENTIALS=/creds.json" \
		"$(IMAGE)" \
		-m vertex_ai_runner

# Start a notebook server
# Expose credentials
# Expose notebooks directory
# Caution: turns off notebook auth!
notebook:
	docker run \
		-it --rm \
		--entrypoint jupyter \
		-p 8888:8888 \
		--volume "$(GOOGLE_APPLICATION_CREDENTIALS):/creds.json:ro" \
		--env "GOOGLE_APPLICATION_CREDENTIALS=/creds.json" \
		--volume "$(shell pwd)/nbs:/nbs:rw" \
		"$(IMAGE)" \
		notebook \
		 -y \
		 --no-browser \
		 --allow-root \
		 --ip=0.0.0.0 --port=8888 --port-retries=0 \
		 --notebook-dir=/nbs \
		 --NotebookApp.password='' \
		 --NotebookApp.token=''


# Auto-format code
black:
	docker run \
		-it --rm \
		--entrypoint black \
		--volume "$(shell pwd)/models:/tfx/src/models:rw" \
		--volume "$(shell pwd)/pipeline.py:/tfx/src/pipeline.py:rw" \
		--volume "$(shell pwd)/local_runner.py:/tfx/src/local_runner.py:rw" \
		--volume "$(shell pwd)/vertex_ai_runner.py:/tfx/src/vertex_ai_runner.py:rw" \
		"$(IMAGE)" \
		models/ \
		pipeline.py \
		local_runner.py \
		vertex_ai_runner.py

# Auto-sort imports
isort:
	docker run \
		-it --rm \
		--entrypoint isort \
		--volume "$(shell pwd)/models:/tfx/src/models:rw" \
		--volume "$(shell pwd)/pipeline.py:/tfx/src/pipeline.py:rw" \
		--volume "$(shell pwd)/local_runner.py:/tfx/src/local_runner.py:rw" \
		--volume "$(shell pwd)/vertex_ai_runner.py:/tfx/src/vertex_ai_runner.py:rw" \
		"$(IMAGE)" \
		--profile black \
		--atomic \
		--overwrite-in-place \
		models/ \
		pipeline.py \
		local_runner.py \
		vertex_ai_runner.py

# Lint the code
lint:
	docker run \
		-it --rm \
		--entrypoint flake8 \
		--volume "$(shell pwd)/models:/tfx/src/models:ro" \
		--volume "$(shell pwd)/pipeline.py:/tfx/src/pipeline.py:ro" \
		--volume "$(shell pwd)/local_runner.py:/tfx/src/local_runner.py:ro" \
		--volume "$(shell pwd)/vertex_ai_runner.py:/tfx/src/vertex_ai_runner.py:ro" \
		"$(IMAGE)" \
		--ignore E501 \
		models/ \
		pipeline.py \
		local_runner.py \
		vertex_ai_runner.py


# Type-check the code
mypy:
	docker run \
		-it --rm \
		--entrypoint mypy \
		--volume "$(shell pwd)/models:/tfx/src/models:ro" \
		--volume "$(shell pwd)/pipeline.py:/tfx/src/pipeline.py:ro" \
		--volume "$(shell pwd)/local_runner.py:/tfx/src/local_runner.py:ro" \
		--volume "$(shell pwd)/vertex_ai_runner.py:/tfx/src/vertex_ai_runner.py:ro" \
		"$(IMAGE)" \
		models/ \
		pipeline.py \
		local_runner.py \
		vertex_ai_runner.py
