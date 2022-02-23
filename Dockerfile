# Start from TFX base image!
FROM gcr.io/tfx-oss-public/tfx:1.6.1

# Install additional dependencies
RUN python -m pip install -q 'tfx[kfp]==1.6.1' tensorflow-text tfx-helper

# Install development tools
RUN python -m pip install ipdb mypy isort flake8


# It takes a while to download BERT, let's display a progress bar during build
ENV TFHUB_DOWNLOAD_PROGRESS=1

# Use ipdb for debugging (insert `breakpoint()` in your code)
ENV PYTHONBREAKPOINT=ipdb.set_trace

# For this project we want to preload BERT preprocessor and encoder into the image

# Copy a minimal subset of file needed for preloading
COPY models/__init__.py models/constants.py models/preload.py ./models/

# Run the preloading script that will pull the BERT files from TF hub
RUN python -m models.preload


# Copy MyPy configuration file
COPY mypy.ini ./mypy.ini

# We copy the pipeline creation code into the image, because we will run
# the pipeline through docker

# Copy the pipeline definition into the image
COPY pipeline.py ./pipeline.py

# Copy the runners into the image
COPY local_runner.py vertex_ai_runner.py ./


# We copy the model code into the image, because TFX will try to import the model
# code during pipeline execution

# Copy your modelling code
COPY models ./models
