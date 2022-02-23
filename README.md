# tfx-sentiment-analysis

End-to-end *TFX* pipeline for sentiment analysis

## Technical details

* E2E *TFX* pipeline from CSV file to serving endpoint,
* runs locally and on Vertex AI,
* uses [TFX helper library](https://github.com/dlabsai/tfx-helper),
* uses *BERT preprocessor* and *BERT encoder* from *Tensorflow Hub*,
* 100% containerized (no local dependencies required).

## Build
Adjust the location of your image in `Makefile` (`IMAGE` variable).

Use

```sh
make build
```

to build the container image.

Use

```sh
make push
```

to push the built image to external repository.

## Running locally:

1. Adjust your settings in `local_runner.py` and `Makefile`.
1. Build and push updated image.
1. To start a local pipeline run execute:

    ```sh
    make local_pipeline
    ```


## Running on Vertex AI

1. Setup you *Google Cloud Platform* project (see article mentioned in `More info` section).
1. Adjust your setting in `vertex_ai_runner.py` and `Makefile`.
1. Build and push updated image.
1. To schedule a pipeline run on *Vertex AI* execute:

    ```sh
    make vertex_ai_pipeline
    ````

## More info

Link to article describing creation of this *TFX* pipeline for sentiment analysis: [LINK_GOES_HERE](https://dlabs.ai/blog/)
