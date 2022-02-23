SEQ_LEN = 128  # maximum sequence length (in tokens) for BERT input

# The BERT preprocessor from TF HUB to use
PREPROCESSOR = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
# The BERT encoder from TF HUB to use
ENCODER = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

# Batch sizes for training and validation
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
