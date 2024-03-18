# Text classifier AI

This is a simple text clasifier

# Installation

Easy to install. First download the latest release source code, then unzip it. Open the folder you have just unzipped in a terminal and run:

```bash
pip install -r "requirements.txt"
```

# How to use

### Preparing data for training

You need a json file with the next format to train

### Training

```
options:
  -h, --help           show this help message and exit
  -tf , --train_file   Specify the path to the training file. This file should be in JSON format containing the training data.
  -n , --name          Specify a dedicated name for your training version. This name will be used to identify your training session.
  -mt , --max_tokens   Set the maximum number of tokens allowed in a sequence. Sequences longer than this will be truncated.
  -ed , --embed_dim    Set the dimensionality of the dense embedding. This parameter defines the size of the vector space in which words will be embedded.
  -oov                 Specify the OOV (out-of-vocabulary) token to replace unknown words in the training data.
  -b , --batch_size    Set the batch size for training. This determines the number of samples processed before the model is updated.
  -e , --epochs        Set the number of epochs for training. An epoch is one complete pass through the entire training dataset.
  -vs , --val_split    Set the validation split ratio. This determines the proportion of the training data to use for validation.
  -s , --shuffle       shuffle the data while training
```

Example use: `python train.py -tf train/custom_data.json -e 100 -b 32`
