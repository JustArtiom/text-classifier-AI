import tensorflow as ts
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os


def readFile(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def initialize():
    config = json.loads(readFile("./config.json"))
    parser = argparse.ArgumentParser(
        description="Welcome to Simple Text Classifier Training model. Here are the options available")

    parser.add_argument(
        "-tf", "--train_file",
        metavar="",
        default=config["train_file"],
        help="Specify the path to the training file. This file should be in JSON format containing the training data."
    )

    parser.add_argument(
        "-n", "--name",
        metavar="",
        default="auto",
        help="Specify a dedicated name for your training version. This name will be used to identify your training session."
    )

    parser.add_argument(
        "-mt", "--max_tokens",
        type=int,
        metavar="",
        default=config["max_tokens"],
        help="Set the maximum number of tokens allowed in a sequence. Sequences longer than this will be truncated."
    )

    parser.add_argument(
        "-ed", "--embed_dim",
        type=int,
        metavar="",
        default=config["embedding_dim"],
        help="Set the dimensionality of the dense embedding. This parameter defines the size of the vector space in which words will be embedded."
    )

    parser.add_argument(
        "-oov",
        metavar="",
        default=config["oov_token"],
        help="Specify the OOV (out-of-vocabulary) token to replace unknown words in the training data."
    )

    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        metavar="",
        default=config["batch_size"],
        help="Set the batch size for training. This determines the number of samples processed before the model is updated."
    )

    parser.add_argument(
        "-e", "--epochs",
        type=int,
        metavar="",
        default=config["epochs"],
        help="Set the number of epochs for training. An epoch is one complete pass through the entire training dataset."
    )

    parser.add_argument(
        "-vs", "--val_split",
        type=float,
        metavar="",
        default=config["validation_split"],
        help="Set the validation split ratio. This determines the proportion of the training data to use for validation."
    )

    parser.add_argument(
        "-s", "--shuffle",
        type=bool,
        metavar="",
        default=config["shuffle"],
        help="shuffle the data while training"
    )

    return parser.parse_args()


def parse_training_data(training_data):
    sentences = []
    labels = []

    for label in training_data:
        if label == "!":
            continue
        for query in training_data[label]["queries"]:
            sentences.append(query.lower())
            labels.append(label.lower())

    return sentences, labels


def create_tokenizer(sentences, oov):
    tokenizer = Tokenizer(oov_token=oov)
    tokenizer.fit_on_texts(sentences)
    return tokenizer


def get_X(sentences, tokenizer, max_tokens):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequence = pad_sequences(
        sequences, maxlen=max_tokens, padding="post")
    return padded_sequence, tokenizer.word_index


def get_Y(labels):
    uniq_lables = list(set(labels))
    label_index = {d: i for i, d in enumerate(uniq_lables)}

    y = np.zeros((len(labels), len(label_index)), dtype=bool)
    for i, label in enumerate(labels):
        y[i, label_index[label]] = 1
    return y, label_index


def create_model(input_dim, embedding_dim, max_tokens, output_size):
    model = Sequential()

    model.add(Embedding(input_dim + 1,
              embedding_dim, input_length=max_tokens))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(output_size, activation="softmax"))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_working_dir(name):
    if not os.path.exists("./models"):
        os.makedirs("./models")

    folder_path = ""
    if name != "auto":
        folder_path = os.path.join("./models", name)
        if os.path.exists(folder_path):
            raise ValueError(f"The folder {folder_path} already exists")
    else:
        folder_number = 0
        while True:
            folder_path = os.path.join("./models", str(folder_number))
            if not os.path.exists(folder_path):
                break
            folder_number += 1

    os.makedirs(folder_path)
    return folder_path


def save_model(model, history, path):
    model.save(os.path.join(path, "model.keras"))

    # Plot and save accuracy graph
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(path, 'accuracy_graph.png'))
    plt.close()

    # Plot and save loss graph
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(path, 'loss_graph.png'))
    plt.close()


def save_json_details(path, args, word_index, label_index):
    with open(os.path.join(path, "model_proprieties.json"), 'w') as file:
        file.write(json.dumps({
            "config": vars(args),
            "word_index": word_index,
            "label_index": label_index
        }, indent=4))


# Declare the main function
def main():
    # Initialize the configuration
    args = initialize()
    training_data = json.loads(readFile(args.train_file))
    sentences, labels = parse_training_data(training_data)
    tokenizer = create_tokenizer(sentences, args.oov)

    X, word_index = get_X(sentences, tokenizer, args.max_tokens)
    y, label_index = get_Y(labels)

    model = create_model(
        input_dim=len(word_index),
        embedding_dim=args.embed_dim,
        max_tokens=args.max_tokens,
        output_size=len(label_index)
    )

    working_dir = get_working_dir(args.name)
    history = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        shuffle=args.shuffle
    )

    save_model(model, history, working_dir)
    save_json_details(working_dir, args, word_index, label_index)

    return 0


if __name__ == "__main__":
    main()
