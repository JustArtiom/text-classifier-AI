from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import numpy as np
import argparse
import random
import json
import sys
import os


def readFile(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def initialize():
    parser = argparse.ArgumentParser(
        description="Welcome to Simple Text Classifier Training model. Here are the options available")

    parser.add_argument(
        "-m", "--model",
        metavar="",
        required=True,
        help="Specify the path to the training model. For example /models/0"
    )

    parser.add_argument(
        "-std",
        metavar="",
        type=bool,
        default=False,
        help="Transfer JSON between a different process trough console"
    )

    return parser.parse_args()


def load_data(path, std):
    model = load_model(os.path.join(path, "model.keras"))

    if std:
        print("MESSAGE::" +
              json.dumps({"event": "load_model", "data": "success"}), flush=True)

    model_prop = json.loads(
        readFile(os.path.join(path, "model_proprieties.json")))

    if std:
        print(
            "MESSAGE::"+json.dumps({"event": "load_model_proprieties", "data": "success"}), flush=True)

    return model, model_prop


def create_tokenizer(word_index, oov):
    tokenizer = Tokenizer(oov_token=oov)
    tokenizer.word_index = word_index
    tokenizer.index_word = {v: i for i, v in word_index.items()}
    return tokenizer


# Declare the main function
def main():
    args = initialize()
    if args.std:
        print("MESSAGE::" +
              json.dumps({"event": "process_start", "data": "success"}), flush=True)
    try:
        model, model_prop = load_data(args.model, args.std)
    except Exception as e:

        if args.std:
            print("MESSAGE::" +
                  json.dumps({"event": "error", "data": str(e)}), flush=True)
        else:
            print(e)

        sys.exit(0)

    tokenizer = create_tokenizer(
        model_prop["word_index"], model_prop["config"]["oov"])

    while True:
        try:
            message = input("" if args.std else "Question: ")

            sequences = tokenizer.texts_to_sequences([message.lower()])
            pad_seq = pad_sequences(
                sequences, maxlen=model_prop["config"]["max_tokens"], padding="post")

            untokenised = tokenizer.sequences_to_texts(sequences)
            prediction = model.predict(pad_seq)[0]
            predicted_label_index = np.argmax(prediction)
            label = {v: i for i, v in model_prop["label_index"].items()}[
                predicted_label_index]

            if args.std:
                print("MESSAGE::"+json.dumps({
                    "event": "job_complete",
                    "data": {
                        "query": untokenised[0],
                        "label": label,
                        "accuracy": str(prediction[predicted_label_index])
                    }
                }), flush=True)
            else:
                print("")
                print("Query: "+untokenised[0])
                print("Label: "+label)
                print("Accuracy: " +
                      str(round(prediction[predicted_label_index] * 100, 2)) + "%")
                print("")
        except EOFError:
            break
        except Exception as e:
            print("MESSAGE::" +
                  json.dumps({"event": "error", "data": str(e)}), flush=True)
            continue

    return 0


if __name__ == "__main__":
    main()
