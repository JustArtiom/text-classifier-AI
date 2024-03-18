from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import numpy as np
import argparse
import json
import sys
import os


def readFile(path: str) -> str:
    """
    Read a file lol

    `Parameters:`
        `path` : str, required - Path towards a file to read

    `Returns:`
        `context` - Returns the file context
    """
    with open(path, "r") as f:
        return f.read()


def initialize():
    """
    Initializes the configuration and sets the flags from the arguments inserted by the user

    `Returns:`
        `args` - Namespace returns the flags data

    """
    parser = argparse.ArgumentParser(
        description="Welcome to Simple Text Classifier Training model. Here are the options available")

    parser.add_argument(
        "-m", "--model",
        metavar="",
        required=True,
        help="Specify the path to the training model. For example /models/0"
    )

    parser.add_argument(
        "-q", "--query",
        metavar="",
        default="",
        help="Allow to insert a custom query"
    )

    parser.add_argument(
        "-std",
        action="store_true",
        help="Transfer JSON between a different process trough console"
    )

    return parser.parse_args()


def load_data(path, std):
    """
    Loads the model and the properties of the model that contains the vocabulary

    `Parameters:`
        `path` : str, required - Path towards the model folder
        `std` : boolean, required - Boolean if this function should log json events or not

    `Returns:`
        `model` - Returns the loaded model
        `model_prop` - Returns the model properties
    """
    model = load_model(os.path.join(path, "model.keras"))

    if std:
        print("MESSAGE::" +
              json.dumps({"event": "load_model", "data": "success"}), flush=True)

    model_prop = json.loads(
        readFile(os.path.join(path, "model_properties.json")))

    if std:
        print(
            "MESSAGE::"+json.dumps({"event": "load_model_properties", "data": "success"}), flush=True)

    return model, model_prop


def create_tokenizer(word_index, oov):
    """
    Create a tokenizer from the word_index object so we could use the current vocabulary to tokenise the sentences

    `Parameters:`
        `word_index` : object, required - an object in the next format `{"string": number, ...}`
        `oov` : string, required - The oov (out-of-vocabulary) token to replace unknown words

    `Returns:`
        `tokenizer` - Returns a Tokenizer built class
    """
    tokenizer = Tokenizer(oov_token=oov)
    tokenizer.word_index = word_index
    tokenizer.index_word = {v: i for i, v in word_index.items()}
    return tokenizer


def clasify(model, model_prop, tokenizer, args):
    """
    Clasifies the query

    `Parameters:`
        `model` : Any, required - an imported tenserflow model
        `model_prop` : object, required - the imported model properties json file
        `tokenizer` : Tokenizer, required - the structured tokeniser class fit with data
        `args.query` : string, optional - A query to classify. IF this is not passed an while loop will run and take input from terminal

    `Returns:`
        `response` : object | string - Returns the data depending if args.std is true or false
    """
    message = args.query or input("" if args.std else "Question: ")

    sequences = tokenizer.texts_to_sequences([message.lower()])
    pad_seq = pad_sequences(
        sequences, maxlen=model_prop["config"]["max_tokens"], padding="post")

    untokenised = tokenizer.sequences_to_texts(sequences)
    prediction = model.predict(pad_seq)[0]
    predicted_label_index = np.argmax(prediction)
    label = {v["index"]: i for i, v in model_prop["labels"].items()}[
        predicted_label_index]

    if args.std:
        return "MESSAGE::"+json.dumps({
            "event": "job_complete",
            "data": {
                "query": untokenised[0],
                "label": label,
                "accuracy": str(prediction[predicted_label_index]),
                "message": model_prop["labels"][label]["response"]
            }
        })
    else:
        return "\nQuery: "+untokenised[0]+"\nLabel: "+label+"\nAccuracy: " + str(round(prediction[predicted_label_index] * 100, 2)) + "%" + "\nmessage: " + model_prop["labels"][label]["response"]+"\n"

# Declare the main function


def main():
    args = initialize()
    if args.std:
        print("MESSAGE::" +
              json.dumps({"event": "process_start", "data": "success"}), flush=True)
    try:
        model, model_prop = load_data(args.model, args.std)

        tokenizer = create_tokenizer(
            model_prop["word_index"], model_prop["config"]["oov"])

        if args.query:
            print(clasify(model, model_prop, tokenizer, args), flush=True)
            return
        while True:
            print(clasify(model, model_prop, tokenizer, args), flush=True)
    except Exception as e:
        print("MESSAGE::" +
              json.dumps({"event": "error", "data": str(e)}), flush=True)

    return 0


if __name__ == "__main__":
    main()
