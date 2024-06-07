# from celery import Celery
# from celery.result import AsyncResult
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import string
import re
from pytorch_models import training
from pytorch_models import modelV0
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch import nn


# CLEANING TEXT FUNCTIONS:
def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def remove_url(text):
    url_pattern = re.compile(
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    return url_pattern.sub(r"", text)


# converting return value from list to string
def clean_text(text):
    delete_dict = {sp_character: "" for sp_character in string.punctuation}
    delete_dict[" "] = " "
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    # print('cleaned:'+text1)
    textArr = text1.split()
    text2 = " ".join(
        [w for w in textArr if (not w.isdigit() and (not w.isdigit() and len(w) > 2))]
    )

    return text2.lower()


# preparing training data from 'text.csv':

train_data = pd.read_csv(
    r"C:\Users\altav\Downloads\how-does-the-internet-feel (1)\how-does-the-internet-feel\backend\text.csv"
)
train_data.dropna(axis=0, how="any", inplace=True)
train_data["Num_words_text"] = train_data["text"].apply(lambda x: len(str(x).split()))
mask = train_data["Num_words_text"] > 2
train_data = train_data[mask]
print("-------Train data--------")
print(train_data["label"].value_counts())
print(len(train_data))
print("-------------------------")
max_train_sentence_length = train_data["Num_words_text"].max()

test_data = pd.read_csv(
    r"C:\Users\altav\Downloads\how-does-the-internet-feel (1)\how-does-the-internet-feel\backend\text.csv"
)
test_data.dropna(axis=0, how="any", inplace=True)
test_data["Num_words_text"] = test_data["text"].apply(lambda x: len(str(x).split()))

max_test_sentence_length = test_data["Num_words_text"].max()

# only takes text that has more than 2 words
mask = test_data["Num_words_text"] > 2
test_data = test_data[mask]

# apply cleaning
test_data["text"] = test_data["text"].apply(remove_emoji)
test_data["text"] = test_data["text"].apply(remove_url)
test_data["text"] = test_data["text"].apply(clean_text)

# split test and train data
X_train, X_test, y_train, y_test = train_test_split(
    train_data["text"].tolist(),
    train_data["label"].tolist(),
    test_size=0.2,
    stratify=train_data["label"].tolist(),
    random_state=0,
)

train_dat = list(zip(y_train, X_train))
test_dat = list(zip(y_test, X_test))

# tokenizing text
tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_dat), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # declare device and use gpu if avilable

model = modelV0(len(vocab), 128).to(
    device
)  # declare model: takes a len 128 tokenized text

# declare loss and optimizer for model 
#(using crossEntropyLoss for multi var neural network) (lr 0.1 for more accuracy)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# run training loop
# train = training()
# train.train_step(model=model,
#          loss_fn=loss,
#          optimizer=optimizer,
#          train_data=train_dat,
#          test_data=test_dat,
#          vocab=vocab,
#          tokenizer=tokenizer)
# torch.save({'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}, "model_save.pth.tar") 
# save model to "model_save.pth.tar"

path = r"C:\Users\altav\Downloads\how-does-the-internet-feel (1)\how-does-the-internet-feel\backend\model_save.pth.tar"

model.load_state_dict(
    torch.load(
        path
    )["state_dict"]
)  # load model once finished

# os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')
# celery = Celery("worker", broker="redis://localhost:6379/0", backend="redis://localhost:6379/1")
app = Flask(__name__)
CORS(app)


# @celery.task(name="analysis")
# def analysis(search_term):
#     tweets = scrape(search_term)
#     while (len(tweets) == 0):
#         tweets = scrape(search_term)

#     results = analyze(tweets)
#     return results


# analysis and testing output of model
def analysis(text):
    with torch.no_grad():
        input = vocab(tokenizer(text))
        analysis = model(torch.tensor(data=input), torch.tensor(data=[0]))
        print(analysis)

    sentiment_label = {
        5: "sadness",
        4: "joy",
        3: "love",
        2: "anger",
        1: "fear",
        0: "surprise",
    }

    print("This is a %s tweet" % sentiment_label[analysis.argmax(1).item()])
    return analysis.argmax(1).item()


@app.route("/", methods=["POST"])
def main():
    content = request.json
    term = content["search_term"]

    return jsonify({"category": analysis(term)}), 200


# task = analysis.delay(term)
# return jsonify({"task_id": task.id}), 202


# @app.route("/<task_id>")
# def get_status(task_id):
#     task_result = AsyncResult(task_id, app=celery)
#     result = {
#         "task_id": task_id,
#         "task_status": task_result.status,
#         "task_result": task_result.result
#     }

#     return jsonify(result), 200

if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")
