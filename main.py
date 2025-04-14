import numpy as np
import pandas as pd
from model import CNN
import argparse
from typing import List
from torch.utils.data import DataLoader
import torch
from utils import *
from evaluate import *
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="bilingual hate speech detection")
    parser.add_argument("--task", type=str, default="HS", help="select HS, TR, or AG as task")
    parser.add_argument("--lang", type=str, default="en", help="select en or es as lang")
    args = parser.parse_args()
    return args


def get_embeddings(glove_path, df, dimension):
    # get glove embeddings and create dictionary
    glove = pd.read_csv(glove_path, sep=" ", quoting=3, header=None, index_col=0)
    glove_embedding = {key: val.values for key, val in glove.T.items()}

    # fit embedding matrix to training vocabulary
    text = df["text"].tolist()
    tokenizer=tf.keras.preprocessing.text.Tokenizer(split=" ")
    tokenizer.fit_on_texts(text)
    word_index = tokenizer.word_index

    embedding_matrix=np.zeros((len(word_index)+1, dimension))
    for word,index in word_index.items():
        if word in glove_embedding:
            embedding_matrix[index]=glove_embedding[word]
    return embedding_matrix


def main():
    args = parse_args()
    lang = args.lang
    task = args.task

    train_df = pd.read_csv(path=f"data/preprocessed/p_{lang}_train.csv", delimiter=",", na_filter=False, encoding="utf-8")
    test_df = pd.read_csv(path=f"data/preprocessed/p_{lang}_test.csv", delimiter=",", na_filter=False, encoding="utf-8")

    # get embedding matrix - adjust args and method if needed
    embedding_matrix = get_embeddings()

    # load data and get word IDs
    train_text, train_features, train_labels = load_data(train_df, task=args.task)
    test_text, test_features, test_labels = load_data(test_df, task=args.task)
    
    train_ids = get_ids_from_words(train_text)
    test_ids = get_ids_from_words(test_text)

    train = DatasetMapper(train_ids, train_features, train_labels)
    # test = DatasetMapper(test_ids, test_features, test_labels)
    loader_train = DataLoader(train, batch_size=32)
    # loader_test = DataLoader(test, batch_size=32)
    
    # create model and train it
    model = CNN(embedding_matrix, 24, 3)
    train(model, loader_train)

    # evaluation
    pred_labels = model(test_ids, test_features)
    predictions_test_binary = list()

    for x in pred_labels:
        y = torch.round(x)
        y = y.item()
        predictions_test_binary.append(y)

    acc_hs, p_hs, p_nohs, r_hs, r_nohs, f1_hs, f1_nohs, p_macro, r_macro, f1_macro = evaluate_a(predictions_test_binary, test_df)
    print("\t".join(["{}".format(x) for x in ["acc.", "P (1)", "P (0)", "R (1)", "R (0)", "F1 (1)", "F1 (0)", "P (avg)", "R (avg)", "F1 (avg)"]]))
    print("\t".join(["{0:.3f}".format(x) for x in [acc_hs, p_hs, p_nohs, r_hs, r_nohs, f1_hs, f1_nohs, p_macro, r_macro, f1_macro]]))

    # once training is complete, pickle model
    filename = f"{lang}_{task}_model.sav"
    pickle.dump(model, open(filename, "wb"))


if __name__ == "__main__":
    main()
