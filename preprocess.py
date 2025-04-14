import argparse
import pandas as pd
import re
import nltk
import spacy
import csv
import re
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import string
import emoji
import ftfy
# pypi pyspellchecker
from spellchecker import SpellChecker
from wordsegment import load, segment
from utils import hashtags as hs, caplist, slang, contractions as con
from nltk.sentiment import SentimentIntensityAnalyzer
from sentiment_analysis_spanish import sentiment_analysis
from typing import List


def lemmatise(tweet: str, nlp) -> str:
    doc = nlp(tweet)
    result = [token.lemma_ for token in doc]  # lemmatization
    tweet = " ".join(result)
    return tweet


def removal(tweet: str) -> str:
    tweet = re.sub(r"http\S+", "", tweet)                               # remove URLS
    tweet = re.sub(r"\S*@\S*\s?", "", tweet)                            #remove emails
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))  #remove punctuation, most non-alphanumerical chars
    return tweet


def demoji(tweet: str, lang: str) -> str:
    if lang == "en":
        tweet = (emoji.demojize(tweet))
    else:
        tweet = (emoji.demojize(tweet, language="es"))
    return tweet


def remove_stop(tweet: str, stopword_list: List[str]) -> str:
    tweet_tok = word_tokenize(tweet)
    my_list = [word for word in tweet_tok if word not in stopword_list]
    stoppito = " ".join(my_list)
    return stoppito


def replace(tweet: str) -> str:
    tweet = re.sub("@[A-Za-z0-9_]+", "user", tweet)     # replace mentions with USER
    tweet = re.sub(r"\d+", "num", str(tweet))           # replace digits with 'NUM'
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)           # replace repeating chars w 2 occurrences
    return tweet


def contractions(tweet: str, con_list: List[str], lang: str) -> str:        # normalises contractions, e.g. I've -> I have
    if lang == "en":
        for key, value in con_list:
            tweet = tweet.replace(key, value)
    return tweet


def fixmalformed(tweet: str) -> str:
    tweet = ftfy.fix_text(tweet)
    tweet = re.sub("&;", "and", tweet)
    return tweet


def spellcheck(tweet, spell) -> str:
    tweet = tweet.split()
    tweet = [spell.correction(x) for x in tweet if x != "user"]
    return " ".join(tweet)


# def spellcheck_es(tweet, spell) -> str:
#     result = []
#     tweet = tweet.split()
#     for x in tweet:
#         if x != "user":
#             result.append(spell.correction(x))
#         else:
#             result.append(x)
#     return " ".join(result)


def flag_all_caps(tweet: str, nlp) -> int:
    caps = re.findall('([A-Z]+(?:(?!\s?[A-Z][a-z])\s?[A-Z])+)', tweet)
    caps = " ".join(caps)
    for key in caplist:  # no caplist for spanish
        caps = caps.replace(key, "")
    caps = " ".join(caps.split())
    doc = nlp(caps)
    counter = 0
    for token in doc:
        counter = counter + 1
    return counter


def normalise_whitespace(tweet: str) -> str:
    return " ".join(tweet.split())


def hashtagsegmentation(tweet, hashlist, lang) -> str:
    if lang == "en":
        for key, value in hashlist:                 # no hashlist for spanish
            tweet = tweet.replace(key, value)
        hashtags = re.findall(r"(#\w+)", tweet)
        for x in hashtags:
            temp = " ".join(segment(x))
            tweet = tweet.replace(x, temp)
    return tweet


def sentimentanalysis(tweet, sentiment, lang: str):
    if lang == "en":
        return sentiment.polarity_scores(tweet)["compound"]
    return sentiment.sentiment(tweet)


def punctuation(tweet: str) -> str:
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    return tweet


def separate_words(text: str) -> str:
    text = text.split()
    A = text[:len(text) // 2]
    A = " ".join(A)
    B = text[len(text) // 2:]
    B = " ".join(B)
    a = (segment(A))
    b = (segment(B))
    tweet = a + b
    tweet = " ".join(tweet)
    return tweet


def replace_slang(tweet, slanglist, lang):
    if lang == "en":
        for key, value in slanglist:
            tweet = tweet.replace(key, value)
    return tweet


def check_hatebase(tweet, hatebase):
    counter = 0
    for i in range(len(hatebase)):
        if hatebase.loc[i, "term"] in tweet:
            counter = counter + 1
    return counter


def load_data(lang: str) -> (pd.DataFrame, pd.DataFrame):
    if lang == "es":
        train_path = "data/original/es/es_train.csv"
        test_path = "data/original/es/es_test.csv"
    else:
        train_path = "data/original/en/en_train.csv"
        test_path = "data/original/en/en_test.csv"
    train_df = pd.read_csv(train_path, delimiter=",", na_filter=False, encoding="utf-8")
    test_df = pd.read_csv(test_path, delimiter=",", na_filter=False, encoding="utf-8")
    return train_df, test_df


def parse_args():
    parser = argparse.ArgumentParser(description="bilingual hate speech detection")
    parser.add_argument("--lang", type=str, default="en", help="select en or es as language")
    args = parser.parse_args()
    return args


def main():
    load()
    args = parse_args()
    lang = args.lang

    if lang == "es":
        stops = stopwords.words("spanish")
        nlp = spacy.load("es_core_web_md")
        spell = SpellChecker(language="es")
        sentiment = sentiment_analysis.SentimentAnalysisSpanish()
        hatebase = pd.read_csv("data/original/es/es_hatebase.csv", delimiter=",", na_filter=False, encoding="utf-8")
    elif lang == "en":
        stops = stopwords.words('english')
        nlp = spacy.load("en_core_web_sm")
        spell = SpellChecker(language="en")
        sentiment = SentimentIntensityAnalyzer()
        hatebase = pd.read_csv("data/original/en/en_hatebase.csv", delimiter=",", na_filter=False, encoding="utf-8")
    else:
        raise ValueError("Please specify lang as either es or en.")

    train, test = load_data(lang)

    hashlist = hs.items()
    listcon = con.items()
    slanglist = slang.items()

    def apply(df: pd.DataFrame) -> pd.DataFrame:
        df["text"] = df["text"].apply(fixmalformed)
        df["text"] = df["text"].apply(removal)
        df["text"] = df["text"].apply(replace)
        df["caps"] = df["text"].apply(lambda x: flag_all_caps(x, nlp))
        # df["text"] = df["text"].apply(lambda x: hashtagsegmentation(x, hashlist, lang))
        df["text"] = df["text"].apply(lambda x: demoji(x, lang))
        df["text"] = df["text"].apply(normalise_whitespace)
        df["sntmt"] = df["text"].apply(lambda x: sentimentanalysis(x, sentiment, lang))
        df["text"] = df["text"].apply(lambda x: x.lower())
        # df["text"] = df["text"].apply(lambda x: replace_slang(x, slanglist, lang))
        # df["text"] = df["text"].apply(lambda x: contractions(x, listcon, lang))
        # print(df["text"][0])
        # df["text"] = df["text"].apply(lambda x: spellcheck(x, spell))
        # df["text"] = df["text"].apply(separate_words)
        # df["text"] = df["text"].apply(lambda x: lemmatise(x, nlp))
        df["text"] = df["text"].apply(lambda x: remove_stop(x, stops))
        # df["text"] = df["text"].apply(punctuation)
        df["text"] = df["text"].apply(normalise_whitespace)
        df["nohs"] = df["text"].apply(lambda x: check_hatebase(x, hatebase))
        return df

    train = apply(train)
    test = apply(test)

    train.to_csv(f"preprocessed_{lang}_train.csv", index=False, encoding="utf-8", sep=",")  # change name accordingly
    test.to_csv(f"preprocessed_{lang}_test.csv", index=False, encoding="utf-8", sep=",")


if __name__ == "__main__":
    main()
