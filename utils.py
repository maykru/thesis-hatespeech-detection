from torch.utils.data import Dataset
from typing import List
import numpy as np
import pandas as pd

class DatasetMapper(Dataset):
    def __init__(self, tweet, features, label):
        self.tweet = tweet            
        self.features = features            
        self.label = label
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, idx):
        return  self.tweet[idx], self.features[idx], self.label[idx]
    

def load_data(df: pd.DataFrame, task: str) -> tuple[List[str], np.ndarray, List[str]]:
    text = df["text"].to_list()
    # might need to be modified - listcomp of pad_truncate 
    text = pad_truncate(text)
    features = df.drop(columns=["text", "HS", "TR", "AG"])
    # convert feature array to torch tensor?
    features = features.to_numpy()
    labels = df[f"{task}"].to_list()
    return text, features, labels


def pad_truncate(tweet: List[str]):
    # implement padding and truncating booooy
    # sentences shorter than 13 padded with zeroes, sentences longer truncated to lenght 13 (average sentence length across datasets)
    temp_tweet = list(tweet.split(" "))
    if len(temp_tweet) == 1:
        for x in range(14):
            temp_tweet.append("0")
        tweet = " ".join(tweet)
    if len(temp_tweet) < 13:
        x = 13 - len(temp_tweet)
        for x in range(x):
            temp_tweet.append("0")
        tweet = " ".join(temp_tweet)
        return tweet
    if len(temp_tweet) > 13:
        y = len(temp_tweet) - 13
        tweet = temp_tweet[y:]
        tweet = " ".join(tweet)
        return tweet
    else:
        return tweet


def get_ids_from_words(samples, assignment_dict):
    final_ids = []
    for x in samples:
        temp = x.split()
        ids = []
        for sample in temp:
            if sample in assignment_dict.keys():
                ids.append(assignment_dict[sample])
            else:
                ids.append(0)
        final_ids.append(ids)
    return torch.tensor(final_ids, dtype=torch.long)

caplist = {"EU", "CIA", "FBI", "NSA", "ICE", "GOP", "USA", "US", "UK", "DACA", "POTUS", "NRA", "SNL", "UN", "TV", "OK", "NFL", "<NUM>", "<USER>"}

slang = {
"gonna" : "going to",
"wanna" : "want to",
"imma" : "i am going to",
"i'mma" : "i am going to",
"lmao" : "laughing my ass off",
"lol" : "laughing out loud",
"idc" : "i do not care",
"idk" : "i do not know",
"smh" : "shaking my head",
" u " : " you ",
" ur " : " your ",
" r " : " are ",
"yo" : "your",
"stfu" : "shut the fuck up",
"gtfo" : "get the fuck out",
"ya" : "you",
"da" : "the",
"sum" : "something",
" n " : "and",
" y " : "why",
" gf " : "girlfriend",
" bf " : "boyfriend",
" af " : "as fuck",
"wtf" : "what the fuck",
"lil" : "little"

}

contractions = {
"ain't": "am",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}