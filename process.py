import string

import numpy as np
import pandas as pd
import settings as S

from collections import Counter, defaultdict
from helpers import make_binary_reviews, split_into_sets, clean_text
from model import Vocabulary
from sklearn.utils import resample, shuffle


class Vectorizer(object):
    def __init__(self, review_df, cutoff=S.VECTORIZER_CUTOFF):
        self.review_vocab = Vocabulary(add_unk=True)
        self.rating_vocab = Vocabulary(add_unk=False)
        
        # add ratings
        for rating in sorted(set(review_df["binary_score"])):
            self.rating_vocab.add_token(rating)
        
        # add top words if count > provided count
        word_counts = Counter()
        for review in review_df["review_text"]:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                if word not in S.STOPWORDS:
                    self.review_vocab.add_token(word)
        
    def vectorize(self, review):
        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)
        
        for token in review.split():
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1
            
        return one_hot

    # show words that appeared frequently enough to
    # appear as keys in review_vocab
    def show_review_vocab_keys(self):
        print("review_vocab keys: ", "\n")
        for key in self.review_vocab.token_to_idx_:
            print(f"{key}")


def annotate_and_clean(df):
    # assign train, val, and test to each observation
    df = split_into_sets(df)
    
    # ensure a clean body of data is fed into vectorizer
    df["review_text"] = df["review_text"].apply(clean_text)

    return df


def balance_data(df):
    # remove class imbalance within sample
    posRating = df[df["binary_score"] == 1].copy()
    negRating = df[df["binary_score"] == 0].copy()
    
    # figure out which of the two labels is less
    # numerous, and undersample the more numerous one
    if len(posRating) >= len(negRating):
        n_equal = len(negRating)
    
    else:
        n_equal = len(posRating)
    
    pos_downsampled = resample(posRating,
                                replace=False,
                                n_samples=n_equal,
                                random_state=1)
    neg_downsampled = resample(negRating,
                                replace=False,
                                n_samples=n_equal,
                                random_state=1)
    balancedSample = pd.concat([pos_downsampled, neg_downsampled])
    balancedSample = shuffle(balancedSample)
    balancedSample.reset_index(drop=True, inplace=True)

    return balancedSample


def get_sample(df):
    print("sampling from & rebalancing data...")
    
    return df.sample(frac=S.SAMPLE_FRAC, random_state=1)


def process_raw_data(data_file):
    print("\n" + "reading in raw txt file...")
    
    # load data into dataframe
    with open(data_file, "r") as f:
        lines = f.readlines()
        data = defaultdict(list)
        for line in lines:
            if ":" in line:
                col, value = line.split(":", 1)
                data[col.strip()].append(value.strip())
    
    review_df = pd.DataFrame(data)
    
    # keep only columns of interest
    review_df.drop(columns=["product/title", "product/price", "review/userId",
                                "review/profileName", "review/helpfulness", "review/time", 
                                "review/summary", "product/productId"],
                        inplace=True)

    # cast review scores into numerical type
    review_df["review/score"] = review_df["review/score"].astype(float)

    # make column names more consistent
    review_df.columns = [ col.replace("/", "_") for col in review_df.columns ]
    
    # generate binary classifications out of review scores
    review_df["binary_score"] = review_df["review_score"].map(make_binary_reviews)

    # drop final unnecessary column
    review_df.drop(columns=["review_score"], inplace=True)

    return review_df


def write(review_df):
    print("writing processed data to csv file..." + "\n")
    review_df.to_csv(S.PROC_DATA_FILE, index=False)


if __name__ == "__main__":
    review_df = process_raw_data(S.RAW_DATA_FILE)
    review_df = get_sample(review_df)
    review_df = balance_data(review_df)
    review_df = annotate_and_clean(review_df)
    write(review_df)