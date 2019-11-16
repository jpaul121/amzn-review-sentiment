import re
import torch

import numpy as np
import settings as S
import pandas as pd

from collections import defaultdict


# clean review text for processing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"([,.?!])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z,.?!]+", r" ", text)
    return text


# flatten output of conv layer
def flatten(x):
    shape = torch.prod(torch.tensor(x.shape[1:])).item()
    return x.view(-1, shape)


# group ratings into positive and negative categories
def make_binary_reviews(value):
    if value >= 3.0:
        return 1
    else:
        return 0


# split sample by rating to create training, validation, and test splits
def split_into_sets(input_df):
    by_rating = defaultdict(list)
    
    for _, row in input_df.iterrows():
        by_rating[row["binary_score"]].append(row.to_dict())
    
    final_list = []
    
    np.random.seed(S.SEED)
    
    for _, item_list in sorted(by_rating.items()):
        np.random.shuffle(item_list)
        
        n_total = len(item_list)
        n_train = int(.7 * n_total)
        n_val = int(.15 * n_total)
        
        # label each observation according to split
        for item in item_list[:n_train]:
            item["split"] = "train"
        for item in item_list[n_train:n_train+n_val]:
            item["split"] = "val"
        for item in item_list[n_train+n_val:]:
            item["split"] = "test"
        
        # add to final list
        final_list.extend(item_list)
    
    output_df = pd.DataFrame(final_list)
    
    return output_df