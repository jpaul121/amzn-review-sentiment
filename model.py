import string

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import settings as S

from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, review_df, vectorizer):
        self.review_df = review_df
        self.vectorizer_ = vectorizer
        
        self.train_df = self.review_df[self.review_df["split"] == "train"]
        self.train_size = len(self.train_df)
        
        self.val_df = self.review_df[self.review_df["split"] == "val"]
        self.val_size = len(self.val_df)
        
        self.test_df = self.review_df[self.review_df["split"] == "test"]
        self.test_size = len(self.test_df)
        
        self.lookup_dict_ = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size)
        }
        
        self.set_split("train")    
    
    def set_split(self, split="train"):
        self.target_split_ = split
        self.target_df_, self.target_size_ = self.lookup_dict_[split]
        
    def __len__(self):
        return self.target_size_
    
    def __getitem__(self, index):
        row = self.target_df_.iloc[index]
        
        review_vector = \
            self.vectorizer_.vectorize(row["review_text"])
        
        rating_index = \
            self.vectorizer_.rating_vocab.lookup_token(row["binary_score"])
        
        return {
            "x_data": review_vector,
            "y_target": rating_index
        }
    
    def get_n_batches(self, batch_size):
        return len(self) // batch_size


class Vocabulary(object):
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx_ = token_to_idx
        self.idx_to_token_ = {
            idx: token
            for token, idx in self.token_to_idx_.items()
        }
        self.add_unk_ = add_unk
        self.unk_token_ = unk_token
        self.unk_index = -1
        if self.add_unk_:
            self.unk_index_ = self.add_token(unk_token)
    
    def add_token(self, token):
        if token in self.token_to_idx_:
            index = self.token_to_idx_[token]
        else:
            index = len(self.token_to_idx_)
            self.token_to_idx_[token] = index
            self.idx_to_token_[index] = token
        return index
    
    def lookup_token(self, token):
        if self.add_unk_:
            return self.token_to_idx_.get(token, self.unk_index)
        else:
            return self.token_to_idx_[token]
    
    def lookup_index(self, index):
        if index not in self.idx_to_token_:
            raise KeyError(f"index ({index}) is not in the Vocabulary")
        return self.idx_to_token_[index]
    
    def __str__(self):
        return f"<Vocabulary(size={len(self)})>"
    
    def __len__(self):
        return len(self.token_to_idx_)


class Classifier(nn.Module):
    def __init__(self, num_features, hidden_dim=S.HIDDEN_DIM):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=1)
        
    def forward(self, x_in, dropout=False, dropout_p=S.DROPOUT_P, apply_sigmoid=False):
        intermediate = F.relu(self.fc1(x_in))
        if dropout == True:
            y_out = self.fc2(F.dropout(intermediate, p=dropout_p)).squeeze()
        else:
            y_out = self.fc2(intermediate).squeeze()
        if apply_sigmoid:
            y_out = F.sigmoid(y_out)
        return y_out