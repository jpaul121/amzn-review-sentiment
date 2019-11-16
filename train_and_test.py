import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import settings as S

from helpers import flatten
from model import Classifier, ReviewDataset
from process import Vectorizer
from torch.utils.data import DataLoader


def run_training_loop(dataset, vectorizer):
    for epoch_index in range(S.NUM_EPOCHS):
        train_state["epoch_index"] = epoch_index
        
        # set up batch generator, initialize loss and 
        # accuracy each outer loop, set train mode on
        dataset.set_split("train")
        dataloader = DataLoader(dataset=dataset,
                                batch_size=S.BATCH_SIZE,
                                drop_last=S.DROP_LAST,
                                shuffle=True)
        
        running_loss = 0.0
        
        classifier.train()
        
        print("\n" + f"EPOCH INDEX: {epoch_index}" + "\n")
        
        for batch_index, batch_dict in enumerate(dataloader):
            print(f"training batch: {batch_index}")
            
            # zero gradients
            optimizer.zero_grad()
            
            # compute output
            y_pred = classifier.forward(x_in=batch_dict["x_data"].float(), dropout=True)
            
            # compute loss
            loss = loss_func(y_pred, batch_dict["y_target"].float())
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)
            
            print(f"avg batch loss: {loss_batch:.4f}" + "\n")
            
            # use loss to produce gradients
            loss.backward()
            
            # use optimizer to take gradient step
            optimizer.step()
            
        train_state["train_loss"].append(running_loss)
        
        # iterate over validation dataset
        
        # set up batch generator, set loss and acc to
        # zero, and set eval mode on
        dataset.set_split("val")
        dataloader = DataLoader(dataset=dataset, batch_size=S.BATCH_SIZE)
        
        running_loss = 0.0
        
        classifier.eval()
        
        for batch_index, batch_dict in enumerate(dataloader):
            print(f"validation batch:  {batch_index}", "\n")
            
            # compute output
            y_pred = classifier.forward(x_in=batch_dict["x_data"])
            
            # compute loss
            loss = loss_func(y_pred.float(), batch_dict["y_target"].float())
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            print(f"avg batch loss: {loss_batch:.4f}" + "\n")

        train_state["val_loss"].append(running_loss)


def setup(vectorizer):
    global classifier, loss_func, optimizer, train_state
    
    # instantiate model
    classifier = Classifier(num_features=len(vectorizer.review_vocab))

    # instantiate loss and optimizer
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=S.LEARNING_RATE)

    # logging, as it were
    train_state = {
        "epoch_index": 0,
        "train_loss": [],
        "val_loss": [],
        "test_loss": -1
    }


def test_model():
    # test model on holdout data
    dataset.set_split("test")
    dataloader = DataLoader(dataset=dataset, batch_size=S.BATCH_SIZE)

    running_loss = 0.0

    classifier.eval()

    for batch_index, batch_dict in enumerate(dataloader):
        # compute the output
        y_pred = classifier(x_in=batch_dict["x_data"].float())
        
        # compute loss
        loss = loss_func(y_pred, batch_dict["y_target"].float())
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_index + 1)
        
    train_state["test_loss"] = running_loss

    # see results of checking loss w/ holdout data
    holdout_loss = train_state["test_loss"]
    print(f"HOLDOUT ACCURACY: {100 * (1 - holdout_loss) :.2f}%" + "\n")


def vectorize_data(data_file):
    # instantiate dataset and vectorizer
    data = pd.read_csv(data_file)
    vectorizer = Vectorizer(data)
    dataset = ReviewDataset(data, vectorizer)
    
    return dataset, vectorizer


if __name__ == "__main__":
    dataset, vectorizer = vectorize_data(S.PROC_DATA_FILE)
    setup(vectorizer)
    run_training_loop(dataset, vectorizer)
    test_model()