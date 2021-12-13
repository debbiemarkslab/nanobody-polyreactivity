import torch
import numpy as np
import pandas as pd
from utils import SequencesToOneHot, OneHotArrayDataset
from models import CNN
from utils import test

def return_scores_cnn_20(test_df,model,filepath):
    test_dataset = OneHotArrayDataset(test_df,'CDRS_withgaps')
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,  
                                                  batch_size = 64,  
                                                  shuffle = False)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    final_scores = test(model, test_loader)
    return final_scores