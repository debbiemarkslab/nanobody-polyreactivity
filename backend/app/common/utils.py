from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import pickle

class SequencesToOneHot():
    def __init__(self, alphabet='protein'):
        if alphabet == 'protein':
            self.aa_list = 'ACDEFGHIKLMNPQRSTVWY'
        else:
            self.aa_list = 'ACGT'
        self.aa_dict = {}
        for i,aa in enumerate(self.aa_list):
            self.aa_dict[aa] = i

    def one_hot_3D(self, s):
        x = np.zeros((len(s), len(self.aa_list)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i , self.aa_dict[letter]] = 1
        return x

    def cdr_seqs_to_arr(self, df_annot, cdr='CDRS_withgaps'):
        onehot_array = np.empty((len(df_annot[cdr]),len(df_annot.iloc[0].loc[cdr]),20))
        for s, seq in enumerate(df_annot[cdr].values):
            if type(seq)==float:
                seq = ''
            seq = seq.upper()
            onehot_array[s] = self.one_hot_3D(seq)
        return onehot_array

class SequencesToOneHot_nonaligned():
    def __init__(self, alphabet='protein', max_len = 39):
        if alphabet == 'protein':
            self.aa_list = 'ACDEFGHIKLMNPQRSTVWY'
        else:
            self.aa_list = 'ACGT'
        self.aa_dict = {}
        for i,aa in enumerate(self.aa_list):
            self.aa_dict[aa] = i
            self.max_len = max_len

    def one_hot_3D(self, s):
        x = np.zeros((len(s), len(self.aa_list)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i , self.aa_dict[letter]] = 1
        return x

    def cdr_seqs_to_arr(self, df_annot, cdr='CDRS_nogaps'):
        onehot_array = np.empty((len(df_annot[cdr]),self.max_len,20))
        for s, seq in enumerate(df_annot[cdr].values):
            if type(seq)==float:
                seq = ''
            new_seq = seq.upper() + '-'*(self.max_len-len(seq))
            onehot_array[s] = self.one_hot_3D(new_seq)
        return onehot_array

class OneHotArrayDataset(Dataset):
    def __init__(self, df, cdr):
        try: df['exp_phenotype_binary']
        except: df['exp_phenotype_binary']=1
        self.samples = []
        k = SequencesToOneHot()
        arr = k.cdr_seqs_to_arr(df,cdr=cdr)
        for x, y in zip(arr, df['exp_phenotype_binary']):
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class NonAlignedOneHotArrayDataset(Dataset):
    def __init__(self, df, cdr,max_len=39):
        try: df['exp_phenotype_binary']
        except: df['exp_phenotype_binary']=1
        self.samples = []
        k = SequencesToOneHot_nonaligned(max_len = max_len)
        arr = k.cdr_seqs_to_arr(df,cdr=cdr)
        for x, y in zip(arr, df['exp_phenotype_binary']):
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def test_rnn(model, test_loader, epoch=1):
    scores = []
    for seq_array, labels in test_loader:
        inputs = seq_array.float()
        labels = labels
        outputs = model(inputs)
        predicted = torch.zeros(outputs.shape[0],outputs.shape[1])
        outputs = outputs.squeeze().tolist()
        if type(outputs) == float:
            outputs = [outputs]
        scores.extend(outputs)
    return scores

def test_cnn(model, test_loader):
    scores = []
    for seq_array, labels in test_loader:
        inputs = torch.reshape(seq_array,(seq_array.shape[0],seq_array.shape[2],seq_array.shape[1])).float()
        labels = labels.reshape(-1,1)
        outputs = model(inputs)
        outputs = outputs.squeeze().tolist()
        if type(outputs) == float:
            outputs = [outputs]
        scores.extend(outputs)
    return scores

def return_scores(test_df,model,filepath,region = 'CDRS_withgaps', model_type = 'cnn', max_len=39):
    if model_type =='cnn':
        test_dataset = OneHotArrayDataset(test_df, region)
    elif model_type =='rnn':
        test_dataset = NonAlignedOneHotArrayDataset(test_df, region,max_len=max_len)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                  batch_size = 64,
                                                  shuffle = False)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    if model_type=='cnn':
        final_scores = test_cnn(model, test_loader)
    elif model_type =='rnn':
        final_scores = test_rnn(model, test_loader)
    return final_scores

def fasta_is_valid(sequences_fasta):
    valid_chars = {'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','U','V','W','Y'}
    if not sequences_fasta:
        return False
    sequences_fasta = sequences_fasta.strip()
    lines = sequences_fasta.split('\n')

    for i in range(0, len(lines), 2):
        if lines[i][0] != '>':
            return False
        if i+1 > len(lines):
            return False
        sequence = lines[i+1].strip()
        if not sequence:
            return False
        for c in sequence:
            if c.upper() not in valid_chars:
                return False
    return True



# def return_scores_sav(df, filepath,batch_size = 1024,cdrs = 'CDRS_nogaps_full'):
#     m = pickle.load(open(filepath,'rb'))
#     df['logistic_regression_3mer_CDRS_full'] = np.zeros(len(df))
#     for batch_tick in np.append(np.arange(batch_size,len(y_s),batch_size),len(y_s)):
#         X_test = cdr_seqs_to_kmer(df['CDRS_nogaps_full'].iloc[batch_tick-batch_size:batch_tick],k=3)
#         y_score = m.decision_function(X_test)
#         df['logistic_regression_3mer_CDRS_full'].iloc[batch_tick-batch_size:batch_tick] = y_score
