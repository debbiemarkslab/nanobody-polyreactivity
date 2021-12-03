import pandas as pd
import subprocess
import pickle
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import re
from pathlib import Path
# from app.double_mutant_generation import generate_doubles
# from app.models import CNN, RNN
import torch
'''
script intakes a one sequence csv created by ANARCI/IMGT and outputs all possible double mutants
'''
aa_list = np.asarray(list('ACDEFGHIKLMNPQRSTVWY-'))

def within_CDR(input_seq, double_mutants_dict, seq, i, a, CDR: str):
    '''
    function generates the mutation within the CDR

    INPUTS:
    input_seq: series that contains existing nb sequence partitioned via CDRS
    double_mutants_dict: dictionary of all mutants
    seq: sequence of CDR
    i: position where first single mutant is
    a: amino acid at pos i
    CDR: CDR where we are generating mutants

    OUTPUTS:
    double_mutants_dict: dictionary of all mutants including the double mutants generated in function
    '''
    for j in range(i+1,len(seq)): # for each single mutant in CDR1, iterating through length of CDR1 sequence but starting after pos i
        for b in aa_list[aa_list != seq[j]]:
            mut_seq = list(seq)
            mut_seq[j] = b
            mut_seq = ''.join(mut_seq)
            key = input_seq['Id']+f'_{CDR}_{seq[i]}{i+1}{a}_{CDR}_{seq[j]}{j+1}{b}'

            if CDR == 'CDR1':
                mut_cdrs = mut_seq+input_seq['CDR2_withgaps']+input_seq['CDR3_withgaps']
                double_mutants_dict[key] = [mut_seq,input_seq['CDR2_withgaps'],input_seq['CDR3_withgaps'],mut_cdrs]
            elif CDR == 'CDR2':
                mut_cdrs = input_seq['CDR1_withgaps']+mut_seq+input_seq['CDR3_withgaps']
                double_mutants_dict[key] = [input_seq['CDR1_withgaps'],mut_seq,input_seq['CDR3_withgaps'],mut_cdrs]
            else:
                mut_cdrs = input_seq['CDR1_withgaps']+input_seq['CDR2_withgaps']+mut_seq
                double_mutants_dict[key] = [input_seq['CDR1_withgaps'],input_seq['CDR2_withgaps'],mut_seq,mut_cdrs]
    return double_mutants_dict

def between_CDRS(input_seq, double_mutants_dict, seq, i, a, CDR1: str, CDR2: str):
    '''
    function generates the mutation between the CDRS

    INPUTS:
    input_seq: series that contains existing nb sequence partitioned via CDRS
    double_mutants_dict: dictionary of all mutants
    seq: sequence of CDR with single mutants
    i: position where first single mutant is
    a: amino acid at pos i
    CDR1: CDR where we are generating single mutants
    CDR2: CDR where we are generating the additional mutant

    OUTPUTS:
    double_mutants_dict: dictionary of all mutants including the double mutants generated in function
    '''
    seq2 = input_seq[f'{CDR2}_withgaps']
    for j in range(len(seq2)): # iterating through length of CDR2 sequence
        for b in aa_list[aa_list!=seq2[j]]:
            mut_seq = list(seq)
            mut_seq2 = list(seq2)
            mut_seq2[j] = b
            mut_seq = ''.join(mut_seq)
            mut_seq2 = ''.join(mut_seq2)
            if CDR1 =='CDR1':
                if CDR2 == 'CDR2':
                    mut_cdrs = mut_seq+mut_seq2+input_seq['CDR3_withgaps']
                    key = input_seq['Id']+f'_{CDR1}_{seq[i]}{i+1}{a}_{CDR2}_{seq2[j]}{j+1}{b}'
                    double_mutants_dict[key] = [mut_seq,mut_seq2,input_seq['CDR3_withgaps'],mut_cdrs]
                else:
                    mut_cdrs = mut_seq+input_seq['CDR2_withgaps']+mut_seq2
                    key = input_seq['Id']+f'_{CDR1}_{seq[i]}{i+1}{a}_{CDR2}_{seq2[j]}{j+1}{b}'
                    double_mutants_dict[key] = [mut_seq,input_seq['CDR2_withgaps'],mut_seq2,mut_cdrs]
            else:
                mut_cdrs = input_seq['CDR1_withgaps'] + mut_seq + mut_seq2
                key = input_seq['Id']+f'_CDR2_{seq[i]}{i+1}{a}_CDR3_{seq2[j]}{j+1}{b}'
                double_mutants_dict[key] = [input_seq['CDR1_withgaps'],mut_seq,mut_seq2,mut_cdrs]  
                
    return double_mutants_dict

def generate_doubles(input_seq_df):
    '''
    generates double mutants df
    input: pandas dataframe with one sequence with all cdrs extracted
    output: pandas dataframe of all doubles in same format
    '''
    input_seq = input_seq_df.iloc[0]
    double_mutants_dict = {}
    key = input_seq['Id']+'_WT'
    double_mutants_dict[key] = [input_seq['CDR1_withgaps'],input_seq['CDR2_withgaps'],input_seq['CDR3_withgaps'],input_seq['CDRS_withgaps']]

    seq = input_seq['CDR1_withgaps']
    for i in range(len(seq)): # iterating through the length of the CDR1 sequence
        for a in aa_list[aa_list != seq[i]]: # iterating through aa list, excluding the aa at that pos already
            mut_seq = list(seq)
            mut_seq[i] = a
            mut_seq = ''.join(mut_seq)
            mut_cdrs = mut_seq+input_seq['CDR2_withgaps']+input_seq['CDR3_withgaps']

            # putting single mutants in key
            key = input_seq['Id']+f'_CDR1_{seq[i]}{i+1}{a}'
            double_mutants_dict[key] = [mut_seq,input_seq['CDR2_withgaps'],input_seq['CDR3_withgaps'],mut_cdrs]
            
            # generating mutants within CDR1
            double_mutants_dict = within_CDR(input_seq, double_mutants_dict, seq,i,a,'CDR1')

            # generating mutants between CDR1 and CDR2
            double_mutants_dict = between_CDRS(input_seq, double_mutants_dict, seq, i, a, 'CDR1', 'CDR2')
            
            # generating mutants between CDR1 and CDR3
            double_mutants_dict = between_CDRS(input_seq, double_mutants_dict, seq, i, a, 'CDR1', 'CDR3')
            
    seq = input_seq['CDR2_withgaps'] 
    for i in range(len(seq)): # iterating through entire length of CDR2
        for a in aa_list[aa_list != seq[i]]:
            mut_seq = list(seq)
            mut_seq[i] = a
            mut_seq = ''.join(mut_seq)
            mut_cdrs = input_seq['CDR1_withgaps']+mut_seq+input_seq['CDR3_withgaps']

            # generate single mutants
            key = input_seq['Id']+f'_CDR2_{seq[i]}{i+1}{a}'
            double_mutants_dict[key] = [input_seq['CDR1_withgaps'],mut_seq,input_seq['CDR3_withgaps'],mut_cdrs]
            
            #generate mutants within CDR2
            double_mutants_dict = within_CDR(input_seq, double_mutants_dict, seq, i, a,'CDR2')

            # generating mutants between CDR2 and CDR3
            double_mutants_dict = between_CDRS(input_seq, double_mutants_dict, seq, i, a, 'CDR2', 'CDR3')

    seq = input_seq['CDR3_withgaps']
    for i in range(len(seq)): # iterating through entire length of CDR3
        for a in aa_list[aa_list != seq[i]]:
            mut_seq = list(seq)
            mut_seq[i] = a
            mut_seq = ''.join(mut_seq)
            mut_cdrs = input_seq['CDR1_withgaps']+input_seq['CDR2_withgaps']+mut_seq
            key = input_seq['Id']+'_CDR3_{}{}{}'.format(seq[i],i+1,a)
            double_mutants_dict[key] = [input_seq['CDR1_withgaps'],input_seq['CDR2_withgaps'],mut_seq,mut_cdrs]

            # generating mutants within CDR3
            double_mutants_dict = within_CDR(input_seq, double_mutants_dict, seq, i, a,'CDR3')

    df_double_muts = pd.DataFrame.from_dict(double_mutants_dict,orient='index',columns=['CDR1_withgaps', 'CDR2_withgaps', 'CDR3_withgaps','CDRS_withgaps'])

    df_double_muts['CDRS_nogaps'] = df_double_muts['CDRS_withgaps'].str.replace('-','')
    df_double_muts['CDR1_nogaps'] = df_double_muts['CDR1_withgaps'].str.replace('-','')
    df_double_muts['CDR2_nogaps'] = df_double_muts['CDR2_withgaps'].str.replace('-','')
    df_double_muts['CDR3_nogaps'] = df_double_muts['CDR3_withgaps'].str.replace('-','')

    df_double_muts['Id'] = df_double_muts.index.astype(str)

    df_double_muts['WT_Id'] = df_double_muts['Id'].str.split('_').str[0]
    df_double_muts['CDRS_nogaps'] = df_double_muts['CDRS_withgaps'].str.replace('-','')
    df_double_muts['CDR1_nogaps'] = df_double_muts['CDR1_withgaps'].str.replace('-','')
    df_double_muts['CDR2_nogaps'] = df_double_muts['CDR2_withgaps'].str.replace('-','')
    df_double_muts['CDR3_nogaps'] = df_double_muts['CDR3_withgaps'].str.replace('-','')

    # labeling if insertion, deletion or missense
    df_double_muts.loc[df_double_muts.Id.str.contains(r'[^(CDR)]+CDR\d_-\d+\w$'),'mut1_type'] = 'insertion'
    df_double_muts.loc[df_double_muts.Id.str.contains(r'[^(CDR)]+CDR\d_\w\d+-$'),'mut1_type'] = 'deletion'
    df_double_muts.loc[df_double_muts.Id.str.contains(r'[^(CDR)]+CDR\d_\w\d+\w$'),'mut1_type'] = 'missense'
    df_double_muts.loc[df_double_muts.Id.str.contains(r'[^(CDR)]+CDR\d_.\d+.$'),'mut1_loc'] = df_double_muts.loc[df_double_muts.Id.str.contains(r'[^(CDR)]+CDR\d_.\d+.$'),'Id'].str.findall(r'[^(CDR)]+CDR\d_.(\d+).$').apply(lambda x: x[0])
    df_double_muts.loc[df_double_muts.Id.str.contains(r'[^(CDR)]+CDR\d_.\d+.$'),'mut1'] = df_double_muts.loc[df_double_muts.Id.str.contains(r'[^(CDR)]+CDR\d_.\d+.$'),'Id'].str.findall(r'[^(CDR)]+CDR\d_.\d+(.)$').apply(lambda x: x[0])

    df_double_muts.loc[df_double_muts.Id.str.contains(r'CDR\d_-\d+\w_CDR\d_.\d+.'),'mut1_type'] = 'insertion'
    df_double_muts.loc[df_double_muts.Id.str.contains(r'CDR\d_\w\d+-_CDR\d_.\d+.'),'mut1_type'] = 'deletion'
    df_double_muts.loc[df_double_muts.Id.str.contains(r'CDR\d_\w\d+\w_CDR\d_.\d+.'),'mut1_type'] = 'missense'
    df_double_muts.loc[df_double_muts.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'mut1_loc'] = df_double_muts.loc[df_double_muts.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'Id'].str.findall(r'CDR\d_.(\d+)._CDR\d_.\d+.').apply(lambda x: x[0])
    df_double_muts.loc[df_double_muts.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'mut1'] = df_double_muts.loc[df_double_muts.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'Id'].str.findall(r'CDR\d_.\d+(.)_CDR\d_.\d+.').apply(lambda x: x[0])

    df_double_muts.loc[df_double_muts.Id.str.contains(r'CDR\d_.\d+._CDR\d_-\d+\w'),'mut2_type'] = 'insertion'
    df_double_muts.loc[df_double_muts.Id.str.contains(r'CDR\d_.\d+._CDR\d_\w\d+-'),'mut2_type'] = 'deletion'
    df_double_muts.loc[df_double_muts.Id.str.contains(r'CDR\d_.\d+._CDR\d_\w\d+\w'),'mut2_type'] = 'missense'
    df_double_muts.loc[df_double_muts.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'mut2_loc'] = df_double_muts.loc[df_double_muts.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'Id'].str.findall(r'CDR\d_.\d+._CDR\d_.(\d+).').apply(lambda x: x[0])
    df_double_muts.loc[df_double_muts.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'mut2'] = df_double_muts.loc[df_double_muts.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'Id'].str.findall(r'CDR\d_.\d+._CDR\d_.\d+(.)').apply(lambda x: x[0])


    seqs_to_drop = (df_double_muts['CDR3_withgaps'].str.contains(r'-[^-]-') | df_double_muts['CDR3_withgaps'].str.contains(r'-[^-][^-]-'))
    print(sum(seqs_to_drop))
    df_double_muts = df_double_muts[~seqs_to_drop]
    return df_double_muts
import torch
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=20,out_channels=32,kernel_size=3,stride=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3,stride=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3,stride=3))
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3,stride=1))
        self.linear = torch.nn.Linear(896, 1, bias=False) # 1024 is 128 channels * 8 width
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
#         print(x.shape)
        out = self.conv1(x)
#         print(out.shape)
        out = self.conv2(out)
#         print(out.shape)
        out = self.conv3(out)
#         print(out.shape)
        out = out.view(x.shape[0],out.shape[1]*out.shape[2])
        out = self.linear(out)
#         print(out.shape)
        out = self.sigmoid(out)
#         print(out.shape)
        return out

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        # or:
        #self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  
        # or:
        #out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        
        out = self.sigmoid(out)
        return out
        
def test(model, test_loader):
    scores = []
    for seq_array, labels in test_loader: 
        inputs = torch.reshape(seq_array,(seq_array.shape[0],seq_array.shape[2],seq_array.shape[1])).float()
        labels = labels.reshape(-1,1)
        outputs = model(inputs)
        predicted = torch.zeros(outputs.shape[0],outputs.shape[1])
        for i, out in enumerate(outputs):
            if out > 0.5:
                predicted[i,0]=1
            else:
                predicted[i,0]=0
        predicted = predicted.squeeze()
        outputs = outputs.squeeze().tolist()
        if type(outputs) == float:
            outputs = [outputs]
        scores.extend(outputs)
    return scores

def return_scores(test_df,model,filepath):
    test_dataset = OneHotArrayDataset(test_df,'CDRS_withgaps')
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,  
                                                  batch_size = 64,  
                                                  shuffle = False)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    final_scores = test(model, test_loader)
    return final_scores



def example_function(seq):
    return f'The sequence is: {seq}'

def get_kmer_list(seq, include_framework='',kmer_len=3):
    if 'C' in include_framework:
        seq = 'C' + seq
    if 'W' in include_framework:
        seq = seq + 'W'
    kmer_counts = {}
    k = 1
    while k <= kmer_len:
        num_chunks = (len(seq)-kmer_len)+1
        for idx in range(0,num_chunks):
            kmer = seq[idx:idx+kmer_len]
            assert len(kmer) == kmer_len
            if kmer in kmer_counts:
                kmer_counts[kmer] += 1
            else:
                kmer_counts[kmer] = 1
        k += 1
    return [(key,val) for key,val in kmer_counts.items()]

def cdr_seqs_to_kmer(seqs, include_framework='',k=3):
    seq_to_kmer_vector = {}
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    kmer_list = [aa for aa in aa_list]
    i = 1
    last_kmer_list = kmer_list
    while i < k:
        more_kmer_list = []
        for ii in last_kmer_list:
            for aa in aa_list:
                more_kmer_list.append(ii+aa)
        kmer_list.extend(more_kmer_list)
        last_kmer_list = more_kmer_list
        i += 1
    kmer_to_idx = {aa: i for i, aa in enumerate(kmer_list)}
    for seq in seqs:
        # Make into kmers
        kmer_data_list = get_kmer_list(seq, include_framework=include_framework,kmer_len=k)
        #print(len(kmer_data_list))

        norm_val = 0.
        for kmer,count in kmer_data_list:
            count = float(count)
            norm_val += count # norm
            # norm_val += (count * count) L2 norm
        # norm_val = np.sqrt(norm_val) L2 norm

        # L2 normalize
        final_kmer_data_list = []
        for kmer,count in kmer_data_list:
            final_kmer_data_list.append((kmer_to_idx[kmer],float(count)/norm_val))

        # save to a dictionary
        seq_to_kmer_vector[seq] = final_kmer_data_list

    kmer_arr = np.zeros((len(seqs), len(kmer_to_idx)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        kmer_vector = seq_to_kmer_vector[seq]
        for j_kmer,val in kmer_vector:
            kmer_arr[i, j_kmer] = val
    return kmer_arr

def one_hot_3D(s):
    """ Transform sequence string into one-hot aa vector"""
    # One-hot encode as row vector
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    aa_dict = {}
    for i,aa in enumerate(aa_list):
        aa_dict[aa] = i
    x = np.zeros((len(s), len(aa_list)))
    for i, letter in enumerate(s):
        if letter in aa_dict:
            x[i , aa_dict[letter]] = 1
    return x

def cdr_seqs_to_onehot(seqs):
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    aa_dict = {}
    for i,aa in enumerate(aa_list):
        aa_dict[aa] = i
    onehot_array = np.empty((len(seqs),len(seqs.iloc[0]),20))
    for s, seq in enumerate(seqs):
        onehot_array[s] = one_hot_3D(seq)
    onehot_array = onehot_array.reshape(len(seqs),-1)
    return onehot_array

def withgap_CDR3(seq):
    if len(seq) < 22:
        nogap_seq = seq.replace('-','')
        if len(nogap_seq)%2 == 0:
            gap_seq = nogap_seq[:len(nogap_seq)//2]+'-'*(22-len(nogap_seq))+nogap_seq[-len(nogap_seq)//2:]
        else:
            gap_seq = nogap_seq[:len(nogap_seq)//2+1]+'-'*(22-len(nogap_seq))+nogap_seq[-(len(nogap_seq)//2):]
    else:
        gap_seq = seq.str[:11]+seq.str[-11:]
    return gap_seq

kd = { 'A': 1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C': 2.5,
       'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I': 4.5,
       'L': 3.8,'K':-3.9,'M': 1.9,'F': 2.8,'P':-1.6,
       'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V': 4.2 }
def hp_index(seq,kd=kd):
    hp_ix = 0
    for a in seq:
        hp_ix += kd[a]
    return hp_ix/len(seq)

N_glycosylation_pattern = 'N[^P][ST][^P]'

# N-linked glycosylation
def find_glyc(seq):
    if re.search(N_glycosylation_pattern,seq):
        return True
    else:
        return False

def extract_cdrs(file):
    df = pd.read_csv(file)
    df['CDR1_nogaps'] = df.loc[:,'27':'38'].fillna('-').apply(lambda x: ''.join(x).replace('-',''), axis=1)
    df['CDR2_nogaps'] = df.loc[:,'55':'65'].fillna('-').apply(lambda x: ''.join(x).replace('-',''), axis=1)
    df['CDR3_nogaps'] = df.loc[:,'105':'117'].fillna('-').apply(lambda x: ''.join(x).replace('-',''), axis=1)
    df['CDR1_withgaps'] = df.loc[:,'27':'38'].fillna('-').apply(lambda x: ''.join(x), axis=1)
    df['CDR2_withgaps'] = df.loc[:,'55':'65'].fillna('-').apply(lambda x: ''.join(x), axis=1)
    df['CDR3_withgaps'] = df.loc[:,'105':'117'].fillna('-').apply(lambda x: ''.join(x), axis=1)
    df['CDRS_nogaps'] = df['CDR1_nogaps'] + df['CDR2_nogaps'] + df['CDR3_nogaps']
    df['CDRS_withgaps'] = df['CDR1_withgaps'] + df['CDR2_withgaps'] + df['CDR3_withgaps']
    df['CDR1_withgaps'] = df['CDR1_withgaps'].str[:4]+df['CDR1_withgaps'].str[-4:]
    df['CDR2_withgaps'] = df['CDR2_withgaps'].str[:5]+df['CDR2_withgaps'].str[-4:]
    df['CDR3_withgaps'] = df['CDR3_withgaps'].apply(withgap_CDR3)
    df['CDRS_withgaps'] = df['CDR1_withgaps'] + df['CDR2_withgaps'] + df['CDR3_withgaps']
    df['CDRS_IP'] = df['CDRS_nogaps'].apply(lambda x: ProteinAnalysis(x).isoelectric_point())
    df['CDRS_HP'] = df['CDRS_nogaps'].apply(hp_index)
    df['CDR1_length'] = df['CDR1_nogaps'].str.len()
    df['CDR2_length'] = df['CDR2_nogaps'].str.len()
    df['CDR3_length'] = df['CDR3_nogaps'].str.len()
    df['CDR1_glycosylation'] = df['CDR1_nogaps'].apply(lambda x: find_glyc(x))
    df['CDR2_glycosylation'] = df['CDR2_nogaps'].apply(lambda x: find_glyc(x))
    df['CDR3_glycosylation'] = df['CDR3_nogaps'].apply(lambda x: find_glyc(x))
    return df

async def score_sequences(
    sequences_filepath: str,
    identifier: str,
):
    results_dir = '/nanobody-polyreactivity/results'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    subprocess.run(f'ANARCI -i {sequences_filepath} -o {results_dir}/{identifier} -s i --csv', shell=True, capture_output=True)

    df = extract_cdrs(f'{results_dir}/{identifier}_H.csv')
    if len(df)==1:
        df = generate_doubles(df)
        
    m = pickle.load(open('/nanobody-polyreactivity/app/models/logistic_regression_onehot_CDRS.sav', 'rb'))
    X_test = cdr_seqs_to_onehot(df['CDRS_withgaps'])
    y_score = m.decision_function(X_test)
    y_pred = m.predict(X_test)
    df['logistic_regression_onehot_CDRS'] = y_score

    m = pickle.load(open('/nanobody-polyreactivity/app/models/logistic_regression_3mer_CDRS.sav', 'rb'))
    X_test = cdr_seqs_to_kmer(df['CDRS_nogaps'],k=3)
    y_score = m.decision_function(X_test)
    y_pred = m.predict(X_test)
    df['logistic_regression_3mer_CDRS'] = y_score

    model = CNN()
    filepath = '/nanobody-polyreactivity/app/models/cnn_20.tar'
    df['cnn_20'] = return_scores(df,model,filepath)

    model = CNN()
    filepath = '/nanobody-polyreactivity/app/models/cnn_CDRS_full_10.tar'
    df['cnn_20'] = return_scores(df,model,filepath)

    model = RNN()
    filepath = '/nanobody-polyreactivity/app/models/rnn_20.tar'
    df['cnn_20'] = return_scores(df,model,filepath)

    model = RNN()
    filepath = '/nanobody-polyreactivity/app/models/rnn_CDRS_full_20.tar'
    df['cnn_20'] = return_scores(df,model,filepath)

    results_filepath = f'{results_dir}/{identifier}_scores.csv'
    df.to_csv(results_filepath)
