import pandas as pd
import subprocess
import pickle
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import re

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

def main():
    seq_strings = '>sample_seq1\nQVQLVESGGGLVQAGGSLRLSCAASGLIFYDNMGWYRQAPGKERELVAAISSSGGSTSYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAADSYPAYLGFVGFDYWGQGTQVTVSS\n>sample_seq2\nQVQLVESGGGLVQAGGSLRLSCAASGFTFVYYVMGWYRQAPGKERELVAAINAGGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCNARVRVRLGRNWSSYYYWGQGTQVTVSS'

    with open('sample_sequences.fa', 'w') as f:
        for l in seq_strings.split('\n'):
            f.write(l+'\n')

    subprocess.run("ANARCI -i sample_sequences.fa -o sample_sequences -s i --csv",shell=True,capture_output=True)

    df = extract_cdrs('sample_sequences_H.csv')

    m = pickle.load(open('./models/logistic_regression_onehot_CDRS.sav', 'rb'))
    X_test = cdr_seqs_to_onehot(df['CDRS_withgaps'])
    y_score = m.decision_function(X_test)
    y_pred = m.predict(X_test)
    df['logistic_regression_onehot_CDRS'] = y_score

    m = pickle.load(open('./models/logistic_regression_3mer_CDRS.sav', 'rb'))
    X_test = cdr_seqs_to_kmer(df['CDRS_nogaps'],k=3)
    y_score = m.decision_function(X_test)
    y_pred = m.predict(X_test)
    df['logistic_regression_3mer_CDRS'] = y_score
    df.to_csv('sample_sequences_scores.csv')

if __name__ == "__main__":
    main()