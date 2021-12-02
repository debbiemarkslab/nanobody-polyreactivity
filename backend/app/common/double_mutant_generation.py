import pandas as pd
import numpy as np
import argparse

'''
script intakes a one sequence csv created by ANARCI/IMGT and outputs all possible double mutants
'''
parser = argparse.ArgumentParser("double mutation generation")
parser.add_argument('--sequence','-s',help = 'csv with one sequence and CDRS already defined via IMGT')
parser.add_argument('--destination','-d',help = 'basename and path of csv output')
input_args = parser.parse_args()

row = pd.read_csv(input_args.sequence).iloc[0]

aa_list = np.asarray(list('ACDEFGHIKLMNPQRSTVWY-'))
def within_CDR(row, double_mutants_dict, seq, i, a, CDR: str):
    '''
    function generates the mutation within the CDR

    INPUTS:
    row: series that contains existing nb sequence partitioned via CDRS
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
            key = row['Id']+f'_{CDR}_{seq[i]}{i+1}{a}_{CDR}_{seq[j]}{j+1}{b}'

            if CDR == 'CDR1':
                mut_cdrs = mut_seq+row['CDR2_withgaps']+row['CDR3_withgaps']
                double_mutants_dict[key] = [mut_seq,row['CDR2_withgaps'],row['CDR3_withgaps'],mut_cdrs]
            elif CDR == 'CDR2':
                mut_cdrs = row['CDR1_withgaps']+mut_seq+row['CDR3_withgaps']
                double_mutants_dict[key] = [row['CDR1_withgaps'],mut_seq,row['CDR3_withgaps'],mut_cdrs]
            else:
                mut_cdrs = row['CDR1_withgaps']+row['CDR2_withgaps']+mut_seq
                double_mutants_dict[key] = [row['CDR1_withgaps'],row['CDR2_withgaps'],mut_seq,mut_cdrs]
    return double_mutants_dict

def between_CDRS(row, double_mutants_dict, seq, i, a, CDR1: str, CDR2: str):
    '''
    function generates the mutation between the CDRS

    INPUTS:
    row: series that contains existing nb sequence partitioned via CDRS
    double_mutants_dict: dictionary of all mutants
    seq: sequence of CDR with single mutants
    i: position where first single mutant is
    a: amino acid at pos i
    CDR1: CDR where we are generating single mutants
    CDR2: CDR where we are generating the additional mutant

    OUTPUTS:
    double_mutants_dict: dictionary of all mutants including the double mutants generated in function
    '''
    seq2 = row[f'{CDR2}_withgaps']
    for j in range(len(seq2)): # iterating through length of CDR2 sequence
        for b in aa_list[aa_list!=seq2[j]]:
            mut_seq = list(seq)
            mut_seq2 = list(seq2)
            mut_seq2[j] = b
            mut_seq = ''.join(mut_seq)
            mut_seq2 = ''.join(mut_seq2)
            if CDR1 =='CDR1':
                if CDR2 == 'CDR2':
                    mut_cdrs = mut_seq+mut_seq2+row['CDR3_withgaps']
                    key = row['Id']+f'_{CDR1}_{seq[i]}{i+1}{a}_{CDR2}_{seq2[j]}{j+1}{b}'
                    double_mutants_dict[key] = [mut_seq,mut_seq2,row['CDR3_withgaps'],mut_cdrs]
                else:
                    mut_cdrs = mut_seq+row['CDR2_withgaps']+mut_seq2
                    key = row['Id']+f'_{CDR1}_{seq[i]}{i+1}{a}_{CDR2}_{seq2[j]}{j+1}{b}'
                    double_mutants_dict[key] = [mut_seq,row['CDR2_withgaps'],mut_seq2,mut_cdrs]
            else:
                mut_cdrs = row['CDR1_withgaps'] + mut_seq + mut_seq2
                key = row['Id']+f'_CDR2_{seq[i]}{i+1}{a}_CDR3_{seq2[j]}{j+1}{b}'
                double_mutants_dict[key] = [row['CDR1_withgaps'],mut_seq,mut_seq2,mut_cdrs]  
                
    return double_mutants_dict


double_mutants_dict = {}
key = row['Id']+'_WT'
double_mutants_dict[key] = [row['CDR1_withgaps'],row['CDR2_withgaps'],row['CDR3_withgaps'],row['CDRS_withgaps']]

seq = row['CDR1_withgaps']
for i in range(len(seq)): # iterating through the length of the CDR1 sequence
    for a in aa_list[aa_list != seq[i]]: # iterating through aa list, excluding the aa at that pos already
        mut_seq = list(seq)
        mut_seq[i] = a
        mut_seq = ''.join(mut_seq)
        mut_cdrs = mut_seq+row['CDR2_withgaps']+row['CDR3_withgaps']

        # putting single mutants in key
        key = row['Id']+f'_CDR1_{seq[i]}{i+1}{a}'
        double_mutants_dict[key] = [mut_seq,row['CDR2_withgaps'],row['CDR3_withgaps'],mut_cdrs]
        
        # generating mutants within CDR1
        double_mutants_dict = within_CDR(row, double_mutants_dict, seq,i,a,'CDR1')

        # generating mutants between CDR1 and CDR2
        double_mutants_dict = between_CDRS(row, double_mutants_dict, seq, i, a, 'CDR1', 'CDR2')
        
        # generating mutants between CDR1 and CDR3
        double_mutants_dict = between_CDRS(row, double_mutants_dict, seq, i, a, 'CDR1', 'CDR3')
         
seq = row['CDR2_withgaps'] 
for i in range(len(seq)): # iterating through entire length of CDR2
    for a in aa_list[aa_list != seq[i]]:
        mut_seq = list(seq)
        mut_seq[i] = a
        mut_seq = ''.join(mut_seq)
        mut_cdrs = row['CDR1_withgaps']+mut_seq+row['CDR3_withgaps']

        # generate single mutants
        key = row['Id']+f'_CDR2_{seq[i]}{i+1}{a}'
        double_mutants_dict[key] = [row['CDR1_withgaps'],mut_seq,row['CDR3_withgaps'],mut_cdrs]
        
        #generate mutants within CDR2
        double_mutants_dict = within_CDR(row, double_mutants_dict, seq, i, a,'CDR2')

        # generating mutants between CDR2 and CDR3
        double_mutants_dict = between_CDRS(row, double_mutants_dict, seq, i, a, 'CDR2', 'CDR3')

seq = row['CDR3_withgaps']
for i in range(len(seq)): # iterating through entire length of CDR3
    for a in aa_list[aa_list != seq[i]]:
        mut_seq = list(seq)
        mut_seq[i] = a
        mut_seq = ''.join(mut_seq)
        mut_cdrs = row['CDR1_withgaps']+row['CDR2_withgaps']+mut_seq
        key = row['Id']+'_CDR3_{}{}{}'.format(seq[i],i+1,a)
        double_mutants_dict[key] = [row['CDR1_withgaps'],row['CDR2_withgaps'],mut_seq,mut_cdrs]

        # generating mutants within CDR3
        double_mutants_dict = within_CDR(row, double_mutants_dict, seq, i, a,'CDR3')

e10_df = pd.DataFrame.from_dict(double_mutants_dict,orient='index',columns=['CDR1_withgaps', 'CDR2_withgaps', 'CDR3_withgaps','CDRS_withgaps'])

e10_df['CDRS_nogaps'] = e10_df['CDRS_withgaps'].str.replace('-','')
e10_df['CDR1_nogaps'] = e10_df['CDR1_withgaps'].str.replace('-','')
e10_df['CDR2_nogaps'] = e10_df['CDR2_withgaps'].str.replace('-','')
e10_df['CDR3_nogaps'] = e10_df['CDR3_withgaps'].str.replace('-','')

e10_df['Id'] = e10_df.index.astype(str)

e10_df['WT_Id'] = e10_df['Id'].str.split('_').str[0]
e10_df['CDRS_nogaps'] = e10_df['CDRS_withgaps'].str.replace('-','')
e10_df['CDR1_nogaps'] = e10_df['CDR1_withgaps'].str.replace('-','')
e10_df['CDR2_nogaps'] = e10_df['CDR2_withgaps'].str.replace('-','')
e10_df['CDR3_nogaps'] = e10_df['CDR3_withgaps'].str.replace('-','')

# labeling if insertion, deletion or missense
e10_df.loc[e10_df.Id.str.contains(r'[^(CDR)]+CDR\d_-\d+\w$'),'mut1_type'] = 'insertion'
e10_df.loc[e10_df.Id.str.contains(r'[^(CDR)]+CDR\d_\w\d+-$'),'mut1_type'] = 'deletion'
e10_df.loc[e10_df.Id.str.contains(r'[^(CDR)]+CDR\d_\w\d+\w$'),'mut1_type'] = 'missense'
e10_df.loc[e10_df.Id.str.contains(r'[^(CDR)]+CDR\d_.\d+.$'),'mut1_loc'] = e10_df.loc[e10_df.Id.str.contains(r'[^(CDR)]+CDR\d_.\d+.$'),'Id'].str.findall(r'[^(CDR)]+CDR\d_.(\d+).$').apply(lambda x: x[0])
e10_df.loc[e10_df.Id.str.contains(r'[^(CDR)]+CDR\d_.\d+.$'),'mut1'] = e10_df.loc[e10_df.Id.str.contains(r'[^(CDR)]+CDR\d_.\d+.$'),'Id'].str.findall(r'[^(CDR)]+CDR\d_.\d+(.)$').apply(lambda x: x[0])

e10_df.loc[e10_df.Id.str.contains(r'CDR\d_-\d+\w_CDR\d_.\d+.'),'mut1_type'] = 'insertion'
e10_df.loc[e10_df.Id.str.contains(r'CDR\d_\w\d+-_CDR\d_.\d+.'),'mut1_type'] = 'deletion'
e10_df.loc[e10_df.Id.str.contains(r'CDR\d_\w\d+\w_CDR\d_.\d+.'),'mut1_type'] = 'missense'
e10_df.loc[e10_df.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'mut1_loc'] = e10_df.loc[e10_df.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'Id'].str.findall(r'CDR\d_.(\d+)._CDR\d_.\d+.').apply(lambda x: x[0])
e10_df.loc[e10_df.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'mut1'] = e10_df.loc[e10_df.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'Id'].str.findall(r'CDR\d_.\d+(.)_CDR\d_.\d+.').apply(lambda x: x[0])

e10_df.loc[e10_df.Id.str.contains(r'CDR\d_.\d+._CDR\d_-\d+\w'),'mut2_type'] = 'insertion'
e10_df.loc[e10_df.Id.str.contains(r'CDR\d_.\d+._CDR\d_\w\d+-'),'mut2_type'] = 'deletion'
e10_df.loc[e10_df.Id.str.contains(r'CDR\d_.\d+._CDR\d_\w\d+\w'),'mut2_type'] = 'missense'
e10_df.loc[e10_df.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'mut2_loc'] = e10_df.loc[e10_df.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'Id'].str.findall(r'CDR\d_.\d+._CDR\d_.(\d+).').apply(lambda x: x[0])
e10_df.loc[e10_df.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'mut2'] = e10_df.loc[e10_df.Id.str.contains(r'CDR\d_.\d+._CDR\d_.\d+.'),'Id'].str.findall(r'CDR\d_.\d+._CDR\d_.\d+(.)').apply(lambda x: x[0])


rows_to_drop = (e10_df['CDR3_withgaps'].str.contains(r'-[^-]-') | e10_df['CDR3_withgaps'].str.contains(r'-[^-][^-]-'))
print(sum(rows_to_drop))
e10_df = e10_df[~rows_to_drop]
e10_df.to_csv(input_args.destination)
