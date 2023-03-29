#######
# This code was written by Nino Verwei and Niek van Hilten at Leiden University, The Netherlands (23 August 2022)
#
# Input:
#   -s: sequence file with a column of peptide sequences (length 7-24, one letter AA abbreviations)
#   -m: trained CNN model (.h5 file)
#
# Output:
#   prediction.txt file with sequences in the first column and the predicted relative membrane binding free energy (in kJ/mol) in the second column
#
# When using this code, please cite:
# Van Hilten, N.; Methorst, J.; Verwei, N.; Risselada, H.J., Science Advances. 2023, 9(11). DOI: 10.1126/sciadv.ade8839
#######

import pickle
import argparse
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import keras as k
import os 
import sys       

flags = argparse.ArgumentParser()
flags.add_argument("-s", "--seqs", default="sequences.txt", 
            help="Input file with sequences", type=str)
flags.add_argument("-m", "--model", default="best_model.h5", 
            help="Trained CNN model", type=str)
args = flags.parse_args()


ALPHABET = "_ARNDCFQEGHILKMPSTWYV"

sequences = []
for line in open(args.seqs, "r"):
    if not line.startswith("#"):
        line = line.strip()
        sequences.append(line.split()[0])
sequences = np.array(sequences)
print(sequences)

def create_dict_OHE():
    alphabet = sorted(ALPHABET)
    let_to_vec = dict((letter, i) for i, letter in enumerate(alphabet))
    return(let_to_vec)
    

def convert_data_str_OHE(sequences, let_to_vec, maxlen):
    def one_hot_encoding(sequence, let_to_vec):
        integer_encoded = [let_to_vec[letter] for letter in sequence]
        onehot_encoded = list()

        for value in integer_encoded:
            letter = [np.int32(0) for _ in range(len(let_to_vec))] 
            letter[value] = np.int32(1)
            onehot_encoded.append(letter)
        return(onehot_encoded)

    OHE_sequences = []
    OHE_sequences_index = []
    for i, sequence in enumerate(sequences):
        if len(sequence) < maxlen:
            sequence += "_" * (maxlen-len(sequence))
        OHE_sequences.append(one_hot_encoding(sequence, let_to_vec))
        OHE_sequences_index.append(i)

    OHE_sequences = np.array(OHE_sequences)
    return(OHE_sequences, OHE_sequences_index)

X, X_ind = convert_data_str_OHE(sequences, create_dict_OHE(), 24)
model = k.models.load_model(args.model)
y = model.predict(X)
X_sequence = sequences[(X_ind)]
predict_dict = {}
for i in range(len(X)):
    predict_dict[X_sequence[i]] = float(y[i][0])

with open("prediction.txt", "wt") as f:
    for key in predict_dict:
        f.write("{}\t{}\n".format(key, predict_dict[key]))
