#######
# This code was written by Nino Verwei and Niek van Hilten at Leiden University, The Netherlands (23 August 2022)
#
# When using this code, please cite:
# Van Hilten, N., Methorst, J., Verwei, N. and Risselada, H.J., (2022). Physics-based inverse design of lipid packing defect sensing peptides; distinguishing sensors from binders.
#######



import numpy as np


# Preparing data #########################################
def get_data(filename): #obtain data and seperate sequences and fitness
        sequence = []
        fitness = []

        with open(filename, "r") as dataset:
            for line in dataset:
                line = line.strip()

                if not line:
                    continue

                line = line.split()
                sequence.append(line[0])
                fitness.append(line[1])
#                for l in line:
#                    if "ddF=" in l:
#                        fitness.append(l.split("ddF=")[-1])
        
        sequences = np.array(sequence)
        seq_len = len(sequences[0])
        fitness_float =[]  #convert strings of fitness to floats in new list
        for index in range(len(fitness)):
            fitness_float.append(np.float32(fitness[index]))
        fitness = np.array(fitness_float)

        return(sequences, fitness, seq_len)


def create_dict_OHE(alph):  #putt in the different letters that there are, a dictionary with letters and places is returned
    alphabet = sorted(alph)
    # define a mapping of letters to integers
    let_to_vec = dict((letter, i) for i, letter in enumerate(alphabet))
    return(let_to_vec)
    

def convert_data_str_OHE(sequences, let_to_vec, maxlen): #list of sting sequences turned into vectors
    def one_hot_encoding(sequence, let_to_vec): # a single sequence is converted to vectors using the dictionairy above.
        integer_encoded = [let_to_vec[letter] for letter in sequence]
        #print(integer_encoded)
        onehot_encoded = list()

        for value in integer_encoded:
            letter = [np.int32(0) for _ in range(len(let_to_vec))] 
            letter[value] = np.int32(1)
            onehot_encoded.append(letter)
        return(onehot_encoded)

    OHE_sequences = []
    #print(sequences)
    #print(let_to_vec)
    for sequence in sequences:
        if len(sequence) < maxlen:
            sequence += "_" * (maxlen-len(sequence))
        OHE_sequences.append(one_hot_encoding(sequence, let_to_vec))
    OHE_sequences = np.array(OHE_sequences)
    #print(OHE_sequences)
    return(OHE_sequences)
