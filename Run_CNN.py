#######
# This code was written by Nino Verwei and Niek van Hilten at Leiden University, The Netherlands (23 August 2022)
# A convolutional neural network (CNN) is trained on MD data (data.txt) of helical peptide sequences with respective relative free energy values as a measure for lipid packing defect sensing
# The trained model can be used to predict relative membrane binding free energy (lipid packing defect sensing) for any given sequence (assuming helical folding and length 7-24)
# Optimized hyperparameters are chosen as default settings
#
# When using this code, please cite:
# Van Hilten, N.; Methorst, J.; Verwei, N.; Risselada, H.J., Science Advances. 2023, 9(11). DOI: 10.1126/sciadv.ade8839
#######

import pickle
import argparse
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import keras as k
import matplotlib
import matplotlib.pyplot as plt
import os
import General_functions as gf
import sys

flags = argparse.ArgumentParser()

flags.add_argument("-f", "--input", default="data.txt", 
            help="Input file with sequences in first column, scores in second column", type=str)
flags.add_argument("-d", "--hist", default="histories.dat", 
            help="Name of the file in which loss over epochs is stored (new file)", type=str)
flags.add_argument("-y", "--Yval", default="Y_Validation.dat", 
            help="Name of the file in which the validation result is stored(new file)", type=str)
flags.add_argument("-x", "--Xval", default="X_Validation.dat", 
            help="Name of the file in which the validation input is stored(new file)", type=str)
flags.add_argument("-a", "--alphabet", default="_ARNDCFQEGHILKMPSTWYV", 
            help="Alphabet for one-hot encoding", type=str)
flags.add_argument("-t", "--test_size", default=0.25, 
            help="The part of the data that is split in validation", type=float)
flags.add_argument("-n", "--n_folds", default=5, 
            help="N-fold cross-validation", type=int)
flags.add_argument("-i", "--epochs", default=1000, 
            help="Initial number of epochs for training", type=int)
flags.add_argument("-k", "--kernel_size", default=5, 
            help="Kernel size used in convolutional layers", type=int)
flags.add_argument("-c", "--conv_filters", default=[64, 64], nargs='+',
            help="A list containing the number of nodes in the convolutional layers", type=int)
flags.add_argument("-o", "--dense_layers", default= [64], nargs='+',
            help="A list containing the number of nodes in the dense layers ", type=int)
flags.add_argument("-u", "--drate", default=0.5,
            help="Dropout rate", type=float)
flags.add_argument("-r", "--lrate", default=0.001,
            help="Learning rate", type=float)
flags.add_argument("-b", "--batch", default=64,
            help="Batch size", type=int)
flags.add_argument("-m", "--dir_name", default='CNN_runs',
            help="Name of output directory", type=str)
flags.add_argument("-v", "--data_file", default='overall_data.dat',
            help="Store data in text form", type=str)

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams['axes.axisbelow'] = True


args = flags.parse_args()

#############################################################################################################################
# define cnn model
def define_model(seq_len, alp_len):
    model = k.models.Sequential()
    for i, filter in enumerate(args.conv_filters):
        model.add(k.layers.Conv1D(filters = filter, kernel_size = args.kernel_size, activation='relu', kernel_initializer='he_uniform',
                        input_shape=(int(seq_len/(2**i)), alp_len), padding = "same"))
        model.add(k.layers.MaxPooling1D(2))

    model.add(k.layers.Flatten())
    model.add(k.layers.Dropout(args.drate))

    for nodes in args.dense_layers:
        model.add(k.layers.Dense(nodes, activation='relu', kernel_initializer='he_uniform'))
    model.add(k.layers.Dense(1, activation=None, kernel_initializer='he_uniform'))

    opt = k.optimizers.Adam(learning_rate= args.lrate)
    model.compile(optimizer= opt, loss='mean_squared_error')
    return model


# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY):
    histories = []
    kfold = KFold(args.n_folds, shuffle=True, random_state=1)

    n = 0
    for train_ix, test_ix in kfold.split(dataX):
        print("KFOLD %02d:" % (n))

        model = define_model(seq_len, len(args.alphabet))

        trainX  = dataX[train_ix]
        trainY  = dataY[train_ix]
        testX   = dataX[test_ix]
        testY   = dataY[test_ix]

        es = k.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        history = model.fit(
                            trainX, trainY, 
                            epochs= args.epochs, 
                            batch_size=args.batch, 
                            validation_data=(testX, testY),
                            verbose=1
        , callbacks=[es])
        histories.append(history)

        model.save(os.path.join(path, 'model_%02d.h5' % (n)))
        n += 1
    return histories

#############################################################################################################################
#folder creation

#name dir new dir is created in
dir_name = args.dir_name

# Parent Directory path
parent_dir = os.getcwd()

#excisting path
first_path = os.path.join(parent_dir, dir_name)

new_dir_name = "{}_{}_{}_{}_{}_{}".format(
                                args.batch,
                                args.kernel_size, 
                                args.conv_filters,
                                args.dense_layers,
                                args.drate,
                                args.lrate,
                                )

#new dir path
path = os.path.join(first_path, new_dir_name)

#creation new path
try:
    os.makedirs(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

#############################################################################################################################
#inital training (kfold)

sequences, fitness, seq_len = gf.get_data(args.input)
plt.hist(fitness, bins=500)
plt.xlabel("$\Delta \Delta F_{sensing}$ (kJ mol$^{-1}$)")
plt.gca().invert_xaxis()
plt.ylabel("Frequency")
plt.savefig(os.path.join(path, "training_data_hist.pdf"))
plt.clf()
plt.close()

X = gf.convert_data_str_OHE(sequences, gf.create_dict_OHE(args.alphabet), 24)
Y = fitness
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size= args.test_size)

# evaluate model
histories = evaluate_model(X_train, Y_train)

#extract best epoch 
average_epoch = 0
for h in histories:
    epoch_dict = {v: k for k,v in enumerate(h.history['val_loss'])}
    best_epoch = epoch_dict[min(epoch_dict.keys())] #find lowest val loss in keys and use key to get epoch number
    average_epoch += best_epoch
average_epoch = int(round(average_epoch / len(histories)))
print("best epoch: {}".format(average_epoch))

for h in histories:
    plt.plot(h.history['loss'], color='blue', label='train')
    plt.plot(h.history['val_loss'], color='orange', label='test')
plt.grid(alpha=0.5)
plt.text(average_epoch ,max(h.history['loss'])/2 ,"Ave low epoch: {}".format(average_epoch))
plt.xlabel("Epoch")
plt.ylabel("Mean squared error")
plt.legend(['Training', 'Testing'])
plt.savefig(os.path.join(path,"loss_test_train.pdf"))
plt.close()

#############################################################################################################################
#final training

model = define_model(seq_len, len(args.alphabet))
es = k.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
mc = k.callbacks.ModelCheckpoint(os.path.join(path, 'best_model.h5'), monitor='loss', mode='min', verbose=1, save_best_only=True)
history = model.fit(
                    X_train, Y_train, 
                    epochs= average_epoch, 
                    batch_size=args.batch, 
                    verbose=1,
                    callbacks=[es, mc])

#############################################################################################################################
# plots final model                    

#plot loss
plt.plot(history.history['loss'], color='blue', label='train')
plt.grid(alpha=0.5)
plt.xlabel("Epoch")
plt.ylabel("Mean squered error")
plt.legend(['Training', 'Testing'])
plt.savefig(os.path.join(path,"loss_full_train.pdf"))
plt.close()

# plot r squared
model = k.models.load_model(os.path.join(path, 'best_model.h5'))
y_test = model.predict(X_val)[:,-1]
plt.scatter(y_test, Y_val)
p = np.poly1d(np.polyfit(y_test, Y_val, 1))
plt.plot(y_test, p(y_test), 'k')
plt.grid(alpha=0.5)
plt.xlabel('Predicted score')
plt.ylabel('Validation score')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

correlation_matrix = np.corrcoef(y_test, Y_val)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print("R_squared = " + str(r_squared))
plt.text(max(y_test)/10 ,max(Y_val)/2 ,"R^2: {:.3f}".format(r_squared))
plt.savefig(os.path.join(path,"scatter_final.pdf"))
plt.close()

#############################################################################################################################
#save Data
 
with open(os.path.join(path, args.hist), "wb") as file:
    pickle.dump([k.history for k in histories], file)
with open(os.path.join(path, args.Yval), "wb") as file:
    pickle.dump(Y_val, file)
with open(os.path.join(path, args.Xval), "wb") as file:
    pickle.dump(X_val, file)
with open(os.path.join(path, "final_data_run_b.dat"), "wb") as file:
    data = [[
            args.batch,
            args.kernel_size,
            args.conv_filters,
            args.dense_layers,
            args.drate,
            args.lrate
            ], 
            [
            average_epoch,
            history.history['loss'][-1],
            r_squared
            ]]
    pickle.dump(data, file)
with open(os.path.join(path, "final_data_run_t.txt"), "wt") as file:
    data = '''batch = {}\nkernel size = {}\nconv_filters = {}\ndense_layers = {}\ndropout_rate = {}\nlearning_rate = {}\naverage_epoch ={}\nLoss = {}\nR_squared = {}'''.format(
            args.batch,
            args.kernel_size,
            args.conv_filters,
            args.dense_layers,
            args.drate,
            args.lrate,
            average_epoch,
            history.history['loss'][-1],
            r_squared
            )
    file.write(data)

print("done")
