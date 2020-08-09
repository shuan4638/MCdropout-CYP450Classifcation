import numpy as np
import pandas as pd
import os
import shutil

import molvs as mv
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem
from sklearn import metrics, model_selection
from sklearn.metrics import roc_auc_score, accuracy_score

import tensorflow.keras as keras
import tqdm
from keras.regularizers import l2
from keras.layers import Input, Dense, Dropout
from keras import Model, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Convert smiles to fingerprint array  
def mol2arr(smiles):
    arr = np.zeros((1,))
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits = 1024)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# Convert the raw inputs into input features and labels
def get_data(file_name, file_df):
    fps_list = []
    answer_list = []
    cnt = 0
    all_len = len(file_df.index)
    print ('Loading dadaset %s, data size: %d, positive: %d, negative: %d' % (file_name, all_len, len(file_df[file_df[1] == 1]),len(file_df[file_df[1] == 0]) ))
    fail_converts = []
    for i in file_df.index:
        smiles = file_df[0][i]
        answer = file_df[1][i]
        try:
            fps = mol2arr(smiles)
            fps_list.append(fps)
            answer_list.append(answer)
        except:
            fail_converts.append(i)
        cnt += 1
        if cnt % 10000 == 0:
            print ('Loading progress: %.3f' % (cnt/all_len))
    data_set = {'fingerprints':fps_list, 'answer': answer_list}
    print ('Finish preparing data set')
    return file_df.drop(fail_converts, axis = 0), data_set

# Train, test, valid set split
def split_var(data_set):   
    data_X = np.array(data_set['fingerprints'])
    data_y = np.array(data_set['answer'])
    train_X, test_X, train_y, test_y = model_selection.train_test_split(data_X, data_y, test_size = 0.1)
    train_X, valid_X, train_y, valid_y = model_selection.train_test_split(train_X, train_y, test_size = 0.11) 
    return train_X, test_X, valid_X, train_y, test_y, valid_y

def MC_NN(train_X, test_X, valid_X, train_y, test_y, valid_y, activation = None, hidden_nodes = 1024, dr = 0, sample_time = 300):
          
    train_X = np.array(train_X)
    valid_X = np.array(valid_X)
    test_X = np.array(test_X)
   
    inputShape=(train_X.shape[1] , )    
    inputs = Input(inputShape, name = 'Input')
    x = Dropout(dr)(inputs)
    x = Dense(hidden_nodes, activation = activation, kernel_regularizer = l2(1e-4))(x)
    x = Dropout(0.5)(x, training = True)
    y = Dense(1, activation = 'sigmoid', name = 'Output')(x)
 
    model = Model(inputs=inputs, outputs = y)
    model.compile(optimizer=optimizers.Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy']) 
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    
    improvement_dir = 'Weights_improvement'
    if not os.path.exists(improvement_dir):
        os.mkdir(improvement_dir)
    filepath="%s/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5" % improvement_dir
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    history = model.fit(train_X, train_y, batch_size= 200, epochs= 300, validation_data=(valid_X, valid_y), callbacks=[checkpoint, early_stopping], verbose=0)
    if os.path.exists(improvement_dir):
        shutil.rmtree(improvement_dir)

    mc_predictions = []
    for i in tqdm.tqdm(range(sample_time)):
        y_p = model.predict(test_X, batch_size=200)
        mc_predictions.append(y_p)
    y_std = np.std(np.array(mc_predictions), axis = 0)
    y_mean = np.mean(np.array(mc_predictions), axis = 0)

    pred_y = y_mean.copy()
    auc = roc_auc_score(test_y, pred_y)
    pred_y[pred_y >= 0.5] = 1
    pred_y[pred_y < 0.5] = 0
    acc = accuracy_score(test_y, pred_y)
    print ('Dropout rate = %s, AUC: %.3f, ACC: %.3f' % (dr, auc, acc))
    return model, history, auc, acc, [mc_predictions,y_mean, y_std] 

