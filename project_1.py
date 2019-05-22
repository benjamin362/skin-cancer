
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
liste = np.array(['index',
    'eventNumber',
    'runNumber',
    'actualInteractionsPerCrossing',
    'averageInteractionsPerCrossing',
    'correctedActualMu',
    'correctedAverageMu',
    'correctedScaledActualMu',
    'correctedScaledAverageMu',
    'NvtxReco',
    'p_nTracks',
    'p_pt_track',
    'p_eta',
    'p_phi',
    'p_charge',
    'p_qOverP',
    'p_z0',
    'p_d0',
    'p_sigmad0',
    'p_d0Sig',
    'p_EptRatio',
    'p_dPOverP',
    'p_z0theta',
    'p_deltaR_tag',
    'p_etaCluster',
    'p_phiCluster',
    'p_eCluster',
    'p_rawEtaCluster',
    'p_rawPhiCluster',
    'p_rawECluster',
    'p_eClusterLr0',
    'p_eClusterLr1',
    'p_eClusterLr2',
    'p_eClusterLr3',
    'p_etaClusterLr1',
    'p_etaClusterLr2',
    'p_phiClusterLr2',
    'p_eAccCluster',
    'p_f0Cluster',
    'p_etaCalo',
    'p_phiCalo',
    'p_eTileGap3Cluster',
    'p_cellIndexCluster',
    'p_phiModCalo',
    'p_etaModCalo',
    'p_dPhiTH3',
    'p_R12',
    'p_fTG3',
    'p_weta2',
    'p_Reta',
    'p_Rphi',
    'p_Eratio',
    'p_f1',
    'p_f3',
    'p_Rhad',
    'p_Rhad1',
    'p_deltaEta1',
    'p_deltaPhiRescaled2',
    'p_TRTPID',
    'p_TRTTrackOccupancy',
    'p_numberOfInnermostPixelHits',
    'p_numberOfPixelHits',
    'p_numberOfSCTHits',
    'p_numberOfTRTHits',
    'p_numberOfTRTXenonHits',
    'p_chi2',
    'p_ndof',
    'p_SharedMuonTrack',
    'Truth',
    'p_truth_E',
    'p_E7x7_Lr2',
    'p_E7x7_Lr3',
    'p_E_Lr0_HiG',
    'p_E_Lr0_LowG',
    'p_E_Lr0_MedG',
    'p_E_Lr1_HiG',
    'p_E_Lr1_LowG',
    'p_E_Lr1_MedG',
    'p_E_Lr2_HiG',
    'p_E_Lr2_LowG',
    'p_E_Lr2_MedG',
    'p_E_Lr3_HiG',
    'p_E_Lr3_LowG',
    'p_E_Lr3_MedG',
    'p_ambiguityType',
    'p_asy1',
    'p_author',
    'p_barys1',
    'p_core57cellsEnergyCorrection',
    'p_deltaEta0',
    'p_deltaEta2',
    'p_deltaEta3',
    'p_deltaPhi0',
    'p_deltaPhi1',
    'p_deltaPhi2',
    'p_deltaPhi3',
    'p_deltaPhiFromLastMeasurement',
    'p_deltaPhiRescaled0',
    'p_deltaPhiRescaled1',
    'p_deltaPhiRescaled3',
    'p_e1152',
    'p_e132',
    'p_e235',
    'p_e255',
    'p_e2ts1',
    'p_ecore',
    'p_emins1',
    'p_etconeCorrBitset',
    'p_ethad',
    'p_ethad1',
    'p_f1core',
    'p_f3core',
    'p_maxEcell_energy',
    'p_maxEcell_gain',
    'p_maxEcell_time',
    'p_maxEcell_x',
    'p_maxEcell_y',
    'p_maxEcell_z',
    'p_nCells_Lr0_HiG',
    'p_nCells_Lr0_LowG',
    'p_nCells_Lr0_MedG',
    'p_nCells_Lr1_HiG',
    'p_nCells_Lr1_LowG',
    'p_nCells_Lr1_MedG',
    'p_nCells_Lr2_HiG',
    'p_nCells_Lr2_LowG',
    'p_nCells_Lr2_MedG',
    'p_nCells_Lr3_HiG',
    'p_nCells_Lr3_LowG',
    'p_nCells_Lr3_MedG',
    'p_pos',
    'p_pos7',
    'p_poscs1',
    'p_poscs2',
    'p_ptconeCorrBitset',
    'p_ptconecoreTrackPtrCorrection',
    'p_r33over37allcalo',
    'p_topoetconeCorrBitset',
    'p_topoetconecoreConeEnergyCorrection',
    'p_topoetconecoreConeSCEnergyCorrection',
    'p_weta1',
    'p_widths1',
    'p_widths2',
    'p_wtots1',
    'p_e233',
    'p_e237',
    'p_e277',
    'p_e2tsts1',
    'p_ehad1',
    'p_emaxs1',
    'p_fracs1',
    'p_DeltaE',
    'p_E3x5_Lr0',
    'p_E3x5_Lr1',
    'p_E3x5_Lr2',
    'p_E3x5_Lr3',
    'p_E5x7_Lr0',
    'p_E5x7_Lr1',
    'p_E5x7_Lr2',
    'p_E5x7_Lr3',
    'p_E7x11_Lr0',
    'p_E7x11_Lr1',
    'p_E7x11_Lr2',
    'p_E7x11_Lr3',
    'p_E7x7_Lr0',
    'p_E7x7_Lr1'])


# In[2]:


# ----------------------------------------------------------------------------------- #
#
#  Python macro for reading ATLAS electron/non-electron file (for small project in
#  MLandBDA2019).
#
#  Author: Troels C. Petersen (NBI) and Lukas Ehrke (NBI)
#  Email:  petersen@nbi.dk
#  Date:   25th of April 2019
#
# ----------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------- #
#
#                         SMALL PROJECT IN MLandBDA2019
#                         -----------------------------
#  The files "train.h5" and "test.h5" contain data from the ATLAS experiment at CERN, more
#  specifically a long list of measurements (i.e. input variables) for electron candidates.
#  There are also two "truth" variables in "train.h5", namely if it is an electron and its energy.
#
#  Using Machine Learning algorithm(s), try to solve at least one of the following
#  problems:
#  1) Identify (i.e. classify) electrons compared to non-electrons based on the target variable
#     'Truth':       0 for non-electrons, 1 for electrons
#  2) Estimate (i.e. make regression for) the energy of electrons based on the target variable
#     'p_truth_E':   Energy (in MeV) of the electrons
#
#  You should use "train.h5" to develop your algorithm, and when you feel satisfied, you should
#  apply it to "test.h5", where you don't know the true values. When training (on "train.h5")
#  remember to divide the sample into a part that you train on, and one that you validate on,
#  such that you don't overtrain (to be discussed in class).
#
#  You "only" have to submit ONE solution for ONE of these problems, though it is typically not
#  hard to rewrite the code to solve the other problem as well. You are welcome to submit up to
#  three solutions for each problem using different algorithms. The solution(s) should NOT USE
#  MORE THAN 30 VARIABLES, but you're welcome to use different variables for each solution. 
#
#  You should hand in (each of) your solution(s) as TWO separate files:
#   * A list of index/event numbers (1, 2, 3, etc.) followed by your estimate on each event, i.e.
#     (Here is shown a classifcation solution. For a regression solution, the last number should
#      be the energy estimate in MeV)
#       0   0.998232
#       1   0.410455
#       2   0.037859
#       3   ...
#   * A list of the variables you've used for each problem, i.e.
#       p_eta
#       p_pt_track
#       ...
#  
#  You should name your file as follows:
#    TypeOfProblemSolved_FirstnameLastname_SolutionName(_VariableList).txt
#  three solution examples of which could for myself be:
#    Classification_TroelsPetersen_SKLearnAlgo1.txt
#    Classification_TroelsPetersen_SKLearnAlgo1_VariableList.txt
#    Classification_TroelsPetersen_XGBoost1.txt
#    Classification_TroelsPetersen_XGBoost1_VariableList.txt
#    Regression_TroelsPetersen_kNN-DidNotReallyWorkWell.txt
#    Regression_TroelsPetersen_kNN-DidNotReallyWorkWell_VariableList.txt
# ----------------------------------------------------------------------------------- #

import numpy as np
import h5py


with h5py.File("data/train.h5", "r") as hf :
    data = hf["train"][:]

for i in range(10) :
    # Basic way of printing in a formatted way (using the "%" sign):
    # print("  %3d    eta (perpend?): %6.3f    FracHad: %6.3f    Track Mom.: %5.1f GeV    Cluster2 E: %5.1f GeV       is Elec: %1d    True E: %5.1f GeV"%(data[i]["index"], data[i]["p_eta"], data[i]["p_Rhad"], data[i]["p_pt_track"]/1000.0, data[i]["p_eClusterLr2"]/1000.0, data[i]["Truth"], data[i]["p_truth_E"]/1000.0))

    # More advanced (and better) way, using the f-strings (one advantage is that what you print goes together with the format):
    print(f"  {data[i]['index']:3d},     Eta (perpend?): {data[i]['p_eta']:6.3f}    FracHad: {data[i]['p_Rhad']:6.3f}    Track Mom.: {data[i]['p_pt_track']:8.1f} MeV    Cluster2 E: {data[i]['p_eClusterLr2']:8.1f} MeV       is Elec: {data[i]['Truth']:1d}    True E: {data[i]['p_truth_E']:8.1f} MeV")
    

# List of variables.
# ------------------
# The list of variables is included below just for reference, so that you can easily read/know what the data file contains.

# There are many variables, and part of the exercise is also to admit, that you don't know, what they are, and that you don't
# really need to know (*). The only thing you REALLY NEED TO KNOW is, which are INPUT variables (i.e. those you use), and
# which are the TARGET variables (i.e. those you want to learn to predict).
# In this case, all the variables are input variables, except the two target variables:
#  * 'Truth':       0 for non-electrons, 1 for electrons (in a CLASSIFICATION PROBLEM)
#  * 'p_truth_E':   Energy (in MeV) of the electrons (in a REGRESSION PROBLEM)

# (*) Well, if you were working alone on this, and wanted to get the absolute best result, you would probably benefit
#     from knowing, but generally one can get very far without knowing.


# In[3]:


from sklearn.decomposition import PCA
import pandas as pd
data_1 = pd.DataFrame(data)
data_1.sample(frac=1);


# ## Shuffle and split into train and test data

# In[4]:


x_all = data_1.drop(columns=['Truth','p_truth_E','index'])
#x_all = data_1.drop(columns=['eventNumber','runNumber','Truth','index','p_truth_E'])
features = x_all.columns
y_p = data_1['p_truth_E'].values
y_e = data_1['Truth'].values


# In[5]:


test_split = 0.9
max_idx = int(np.floor(test_split * len(data)))
X_train, X_test, y_p_train, y_p_test, y_e_train, y_e_test = x_all[:max_idx], x_all[max_idx:], y_p[:max_idx], y_p[max_idx:], y_e[:max_idx], y_e[max_idx:] 


# ## Train and test a boosted decision tree

# In[6]:


from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc
import shap


# In[52]:


model = XGBClassifier(objective='binary:logistic',eval_metric='logloss')
bst = model.fit(x_all, y_e.astype(int))


# In[55]:


y_e_pred = model.predict_proba(X_test)
predictions = (y_e_pred > 0.5)[:,1]


# In[56]:


accuracy = accuracy_score(y_e_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[57]:


fpr, tpr, thresholds = roc_curve(y_e_test, y_e_pred[:,1])
roc_fig, roc_ax = plt.subplots(figsize=(8,6))
roc_ax.plot(tpr,fpr,linestyle='--')


# In[58]:


auc(fpr,tpr)


# ## Do it with k-fold cross validation

# In[ ]:


#model = XGBClassifier(objective='binary:logistic',eval_metric='logloss')
#kfold = KFold(n_splits=3, random_state=7)
#results = cross_val_score(model, x_all, y_e, cv=kfold)


# In[ ]:


#results


# In[59]:


select30 = SelectFromModel(model,prefit=True, threshold=-np.inf, max_features=30)
X_selected_1 = select30.transform(x_all)
features_idx = select30.get_support()
feature_name =  features[features_idx]


# In[60]:


feature_name


# In[61]:


scores = model.get_booster().get_score(importance_type= 'gain')


# In[62]:


len(scores.values())


# In[63]:


import xgboost as xgb


# In[64]:


xgb.plot_importance(model,importance_type='gain')
plt.figure(figsize=(20,10))


# In[65]:


feat_idx = np.argsort(model.feature_importances_)


# In[66]:


importance_1 = features[feat_idx[:30]]


# In[67]:


get_ipython().run_line_magic('time', 'shap_values = shap.TreeExplainer(model).shap_values(x_all)')


# In[68]:


importance_2 = features[np.argsort(-np.abs(shap_values).mean(axis=0))[:30]]


# In[69]:


len(np.intersect1d(feature_name,importance_2))


# In[70]:


importance_2


# In[71]:


x_imp1 = data_1[feature_name]
X_train_1, X_test_1 = x_imp1[:max_idx], x_imp1[max_idx:] 
x_imp2 = data_1[importance_2]
X_train_2, X_test_2 = x_imp2[:max_idx], x_imp2[max_idx:] 


# In[72]:


model_1 = XGBClassifier()
model_1.fit(X_train_1, y_e_train.astype(int))
y_e_pred_1 = model_1.predict_proba(X_test_1)
predictions_1 = (y_e_pred_1 > 0.5)[:,1]
accuracy_1 = accuracy_score(y_e_test, predictions_1)
print("Accuracy: %.2f%%" % (accuracy_1 * 100.0))


# In[73]:


fpr, tpr, thresholds = roc_curve(y_e_test, y_e_pred_1[:,1])
roc_fig, roc_ax = plt.subplots(figsize=(8,6))
roc_ax.plot(tpr,fpr,linestyle='--')
print(auc(fpr,tpr))


# In[74]:


model_2 = XGBClassifier()
model_2.fit(X_train_2, y_e_train.astype(int))
y_e_pred_2 = model_2.predict_proba(X_test_2)
predictions_2 = (y_e_pred_2 > 0.5)[:,1]
accuracy_2 = accuracy_score(y_e_test, predictions_2)
print("Accuracy: %.2f%%" % (accuracy_2 * 100.0))


# In[75]:


fpr, tpr, thresholds = roc_curve(y_e_test, y_e_pred_2[:,1])
roc_fig, roc_ax = plt.subplots(figsize=(8,6))
roc_ax.plot(tpr,fpr,linestyle='--')
print(auc(fpr,tpr))


# ## Now classify the unknown data

# In[12]:


with h5py.File("data/test.h5", "r") as hf :
    data_test = hf["test"][:]


# In[13]:


data_test = pd.DataFrame(data_test)
test_1_e = data_test[feature_name]
test_2_e = data_test[importance_2]


# ## Delete the old models and train them on all labeled data

# In[78]:


del model_1, model_2


# In[80]:


model_1 = XGBClassifier()
model_1.fit(x_imp1, y_e.astype(int))


# In[79]:


model_2 = XGBClassifier()
model_2.fit(x_imp2, y_e.astype(int))


# ## Save the feature lists

# In[99]:


feat_1 = feature_name.values
feat_2 = importance_2.values
np.savetxt('hand_in/feat1.txt',feat_1,fmt='%s')
np.savetxt('hand_in/feat2.txt',feat_2,fmt='%s')


# ## Predict the probabilites for the calsses of the unknown data

# In[81]:


pred_1 = model_1.predict_proba(test_1_e)
pred_1 = pred_1[:,1]
pred_2 = model_2.predict_proba(test_2_e)
pred_2 = pred_2[:,1]


# In[82]:


predictions_1 = (pred_1 > 0.5)
predictions_2 = (pred_2 > 0.5)


# In[83]:


len(np.where(predictions_1 != predictions_2)[0]) / test_1_e.shape[0]


# ## Save the predicted class probabilties to a file

# In[103]:


np.savetxt('hand_in/Classification_BenjaminHeuser_XGBoost_feat1.txt',pred_1)
np.savetxt('hand_in/Classification_BenjaminHeuser_XGBoost_feat2.txt',pred_2)


# ## Neural network for binary classification

# In[7]:


import keras
import tensorflow as tf
from keras.optimizers import SGD
from sklearn.utils import class_weight
from sklearn import preprocessing
encoded_y_e_train = np.array(y_e_train,dtype=int)


# ## Normalize the train data

# In[105]:


x = x_imp2.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_norm = pd.DataFrame(x_scaled)


# In[106]:


model_nn = keras.Sequential()
model_nn.add(keras.layers.Dense(30, input_dim=30,activation='relu'))
model_nn.add(keras.layers.Dense(30,activation='relu'))
model_nn.add(keras.layers.Dropout(0.1, noise_shape=None, seed=None))
model_nn.add(keras.layers.Dense(1,activation='sigmoid'))


# In[107]:


class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_e),
                                                 y_e)


# In[108]:


model_nn.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[109]:


model_nn.fit(X_norm, y_e, epochs=10,class_weight=class_weights)


# ## Normalize the test data and evaluate

# In[110]:


x = test_2_e.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_test_norm = pd.DataFrame(x_scaled)


# ## Predict the probabiltiy

# In[120]:


pred_3 = model_nn.predict_proba(X_test_norm)


# In[126]:


predictions_3 = (pred_3 > 0.5).flatten()


# In[128]:


len(np.where(predictions_1 != predictions_3)[0]) / test_1_e.shape[0]


# In[129]:


np.savetxt('hand_in/Classification_BenjaminHeuser_NN_2lay_relu_feat2.txt',pred_3)


# # Regression

# In[9]:


with h5py.File("data/train.h5", "r") as hf :
    data = hf["train"][:]


# In[8]:


from sklearn.decomposition import PCA
import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
data_1 = pd.DataFrame(data)
data_1.sample(frac=1);


# In[10]:


y_p_all = data_1['p_truth_E'].values
y_e = data_1['Truth'].values
x_all = data_1.drop(columns=['Truth','p_truth_E','index'])
x_reg = data_1[data_1['Truth'] == 1]
y_p = data_1[data_1['Truth'] == 1]['p_truth_E'].values


# In[11]:


test_split = 0.9
max_idx = int(np.floor(test_split * len(y_p)))
X_train_reg, X_test_reg, y_p_train, y_p_test= x_all[:max_idx], x_all[max_idx:], y_p[:max_idx], y_p[max_idx:] 


# In[12]:


reg_xgb = XGBRegressor()


# In[13]:


reg_xgb.fit(x_reg, y_p)


# ## Get the importance

# In[14]:


feat_idx = np.argsort(reg_xgb.feature_importances_)
features = x_all.columns
reg_feat_1 = features[feat_idx[:30]]


# In[15]:


get_ipython().run_line_magic('time', 'shap_values_reg = shap.TreeExplainer(reg_xgb).shap_values(x_reg)')


# In[16]:


reg_feat_2 = features[np.argsort(-np.abs(shap_values_reg).mean(axis=0))[:30]]


# ## Define two subsamples for the data 

# In[17]:


test_split = 0.9
max_idx = int(np.floor(test_split * len(y_p)))


# In[18]:


x_reg1 = x_reg[reg_feat_1]
X_reg_1_train, X_reg_1_test = x_reg1[:max_idx], x_reg1[max_idx:] 
x_reg2 = x_reg[reg_feat_2]
X_reg_2_train, X_reg_2_test = x_reg2[:max_idx], x_reg2[max_idx:]
y_p_train, y_p_test = y_p[:max_idx], y_p[max_idx:] 


# ## And train two different models on those features

# In[19]:


def mylogcosh(true, pred):
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss) / len(pred)


# In[20]:


reg_xgb_1 = XGBRegressor(feval = mylogcosh)
reg_xgb_1.fit(X_reg_1_train, y_p_train)


# In[21]:


reg_xgb_2 = XGBRegressor(feval = mylogcosh)
reg_xgb_2.fit(X_reg_2_train, y_p_train)


# ## And evaluate their errors

# In[22]:


pred_p_1 = reg_xgb_1.predict(X_reg_1_test)
error_reg_1 = (pred_p_1 - y_p_test) / y_p_test


# In[23]:


pred_p_2 = reg_xgb_2.predict(X_reg_2_test)
error_reg_2 = (pred_p_2 - y_p_test) / y_p_test


# In[24]:


plt.scatter(pred_p_1,y_p_test,s=0.1,label='Features 1')
plt.scatter(pred_p_2,y_p_test,s=0.1,label='Features 2')
plt.legend()


# In[25]:


plt.hist(error_reg_1,bins=100,label='Features 1');
plt.hist(error_reg_2,bins=100,label='Features 2');
plt.legend()


# In[26]:


print(error_reg_1.std())
print(error_reg_2.std())


# In[27]:


reg_feat_1


# In[28]:


reg_feat_2


# In[29]:


np.intersect1d(reg_feat_1,reg_feat_2)


# ## Now take the better performing features and train the XGB with the whole data

# In[30]:


reg_xgb_final = XGBRegressor()
reg_xgb_final.fit(x_reg1, y_p)


# In[30]:


np.savetxt('hand_in/feat3.txt',reg_feat_1,fmt='%s')


# In[32]:


with h5py.File("data/test.h5", "r") as hf :
    data_test = hf["test"][:]
data_test = pd.DataFrame(data_test)


# In[36]:


electron_idx = data_test['index']
#electron_data = data_test[predictions_2]
e_reduced = data_test[reg_feat_1]


# In[37]:


predicted_p = reg_xgb_final.predict(e_reduced)


# In[38]:


plt.hist(y_p[:len(predicted_p)],bins=100);
plt.hist(predicted_p,bins=100);


# ## Save the index numbers with the corresponding predicted momentum

# In[40]:


regression_sol_1 = np.vstack([electron_idx,predicted_p])
np.savetxt('hand_in/Regression_BenjaminHeuser_XGB_feat3.txt',regression_sol_1.T)
#reg_feat_1 = np.loadtxt('hand_in/feat3.txt',dtype=str)


# ## NN
# ## Normailze the training data

# In[41]:


import keras
import tensorflow as tf
from keras.optimizers import SGD
from sklearn.utils import class_weight
from sklearn import preprocessing


# In[42]:


x_reg1 = x_reg[reg_feat_1]
x_reg2 = x_reg[reg_feat_2]
test_split = 0.9
max_idx = int(len(y_p) * test_split)
y_p_train, y_p_test = y_p[:max_idx], y_p[max_idx:] 


# In[43]:


x = x_reg1.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x_reg1_norm = pd.DataFrame(x_scaled)


# In[44]:


x = x_reg2.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x_reg2_norm = pd.DataFrame(x_scaled)


# In[45]:


X_reg_1_train, X_reg_1_test = x_reg1_norm[:max_idx], x_reg1_norm[max_idx:] 
X_reg_2_train, X_reg_2_test = x_reg2_norm[:max_idx], x_reg2_norm[max_idx:]


# In[46]:


model_nn = keras.Sequential()
model_nn.add(keras.layers.Dense(50, input_dim=30,activation='relu'))
model_nn.add(keras.layers.Dense(50,activation='relu'))
#model_nn.add(keras.layers.Dropout(0.1, noise_shape=None, seed=None))
model_nn.add(keras.layers.Dense(1,activation='linear'))
model_nn.compile(loss="logcosh", optimizer='adam')


# In[47]:


model_nn.fit(X_reg_1_train,y_p_train,epochs=20)


# In[48]:


preds = model_nn.predict(X_reg_1_test)
diff = preds - y_p_test
percentDiff = (diff / y_p_test) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)


# In[49]:


error = (preds.flatten() - y_p_test) / y_p_test
print(error.std())


# In[50]:


plt.scatter(preds,y_p_test,s=0.1);


# In[53]:


plt.scatter(preds,y_p_test,s=0.1);
plt.plot(np.arange(300000),np.arange(300000),c='r');


# In[54]:


model_nn_final = keras.Sequential()
model_nn_final.add(keras.layers.Dense(50, input_dim=30,activation='relu'))
model_nn_final.add(keras.layers.Dense(50,activation='relu'))
#model_nn.add(keras.layers.Dropout(0.1, noise_shape=None, seed=None))
model_nn_final.add(keras.layers.Dense(1,activation='linear'))
model_nn_final.compile(loss="logcosh", optimizer='adam')


# In[55]:


X_1_reg = x_reg[reg_feat_1]
model_nn_final.fit(X_1_reg,y_p,epochs=30)


# In[56]:


predicted_final_p = model_nn_final.predict(e_reduced)


# In[57]:


write_out_2 = np.vstack([electron_idx,predicted_final_p.flatten()])


# In[58]:


write_out_2


# In[59]:


np.savetxt('hand_in/Regression_BenjaminHeuser_NN_feat3.txt',write_out_2.T)

