# Analysis scripts for the latent EEG space identification
# Qianying Wu, Sept 25, 2024

# Need to set the current directory to be working directory
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import unravel_index
from sklearn.decomposition import NMF
from sklearn import preprocessing
from kfda import Kfda
from sklearn.model_selection import KFold
from numpy import unravel_index
from sklearn.model_selection import StratifiedKFold
import seaborn as sns

# 1. load PSD data
freqnames = ['theta','alpha','beta','gamma']
data_PSD_raw = []
data_PSD_norm = []
NMF_scores = []
NMF_transmat = []

for freq in freqnames:
    data_PSD_raw.append(pd.read_csv ('../data/PSD_'+freq+'_raw.csv',header=None))
    data_PSD_norm.append(pd.read_csv ('../data/PSD_'+freq+'_norm.csv',header=None))
# load data label
data_label = pd.read_csv('../data/PSD_label.csv')
data_label["subcond"] = data_label.subname.astype(str) + '_' + data_label.condition.astype(str)

# 2. loop over all possible NMF and LDA components
# function
# NMF: try 10 repetitions of the k-fold cross validation
def NormPSD_NMF_LDA_K_fold(y_data,nfold,n_repeat,range_lda,k_nmf):
    mean_train_rep = np.zeros(shape=(n_repeat, len(range_lda)))
    mean_test_rep = np.zeros(shape=(n_repeat, len(range_lda)))
    for rep in range(n_repeat):
        print('Repetition', rep)
        # do 5 fold cross validation
        kf = KFold(n_splits=nfold, shuffle=True)
        score_train = np.zeros(shape=(nfold, len(range_lda)))
        score_test = np.zeros(shape=(nfold, len(range_lda)))
        i = 0
        for train_index, test_index in kf.split(data_label):
            NMF_scores_train = []
            NMF_scores_test = []
        # 1. normalize the PSD, which is the normalized PSD + the mean of each channel within trainning or testing
            for f in range(4):
                # all PSD in 4 freq bands
                PSD_norm_freq = np.array(data_PSD_norm[f])
                PSD_raw_freq = np.array(data_PSD_raw[f])
                # training set PSD
                PSD_norm_freq_train = PSD_norm_freq[train_index,:]
                PSD_raw_freq_train = PSD_raw_freq[train_index, :]
                # testing set PSD
                PSD_norm_freq_test = PSD_norm_freq[test_index,:]
                PSD_raw_freq_test = PSD_raw_freq[test_index, :]

                # calculate mean PSD of each subject across all channels - train
                sub_train = set(y_data.subname[train_index])
                data_label_train = y_data.iloc[train_index]
                PSD_subnorm_freq_train = np.zeros((len(train_index),128))
                for sub in sub_train:
                    sub_idx_train = np.where(data_label_train['subname'] == sub)
                    PSD_raw_sub_train = PSD_raw_freq_train[sub_idx_train]
                    PSD_norm_sub_train = PSD_norm_freq_train[sub_idx_train]
                    mean_sub_PSD_train = np.mean(PSD_raw_sub_train,0)
                    PSD_subnorm_freq_train[sub_idx_train[0],:]=PSD_norm_sub_train + mean_sub_PSD_train

                # calculate mean PSD of each subject across all channels - test
                sub_test = set(y_data.subname[test_index])
                data_label_test = y_data.iloc[test_index]
                PSD_subnorm_freq_test = np.zeros((len(test_index), 128))
                for sub in sub_test:
                    sub_idx_test = np.where(data_label_test['subname'] == sub)
                    PSD_raw_sub_test = PSD_raw_freq_test[sub_idx_test]
                    PSD_norm_sub_test = PSD_norm_freq_test[sub_idx_test]
                    mean_sub_PSD_test = np.mean(PSD_raw_sub_test, 0)
                    PSD_subnorm_freq_test[sub_idx_test[0], :] = PSD_norm_sub_test + mean_sub_PSD_test

                # 2. NMF on each frequency for both training and testing separately
                # 2.1.1 transform PSD to be nonnegative
                min_PSD_train = np.min(np.min(PSD_subnorm_freq_train))
                X_train = PSD_subnorm_freq_train - min_PSD_train
                # 2.1.2 fit NMF model
                model1 = NMF(n_components=k_nmf, init='nndsvd', random_state=0, max_iter=10000)
                model1.fit(X_train)
                NMF_scores_train.append(model1.transform(X_train))
                # 2.2.1 transform PSD to be nonnegative
                min_PSD_test = np.min(np.min(PSD_subnorm_freq_test))
                X_test = PSD_subnorm_freq_test - min_PSD_test
                # 2.2.2 fit NMF model
                NMF_scores_test.append(model1.transform(X_test))

            # merge all 4 freq band NMF
            NMF_train = np.concatenate((NMF_scores_train[0], NMF_scores_train[1], NMF_scores_train[2], NMF_scores_train[3]), axis=1)
            NMF_test = np.concatenate((NMF_scores_test[0], NMF_scores_test[1], NMF_scores_test[2], NMF_scores_test[3]), axis=1)
            y_train = y_data.iloc[train_index]['subcond']
            y_test = y_data.iloc[test_index]['subcond']

            l = 0
            for n_comp in range_lda:
                try:
                    cls = Kfda(kernel='linear', n_components=n_comp)
                    cls.fit(NMF_train, y_train)
                    score_train[i, l] = cls.score(NMF_train, y_train)
                    score_test[i, l] = cls.score(NMF_test, y_test)
                except Exception:
                    score_train[i, l] = float("nan")
                    score_test[i, l] = float("nan")
                l += 1

            i += 1

        # plot training/testing accuracy vs parameters
        mean_train_rep[rep, :] = np.nanmean(score_train, axis=0)
        mean_test_rep[rep, :] = np.nanmean(score_test, axis=0)

    return mean_train_rep,mean_test_rep

# Call the function
range_nmf = np.arange(10,80,5)
range_lda = np.arange(1,31,1)
for comp_nmf in range_nmf:
    mean_train_rep, mean_test_rep = NormPSD_NMF_LDA_K_fold(
        data_label, nfold=5, n_repeat=100, range_lda=range_lda,k_nmf=comp_nmf)
    # from the numpy module
    filename1 = '../results/NMF_LDA/nmf_' \
               + str(comp_nmf) +'_lda2to30_train.csv'
    filename2 = '../results/NMF_LDA/nmf_' \
               + str(comp_nmf) +'_lda2to30_test.csv'
    np.savetxt(filename1,mean_train_rep.transpose(),delimiter=", ",fmt='% s')
    np.savetxt(filename2, mean_test_rep.transpose(), delimiter=", ", fmt='% s')

# 3. Parameter selection
# check AUC of the train & validate results
from sklearn import metrics
train_nmf_lda = np.zeros((len(range_lda),len(range_nmf)))
test_nmf_lda = np.zeros((len(range_lda),len(range_nmf)))
auc_train = np.zeros((len(range_nmf),1))
auc_test = np.zeros((len(range_nmf),1))
for c,comp_nmf in enumerate(range_nmf):
    train_nmf = pd.read_csv ('../results/NMF_LDA/nmf_' \
               + str(comp_nmf) +'_lda2to30_train.csv', header = None)
    train_nmf_lda[:,c] = np.mean(train_nmf,axis=1)
    auc_train[c] = metrics.auc(range_lda[0:20], train_nmf_lda[0:20,c])
    test_nmf = pd.read_csv ('../results/NMF_LDA/nmf_' \
               + str(comp_nmf) +'_lda2to30_test.csv', header = None)
    test_nmf_lda[:,c] = np.mean(test_nmf,axis=1)
    auc_test[c] = metrics.auc(range_lda[0:20], test_nmf_lda[0:20, c])

# heatmap
heatmap_train = train_nmf_lda[0:20,:]
with open('../results/heatmap_train.npy', 'wb') as f:
    np.save(f, heatmap_train)
sns.heatmap(train_nmf_lda[0:20,:], vmin=0, vmax=1,  yticklabels=range_lda[0:20], xticklabels=range_nmf)
plt.title('Sub*cond classification -train')
plt.xlabel('NMF components')
plt.ylabel('LDA components')
plt.show()

heatmap_val = test_nmf_lda[0:20,:]
with open('../results/heatmap_val.npy', 'wb') as f:
    np.save(f, heatmap_val)
sns.heatmap(test_nmf_lda[0:20,:], vmin=0, vmax=1,  yticklabels=range_lda[0:20], xticklabels=range_nmf)
plt.title('Sub*cond classification - test')
plt.xlabel('NMF components')
plt.ylabel('LDA components')
plt.show()

# AUC
plt.plot(range_nmf,auc_train,label = 'train')
plt.plot(range_nmf,auc_test, label = 'test')
plt.xlabel('NMF components')
plt.ylabel('AUC')
plt.legend()
plt.show()

# decide the knee point of the LDA curve
knee_data = test_nmf_lda[0:20,6]
from kneed import KneeLocator
kneedle = KneeLocator(range_lda[0:20],test_nmf_lda[0:20,6], S=1.0, curve="concave", direction="increasing")
print(kneedle.elbow)
kneedle.plot_knee()
plt.show()
# elbow of LDA
plt.plot(range_lda,train_nmf_lda[:,6],label='train')
plt.plot(range_lda,test_nmf_lda[:,6],label='test')
plt.xlabel('LDA components')
plt.ylabel('Classification accuracy')
plt.vlines(x=9,ymin=0,ymax=1,linestyles='--',colors='black')
plt.legend()
plt.title('NMF =' + str(range_nmf[6]))
plt.show()

# 4. Train on all the data, and visualize
NMF_scores = []
H = []
# 1. normalize the PSD, which is the normalized PSD + the mean of each channel within trainning or testing
for f in range(4):
    # all PSD in 4 freq bands
    PSD_norm_freq = np.array(data_PSD_norm[f])
    PSD_raw_freq = np.array(data_PSD_raw[f])

    # calculate mean PSD of each subject across all channels - train
    sub_train = set(data_label.subname)
    PSD_subnorm_freq = np.zeros((348,128))
    for sub in sub_train:
        sub_idx = np.where(data_label['subname'] == sub)
        PSD_raw_sub = PSD_raw_freq[sub_idx]
        PSD_norm_sub = PSD_norm_freq[sub_idx]
        mean_sub_PSD_train = np.mean(PSD_raw_sub,0)
        PSD_subnorm_freq[sub_idx[0],:]=PSD_norm_sub + mean_sub_PSD_train


    # 2. NMF on each frequenc
    # 2.1.1 transform PSD to be nonnegative
    min_PSD = np.min(np.min(PSD_subnorm_freq))
    X_train = PSD_subnorm_freq - min_PSD
    # 2.1.2 fit NMF model
    model1 = NMF(n_components=40, init='nndsvd', random_state=0, max_iter=100000)
    model1.fit(X_train)
    NMF_scores.append(model1.transform(X_train))
    H.append(model1.components_) # the mapping matrix

from scipy.io import savemat
savemat('../data/NMF_H.mat',{"H":H})
savemat('../data/NMF_scores.mat',{"NMF_scores":NMF_scores})

# merge all 4 freq band NMF
NMF_train = np.concatenate((NMF_scores[0], NMF_scores[1], NMF_scores[2], NMF_scores[3]), axis=1)
# zscore standardization of the NMF data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
NMF_zscore = scaler.fit_transform(NMF_train)
# run LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis(n_components=7,store_covariance = True)
LDA_all = clf.fit_transform(NMF_zscore,data_label.subcond)
exp_var = clf.explained_variance_ratio_
plt.plot(np.arange(1,8),np.cumsum(exp_var))
plt.xlabel('LDA components',fontsize=15)
plt.ylabel('Explained variance',fontsize=15)
np.savetxt("../results/LD_exp_variance.csv", exp_var, delimiter=",")
# plot the in-sample classification accuracy
l = 0
score_insample = np.zeros(shape=(7))
for n_comp in range(1,8):
    clf_i = Kfda(kernel='linear', n_components = n_comp)
    clf_i.fit(NMF_zscore, data_label.subcond)
    score_insample[l] = clf_i.score(NMF_zscore,data_label.subcond)
    l += 1
plt.rcParams["figure.figsize"] = (6,3)
plt.plot(range(1,8),score_insample)
plt.xlabel('Latent space dimension',fontsize=15)
plt.ylabel('Classification accuracy',fontsize=15)
plt.ylim([0,1.1])
np.savetxt("../results/Acc_insample.csv", score_insample, delimiter=",")

# plot the 3D LD space
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
colist = le.fit_transform(data_label.subname)
from palettable.cartocolors.qualitative import Pastel_10
plt.rcParams.update({'font.family':'avenir'})
plt.rcParams["figure.figsize"] = (20,20)
fig = plt.figure()
ax = plt.axes(projection="3d")
# reverse the LD order
ax.scatter(LDA_all[data_label.condition == 'tm',0], LDA_all[data_label.condition == 'tm',1], LDA_all[data_label.condition == 'tm',2], c = colist[data_label.condition == 'tm'],
             cmap = Pastel_10.mpl_colormap, marker = 'o', s=100, alpha = 0.7)
ax.scatter(LDA_all[data_label.condition == 'tmoc',0], LDA_all[data_label.condition == 'tmoc',1], LDA_all[data_label.condition == 'tmoc',2], c = colist[data_label.condition == 'tmoc'],
             cmap = Pastel_10.mpl_colormap, marker = '^', s=100, alpha = 0.7)
ax.scatter(LDA_all[data_label.condition == 'tmrv',0], LDA_all[data_label.condition == 'tmrv',1], LDA_all[data_label.condition == 'tmrv',2], c = colist[data_label.condition == 'tmrv'],
             cmap = Pastel_10.mpl_colormap, marker = 's', s=100, alpha = 0.7)
ax.set_xlabel('LD1',fontsize = 20)
ax.set_ylabel('LD2',fontsize = 20)
ax.set_zlabel('LD3',fontsize = 20)
#ax.grid(False)

np.savetxt("../results/condsub_7LD.csv", LDA_all, delimiter=",")

# 5. Interpretability
# plot correlation between LDs and original features
k=0
plt.rcParams["figure.figsize"] = (8,4)
corrcoef = np.ndarray(shape = (LDA_all.shape[1],NMF_zscore.shape[1]))
corrcoef_thres = np.zeros(shape = (LDA_all.shape[1],NMF_zscore.shape[1]))

for i in range(LDA_all.shape[1]):
    for j in range(NMF_zscore.shape[1]):
        corrcoef[i,j] = np.corrcoef(NMF_zscore[:,j], LDA_all[:,i])[1,0]
        if abs(corrcoef[i,j]) > 0.4:
            corrcoef_thres[i,j] = corrcoef[i,j]

plt.imshow(corrcoef_thres, aspect='auto', vmin=-1, vmax=1, cmap='bwr')
plt.xlabel('Original feature')
plt.ylabel('LD')
plt.title('Correlation between original features and hidden dimensions')
plt.colorbar()
plt.show()

# projection matrix
feat_inv = np.linalg.pinv(NMF_zscore)
proj = feat_inv @ LDA_all
recover = NMF_zscore @ proj # can recover the projected space
plt.imshow(np.transpose(proj), aspect='auto', vmax = 25, vmin = -25, cmap='bwr')
plt.xlabel('Original feature')
plt.ylabel('LD')
plt.title('Projection between original features and hidden dimensions')
plt.colorbar()
# calculate the mean and sd of the weights
w_mean_abs  = np.mean(np.abs(proj),axis=0)
w_std_abs = np.std(np.abs(proj),axis = 0)
w_keep = np.zeros((160,7))
for ld in range(7):
    w_keep[:,ld] = proj[:,ld] * (np.abs(proj[:,ld]) > w_mean_abs[ld] + 3*w_std_abs[ld])
plt.imshow(w_keep, aspect='auto', vmax = 25, vmin = -25, cmap='bwr')
plt.ylabel('Original feature')
plt.xlabel('LD')
plt.title('Projection filtered by highest weights')
plt.colorbar()

savemat('../data/LD_NMF_important.mat',{"NMF_per_LD":w_keep})
savemat('../data/LD_NMF_project.mat',{"LD_loading":proj})

# 6. permutation
range_lda = range(7,8)
comp_nmf = 40
# 6.1 shuffle both subject and condition labels
val_acc_perm1_rep = []
for rep in range(100):
    data_per = np.random.permutation(data_label)
    data_per_df = pd.DataFrame(data_per, columns = ['condition','subname','trial','subcond'])
    mean_train_perm1, mean_test_perm1 = NormPSD_NMF_LDA_K_fold(
        data_per_df, nfold=5, n_repeat=1, range_lda=range_lda, k_nmf=comp_nmf)
    mean_acc_perm1_rep.append(mean_train_perm1[0])
    val_acc_perm1_rep.append(mean_test_perm1[0])

np.savetxt("../results/perm_1_val_acc.csv", val_acc_perm1_rep, delimiter=",")


# 6.2 shuffle only condition labels
def perm_state(data_label):
    sub_ori = data_label.subname
    subset = set(sub_ori)
    y_per = data_label.subcond.copy(deep = True)
    y_ori = data_label.subcond.copy(deep=True)
    cond_per = data_label.condition.copy(deep=True)
    for sub in subset:
        labs = y_ori[sub_ori == sub].index
        vals = y_ori[sub_ori == sub].values
        vals_per = np.random.permutation(vals)

        # replace original values with permuted ones
        for i in range(vals_per.shape[0]):
            y_per.at[labs[i]] = vals_per[i]
            cond_per.at[labs[i]] = vals_per[i].split('_')[1]

    data_per = data_label.copy(deep=True)
    data_per.subcond = y_per
    data_per['condition'] = cond_per
    return(data_per)

val_acc_perm2_rep = []
for rep in range(1):
    data_per2 = perm_state(data_label)
    mean_train_perm2, mean_test_perm2 = NormPSD_NMF_LDA_K_fold(
        data_per2, nfold=5, n_repeat=1, range_lda=range_lda, k_nmf=comp_nmf)
    val_acc_perm2_rep.append(mean_test_perm2[0])
    mean_train_perm2_rep.append(mean_train_perm2[0])

np.savetxt("../results/perm_2_val_acc.csv", val_acc_perm2_rep, delimiter=",")

# 6.3 shuffle only subject labels
def perm_sub(data_label):
    cond_ori = data_label.condition
    subset = set(cond_ori)
    y_per = data_label.subcond.copy(deep = True)
    y_ori = data_label.subcond.copy(deep=True)
    sub_per = data_label.subname.copy(deep=True)
    for cond in subset:
        labs = y_ori[cond_ori == cond].index
        vals = y_ori[cond_ori == cond].values
        vals_per = np.random.permutation(vals)

        # replace original values with permuted ones
        for i in range(vals_per.shape[0]):
            y_per.at[labs[i]] = vals_per[i]
            sub_per.at[labs[i]] = vals_per[i].split('_')[0]

    data_per = data_label.copy(deep=True)
    data_per.subcond = y_per
    data_per['subname'] = sub_per
    return(data_per)

val_acc_perm3_rep = []
train_acc_perm3_rep = []
for rep in range(1):
    data_per3 = perm_sub(data_label)
    mean_train_perm3, mean_test_perm3 = NormPSD_NMF_LDA_K_fold(
        data_per3, nfold=5, n_repeat=1, range_lda=range_lda, k_nmf=comp_nmf)
    val_acc_perm3_rep.append(mean_test_perm3[0])
    train_acc_perm3_rep.append(mean_train_perm3[0])


np.savetxt("../results/perm_3_val_acc.csv", val_acc_perm3_rep, delimiter=",")
