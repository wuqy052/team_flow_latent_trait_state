
# Import the libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

# 2. Identification of the latent space
# 2A. training & validation heatmap
# training
range_nmf = np.arange(10,80,5)
heatmap_train = np.load('heatmap_train.npy')
plt.rcParams.update({'font.family':'avenir'})
plt.rcParams["figure.figsize"] = (6,5)

sns.heatmap(heatmap_train, vmin=0, vmax=1,cmap=sns.color_palette("blend:#7AB,#EDA", as_cmap=True),
            yticklabels=np.arange(1,21), xticklabels=range_nmf)
plt.title('Training accuracy',fontsize=20)
plt.xlabel('Number of NMF components',fontsize=16)
plt.ylabel('Number of LDs',fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
plt.tight_layout()
plt.savefig("output/Fig2A_1.jpeg",dpi=300)
# validation
heatmap_val = np.load('heatmap_val.npy')
plt.rcParams.update({'font.family':'avenir'})
plt.rcParams["figure.figsize"] = (6,5)
sns.heatmap(heatmap_val, vmin=0, vmax=1,cmap=sns.color_palette("blend:#7AB,#EDA", as_cmap=True),
            yticklabels=np.arange(1,21), xticklabels=range_nmf)
plt.title('Validation accuracy',fontsize=20)
plt.xlabel('Number of NMF components',fontsize=16)
plt.ylabel('Number of LDs',fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
plt.tight_layout()
plt.savefig("output/Fig2A_2.jpeg",dpi=300)
# 2B. AUC
data_AUC = pd.read_csv('AUC.csv')
plt.rcParams["figure.figsize"] = (5,4)
plt.rcParams.update({'font.family':'avenir'})
plt.plot(data_AUC['NMF'],data_AUC['AUC_train'],c=[244/256,165/256,130/256],label='Training')
plt.plot(data_AUC['NMF'],data_AUC['AUC_val'],c= '#41b6c4',label='Validation')
plt.legend(fontsize=14,loc=4)
plt.xlabel('Number of NMF components',fontsize=16)
plt.ylabel('Accuracy AUC',fontsize=16)
plt.xticks(fontsize=14)
plt.axvline(x=40,linestyle='--',c='black',ymin=0.1,ymax=0.98,linewidth=1)
plt.yticks([0,5,10,15,20],[0,5,10,15,20],fontsize=14)
plt.ylim([0,20])
for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.tight_layout()
plt.savefig("output/Fig2B.jpeg",dpi=300)
# 2C. Knee point
data_knee = pd.read_csv('Knee.csv')
plt.rcParams["figure.figsize"] = (5,4)
plt.rcParams.update({'font.family':'avenir'})
plt.plot(data_knee['LD'],data_knee['ACC'],c='#41b6c4')
plt.axvline(x=7,linestyle='--',c='black',ymin=0.1,ymax=0.9,linewidth=1)
plt.xlabel('Number of LDs',fontsize=16)
plt.ylabel('Validation accuracy',fontsize=16)
plt.xticks([1,4,7,10,13,16,19],[1,4,7,10,13,16,19], fontsize=14)
plt.yticks(fontsize=14)
plt.ylim([0,0.6])
for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.tight_layout()
plt.savefig("output/Fig2C.jpeg",dpi=300)
# 2D. 3D visualization
from sklearn import preprocessing
from palettable.colorbrewer.qualitative import Paired_12
from palettable.cartocolors.qualitative import Pastel_10
data_label = pd.read_csv('PSD_label.csv')
data_label["subcond"] = data_label.subname.astype(str) + '_' + data_label.condition.astype(str)
LDA_all = pd.read_csv('condsub_7LD.csv',header = None)
LDA_all = np.array(LDA_all)
le = preprocessing.LabelEncoder()
colist = le.fit_transform(data_label.subname)
plt.rcParams.update({'font.family':'avenir'})
plt.rcParams["figure.figsize"] = (10,10)
ax = plt.axes(projection="3d")
# reverse the LD order
ax.scatter(LDA_all[data_label.condition == 'tm',0], LDA_all[data_label.condition == 'tm',1], LDA_all[data_label.condition == 'tm',2], c = colist[data_label.condition == 'tm'],
             cmap = Pastel_10.mpl_colormap, marker = 'o', s=100, alpha = 0.7)
ax.scatter(LDA_all[data_label.condition == 'tmoc',0], LDA_all[data_label.condition == 'tmoc',1], LDA_all[data_label.condition == 'tmoc',2], c = colist[data_label.condition == 'tmoc'],
             cmap = Pastel_10.mpl_colormap, marker = '^', s=100, alpha = 0.7)
ax.scatter(LDA_all[data_label.condition == 'tmrv',0], LDA_all[data_label.condition == 'tmrv',1], LDA_all[data_label.condition == 'tmrv',2], c = colist[data_label.condition == 'tmrv'],
             cmap = Pastel_10.mpl_colormap, marker = 's', s=100, alpha = 0.7)
ax.set_xlabel('LD1',fontsize = 24)
ax.set_ylabel('LD2',fontsize = 24)
ax.set_zlabel('LD3',fontsize = 24)
ax.set_zticks([-125,-75,-25,25,75])
ax.set_yticks([-80,-40,0,40,80])
ax.set_xticks([-80,-40,0,40,80])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
#ax.set_xlim([-60,-20])
#ax.set_ylim([-80,-40])
#ax.set_zlim([-25,25])
ax.grid(True)
ax.view_init(40, -57)
plt.show()
plt.savefig("output/Fig2D.jpeg",dpi=300)


# 3. Trait-state hierarchy
bwd = pd.read_csv('Figure_hierarchy.csv')
plt.rcParams["figure.figsize"] = (7,4)
plt.hist(bwd.bwd_perm,bins=100,color='#41b6c4' )
plt.xlabel('Petmuted inter/intra individual distance ratio',fontsize=20)
plt.xticks([0,2,4,6,8],[0,2,4,6,8],fontsize = 17)
plt.ylabel('Count',fontsize=20)
plt.yticks([0,40,80,120,160],[0,40,80,120,160],fontsize=17)
plt.axvline(x=7.0808,c=[244/256,165/256,130/256],lw=2)
plt.tight_layout()
plt.savefig("output/Fig3D.jpeg")

# 4. RSA
# 4.1 heatmap trial EEG
#sns.set_theme()
edist_trial = pd.read_csv('Figure_RSA_edist.csv',header=None)
plt.rcParams.update({'font.family':'avenir'})
plt.rcParams["figure.figsize"] = (10.5,8)
sns.set(font = 'avenir',font_scale=1.5)
ax = sns.heatmap(edist_trial, cmap="YlGnBu",vmin = 0, vmax = 12, xticklabels=False, yticklabels = False)
plt.savefig("output/Fig4A.jpeg")
plt.show()
# 4.4.1 heatmap trial beh
behdist_trial = pd.read_csv('Figure_RSA_behdist.csv',header=None)
plt.rcParams["figure.figsize"] = (10.5,8)
sns.set(font = 'avenir',font_scale=1.5)
ax = sns.heatmap(behdist_trial, cmap="YlGnBu",vmin = 0, vmax = 12, xticklabels=False, yticklabels = False)
plt.savefig("output/Fig4B.jpeg")
plt.show()
# 4.3 correlation
rsa_corr_trial = pd.read_csv('Figure_RSA_corr.csv')
plt.rcParams.update({'font.family':'avenir'})
plt.rcParams["figure.figsize"] = (6,5)
plt.scatter(rsa_corr_trial.edist,rsa_corr_trial.beh,s=10,alpha=0.05,c='#41b6c4')
#find line of best fit
a3, b3 = np.polyfit(rsa_corr_trial.edist,rsa_corr_trial.beh, 1)
plt.plot( rsa_corr_trial.edist,a3*rsa_corr_trial.edist+b3,c=[221/256,221/256,221/256])
plt.ylabel('Skill/Cognition Distance',fontsize=20)
plt.yticks([0,2,4,6,8],[0,2,4,6,8],fontsize = 17)
plt.xlabel('Latent EEG distance',fontsize=20)
plt.xticks([0,2,4,6],[0,2,4,6],fontsize=17)
plt.tight_layout()
plt.savefig("output/Fig4C.jpeg")
plt.show()
corr_coef = stats.pearsonr(rsa_corr_trial.edist,rsa_corr_trial.beh)
# 4.4 histogram
rsa_perm_trial = pd.read_csv('Figure_RSA_permutation.csv')
plt.rcParams["figure.figsize"] = (6,5)
plt.hist(rsa_perm_trial.r_perm,bins=100,color='#41b6c4')
plt.xlabel('Permuted R',fontsize=20)
plt.xticks([-0.1,0,0.1,0.2,0.3],[-0.1,0,0.1,0.2,0.3],fontsize = 17)
plt.ylabel('Count',fontsize=20)
plt.yticks([0,40,80,120,160],[0,40,80,120,160],fontsize=17)
plt.axvline(x=corr_coef[0],c=[244/256,165/256,130/256],lw=2)
plt.tight_layout()
plt.savefig("output/Fig4D.jpeg")

# 5. Inter-individual & Inter condition
# 5A: t-test for each LD
interind = pd.read_csv('Figure_BetweenInd.csv')
pal_ld = {"Null": "#bdbdbd", "LD1":"#41b6c4","LD2":"#d5f0b3","LD3":"#41b6c4",
          "LD4":"#d5f0b3","LD5":"#41b6c4","LD6":"#d5f0b3","LD7":"#d5f0b3"}
order_ld = ["Null","LD1","LD3","LD5","LD4","LD6","LD2","LD7"]
plt.rcParams["figure.figsize"] = (9,5)
plt.rcParams.update({'font.family':'avenir'})
sns.boxplot(data=interind, x="LD", y="Ratio", palette=pal_ld,showfliers=False, width=0.6,order=order_ld,
            showmeans=True,meanprops={"markeredgecolor": "black","markerfacecolor":"white", "markersize": "12"})
sns.despine()
plt.axhline(y=1.3467,c=[244/256,165/256,130/256],lw=2,linestyle='--')
plt.xlabel(' ')
plt.xticks(fontsize = 18)
plt.ylabel('Repeated/Non-repeated \nparticipation distance ratio',fontsize=20)
plt.yticks(fontsize=17)
plt.tight_layout()
plt.savefig("output/Fig5A.jpeg")
# 5B: ANOVA across 3 conditions
intercond = pd.read_csv('Figure_InterConditionANOVA.csv')
pal_cond = {"Team Flow": "#b8d98b", "Team Only": "#fcd95b", "Flow Only":"#6cb7e6"}
# LD1
plt.rcParams["figure.figsize"] = (5,5)
plt.rcParams.update({'font.family':'avenir'})
sns.boxplot(data=intercond, x="Condition", y="LD1", palette=pal_cond, showfliers=False, width=0.6,
            showmeans=True,meanprops={"markeredgecolor": "black","markerfacecolor":"white", "markersize": "12"})
sns.despine()
plt.ylabel('Normalized LD1',fontsize=20)
plt.xticks(fontsize = 17)
plt.yticks([-10,-5,0,5,10],[-10,-5,0,5,10],fontsize = 17)
plt.ylim([-15,15])
plt.xlabel('')
plt.tight_layout()
plt.savefig("output/Fig5B_1.jpeg")
# LD3
plt.rcParams["figure.figsize"] = (5,5)
plt.rcParams.update({'font.family':'avenir'})
sns.boxplot(data=intercond, x="Condition", y="LD3", palette=pal_cond,showmeans=True,width=0.6,
            showfliers=False,meanprops={"markeredgecolor": "black","markerfacecolor":"white", "markersize": "12"})
sns.despine()
plt.ylabel('Normalized LD3',fontsize=20)
plt.xticks(fontsize = 17)
plt.yticks([-10,-5,0,5,10],[-10,-5,0,5,10],fontsize = 17)
plt.ylim([-12,12])
plt.xlabel('')
plt.tight_layout()
plt.savefig("output/Fig5B_2.jpeg")
# LD5
plt.rcParams["figure.figsize"] = (5,5)
plt.rcParams.update({'font.family':'avenir'})
sns.boxplot(data=intercond, x="Condition", y="LD5", palette=pal_cond,showmeans=True,width=0.6,
            showfliers=False,meanprops={"markeredgecolor": "black","markerfacecolor":"white", "markersize": "12"})
sns.despine()
plt.ylabel('Normalized LD5',fontsize=20)
plt.xticks(fontsize = 17)
plt.yticks([-10,-5,0,5,10],[-10,-5,0,5,10],fontsize = 17)
plt.ylim([-12,12])
plt.xlabel('')
plt.tight_layout()
plt.savefig("output/Fig5B_3.jpeg")

# 6. Projection matrix
import scipy.io as sio
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.io import savemat
mat_proj = sio.loadmat('../data/LD_NMF_project.mat')
proj = mat_proj['LD_loading']
bottom = cm.get_cmap('Oranges', 128)
top = cm.get_cmap('Blues_r', 128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')
cmapmat={"colormap": newcolors}
savemat("../analysis/bluewhiteorange.mat", cmapmat)
# the full projection matrix
plt.rcParams.update({'font.family':'avenir'})
plt.rcParams["figure.figsize"] = (12,6)
fig, ax = plt.subplots(1,1)
img = ax.imshow(proj.T, aspect='8', vmax = 25, vmin = -25,  cmap=newcmp)
y_label_list = ['LD1', 'LD2', 'LD3','LD4','LD5','LD6','LD7']
x_label_list = ['10','20','30','40','10','20','30','40','10','20','30','40','10','20','30','40']
ax.set_yticks([0, 1, 2, 3,4,5, 6])
ax.set_xticks([9,19,29,39,49,59,69,79,89,99,109,119,129,139,149,159])
ax.set_xticklabels(x_label_list, fontsize = 14)
ax.set_yticklabels(y_label_list, fontsize = 18)
plt.xlabel('NMF',fontsize=18)
# plt.title('Projection between original features and hidden dimensions')
# plt.colorbar()
fig.colorbar(img, shrink=0.5)
plt.show()
plt.savefig("output/Fig6A_full.jpeg",dpi=300)
# LD 1, 3, 5, beta & gamma only
mat_keep = sio.loadmat('../data/LD_NMF_important.mat')
keep = mat_keep['NMF_per_LD']
proj_part = keep[80:160,[0,2,4]]#proj[80:160,[0,2,4]]
plt.rcParams.update({'font.family':'avenir'})
fig, ax = plt.subplots(figsize=(15,4))
img = ax.imshow(proj_part.T, aspect='2', vmax = 25, vmin = -25,  cmap=newcmp)
y_label_list = ['LD1','LD3','LD5']
x_label_list = ['10','20','30','40','10','20','30','40']
ax.set_yticks([0, 1, 2])
ax.set_xticks([9,19,29,39,49,59,69,79])
ax.set_xticklabels(x_label_list, fontsize = 14)
ax.set_yticklabels(y_label_list, fontsize = 18)
plt.xlabel('NMF',fontsize=18)
# plt.title('Projection between original features and hidden dimensions')
# plt.colorbar()
#fig.colorbar(img, shrink=0.4)
plt.show()
plt.savefig("output/Fig6A_part.jpeg",dpi=300)

# Supplementary
# S1. Permutation test
# A. permutation 1
perm_1 = np.genfromtxt('../results/perm_1_val_acc.csv', delimiter=',')
plt.rcParams["figure.figsize"] = (5,4)
plt.hist(perm_1*100,bins=15,color='#41b6c4')
plt.xlabel('Validation accuracy (%)',fontsize=18)
plt.xticks(fontsize = 15)
plt.ylabel('Count',fontsize=18)
plt.yticks(fontsize=15)
plt.axvline(x=44.95,c=[244/256,165/256,130/256],lw=2)
plt.tight_layout()
plt.savefig("output/FigS1A.jpeg",dpi=300)
# B. permutation 2
perm_2 = np.genfromtxt('../results/perm_2_val_acc.csv', delimiter=',')
plt.rcParams["figure.figsize"] = (5,4)
plt.hist(perm_2*100,bins=15,color='#41b6c4')
plt.xlabel('Validation accuracy (%)',fontsize=18)
plt.xticks(fontsize = 15)
plt.ylabel('Count',fontsize=18)
plt.yticks(fontsize=15)
plt.axvline(x=44.95,c=[244/256,165/256,130/256],lw=2)
plt.tight_layout()
plt.savefig("output/FigS1B.jpeg",dpi=300)
# B. permutation 3
perm_3 = np.genfromtxt('../results/perm_3_val_acc.csv', delimiter=',')
plt.rcParams["figure.figsize"] = (5,4)
plt.hist(perm_3*100,bins=15,color='#41b6c4')
plt.xlabel('Validation accuracy (%)',fontsize=18)
plt.xticks(fontsize = 15)
plt.ylabel('Count',fontsize=18)
plt.yticks(fontsize=15)
plt.axvline(x=44.95,c=[244/256,165/256,130/256],lw=2)
plt.tight_layout()
plt.savefig("output/FigS1C.jpeg",dpi=300)



# S2. LDA
# A. explained variance
exp_var = np.genfromtxt('../results/LD_exp_variance.csv', delimiter=',')
plt.rcParams.update({'font.family':'avenir'})
plt.rcParams["figure.figsize"] = (5,4)
plt.plot(np.arange(1,8),np.cumsum(exp_var)*100,color='#41b6c4')
plt.ylim([0,100])
plt.xlabel('LDA components',fontsize=15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize=12)
plt.ylabel('Cumulative explained variance (%)',fontsize=15)
plt.tight_layout()
plt.savefig("output/FigS1A.jpeg",dpi=300)
# B. in-sample classification accuracy
score_insample = np.genfromtxt('../results/Acc_insample.csv',delimiter=',')
plt.rcParams.update({'font.family':'avenir'})
plt.rcParams["figure.figsize"] = (5,4)
plt.plot(range(1,8),score_insample*100,color='#41b6c4')
plt.xlabel('Latent space dimension',fontsize=15)
plt.ylabel('Classification accuracy (%)',fontsize=15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize=12)
plt.ylim([0,110])
plt.tight_layout()
plt.savefig("output/FigS1B.jpeg",dpi=300)

# S4
# bar plots for all LDs across 3 conditions
intercond = pd.read_csv('Figure_InterConditionANOVA.csv')
pal_cond = {"Team Flow": "#b8d98b", "Team Only": "#fcd95b", "Flow Only":"#6cb7e6"}
LD_list = ['LD1','LD2','LD3','LD4','LD5','LD6','LD7']
Letter_list = ['A','B','C','D','E','F','G']
for (ld,letter) in zip(LD_list,Letter_list):
    plt.rcParams["figure.figsize"] = (5,4)
    plt.rcParams.update({'font.family':'avenir'})
    sns.barplot(data=intercond, x="Condition", y=ld, hue="Condition", palette=pal_cond, width=0.6,errorbar='se')
    sns.despine()
    plt.ylabel('Normalized '+ld,fontsize=20)
    plt.xticks(fontsize = 17)
    plt.yticks(fontsize = 17)
    plt.ylim([-3,3])
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig("output/FigS4"+letter+".jpeg",dpi=300)
    plt.clf()