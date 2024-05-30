from cajal import *
from cajal.utilities import *
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.stats import fisher_exact


df = pd.read_csv("/home/patn/kz/5_14/e1260_all_ctrls/e1260_all_ctrls_features.csv")
dfm = pd.read_csv("/home/patn/kz/5_14/e1260_matching_ctrls/e1260_matching_ctrls_features.csv")

from cajal.laplacian_score import laplacian_score_no_covariates

_,dist_dict = read_gw_dists("/home/patn/kz/5_14/geodesic_100_all_gw.csv",header=True)

dmat = dist_mat_of_dict(dist_dict, df['cell_name'])
dmat_m = dist_mat_of_dict(dist_dict, dfm['cell_name'])

quantile = np.quantile(squareform(dmat),.9)

def fisher(dmat, quantile, features):
    if isinstance(features, pd.Series):
        features = features.to_numpy()
    tmp= dmat < quantile
    np.fill_diagonal(tmp,False)
    index_fst, index_snd = np.where(tmp)
    fst = features[index_fst]
    snd = features[index_snd]
    x1 = ((fst == 0) & (snd == 0)).sum()
    x2 = ((fst == 0) & (snd == 1)).sum()
    y1 = ((fst == 1) & (snd == 0)).sum()
    y2 = ((fst == 1) & (snd == 1)).sum()
    print(x1,x2,y1,y2)
    return fisher_exact( np.array( [[x1,x2],[y1,y2]] ))

qs = np.quantile(squareform(dmat_m), np.arange(0.01,0.99,.01))
ell = []
for q in qs:
    ell.append(fisher(dmat_m,q, dfm['e1260'].to_numpy() ).pvalue)
 
import matplotlib.pyplot as plt

plt.plot(ell)
plt.savefig("e1260_matching_fisher_exact.png")