import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist
from sklearn.decomposition import NMF
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
import time
from quantile_normalize_yoo import quantile_normalize

def various_nmf(inputs, InitA=None, InitS=None, cal_type = 'original', with_sqrt = True):
    X, k, maxiter = inputs
    # figure out the size of data matrix
    m, N = X.shape
    # initial conditions
    if InitA is None:
        A = np.random.rand(m, k)
    else:
        A = InitA

    if InitS is None:
        S = np.random.rand(k, N)
    else:
        S = InitS
    eps = 2.2204e-16
    for its in range(maxiter):
        if cal_type == 'nmf':
            if with_sqrt:
                S = S * np.sqrt((A.T @ X + eps) / (A.T @ A @ S + eps))
                A = A * np.sqrt((X @ S.T + eps) / (A @ S @ S.T + eps))
            else:
                S = S * ((A.T @ X + eps) / (A.T @ A @ S + eps))
                A = A * ((X @ S.T + eps) / (A @ S @ S.T + eps))
                
        elif cal_type == 'bionmf':
            S = S * np.sqrt((A.T @ X + eps) / (S @ X.T @ A @ S + eps))
            A = A * np.sqrt((X @ S.T + eps) / ((A @ S) @ (X.T @ A) + eps))
        elif cal_type == 'aonmf_original':
            if with_sqrt:
                S = S * np.sqrt((A.T @ X + eps) / (A.T @ A @ S + eps))
                A = A * np.sqrt((X @ S.T + eps) / ((A @ S) @ (X.T @ A) + eps))
                A = A / np.sum(A, axis = 0).reshape(1, -1)
            else:
                S = S * ((A.T @ X + eps) / (A.T @ A @ S + eps))
                A = A * ((X @ S.T + eps) / ((A @ S) @ (X.T @ A) + eps))
                A = A / np.sum(A, axis = 0).reshape(1, -1)
        elif cal_type == 'aonmf_A_norm':
            A = A * np.sqrt((X @ S.T + eps) / ((A @ S) @ (X.T @ A) + eps))
            A = A / np.sum(A, axis = 0).reshape(1, -1)
            S = S * np.sqrt((A.T @ X + eps) / (A.T @ A @ S + eps))
        elif cal_type == 'aonmf_S_norm':
            S = S * np.sqrt((A.T @ X + eps) / (A.T @ A @ S + eps))
            S = S / np.sum(S, axis = 1).reshape(-1, 1)
            A = A * np.sqrt((X @ S.T + eps) / ((A @ S) @ (X.T @ A) + eps))
        elif cal_type == 'aonmf_no_norm':
            S = S * np.sqrt((A.T @ X + eps) / (A.T @ A @ S + eps))
            A = A * np.sqrt((X @ S.T + eps) / ((A @ S) @ (X.T @ A) + eps))
        elif cal_type == 'sonmf_A_norm':
            if with_sqrt:
                A = A * np.sqrt((X @ S.T + eps) / (A @ S @ S.T + eps))
                A = A / np.sum(A, axis = 0).reshape(1, -1)
                S = S * np.sqrt((A.T @ X + eps) / (S @ X.T @ A @ S + eps))
            else:
                A = A * ((X @ S.T + eps) / (A @ S @ S.T + eps))
                A = A / np.sum(A, axis = 0).reshape(1, -1)
                S = S * ((A.T @ X + eps) / (S @ X.T @ A @ S + eps))
                
        elif cal_type == 'sonmf_S_norm':
            S = S * np.sqrt((A.T @ X + eps) / (S @ X.T @ A @ S + eps))
            S = S / np.sum(S, axis = 1).reshape(-1, 1)
            A = A * np.sqrt((X @ S.T + eps) / (A @ S @ S.T + eps))
        elif cal_type == 'sonmf_no_norm':
            S = S * np.sqrt((A.T @ X + eps) / (S @ X.T @ A @ S + eps))
            A = A * np.sqrt((X @ S.T + eps) / (A @ S @ S.T + eps))

    ind = np.argmax(A, axis=1)
    C = (ind[:, None] == ind).astype(int)
    return A, S, C


def aoNMF_subtyping(data, Nbasis, Nmfiter, Numiter, time_start):
    # data (row: sample, column: gene)
    X = data.copy()
    X = np.hstack((X, -X))
    print(X.shape)
    X[X < 0] = 0
    # AONMF
    Ss = []
    Aa = []
    total_C = np.zeros((data.shape[0], data.shape[0]))

    for k in range(Numiter):
        if (k+1) % 10 == 0:
            t_now = time.time()-time_start
            print(f"{k}/{Numiter} : {t_now//60}min {t_now%60 : .3f}sec")
        A,S,C =various_nmf((X,Nbasis,Nmfiter), cal_type = 'aonmf_original', with_sqrt = True)
        # A,S,C =various_nmf((X,Nbasis,Nmfiter), cal_type = 'aonmf_original', with_sqrt = False)
        Ss.append(S)
        Aa.append(A)
        total_C += C

    ave_C = total_C / Numiter
    Z = linkage(ave_C, method='average')
    Y = pdist(ave_C)
    coph_cor, _ = cophenet(Z, Y)
    dend_result = dendrogram(Z, no_plot = True,color_threshold=Z[-(k-1),2])

    return coph_cor, ave_C, Aa, Ss, dend_result, Z

if __name__ == "__main__":
    time_start = time.time()
    args = sys.argv[1:]
    MAD_num = int(args[0])
    MAD_quantile = (100-MAD_num)/100
    k_in = int(args[1])
    save_loc = args[2]
    df_name = args[3]
    file_pri = args[4]
    if len(args) > 5:
        Filter_file = args[5]
        if Filter_file != "False":
            Filter_sample = True
        else:
            Filter_sample = False
    else:
        Filter_sample = False
    print(MAD_num,k_in)
    file_list = os.listdir(save_loc)
    file_name = f'{file_pri}_MAD{MAD_num}_k_{k_in}.pkl'
    # if file_name in file_list:
        # print('file_exist')
    if False:
        pass
    else:
        # Specify the file path to save the pickled data
        file_path = f'{save_loc}{file_name}'
        
        final_df = pd.read_csv(df_name).set_index('Unnamed: 0')
        final_df = final_df[final_df.isna().sum(axis = 1)==0]
        print(final_df.shape)
        final_df_sub = final_df.copy().T
        if Filter_sample:
            print(final_df_sub.shape)
            print(f'additional_filtering: {Filter_file}')
            filtering_df = pd.read_csv(Filter_file).set_index('Unnamed: 0')
            filter_col = filtering_df.index
            filter_index = filtering_df.columns

            
        if Filter_sample:
            final_df_sub_mad = final_df_sub.loc[:,filter_col]
            final_df_sub_mad = (final_df_sub_mad-final_df_sub_mad.median()).abs().median()
            final_df_sub_mad = final_df_sub_mad>final_df_sub_mad.quantile(MAD_quantile)
            final_df_sub_mad = final_df_sub_mad[final_df_sub_mad].index
            
            final_df_sub_ckm = final_df_sub[final_df_sub_mad].copy()
        elif MAD_quantile == 0:
            final_df_sub_ckm = final_df_sub.copy()
        else:
            final_df_sub_mad = (final_df_sub-final_df_sub.median()).abs().median()
            final_df_sub_mad = final_df_sub_mad>final_df_sub_mad.quantile(MAD_quantile)
            final_df_sub_mad = final_df_sub_mad[final_df_sub_mad].index

            final_df_sub_ckm = final_df_sub[final_df_sub_mad].copy()
        final_df_sub_ckm = (final_df_sub_ckm -final_df_sub_ckm.mean())/(final_df_sub_ckm.std())
        if Filter_sample:
            final_df_sub_ckm = final_df_sub_ckm.loc[filter_index,:]
        final_df_sub_ckm = final_df_sub_ckm.fillna(0)
        

        print(final_df_sub_ckm.shape)
        
        [coph_cor_C, ave_C, _, _, dendro_C, Z_C] = aoNMF_subtyping(final_df_sub_ckm.values,k_in,1000,100,time_start)
        
        # Specify the file path to save the pickled data

        # Create a dictionary to store the data
        data = {
            'coph_cor_C': coph_cor_C,
            'ave_C': ave_C,
            'dendro_C': dendro_C,
            'Z_C' : Z_C
        }

        # Pickle dump the data
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f'MAD{MAD_num}_k_{k_in} DONE!!')