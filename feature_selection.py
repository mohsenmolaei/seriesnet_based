from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import statsmodels.tsa.stattools as stattools
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import lingam

def select_matrix(cond , cryptoname="price_usd_close"):
    features =cond.columns
    matrix_granger = matrix_correlation=matrix_observed=matrix_latent =matric_mi = pd.DataFrame()
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(cond.to_numpy())
    df_scaled = pd.DataFrame(df_scaled,columns=cond.columns)
    df_scaled.fillna(method='ffill', inplace=True) 

    # # correlation
    matrix_correlation = (abs(df_scaled.corr(method="pearson")[cryptoname]) >= 0.5)  
    matrix_correlation = pd.DataFrame(matrix_correlation.astype(int)[1:])

    #Lingam
    model = lingam.VARLiNGAM()
    model.fit(cond)
    #observed
    a = pd.DataFrame( model.adjacency_matrices_[0], index=cond.columns , columns=cond.columns)[cryptoname]
    matrix_observed  = (abs(a) >= 0.01)
    matrix_observed  = pd.DataFrame(matrix_observed.astype(int)[1:])
    #latent
    b = pd.DataFrame( model.adjacency_matrices_[1],index=cond.columns , columns=cond.columns)[cryptoname]
    matrix_latent  = (abs(b) >= 0.00001)
    matrix_latent = pd.DataFrame(matrix_latent.astype(int)[1:]) 

    alpha = 0.05
    matrix_granger = pd.DataFrame(columns=features , index=[cryptoname])
        #Granger
    for  feature in features:
        result = stattools.grangercausalitytests(df_scaled[[cryptoname,feature]], maxlag=2, verbose=False)
        p_value = result[2][0]['ssr_ftest'][1]
        matrix_granger[feature][cryptoname] = p_value < alpha
        matrix_granger[feature][cryptoname] = matrix_granger[feature][cryptoname].astype(int)
    matrix_granger = pd.DataFrame(matrix_granger.astype(int).T[1:])

    #mutual information
    autoscaler = StandardScaler()
    features = autoscaler.fit_transform(cond)
    Y = df_scaled[cryptoname]
    label_encoder = LabelEncoder()
    n_bins = 50
    bins = np.linspace(Y.min(), Y.max(), n_bins + 1)
    bin_indices = np.digitize(Y, bins)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(bin_indices)
    mi= pd.DataFrame()
    mi = mutual_info_classif(df_scaled.drop(cryptoname, axis=1), y_train)
    mi = pd.Series(mi >=0.5)
    mi.index = df_scaled.drop(cryptoname, axis=1).columns
    mi.columns = [cryptoname]
    matric_mi[cryptoname] = pd.DataFrame(mi).astype(int)
    
    sum_matrix = (matrix_correlation + matrix_observed + matrix_latent + matrix_granger + matric_mi).T
    return sum_matrix , matrix_correlation , matrix_observed , matrix_latent , matrix_granger , matric_mi