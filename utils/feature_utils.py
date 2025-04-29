import numpy as np



def get_feature(feature_name:str):
    if feature_name=="fourier_feature":
        return fourier_featrue
    if feature_name=="sparse_feature":
        return sparse_feature
    if feature_name=="non_negative_feature":
        return non_negative_feature
    raise ValueError(f'Unknown feature name: {feature_name}. Choose among ["fourier_feature","sparse_feature","non_negative_feature"]')


def fourier_featrue(num_states:int,num_features:int):
    X = np.round(np.random.uniform(-3,3,(num_states,num_features)),2)
    X = np.round(np.cos(np.pi*X),2)
    return X

def sparse_feature(num_states:int,num_features:int):
    X = np.round(np.random.uniform(-1,1,(num_states,num_features)),2)
    X = np.where(np.logical_or(X>0.5,X<-0.5),X,0)
    return X


def non_negative_feature(num_states:int,num_features:int):
    X = np.round(np.random.uniform(0,3,(num_states,num_features)),2)
    X = np.where(np.logical_or(X>1,X<-1),X,0)
    return X