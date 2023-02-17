import numpy as np
from geomloss import SamplesLoss
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
import io
import pickle
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
np.random.seed(42)

def compute_distance(feat1, feat2):
    """
    Compute the sliced Wasserstein distance and t-SNE distance between the means
    of two data distributions.

    Args:
        feat1 (np.ndarray): numpy array of shape (sample_num, m, n)
        feat2 (np.ndarray): numpy array of shape (sample_num, n)

    Returns:
        swd_distance (float): the sliced Wasserstein distance between the means of feat1 and feat2
        tsne_distance (float): the t-SNE distance between the means of feat1 and feat2
    """
    
    n_components = 50 # number of dimensions for the reduced features
    pca = PCA(n_components=n_components)
    tsne = TSNE(n_components=2, random_state=42, learning_rate="auto", n_iter=3000)

    feat1_reduced = tsne.fit_transform(feat1)
    feat2_reduced = tsne.fit_transform(feat2)

    print(feat1_reduced.shape)
    tsne_distance = np.linalg.norm(np.median(feat1_reduced, axis=0) - np.median(feat2_reduced, axis=0))

    # print('Sliced Wasserstein distance:', swd)
    print('t-SNE distance:', tsne_distance)


    return  feat1_reduced, feat2_reduced, tsne_distance


def plot_scatter(feat1_reduced, feat2_reduced,random_reduced, model):
    fig, ax = plt.subplots()
    dots_size = 10
    ax.scatter(feat1_reduced[:, 0], feat1_reduced[:, 1], label='librispeech', s = dots_size)
    ax.scatter(feat2_reduced[:, 0], feat2_reduced[:, 1], label='speech command', s = dots_size)
    ax.scatter(random_reduced[:, 0], random_reduced[:, 1], label='random', s = dots_size)
    ax.set_title('t-SNE visualization of '+model)
    ax.legend() 
    plt.savefig("figs/"+model+"_tsne.png") 

models = ["hubert", "wav2vec2", "decoar2"]#sys.argv[1]


random_feature = np.random.rand(1001, 768)

for model in models:
    feat1_path = f"/home/virginiakm1988/s3prl/s3prl/result/KS_features/PR_{model}_baseline.pkl"
    feat2_path = f"/home/virginiakm1988/s3prl/s3prl/result/KS_features/KS_{model}_baseline.pkl"

    with open(feat1_path, "rb") as fp:   # Unpickling
        pickle_feature_1 = CPU_Unpickler(fp).load()
    with open(feat2_path, "rb") as fp:   # Unpickling
        pickle_feature_2 = CPU_Unpickler(fp).load()
    feat1 = np.asarray(pickle_feature_1)
    feat1_lst = []
    for idx,features in enumerate(feat1):
        feat1_lst.append(np.mean(features, axis=0))
    feat1 = np.matrix(feat1_lst)
    feat2 = np.mean(np.asarray(pickle_feature_2), axis =1)
    feat1_reduced, feat2_reduced, tsne_distance = compute_distance(feat1, feat2)
    normal_reduced,_, ks_random_tsne_distance = compute_distance(random_feature,feat2)
    _,_, pr_random_tsne_distance = compute_distance(random_feature,feat1)
    print("distacne of "+model+": "+str(tsne_distance))
    print("ks vs random distacne of "+model+": "+str(ks_random_tsne_distance))
    print("pr vs random distacne of "+model+": "+str(pr_random_tsne_distance))
    plot_scatter(feat1_reduced, feat2_reduced,normal_reduced, model)