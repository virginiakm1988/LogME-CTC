import numpy as np
from geomloss import SamplesLoss
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
import io
import ot as ot
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
    n = len(feat1)
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    seed = 42
    n_projections = 50
    swd_distance = ot.sliced.sliced_wasserstein_distance(feat1, feat2, a, b, n_projections, seed=seed)

    return   swd_distance


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
        feat1_lst.append(np.median(features, axis=0))
    feat1 = np.array(np.matrix(feat1_lst))
    print(feat1.shape, "shape of feat1")
    feat2 = np.median(np.asarray(pickle_feature_2), axis =1)
    distance = compute_distance(feat1, feat2)
    ks_random_distance = compute_distance(random_feature,feat2)
    pr_random_distance = compute_distance(random_feature,feat1)
    print("distacne of "+model+": "+str(distance))
    print("ks vs random distacne of "+model+": "+str(ks_random_distance))
    print("pr vs random distacne of "+model+": "+str(pr_random_distance))
    #plot_scatter(feat1_reduced, feat2_reduced,normal_reduced, model)