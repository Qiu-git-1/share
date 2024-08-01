
from __future__ import print_function

import numpy as np
import faiss
from sklearn.covariance import ShrunkCovariance

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import DistanceMetric


'''
The Python code contains two classes, kNN_shrunk_l1 and kNN_shrunk, which implement k nearest neighbors (kNN) based on L1 distances, respectively.
Algorithm and kNN algorithm based on covariance contraction. The main purpose of this code is to perform a near neighbor search on high dimensional data, and you can choose whether to do it or not
Bleaching treatment to reduce noise and improve performance.

Here is a detailed description of each class:

kNN_shrunk_l1
This class uses the NearestNeighbors class from the Scikit-learn library to implement the kNN search. It accepts L1 distance as a metric,
And it can run on the CPU.
The -__init__ method initializes the kNN classifier and can choose whether or not to whiten the data.
The -train method is an empty method, and no training process is actually required in this type of kNN implementation.
The -score method calculates the distance between a given source data and the training dataset and can return an index of the nearest neighbors.

kNN_shrunk
This class uses the Faiss library to implement kNN search and supports GPU acceleration. It can handle vector data or higher dimensional data, and can choose
Whether to whiten or not.
The - __init__ method initializes the Faiss index, which can be either a CPU index or a GPU index, and can choose whether to whiten the data.
The -train method is also an empty method because the Faiss index is already created at initialization.
The -score method uses the Faiss index to calculate the distance between the source data and the training dataset, and can return the index of the nearest neighbor.


'''
class kNN_shrunk_l1:
    def __init__(self, target, K, is_cpu, is_whitening, is_vector = False, shrinkage_factor = 0.1):
        """
                Initialize the kNN_shrunk_l1 class.

                Args:
                - target: indicates the target vector of the training data
                -K: indicates the number of near neighbors
                - is_cpu: indicates whether the CPU index is used
                - is_whitening: Specifies whether whitening is processed
                - is_vector: Whether it is a vector
                - shrinkage_factor: shrinkage_factor
                """

        dist = DistanceMetric.get_metric('l1')
        # Create KNN Classifier
        self.knn = NearestNeighbors(n_neighbors=1, metric='l1')

        # Train the model using the training sets
        self.knn.fit(target)


    def train(self, type):

        pass

    def score(self, src, is_return_ind = False):
        """
                计Calculate the score of the source data on the training model.

                Args:
                - src: indicates the source data
                - is_return_ind: indicates whether the index is returned

                Returns:
                -D: indicates the distance between the source data and the nearest neighbor data
                -I: nearest neighbor index of the source data
                """
        D, I = self.knn.kneighbors(src, return_distance=True)

        #D, I = self.gpu_index.search(np.ascontiguousarray(src.astype('float32')), self.K)

        if is_return_ind:
            return D, I
        else:
            return D



class kNN_shrunk:
    def __init__(self, target, K, is_cpu, is_whitening, is_vector = False, shrinkage_factor = 0.1):
        """
                Initialize the kNN_shrunk class.

                Args:
                - target: indicates the target vector of the training data
                -K: indicates the number of near neighbors
                - is_cpu: indicates whether the CPU index is used
                - is_whitening: Specifies whether whitening is processed
                - is_vector: Whether it is a vector
                - shrinkage_factor: shrinkage_factor
                """
        self.K = K
        self.is_whitening = is_whitening
        self.target = target
        self.shrinkage_factor = shrinkage_factor

        # If the bleaching process
        if is_whitening:
            # ShrunkCovariance was used for covariance estimation
            cov = ShrunkCovariance(shrinkage = self.shrinkage_factor).fit(target).covariance_
            try:
                cov_inv = np.linalg.inv(cov)
            except:
                cov_inv = np.linalg.pinv(cov)
            # Whiten the target data
            target = target.dot(cov_inv)
            self.cov_inv = cov_inv
        # Initialize Faiss GPU resources
        res = faiss.StandardGpuResources()
        # Select the index type based on whether it is a vector
        if is_vector:
            index = faiss.IndexFlatL2(target.shape[1])
        else:
            index = faiss.IndexFlatL2(target.shape[2])
        # Move the Faiss index to the GPU
        self.gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        # If it is not a vector, you need to concatenate the target data
        if is_vector is not True:
            target =  np.concatenate(target,0)
        # If it is a CPU index or a vector index, the CPU index is used
        if is_cpu or is_vector:
            self.gpu_index = index

        # Add the target data to the Faiss index
        self.gpu_index.add(np.ascontiguousarray(target.astype('float32')))

    def train(self, type):

        pass

    def score(self, src, is_return_ind = False):
        #print("src",src.shape)
        """
                Calculate the score of the source data on the training model.

                Args:
                - src: indicates the source data
                - is_return_ind: indicates whether the index is returned

                Returns:
                -D: indicates the distance between the source data and the nearest neighbor data
                -I: nearest neighbor index of the source data
                """
        # If whitened, the source data is whitened
        if self.is_whitening:
            src = src.dot(self.cov_inv)

        # The Faiss index is used to calculate the nearest neighbors of the source data
        D, I = self.gpu_index.search(np.ascontiguousarray(src.astype('float32')), self.K)

        if is_return_ind:
            return D, I
        else:
            return D
