from sklearn.base import BaseEstimator, TransformerMixin
import sys
import numpy as np
from scipy import sparse
from FeatureWeights import FeatureWeights
from sklearn.neighbors import KNeighborsRegressor
from sklearn.grid_search import GridSearchCV
import pandas as pd
import os
import cvxpy as cvx
from sklearn.pipeline import Pipeline

class FeatureWeightsRegressor(BaseEstimator, TransformerMixin):
    '''Compute feature weights based on pairwise differences and a target

    Parameters
    ----------
    Attributes
    ----------
    layer_names: list
    Examples
    --------
    from FeatureWeights import FeatureWeights
    FW = FeatureWeights()
    FW.fit(X, y)
    X_weighted = FW.transform(X)
    '''
    def __init__(self,num_sample_neighbors = 5, n_iter = 10):
        self.num_sample_neighbors = num_sample_neighbors
        self.n_iter = n_iter


    def fit(self, X_train=None, y_train=None):
        self.X_train = X_train
        self.y_train = y_train

        return self

    def _lexsort_based(self, data):
        sorted_data =  data[np.lexsort(data.T),:]
        row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
        return sorted_data[row_mask]

    def find_nearest(self, X):
        num_neighbors = self.num_sample_neighbors
        X_train_weighted = self.X_train_weighted
        X_train = self.X_train
        y_train = self.y_train
        X_nearest = np.empty((0,X.shape[1]))
        y_nearest = np.empty(0)
        for i in range(X.shape[0]):
            dists = np.sum((X[i,:] - X_train_weighted)**2, axis = 1)
            if i < 0:
                print "nearest neighbors are:", np.argsort(dists)[:num_neighbors]
            X_nearest = np.vstack((X_nearest, X_train[np.argsort(dists)[:num_neighbors],:]))
            #X_nearest = np.vstack((X_nearest, X_train_weighted[np.argsort(dists)[:num_neighbors],:]))
            X_train_weighted = np.delete(X_train_weighted, np.argsort(dists)[:num_neighbors], axis = 0)
            y_nearest = np.hstack((y_nearest, y_train[np.argsort(dists)[:num_neighbors]]))
            y_train = np.delete(y_train, np.argsort(dists)[:num_neighbors])

        print "unique rows for X_nearest", self._lexsort_based(X_nearest).shape
        return X_nearest, y_nearest


    def predict_tester(self, X, y):
        self.X_train_weighted = self.X_train
        self.errs = []
        X_weighted = X
        self.KNN = GridSearchCV(estimator=KNeighborsRegressor(), 
                param_grid=dict(n_neighbors=range(1,11)), n_jobs = -1, 
                scoring = 'mean_absolute_error')
        self.KNN.fit(self.X_train_weighted, self.y_train)
        preds = self.KNN.predict(X_weighted)
        self.errs.append(np.mean(abs(preds-y)))

        FW = FeatureWeights(round_weights=False, upper_bound=False, cvx_solver=cvx.SCS,
            sparsify_weights=False)
        self.weights = []
        self.cvx_status = []
        for i in range(self.n_iter):
            print '****************************************'
            print "iter", i
            X_train_nearest, y_train_nearest = self.find_nearest(X_weighted)

            FW.fit(X_train_nearest, y_train_nearest)
            self.X_train_weighted = FW.transform(self.X_train_weighted)
            X_weighted = FW.transform(X_weighted)
            self.weights.append(FW.weights)
            self.cvx_status.append(FW.statusprob)


            self.KNN.fit(self.X_train_weighted, self.y_train)
            preds = self.KNN.predict(X_weighted)
            self.errs.append(np.mean(abs(preds-y)))
            print FW.statusprob, np.mean(abs(preds-y))


    def predict(self, X):
        self.X_train_weighted = self.X_train
        X_weighted = X
        FW = FeatureWeights(round_weights=False, upper_bound=False, cvx_solver=cvx.SCS,
            sparsify_weights=False)
        self.weights = []
        self.cvx_status = []
        for i in range(self.n_iter):
            print '****************************************'
            print "iter", i
            X_train_nearest, y_train_nearest = self.find_nearest(X_weighted)
            #print "train nearest shape", X_train_nearest.shape
            #self.X_train_weighted = FW.fit_transform(X_train_nearest, y_train_nearest)
            FW.fit(X_train_nearest, y_train_nearest)
            self.X_train_weighted = FW.transform(self.X_train_weighted)
            #print "train weighted shape", self.X_train_weighted.shape
            X_weighted = FW.transform(X_weighted)
            self.weights.append(FW.weights)
            print "cvx status", FW.statusprob
            self.cvx_status.append(FW.statusprob)

        X_train_nearest, y_train_nearest = self.find_nearest(X_weighted)
        #print X_train_nearest.shape
        self.KNN = GridSearchCV(estimator=KNeighborsRegressor(), 
            param_grid=dict(n_neighbors=range(1,11)), n_jobs = -1, 
            scoring = 'mean_absolute_error')
        self.FW = FeatureWeights(upper_bound=False, cvx_solver=cvx.SCS, obj_norm=2)

        #metric_KNN = Pipeline([('metric', self.FW), ('knn', KNN)])
        #metric_KNN.fit(X_train_nearest, y_train_nearest)
        self.FW.fit(X_train_nearest, y_train_nearest)
        X_train_transformed = self.FW.transform(self.X_train_weighted)
        print "X_train_transformed shape:", X_train_transformed.shape
        X_test_transformed = self.FW.transform(X_weighted)
        self.KNN.fit(X_train_transformed, self.y_train)
        return self.KNN.predict(X_test_transformed)
        #return metric_KNN.predict(X_weighted)



if __name__ == "__main__":
  print "nothing to do"