from sklearn.base import BaseEstimator, TransformerMixin
import sys
import numpy as np
import pandas as pd
import os
import cvxpy as cvx

class FeatureWeights(BaseEstimator, TransformerMixin):
    '''Compute feature weights based on pairwise differences and a target

    Parameters
    ----------
    caffe_root: path destination of where caffe is located
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
    def __init__(self,round_weights = True, normalize_condition_number = True, cvx_solver = cvx.CVXOPT, obj_norm = 2):
        self.obj_norm = obj_norm
        self.cvx_solver = cvx_solver
        self.round_weights = round_weights
        self.normalize_condition_number = normalize_condition_number

    def pairwise_diffs(self, np_arr):
        np_diffs = np.empty((len(np_arr)*(len(np_arr)-1)/2, np_arr.shape[1]))
        start_ind = 0
        for i in range(len(np_arr)-1):
            sample = np_arr[i,:]
            diffs = np.sqrt((np_arr[i+1:,:] - sample)**2)
            end_ind = start_ind+len(diffs)
            np_diffs[start_ind:end_ind,:] = diffs
            start_ind = end_ind
        return np_diffs


    def optimize_weights(self, X_diffs, y_diffs):
        #sc = (np.linalg.norm(np.dot(X_diffs.T,X_diffs)))**.5
        if self.normalize_condition_number:
          sc = np.linalg.norm(X_diffs)
        else:
          sc = 1.0
        A = X_diffs/sc
        b = y_diffs/sc
        w = cvx.Variable(X_diffs.shape[1])
        objective = cvx.Minimize(cvx.norm(A*w - b, self.obj_norm))
        constraints = [0 <= w, w <= 1]

        prob = cvx.Problem(objective, constraints)
        prob.solve(solver = self.cvx_solver)

        self.prob_status = prob.status
        self.w_value = w.value

        return self.prob_status, self.w_value

    def fit(self, X=None, y=None):
        X_diffs = self.pairwise_diffs(X)
        y_diffs = self.pairwise_diffs(y[np.newaxis].T)
        self.statusprob, self.weights = self.optimize_weights(X_diffs, y_diffs)
        if self.round_weights:
          self.weights = np.round(self.weights)
        return self


    def transform(self, X):
        found_weights = np.asarray(self.weights).squeeze()
        non_zero_weights = found_weights[found_weights!=0]
        X_rel = X[:,found_weights!=0]*non_zero_weights
        #X_rel = X[:,found_weights!=0]

        return X_rel


if __name__ == "__main__":
  print "nothing to do"