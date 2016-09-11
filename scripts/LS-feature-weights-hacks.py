import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.cross_validation import KFold
import numpy as np
from scipy.stats import hmean
from sklearn import metrics
from FWHacks import FWHacks
from sklearn.pipeline import Pipeline
import os

# In[2]:

src_path = '../data/'
file_types = ('.csv')
 
file_paths = []  
for root, dirs, files in os.walk(src_path):
    file_paths.extend([os.path.join(root, f) for f in files if f.endswith(file_types)])
    
print 'number of files is', len(file_paths)


# In[4]:

biomarkers = ["Stool.Lysozyme",
              "Stool.Lactoferrin", 
              "Stool.SIgA",
              "X..SCFA.Acetate",
              "X..SCFA.Propionate",
              "X..SCFA.Valerate",
              "X..SCFA.Butyrate",
              "Total.SCFA",
              "Butyrate",
              "Stool.pH",
              "Neutrophil.Count",
              "Lymphocyte.Count" ,
              "Monocyte.Count",
              "Esoinophil.Count"]

# In[5]:

scoring_metric = "mean_absolute_error"


def CV_scores(regressor, X, y, data_type, estimator_name):
    y_n = (y - np.mean(y))/np.std(y)
    scores = -1*cross_validation.cross_val_score(regressor, X, y, cv=10, scoring = scoring_metric)
    results = {'data': data_type, "model": estimator_name, "Mean score": np.mean(scores), "STD score": np.std(scores)}
    return results

for biomarker in biomarkers:
    print "working on", biomarker
    summary_stats = []
    for file_path in file_paths:
        #
        # data load and prep
        #
        df = pd.read_csv(file_path)
        fields = list(df.columns)
        microbiome_indx = [i for i, field in enumerate(fields) if "__" in field]
        column_names = list(df.columns)
        target_indx = [i for i, column_name in enumerate(column_names) if column_name == biomarker]
        df_rel = df.iloc[:,target_indx + range(microbiome_indx[0], df.shape[1])]
        df_rel[[biomarker]] = df_rel[[biomarker]].apply(lambda x: pd.to_numeric(x, errors = "coerce"))
        df_clean = df_rel.dropna()
        y = np.array(df_clean[biomarker])
        X = np.array(df_clean.iloc[:,1:])
        #
        # Metric KNN 
        #        
        FW = FWHacks()
        KNN = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=dict(n_neighbors=range(1,11), p=[1,2]), n_jobs = -1, 
                           scoring = scoring_metric)        
        metric_KNN = Pipeline([('metric', FW), ('knn', KNN)])
        summary_stats.append(CV_scores(metric_KNN, X, y, data_type = file_path, estimator_name = "Metric KNN"))
    
    results_df = pd.DataFrame(summary_stats)
    results_df.to_csv("../results/LS/metric_hacks_" + biomarker + ".csv", index = False)


