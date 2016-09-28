import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
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
from sklearn.pipeline import Pipeline
import os


src_path = '../data/LS/81-samples/'
file_types = ('.csv')
 
file_paths = []  
for root, dirs, files in os.walk(src_path):
    file_paths.extend([os.path.join(root, f) for f in files if f.endswith(file_types)])
    
print 'number of files is', len(file_paths)


# biomarkers = ["Stool.Lysozyme",
#               "Stool.Lactoferrin", 
#               "Stool.SIgA",
#               "X..SCFA.Acetate",
#               "X..SCFA.Propionate",
#               "X..SCFA.Valerate",
#               "X..SCFA.Butyrate",
#               "Total.SCFA",
#               "Butyrate",
#               "Stool.pH",
#               "Neutrophil.Count",
#               "Lymphocyte.Count" ,
#               "Monocyte.Count",
#               "Esoinophil.Count"]


biomarkers = ["BMI","Calprotectin", "Lysozyme", "Lactoferrin", "Sec.IgA"]



scoring_metric = "mean_absolute_error"


def CV_scores(regressor, X, y, data_type, estimator_name):
    num_folds = 5
    scores = -1*cross_validation.cross_val_score(regressor, X, y, cv=num_folds, scoring = scoring_metric)
    results = {'data': data_type, "model": estimator_name, "Mean score": np.mean(scores), "STD score": np.std(scores), 
               "num_folds": num_folds}
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
        # Elastic Net
        #
        ENet = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], eps=0.001, n_alphas=100, normalize=False, 
                            max_iter=1000, tol=0.0001, cv=3, copy_X=True, n_jobs=-1)
        summary_stats.append(CV_scores(ENet,X, y, data_type = file_path, estimator_name = "Elastic Net"))
        print "ENet done"
        #
        # Linear SVM
        # 
        linear_SVM = GridSearchCV(estimator=SVR(kernel='linear'), param_grid=dict(C=np.logspace(-2,10,100)), n_jobs = -1, 
                                  scoring = scoring_metric)
        summary_stats.append(CV_scores(linear_SVM,X, y, data_type = file_path, estimator_name = "Linear SVM"))
        print "linear SVM done"
        #    
        # RF
        #
        RF = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
        summary_stats.append(CV_scores(RF,X, y, data_type = file_path, estimator_name = "RF"))
        print "RF done"
        #
        # KNN 
        #
        KNN = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=dict(n_neighbors=range(1,11), p=[1,2]), n_jobs = -1, 
                           scoring = scoring_metric)        
        summary_stats.append(CV_scores(KNN,X, y, data_type = file_path, estimator_name = "KNN"))
        print "KNN done"
        
    
    results_df = pd.DataFrame(summary_stats)
    results_df.to_csv("../results/LS/81-samples/5-fold/benchmark_" + biomarker + ".csv", index = False)




