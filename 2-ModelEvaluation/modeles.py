import os
import numpy as np
import pickle
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier



class ModelEvaluation():
    def __init__(self,out_dir):
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        self.models = [ xgb.XGBClassifier(n_jobs=-1, n_estimators=1000, max_depth=10, **{'gpu_id': 0, 'tree_method': 'gpu_hist'} ),
                        SVC(gamma='auto'), NuSVC(gamma='auto'), LinearSVC(),
                        SGDClassifier(max_iter=100, tol=1e-3), 
                        LogisticRegression(solver='lbfgs'), #LogisticRegressionCV(cv=3, n_jobs=-1),
                        BaggingClassifier(verbose=True), 
                        ExtraTreesClassifier(n_estimators=300, verbose=True, n_jobs=-1),
                        MLPClassifier(solver='lbfgs', hidden_layer_sizes=[100,100], verbose=True),
                        RandomForestClassifier(n_estimators=1000, max_depth=15,max_features=128,verbose=True, n_jobs=-1)
                      ]

        self.cmap = plt.cm.get_cmap('hsv', len(self.models)+5)
        self.out_dir = out_dir

    def train(self, X_train, y_train):
        for model in self.models:
            print('training %s'%model.__class__.__name__)
            model.fit(X_train,y_train)

    def evaluate_plot(self, X_test, y_test):
        figr = plt.figure(figsize=(10,10))
        for i, model in enumerate(self.models):
            name = model.__class__.__name__
            print('testing %s'%name)
            self.roc_plot(name, model, X_test, y_test, self.cmap(i), figr)
        plt.legend(loc='best')
        plt.show()

        #save
        plt.savefig(os.path.join(self.out_dir,'roc.png'))
        for model in self.models:
            name = model.__class__.__name__
            filename = os.path.join(self.out_dir,'%s.sav'%name)
            pickle.dump(model, open(filename, 'wb'))
        

    def evaluate_score(self, X_test, y_test):
        for model in self.models:
            self.score_model(X_test, y_test, model)

    def score_model(self, X, y, estimator):
        """
        Test various estimators.
        """
        expected  = y
        predicted = estimator.predict(X)

        # Compute and return F1 (harmonic mean of precision and recall)
        print("{}: {}".format(estimator.__class__.__name__, f1_score(expected, predicted)))
    
    def roc_plot(self, name, mdl, X_test, y_test, color, figure=None):
        features, target = X_test, y_test

        # Get predicted probabilities
        target_probabilities = mdl.predict_proba(features)[:,1]

        # Create true and false positive rates
        false_positive_rate, true_positive_rate, threshold = roc_curve(target, target_probabilities)
        
        if figure is None:
            plt.figure(figsize=(10,10))
        
        y_test_pred = mdl.predict(features)
        f1_test = f1_score(target, y_test_pred) 

        # Plot ROC curve
        plt.plot(false_positive_rate, true_positive_rate, label=name+"(f1:%.3f)"%f1_test, c=color)
    
    def cv(self, models, train,  y_train, n_folds):
        kf = KFold(n_folds, shuffle=False, random_state=42).get_n_splits(train.values)
        scores = []
        for model in models :
            scores.append(cross_val_score(model, train.values, y_train, scoring="f1_macro", cv = kf))    
        return(scores)

    def roc_curve(self, models, X, y, n_folds):
        cv = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
        results = pd.DataFrame(columns=['training_score', 'test_score'])
        fprs, tprs, scores = [], [], []
        fprm, tprm = [], []
        for model in models :
            for (train, test), i in zip(cv.split(X, y), range(n_folds)):
                model.fit(X.iloc[train], y.iloc[train])
                fpr, tpr, _ = compute_roc_auc(test)
                fprs.append(fpr)
                tprs.append(tpr)
            fprm.append(np.mean(fprs, axis = 1))
            tprm.append(np.mean(tprs, axis = 1))
        plot_roc_curve(fprm, tprm);
        


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", type = str, default= "", help = "X_train embeddings")
    parser.add_argument("--y", type = str, default= "", help = "y is the groudthruth")
    parser.add_argument("--test", type = str, default= "", help = "Test set")
    parser.add_argument("--out", type = str, default= "", help = "output directory")
    args = parser.parse_args()

    assert args.X != "" and os.path.isfile(args.X)
    assert args.y != "" and os.path.isfile(args.y)
    X = np.load(args.X)
    y = np.load(args.y)
    test = np.load(args.test)

    #Split data
    idx_0 = np.where(y==0)[0]
    idx_1 = np.where(y==1)[0]

    train_i0, test_i0, _, _ = train_test_split(idx_0, idx_0, test_size=0.1)
    train_i1, test_i1, _, _ = train_test_split(idx_1, idx_1, test_size=0.1)
    train_i = np.hstack((train_i0, train_i1))
    test_i = np.hstack((test_i0, test_i1))

    for i in range(7):
        np.random.shuffle(train_i)
        np.random.shuffle(test_i)
    
    X_train, y_train = X[train_i], y[train_i]
    X_test, y_test = X[test_i], y[test_i]

    #Normalize data
    X_scaled = preprocessing.scale(X_train)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    s_test = test

    evaluate = ModelEvaluation(args.out)
    evaluate.train(X_train, y_train)
    evaluate.evaluate_score(X_test, y_test)
    evaluate.evaluate_plot(X_test, y_test)

    for model in evaluate.models:
        name = model.__class__.__name__
        pred = model.predict(s_test)
        predstr = [str(pred[i]) for i in range(len(pred))]
        predictions = zip(range(len(predstr)), predstr)

        # Write the output in the format required by Kaggle
        import csv
        p = os.path.join(args.out, "prediction")
        if not os.path.isdir(p):
            os.makedirs(p)
        with open(os.path.join(p, "predictions_%s.csv"%name),"w") as pred:
            csv_out = csv.writer(pred)
            csv_out.writerow(['id','predicted'])
            for row in predictions:
                csv_out.writerow(row) 




    