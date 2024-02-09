import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from  sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def getmetrics(labels, predictions, average="macro", plotcm: bool = False, head=False):
    if head: print('accuracy, precision, racall, f1-score')
    acc, pre, rec, f1 = accuracy_score(labels, predictions), precision_score(labels, predictions, average=average), recall_score(labels, predictions, average=average), f1_score(labels, predictions, average=average)
    if plotcm:
        print('Confusion matrix: ')
        cm = confusion_matrix(labels, predictions)
        print(cm)
    return round(acc*100,2), round(pre*100,2), round(rec*100,2), round(f1*100,2)

def fitsvc(X_train, y_train, X_test, y_test,
           random_state=None, 
           gridsearch: bool = False, 
           randomsearch: bool = False, 
           params_grid : dict = None, 
           cv_folds:int=3, 
           n_iter:int=100, 
           plotcm:bool=False, 
           verbose=3):
    assert (gridsearch and randomsearch) != True
    if gridsearch:
        model_s = GridSearchCV(
            estimator = SVC(random_state=random_state),
            param_grid = params_grid,
            cv = cv_folds, 
            verbose=verbose,
            n_jobs = -1
            )
    elif randomsearch:
        model_s = RandomizedSearchCV(
            estimator = SVC(random_state=random_state),
            param_distributions = params_grid,
            cv = cv_folds, 
            verbose=verbose,
            n_jobs = -1,
            random_state=random_state,
            n_iter=n_iter)
    model_s.fit(X_train,y_train)
    getmetrics(y_test, model_s.predict(X_test), plotcm=plotcm)
    return model_s

def fitknn(X_train, y_train, X_test, y_test, 
           random_state=None, 
           gridsearch: bool = False, 
           randomsearch: bool = False, 
           params_grid : dict = None, 
           cv_folds:int=3, 
           n_iter:int=100, 
           plotcm:bool=False, 
           verbose=3):
    assert (gridsearch and randomsearch) != True
    if gridsearch:
        model_s = GridSearchCV(
            estimator = KNeighborsClassifier(),
            param_grid = params_grid,
            cv = cv_folds, 
            verbose=verbose,
            n_jobs = -1
            )
    elif randomsearch:
        model_s = RandomizedSearchCV(
            estimator = KNeighborsClassifier(),
            param_distributions = params_grid,
            cv = cv_folds, 
            verbose=verbose,
            n_jobs = -1,
            random_state=random_state,
            n_iter=n_iter)
    model_s.fit(X_train,y_train)
    test_metrics = getmetrics(y_test, model_s.predict(X_test), plotcm=plotcm)
    return model_s, test_metrics
    
def fitrf(X_train, y_train, X_test, y_test, 
          random_state=None, 
          gridsearch: bool = False, 
          randomsearch: bool = False, 
          params_grid : dict = None, 
          cv_folds:int=3, 
          n_iter:int=100, 
          plotcm:bool=False, 
          verbose=3):
    assert (gridsearch and randomsearch) != True
    if gridsearch:
        model_s = GridSearchCV(
            estimator = RandomForestClassifier(random_state=random_state),
            param_grid = params_grid,
            cv = cv_folds, 
            verbose=verbose,
            n_jobs = -1
            )
    elif randomsearch:
        model_s = RandomizedSearchCV(
            estimator = RandomForestClassifier(random_state=random_state),
            param_distributions = params_grid,
            cv = cv_folds, 
            verbose=verbose,
            n_jobs = -1,
            random_state=random_state,
            n_iter=n_iter)
    model_s.fit(X_train,y_train)
    test_metrics = getmetrics(y_test, model_s.predict(X_test), plotcm=plotcm)
    return model_s, test_metrics
    
def fitgb(X_train, y_train, X_test, y_test, 
          random_state=None, 
          gridsearch: bool = False, 
          randomsearch: bool = False, 
          params_grid : dict = None, 
          cv_folds:int=3, 
          n_iter:int=100, 
          plotcm:bool=False, 
          verbose=3):
    assert (gridsearch and randomsearch) != True
    if gridsearch:
        model_s = GridSearchCV(
            estimator = GradientBoostingClassifier(random_state=random_state),
            param_grid = params_grid,
            cv = cv_folds, 
            verbose=verbose,
            n_jobs = -1
            )
    elif randomsearch:
        model_s = RandomizedSearchCV(
            estimator = GradientBoostingClassifier(random_state=random_state),
            param_distributions = params_grid,
            cv = cv_folds, 
            verbose=verbose,
            n_jobs = -1,
            random_state=random_state,
            n_iter=n_iter
            )
    model_s.fit(X_train,y_train)
    test_metrics = getmetrics(y_test, model_s.predict(X_test), plotcm=plotcm)
    return model_s, test_metrics