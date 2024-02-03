import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from  sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def getmetrics(labels, predictions, average="macro", plotcm: bool = False, classes=None, head=False):
    if head: print('accuracy, precision, racall, f1-score')
    print(f'{accuracy_score(labels, predictions)*100:.2f},{precision_score(labels, predictions, average=average)*100:.2f},{recall_score(labels, predictions, average=average)*100:.2f},{f1_score(labels, predictions, average=average)*100:.2f}')
    if plotcm:
        print('Confusion matrix: ')
        cm = confusion_matrix(labels, predictions)
        print(cm)

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

    model_base = SVC(random_state=random_state)
    model_base.fit(X_train,y_train)
    print(f'- BASE MODEL performance:')
    getmetrics(y_test, model_base.predict(X_test))

    if (gridsearch or randomsearch) != False:
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
        print(f'best parameters: {model_s.best_params_}')
        print('- SEARCH MODEL performance:')
        getmetrics(y_test, model_s.predict(X_test), plotcm=plotcm)
        return model_s
    else:
        return model_base

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
    model_base = KNeighborsClassifier()
    model_base.fit(X_train,y_train)
    print(f'- BASE MODEL performance:')
    getmetrics(y_test, model_base.predict(X_test))
    if (gridsearch or randomsearch) != False:
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
        print(f'best parameters: {model_s.best_params_}')
        print('- SEARCH MODEL performance:')
        getmetrics(y_test, model_s.predict(X_test), plotcm=plotcm)
        return model_s
    else:
        return model_base
    
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
    model_base = RandomForestClassifier(random_state=random_state)
    model_base.fit(X_train,y_train)
    print(f'- BASE MODEL performance:')
    getmetrics(y_test, model_base.predict(X_test))
    if (gridsearch or randomsearch) != False:
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
        print(f'best parameters: {model_s.best_params_}')
        print('- SEARCH MODEL performance:')
        getmetrics(y_test, model_s.predict(X_test), plotcm=plotcm)
        return model_s
    else:
        return model_base
    
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
    model_base = GradientBoostingClassifier(random_state=random_state)
    model_base.fit(X_train,y_train)
    print(f'- BASE MODEL performance:')
    getmetrics(y_test, model_base.predict(X_test))
    if (gridsearch or randomsearch) != False:
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
        print(f'best parameters: {model_s.best_params_}')
        print('- SEARCH MODEL performance:')
        getmetrics(y_test, model_s.predict(X_test), plotcm=plotcm)
        return model_s
    else:
        return model_base