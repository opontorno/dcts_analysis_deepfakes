from fitmodels import *
from settings import X_train, X_test, y_test, y_train, params_grids
import pickle

svc=True
knn=True
rf=True
gb=True

if svc:
    print('SVC\n')
    svc_all = fitsvc(X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    random_state=13,
                    gridsearch=False,
                    randomsearch=True,
                    params_grid=params_grids['svc'],
                    n_iter=50,
                    cv_folds=3,
                    verbose=3,
                    plotcm=True)

    pickle.dump(svc_all, open('models/all_betas/svc_all.pkl', 'wb'))

if knn:
    print('KNN\n')
    knn_all = fitknn(X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    random_state=13,
                    gridsearch=True,
                    randomsearch=False,
                    params_grid=params_grids['knn'],
                    n_iter=100,
                    cv_folds=3,
                    verbose=3,
                    plotcm=True)

    pickle.dump(knn_all, open('models/all_betas/knn_all.pkl', 'wb'))

if rf:
    print('RANDOM FOREST')
    rf_all = fitrf(X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    random_state=13,
                    gridsearch=False,
                    randomsearch=True,
                    params_grid=params_grids['rf'],
                    n_iter=100,
                    cv_folds=3,
                    verbose=3,
                    plotcm=True)

    pickle.dump(rf_all, open('models/all_betas/rf_all.pkl', 'wb'))

if gb:
    print('GRADIENT BOOSTING')
    gb_all = fitgb(X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    random_state=13,
                    gridsearch=False,
                    randomsearch=True,
                    params_grid=params_grids['rf'],
                    n_iter=100,
                    cv_folds=3,
                    verbose=3,
                    plotcm=True)

    pickle.dump(gb_all, open('models/all_betas/gb_all.pkl', 'wb'))