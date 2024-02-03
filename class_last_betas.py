import numpy as np
import pickle
from fitmodels import *
from settings import *
from sklearn.model_selection import train_test_split

svc=False
knn=True
rf=True
gb=True

save_models = False

QFmodel=dcts_path[-4:] if dcts_path.endswith('0') else 'RAW'
print(f'\n--  {QFmodel}')

#%% SETTINGS FOR CLASSIFICATION USING THE LAST BETAS
for last in range(35,1,-1):
    print(f'\n--    USING LAST {last} COEFFICIENTS:\n')
    Xls_train = train_betas[:,(63-last):]
    yls_train = train_targets

    Xls_test = test_betas[:,(63-last):]
    yls_test = test_targets
    if knn:
        print('\n-  KNN\n')
        knn_l = fitknn(X_train=Xls_train,
                        y_train=yls_train,
                        X_test=Xls_test,
                        y_test=yls_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['knn'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
        if save_models: pickle.dump(knn_l, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/knn_l'+str(last)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,(63-last):]
            getmetrics(y_test, knn_l.predict(X_test))

    if rf:
        print('\nRANDOM FOREST\n')
        rf_l = fitrf(X_train=Xls_train,
                        y_train=yls_train,
                        X_test=Xls_test,
                        y_test=yls_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['rf'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
        if save_models: pickle.dump(rf_l, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/rf_l'+str(last)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,(63-last):]
            getmetrics(y_test, rf_l.predict(X_test))

    if gb:
        print('\nGRADIENT BOOSTING\n')
        gb_l = fitgb(X_train=Xls_train,
                        y_train=yls_train,
                        X_test=Xls_test,
                        y_test=yls_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['gb'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
        if save_models: pickle.dump(gb_l, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/gb_l'+str(last)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,(63-last):]
            getmetrics(y_test, gb_l.predict(X_test))

    if svc:
        print('\nSVC\n')
        svc_l = SVC(gamma=0.1, kernel='poly', degree=2).fit(Xls_train, yls_train)
        print('MODEL PERFOMANCE:')
        getmetrics(yls_test, svc_l.predict(Xls_test), plotcm=True)
        if save_models: pickle.dump(svc_l, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/svc_l'+str(last)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,(63-last):]
            getmetrics(y_test, svc_l.predict(X_test))