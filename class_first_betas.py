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

#%% SETTINGS FOR CLASSIFICATION USING THE FIRST BETAS
for first in range(30,1,-1):
    print(f'\n--    USING FIRST {first} COEFFICIENTS:\n')
    Xsh_train = train_betas[:,:first]
    ysh_train = train_targets

    Xsh_test = test_betas[:,:first]
    ysh_test = test_targets

    if knn:
        print('\n-  KNN\n')
        knn_f = fitknn(X_train=Xsh_train,
                        y_train=ysh_train,
                        X_test=Xsh_test,
                        y_test=ysh_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['knn'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
        if save_models: pickle.dump(knn_f, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/knn_f'+str(first)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,:first]
            getmetrics(y_test, knn_f.predict(X_test))

    if rf:
        print('\n-  RANDOM FOREST\n')
        rf_f = fitrf(X_train=Xsh_train,
                        y_train=ysh_train,
                        X_test=Xsh_test,
                        y_test=ysh_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['rf'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)

        if save_models: pickle.dump(rf_f, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/rf_f'+str(first)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,:first]
            getmetrics(y_test, rf_f.predict(X_test))

    if gb:
        print('\n-  GRADIENT BOOSTING\n')
        gb_f = fitgb(X_train=Xsh_train,
                        y_train=ysh_train,
                        X_test=Xsh_test,
                        y_test=ysh_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['gb'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
        if save_models: pickle.dump(gb_f, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/gb_f'+str(first)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,:first]
            getmetrics(y_test, gb_f.predict(X_test))
    if svc:
        print('\n-  SVC\n')
        svc_f = SVC(gamma=0.1, kernel='poly', degree=2).fit(Xsh_train, ysh_train)
        print('MODEL PERFOMANCE:')
        getmetrics(ysh_test, svc_f.predict(Xsh_test), plotcm=True)
        if save_models: pickle.dump(svc_f, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/svc_f'+str(first)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,:first]
            getmetrics(y_test, svc_f.predict(X_test))