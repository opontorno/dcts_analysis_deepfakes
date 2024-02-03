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

#%% SETTINGS FOR CLASSIFICATION USING THE CENTER BETAS
for center in range(1,16):
    print(f'\n--    USING CENTERED {center} COEFFICIENTS:\n')
    Xc_train = train_betas[:,(31-center):(31+center+1)]
    yc_train = train_targets

    Xc_test = test_betas[:,(31-center):(31+center+1)]
    yc_test = test_targets

    if knn:
        print('\n-  KNN\n')
        knn_c = fitknn(X_train=Xc_train,
                        y_train=yc_train,
                        X_test=Xc_test,
                        y_test=yc_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['knn'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
        if save_models: pickle.dump(knn_c, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/knn_c'+str(center)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,(31-center):(31+center+1)]
            getmetrics(y_test, knn_c.predict(X_test))

    if rf:
        print('\n-  RANDOM FOREST\n')
        rf_c = fitrf(X_train=Xc_train,
                        y_train=yc_train,
                        X_test=Xc_test,
                        y_test=yc_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['rf'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
        if save_models: pickle.dump(rf_c, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/rf_c'+str(center)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,(31-center):(31+center+1)]
            getmetrics(y_test, rf_c.predict(X_test))
    if gb:
        print('\n-  GRADIENT BOOSTING\n')
        gb_c = fitgb(X_train=Xc_train,
                        y_train=yc_train,
                        X_test=Xc_test,
                        y_test=yc_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['gb'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)

        if save_models: pickle.dump(gb_c, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/gb_c'+str(center)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,(31-center):(31+center+1)]
            getmetrics(y_test, gb_c.predict(X_test))

    if svc:
        print('\n-  SVC\n')
        svc_c = SVC(gamma=0.1, kernel='poly', degree=2).fit(Xc_train, yc_test)
        print('MODEL PERFOMANCE:')
        getmetrics(yc_test, svc_c.predict(Xc_test), plotcm=True)

        if save_models: pickle.dump(svc_c, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/svc_c'+str(center)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,(31-center):(31+center+1)]
            getmetrics(y_test, svc_c.predict(X_test))