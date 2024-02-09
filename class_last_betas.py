import numpy as np
import pickle
from fitmodels import *
from settings import *
from sklearn.model_selection import train_test_split

knn=True
rf=True
gb=True

save_models = False

QFmodel=dcts_path[-4:] if dcts_path.endswith('0') else 'RAW'
print(f'\n--  {QFmodel}')

if knn:
    print('\n-  KNN\n')
    for last in range(35,1,-1):
        metrics = []
        Xls_train = train_betas[:,(63-last):]
        yls_train = train_targets
        Xls_test = test_betas[:,(63-last):]
        yls_test = test_targets
        knn_l, (acc,_,_,f1) = fitknn(X_train=Xls_train,
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
                        plotcm=False)
        metrics.extend([str(acc)+'/'+str(f1)])
        if save_models: pickle.dump(knn_l, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/knn_l'+str(last)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,(63-last):]
            qftest = getmetrics(y_test, knn_l.predict(X_test))
            metrics.extend([str(qftest[0])+'/'+str(qftest[3])])
        print(*metrics, sep=',')
        

if rf:
    print('\n-  RANDOM FOREST\n')
    for last in range(35,1,-1):
        metrics = []
        Xls_train = train_betas[:,(63-last):]
        yls_train = train_targets
        Xls_test = test_betas[:,(63-last):]
        yls_test = test_targets
        rf_l, (acc,_,_,f1) = fitrf(X_train=Xls_train,
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
                        plotcm=False)
        metrics.extend([str(acc)+'/'+str(f1)])
        if save_models: pickle.dump(rf_l, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/rf_l'+str(last)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,(63-last):]
            qftest = getmetrics(y_test, rf_l.predict(X_test))
            metrics.extend([str(qftest[0])+'/'+str(qftest[3])])
        print(*metrics, sep=',')

if gb:
    print('\n-  GRADIENT BOOSTING\n')
    for last in range(35,1,-1):
        metrics = []
        Xls_train = train_betas[:,(63-last):]
        yls_train = train_targets
        Xls_test = test_betas[:,(63-last):]
        yls_test = test_targets
        gb_l, (acc,_,_,f1) = fitgb(X_train=Xls_train,
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
                        plotcm=False)
        metrics.extend([str(acc)+'/'+str(f1)])
        if save_models: pickle.dump(gb_l, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/gb_l'+str(last)+'.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            X_test = X_test[:,(63-last):]
            qftest = getmetrics(y_test, gb_l.predict(X_test))
            metrics.extend([str(qftest[0])+'/'+str(qftest[3])])
        print(*metrics, sep=',')