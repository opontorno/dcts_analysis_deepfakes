import numpy as np
import pickle
import argparse
from fitmodels import *
from settings import *
from sklearn.model_selection import train_test_split

def getparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-svc', '--fit_svc', type=bool, default=False)
    parser.add_argument('-knn', '--fit_knn', type=bool, default=False)
    parser.add_argument('-rf', '--fit_rf', type=bool, default=False)
    parser.add_argument('gb', '--fit_gb', type=bool, default=False)
    parser.add_argument('-save_m', '--save_models', type=bool, default=False)

    args = parser.parse_args()
    return args



def main():
    parser = get_parser()
    QFmodel=dcts_path[-4:] if dcts_path.endswith('0') else 'RAW'
    print(f'\n--  {QFmodel}')
    if parser.fit_knn:
        print('\n-  KNN\n')
        for center in range(15,0,-1):
            metrics = []
            Xc_train = train_betas[:,(31-center):(31+center+1)]
            yc_train = train_targets
            Xc_test = test_betas[:,(31-center):(31+center+1)]
            yc_test = test_targets
            knn_c, (acc,_,_,f1) = fitknn(X_train=Xc_train,
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
                            plotcm=False)
            metrics.extend([str(acc)+'/'+str(f1)])
            if parser.save_models: pickle.dump(knn_c, open(working_dir+'models/'+QFmodel+'/knn_c'+str(center)+'.pkl', 'wb'))
            for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
                dct_path=dcts_path+'_'+qf
                X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
                X_test = X_test[:,(31-center):(31+center+1)]
                qftest = getmetrics(y_test, knn_c.predict(X_test))
                metrics.extend([str(qftest[0])+'/'+str(qftest[3])])
            print(*metrics, sep=',')
            

    if parser.fit_rf:
        print('\n-  RANDOM FOREST\n')
        for center in range(15,0,-1):
            metrics = []
            Xc_train = train_betas[:,(31-center):(31+center+1)]
            yc_train = train_targets
            Xc_test = test_betas[:,(31-center):(31+center+1)]
            yc_test = test_targets
            rf_c, (acc,_,_,f1) = fitrf(X_train=Xc_train,
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
                            plotcm=False)
            metrics.extend([str(acc)+'/'+str(f1)])
            if parser.save_models: pickle.dump(rf_c, open(working_dir+'models/'+QFmodel+'/rf_c'+str(center)+'.pkl', 'wb'))
            for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
                dct_path=dcts_path+'_'+qf
                X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
                X_test = X_test[:,(31-center):(31+center+1)]
                qftest = getmetrics(y_test, rf_c.predict(X_test))
                metrics.extend([str(qftest[0])+'/'+str(qftest[3])])
            print(*metrics, sep=',')

    if parser.fit_gb:
        print('\n-  GRADIENT BOOSTING\n')
        for center in range(15,0,-1):
            metrics = []
            Xc_train = train_betas[:,(31-center):(31+center+1)]
            yc_train = train_targets
            Xc_test = test_betas[:,(31-center):(31+center+1)]
            yc_test = test_targets
            gb_c, (acc,_,_,f1) = fitgb(X_train=Xc_train,
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
                            plotcm=False)
            metrics.extend([str(acc)+'/'+str(f1)])
            if parser.save_models: pickle.dump(gb_c, open(working_dir+'models/'+QFmodel+'/gb_c'+str(center)+'.pkl', 'wb'))
            for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
                dct_path=dcts_path+'_'+qf
                X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
                X_test = X_test[:,(31-center):(31+center+1)]
                qftest = getmetrics(y_test, gb_c.predict(X_test))
                metrics.extend([str(qftest[0])+'/'+str(qftest[3])])
            print(*metrics, sep=',')

if __name__=='__main__':
    main()