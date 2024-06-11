#%% IMPORT LIBRARIES
import numpy as np
import pickle
from fitmodels import *
from settings import *
from wd import working_dir
from sklearn.model_selection import train_test_split

def getparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-svc', '--fit_svc', type=bool, default=False)
    parser.add_argument('-knn', '--fit_knn', type=bool, default=False)
    parser.add_argument('-rf', '--fit_rf', type=bool, default=False)
    parser.add_argument('-gb', '--fit_gb', type=bool, default=False)

    args = parser.parse_args()
    return args

def main():
    parser = get_parser()
    QFmodel=dcts_path[-4:] if dcts_path.endswith('0') else 'RAW'
    print(f'\n--  {QFmodel}')
    if parser.fit_knn:
        print('\nKNN\n')
        knn_all = fitknn(X_train=train_betas,
                        y_train=train_targets,
                        X_test=test_betas,
                        y_test=test_targets,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['knn'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
        pickle.dump(knn_all, open(working_dir+'models/'+QFmodel+'/knn_all.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            getmetrics(y_test, knn_all.predict(X_test))

    if parser.fit_rf:
        print('\nRANDOM FOREST\n')
        rf_all = fitrf(X_train=train_betas,
                        y_train=train_targets,
                        X_test=test_betas,
                        y_test=test_targets,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['rf'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
        pickle.dump(rf_all, open(working_dir+'models/'+QFmodel+'/rf_all.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            getmetrics(y_test, rf_all.predict(X_test))

    if parser.fit_gb:
        print('\nGRADIENT BOOSTING\n')
        gb_all = fitgb(X_train=train_betas,
                        y_train=train_targets,
                        X_test=test_betas,
                        y_test=test_targets,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['gb'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
        pickle.dump(gb_all, open(working_dir+'models/'+QFmodel+'/gb_all.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            getmetrics(y_test, gb_all.predict(X_test))

    if parser.fit_svc:
        print('\nSVC\n')
        svc_all = SVC(gamma=0.1, kernel='poly', degree=2).fit(train_betas, train_betas)
        print('MODEL PERFOMANCE:')
        getmetrics(test_targets, svc_all.predict(test_betas), plotcm=True)
        pickle.dump(svc_all, open(working_dir+'models/'+QFmodel+'/svc_all.pkl', 'wb'))
        for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
            print(f'\nJPEG-TESTING {qf}: ')
            dct_path=dcts_path+'_'+qf
            X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
            getmetrics(y_test, svc_all.predict(X_test))

if __name__=='__main__':
    main()