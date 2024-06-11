#%% IMPORT LIBRARIES
import numpy as np
import pickle
from fitmodels import *
from settings import *
from wd import working_dir
from sklearn.model_selection import train_test_split

def getparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-subset', '--lime_subset', type=str, choices=['pos_lime', 'abs_lime'])
    parser.add_argument('-svc', '--fit_svc', type=bool, default=False)
    parser.add_argument('-knn', '--fit_knn', type=bool, default=False)
    parser.add_argument('-rf', '--fit_rf', type=bool, default=False)
    parser.add_argument('-gb', '--fit_gb', type=bool, default=False)
    parser.add_argument('-save_m', '--save_models', type=bool, default=False)

    args = parser.parse_args()
    return args

def main():
    parser = get_parser()
    QFmodel=dcts_path[-4:] if dcts_path.endswith('0') else 'RAW'
    print(f'\n--  {QFmodel}')

    attrs_all = np.load(f'attrs{QFmodel}.npy')
    avg_arr = np.mean(attrs_all,axis=0)

    if parser.lime_subset == 'pos_lime':
        positive_all = np.where(avg_arr>0)[0].tolist()

        print(f'\n--    USING ALL POSITIVE LIME COEFFICIENTS:\n')
        Xlime_pos_train = train_betas[:,positive_all]
        ylime_pos_train = train_targets

        Xlime_pos_test = test_betas[:,positive_all]
        ylime_pos_test = test_targets

        if parser.fit_knn:
            print('\n-  KNN\n')
            knn_poslime, (acc,_,_,f1) = fitknn(X_train=Xlime_pos_train,
                        y_train=ylime_pos_train,
                        X_test=Xlime_pos_test,
                        y_test=ylime_pos_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['knn'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
            if parser.save_models: pickle.dump(knn_poslime, open(working_dir+'models/'+QFmodel+'/knn_all-pos_lime.pkl', 'wb'))
            for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
                print(f'\nJPEG-TESTING {qf}: ')
                dct_path=dcts_path+'_'+qf
                X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
                
        if parser.fit_rf:
            print('\n-  RANDOM FOREST\n')
            rf_poslime, (acc,_,_,f1) = fitrf(X_train=Xlime_pos_train,
                        y_train=ylime_pos_train,
                        X_test=Xlime_pos_test,
                        y_test=ylime_pos_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['rf'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
            if parser.save_models: pickle.dump(rf_poslime, open(working_dir+'models/'+QFmodel+'/rf_all-pos_lime.pkl', 'wb'))
            for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
                print(f'\nJPEG-TESTING {qf}: ')
                dct_path=dcts_path+'_'+qf
                X_test, y_test = get_train_test(dct_path, train=False, )
                X_test = X_test[:,positive_all]
                getmetrics(y_test, rf_poslime.predict(X_test))

        if parser.fit_gb:
            print('\n-  GRADIENT BOOSTING\n')
            gb_poslime, (acc,_,_,f1) = fitgb(X_train=Xlime_pos_train,
                        y_train=ylime_pos_train,
                        X_test=Xlime_pos_test,
                        y_test=ylime_pos_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['gb'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)

            if parser.save_models: pickle.dump(gb_poslime, open(working_dir+'models/'+QFmodel+'/gb_all-pos_lime.pkl', 'wb'))
            for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
                print(f'\nJPEG-TESTING {qf}: ')
                dct_path=dcts_path+'_'+qf
                X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
                X_test = X_test[:,positive_all]
                getmetrics(y_test, gb_poslime.predict(X_test))

    if parser.lime_subset == 'abs_lime':
        abs_sign = np.where(abs(avg_arr)>np.median(abs(avg_arr)))[0].tolist()

        print(f'\n--    USING ALL ABS-SIGNIFICATIVE LIME COEFFICIENTS:\n')
        Xlime_abs_sign_train = train_betas[:,abs_sign]
        ylime_abs_sign_train = train_targets

        Xlime_abs_sign_test = test_betas[:,abs_sign]
        ylime_abs_sign_test = test_targets

        if parser.fit_knn:
            print('\nKNN\n')
            knn_abs_sign, (acc,_,_,f1) = fitknn(X_train=Xlime_abs_sign_train,
                        y_train=ylime_abs_sign_train,
                        X_test=Xlime_abs_sign_test,
                        y_test=ylime_abs_sign_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['knn'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
            if parser.save_models: pickle.dump(knn_abs_sign, open(working_dir+'models/'+QFmodel+'/knn_all-absign_lime.pkl', 'wb'))
            for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
                print(f'\nJPEG-TESTING {qf}: ')
                dct_path=dcts_path+'_'+qf
                X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
                X_test = X_test[:,abs_sign]
                getmetrics(y_test, knn_abs_sign.predict(X_test))

        if parser.fit_rf:
            print('\nRANDOM FOREST\n')
            rf_abs_sign, (acc,_,_,f1) = fitrf(X_train=Xlime_abs_sign_train,
                        y_train=ylime_abs_sign_train,
                        X_test=Xlime_abs_sign_test,
                        y_test=ylime_abs_sign_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['rf'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
            if parser.save_models: pickle.dump(rf_abs_sign, open(working_dir+'models/'+QFmodel+'/rf_all-absign_lime.pkl', 'wb'))
            for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
                print(f'\nJPEG-TESTING {qf}: ')
                dct_path=dcts_path+'_'+qf
                X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
                X_test = X_test[:,abs_sign]
                getmetrics(y_test, rf_abs_sign.predict(X_test))

        if parser.fit_gb:
            print('\nGRADIENT BOOSTING\n')
            gb_abs_sign, (acc,_,_,f1) = fitgb(X_train=Xlime_abs_sign_train,
                        y_train=ylime_abs_sign_train,
                        X_test=Xlime_abs_sign_test,
                        y_test=ylime_abs_sign_test,
                        random_state=13,
                        gridsearch=False,
                        randomsearch=True,
                        params_grid=params_grids['gb'],
                        n_iter=100,
                        cv_folds=3,
                        verbose=0,
                        plotcm=True)
            if parser.save_models: pickle.dump(gb_abs_sign, open(working_dir+'models/'+QFmodel+'/gb_all-absign_lime.pkl', 'wb'))
            for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
                print(f'\nJPEG-TESTING {qf}: ')
                dct_path=dcts_path+'_'+qf
                X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
                X_test = X_test[:,abs_sign]
                getmetrics(y_test, gb_abs_sign.predict(X_test))
            
if __name__=='__main__':
    main()