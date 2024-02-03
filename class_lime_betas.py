#%% IMPORT LIBRARIES
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

#%% SETTINGS FOR CLASSIFICATION USING THE ALL POSITIVE LIME BETAS
attrs_all = np.load(f'attrs{QFmodel}.npy')
avg_arr = np.mean(attrs_all,axis=0)

positive_all = np.where(avg_arr>0)[0].tolist()

perc_train=0.7
print(f'\n--    USING ALL POSITIVE LIME COEFFICIENTS:\n')
Xlime_pos_train = train_betas[:,positive_all]
ylime_pos_train = train_targets

Xlime_pos_test = test_betas[:,positive_all]
ylime_pos_test = test_targets

if knn:
    print('\n-  KNN\n')
    knn_poslime = fitknn(X_train=Xlime_pos_train,
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
    if save_models: pickle.dump(knn_poslime, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/knn_all-pos_lime.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
        
if rf:
    print('\n-  RANDOM FOREST\n')
    rf_poslime = fitrf(X_train=Xlime_pos_train,
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
    if save_models: pickle.dump(rf_poslime, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/rf_all-pos_lime.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, )
        X_test = X_test[:,positive_all]
        getmetrics(y_test, rf_poslime.predict(X_test))

if gb:
    print('\n-  GRADIENT BOOSTING\n')
    gb_poslime = fitgb(X_train=Xlime_pos_train,
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

    if save_models: pickle.dump(gb_poslime, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/gb_all-pos_lime.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
        X_test = X_test[:,positive_all]
        getmetrics(y_test, gb_poslime.predict(X_test))

if svc:
    print('\n-  SVC\n')
    svc_l = SVC(gamma=0.1, kernel='poly', degree=2).fit(Xlime_pos_train, ylime_pos_test)
    print('MODEL PERFOMANCE:')
    getmetrics(ylime_pos_test, svc_l.predict(Xlime_pos_test), plotcm=True)

    if save_models: pickle.dump(svc_l, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/svc_all-pos_lime.pkl', 'wb'))

#%% SETTINGS FOR CLASSIFICATION USING THE ALL ABS-SIGNIFICATIVE LIME BETAS
abs_sign = np.where(abs(avg_arr)>np.median(abs(avg_arr)))[0].tolist()

print(f'\n--    USING ALL ABS-SIGNIFICATIVE LIME COEFFICIENTS:\n')
Xlime_abs_sign_train = train_betas[:,abs_sign]
ylime_abs_sign_train = train_targets

Xlime_abs_sign_test = test_betas[:,abs_sign]
ylime_abs_sign_test = test_targets

if knn:
    print('\nKNN\n')
    knn_abs_sign = fitknn(X_train=Xlime_abs_sign_train,
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
    if save_models: pickle.dump(knn_abs_sign, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/knn_all-absign_lime.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
        X_test = X_test[:,abs_sign]
        getmetrics(y_test, knn_abs_sign.predict(X_test))

if rf:
    print('\nRANDOM FOREST\n')
    rf_abs_sign = fitrf(X_train=Xlime_abs_sign_train,
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
    if save_models: pickle.dump(rf_abs_sign, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/rf_all-absign_lime.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
        X_test = X_test[:,abs_sign]
        getmetrics(y_test, rf_abs_sign.predict(X_test))

if gb:
    print('\nGRADIENT BOOSTING\n')
    gb_abs_sign = fitgb(X_train=Xlime_abs_sign_train,
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
    if save_models: pickle.dump(gb_abs_sign, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/gb_all-absign_lime.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
        X_test = X_test[:,abs_sign]
        getmetrics(y_test, gb_abs_sign.predict(X_test))

if svc:
    print('\nSVC\n')
    svc_l = SVC(gamma=0.1, kernel='poly', degree=2).fit(Xlime_abs_sign_train, ylime_abs_sign_test)
    print('MODEL PERFOMANCE:')
    getmetrics(ylime_abs_sign_test, svc_l.predict(Xlime_abs_sign_test), plotcm=True)

    if save_models: pickle.dump(svc_l, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/'+QFmodel+'/svc_all-absign_lime.pkl', 'wb'))