#%% IMPORT LIBRARIES
import numpy as np
import pickle
import os
from settings import params_grids
from imblearn.under_sampling import RandomUnderSampler 
from fitmodels import *
from dcts_analysis import *

dcts_path = '/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/datasets/dcts'
train_betas, train_targets = get_train_test(dcts_path, guidance, train=True)
test_betas, test_targets = get_train_test(dcts_path, guidance, train=False)
dset = np.unique(np.concatenate((train_betas, train_targets.reshape(-1, 1)), axis=1), axis=0)
random_seed = 13
train_betas = dset[:,:63]
train_targets = dset[:,63].astype(int)

## MAKE THE DATASET BALANCED
class_counts = {label: count for label, count in zip(*np.unique(train_targets, return_counts=True))}
min_class_count = min(class_counts.values())
min_class_label = min(class_counts, key=class_counts.get)
undersampler = RandomUnderSampler(sampling_strategy={label: min_class_count for label in class_counts})
betas_resampled, targets_resampled = undersampler.fit_resample(train_betas, train_targets)
print(f'\nRESAMPLED BETAS: \ntype: {type(betas_resampled)} \nsize: {betas_resampled.shape}\n')
print(f'RESAMPLED TARGETS: \ntype: {type(targets_resampled)} \nsize: {targets_resampled.shape} \nunique values: {np.unique(targets_resampled, return_counts=True)}')


# %% JPEG-TESTING USING LIME COEFFICIENTS
print('\n-  GRADIENT BOOSTING\n')
attrs_all = np.load('attrs.npy')
avg_arr = np.mean(attrs_all,axis=0)
positive_all = np.where(avg_arr>0)[0].tolist()
abs_sign = np.where(abs(avg_arr)>np.median(abs(avg_arr)))[0].tolist()

print(f'\n--    USING ALL POSITIVE LIME COEFFICIENTS:\n')
Xlime_pos_train = betas_resampled[:,positive_all]
ylime_pos_train = targets_resampled

Xlime_pos_test = test_betas[:,positive_all]
ylime_pos_test = test_targets


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
pickle.dump(gb_poslime, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/gb_all-pos_lime.pkl', 'wb'))

for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
    print(f'\nJPEG-TESTING {qf}: ')
    dct_path=dcts_path+'_'+qf
    X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
    X_test = X_test[:,positive_all]
    getmetrics(y_test, gb_poslime.predict(X_test))

print(f'\n--    USING ALL ABS-SIGNIFICATIVE LIME COEFFICIENTS:\n')
Xlime_abs_sign_train = betas_resampled[:,abs_sign]
ylime_abs_sign_train = targets_resampled

Xlime_abs_sign_test = test_betas[:,abs_sign]
ylime_abs_sign_test = test_targets


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
pickle.dump(gb_abs_sign, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/gb_all-absign_lime.pkl', 'wb'))

for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
    print(f'\nJPEG-TESTING {qf}: ')
    dct_path=dcts_path+'_'+qf
    X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
    X_test = X_test[:,abs_sign]
    getmetrics(y_test, gb_abs_sign.predict(X_test))

print('\n-  RANDOM FOREST\n')
attrs_all = np.load('attrs.npy')
avg_arr = np.mean(attrs_all,axis=0)
positive_all = np.where(avg_arr>0)[0].tolist()
abs_sign = np.where(abs(avg_arr)>np.median(abs(avg_arr)))[0].tolist()

print(f'\n--    USING ALL POSITIVE LIME COEFFICIENTS:\n')
Xlime_pos_train = betas_resampled[:,positive_all]
ylime_pos_train = targets_resampled

Xlime_pos_test = test_betas[:,positive_all]
ylime_pos_test = test_targets


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
pickle.dump(rf_poslime, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/rf_all-pos_lime.pkl', 'wb'))

for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
    print(f'\nJPEG-TESTING {qf}: ')
    dct_path=dcts_path+'_'+qf
    X_test, y_test = get_train_test(dct_path, train=False, )
    X_test = X_test[:,positive_all]
    getmetrics(y_test, rf_poslime.predict(X_test))

print(f'\n--    USING ALL ABS-SIGNIFICATIVE LIME COEFFICIENTS:\n')
Xlime_abs_sign_train = betas_resampled[:,abs_sign]
ylime_abs_sign_train = targets_resampled

Xlime_abs_sign_test = test_betas[:,abs_sign]
ylime_abs_sign_test = test_targets


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
pickle.dump(rf_abs_sign, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/rf_all-absign_lime.pkl', 'wb'))
for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
    print(f'\nJPEG-TESTING {qf}: ')
    dct_path=dcts_path+'_'+qf
    X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
    X_test = X_test[:,abs_sign]
    getmetrics(y_test, rf_abs_sign.predict(X_test))

print('\n-  KNN\n')
attrs_all = np.load('attrs.npy')
avg_arr = np.mean(attrs_all,axis=0)
positive_all = np.where(avg_arr>0)[0].tolist()
abs_sign = np.where(abs(avg_arr)>np.median(abs(avg_arr)))[0].tolist()

print(f'\n--    USING ALL POSITIVE LIME COEFFICIENTS:\n')
Xlime_pos_train = betas_resampled[:,positive_all]
ylime_pos_train = targets_resampled

Xlime_pos_test = test_betas[:,positive_all]
ylime_pos_test = test_targets


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
pickle.dump(knn_poslime, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/knn_all-pos_lime.pkl', 'wb'))
for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
    print(f'\nJPEG-TESTING {qf}: ')
    dct_path=dcts_path+'_'+qf
    X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
    X_test = X_test[:,positive_all]
    getmetrics(y_test, knn_poslime.predict(X_test))

print(f'\n--    USING ALL ABS-SIGNIFICATIVE LIME COEFFICIENTS:\n')
Xlime_abs_sign_train = betas_resampled[:,abs_sign]
ylime_abs_sign_train = targets_resampled

Xlime_abs_sign_test = test_betas[:,abs_sign]
ylime_abs_sign_test = test_targets


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
pickle.dump(knn_abs_sign, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/knn_all-absign_lime.pkl', 'wb'))
for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
    print(f'\nJPEG-TESTING {qf}: ')
    dct_path=dcts_path+'_'+qf
    X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
    X_test = X_test[:,abs_sign]
    getmetrics(y_test, knn_abs_sign.predict(X_test))

# %% JPEG-TESTING USING LAST COEFFICIENTS
print('\n--     GRADIENT BOOSTING\n')
for last in range(35,30,-1):
    print(f'\n--    USING LAST {last} COEFFICIENTS:\n')
    Xls_train = betas_resampled[:,(63-last):]
    yls_train = targets_resampled

    Xls_test = test_betas[:,(63-last):]
    yls_test = test_targets

    
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

    pickle.dump(gb_l, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/gb_l'+str(last)+'.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
        X_test = X_test[:,(63-last):]
        getmetrics(y_test, gb_l.predict(X_test))

print('\n--     RANDOM FOREST\n')
for last in range(35,30,-1):
    print(f'\n--    USING LAST {last} COEFFICIENTS:\n')
    Xls_train = betas_resampled[:,(63-last):]
    yls_train = targets_resampled

    Xls_test = test_betas[:,(63-last):]
    yls_test = test_targets

    
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

    pickle.dump(rf_l, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/rf_l'+str(last)+'.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
        X_test = X_test[:,(63-last):]
        getmetrics(y_test, rf_l.predict(X_test))

print('\n--     KNN\n')
for last in range(35,30,-1):
    print(f'\n--    USING LAST {last} COEFFICIENTS:\n')
    Xls_train = betas_resampled[:,(63-last):]
    yls_train = targets_resampled

    Xls_test = test_betas[:,(63-last):]
    yls_test = test_targets

    
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

    pickle.dump(knn_l, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/knn_l'+str(last)+'.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
        X_test = X_test[:,(63-last):]
        getmetrics(y_test, knn_l.predict(X_test))

# %% JPEG-TESTING USING FIRST COEFFICIENTS
print('\n--     GRADIENT BOOSTING\n')
for first in range(30,27,-1):
    print(f'\n USING FIRST {first} COEFFICIENTS:\n')
    Xf_train = betas_resampled[:,:first]
    yf_train = targets_resampled

    Xf_test = test_betas[:,:first]
    yf_test = test_targets

    
    gb_f = fitgb(X_train=Xf_train,
                    y_train=yf_train,
                    X_test=Xf_test,
                    y_test=yf_test,
                    random_state=13,
                    gridsearch=False,
                    randomsearch=True,
                    params_grid=params_grids['gb'],
                    n_iter=100,
                    cv_folds=3,
                    verbose=0,
                    plotcm=True)

    pickle.dump(gb_f, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/gb_f'+str(first)+'.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
        X_test = X_test[:,:first]
        getmetrics(y_test, gb_f.predict(X_test))

print('\n--     RANDOM FOREST\n')
for first in range(30,27,-1):
    print(f'\n USING FIRST {first} COEFFICIENTS:\n')
    Xf_train = betas_resampled[:,:first]
    yf_train = targets_resampled

    Xf_test = test_betas[:,:first]
    yf_test = test_targets

    
    rf_f = fitrf(X_train=Xf_train,
                    y_train=yf_train,
                    X_test=Xf_test,
                    y_test=yf_test,
                    random_state=13,
                    gridsearch=False,
                    randomsearch=True,
                    params_grid=params_grids['rf'],
                    n_iter=100,
                    cv_folds=3,
                    verbose=0,
                    plotcm=True)

    pickle.dump(rf_f, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/rf_f'+str(first)+'.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
        X_test = X_test[:,:first]
        getmetrics(y_test, rf_f.predict(X_test))

print('\n--     KNN\n')
for first in range(30,27,-1):
    print(f'\n USING FIRST {first} COEFFICIENTS:\n')
    Xf_train = betas_resampled[:,:first]
    yf_train = targets_resampled

    Xf_test = test_betas[:,:first]
    yf_test = test_targets

    
    knn_f = fitknn(X_train=Xf_train,
                    y_train=yf_train,
                    X_test=Xf_test,
                    y_test=yf_test,
                    random_state=13,
                    gridsearch=False,
                    randomsearch=True,
                    params_grid=params_grids['knn'],
                    n_iter=100,
                    cv_folds=3,
                    verbose=0,
                    plotcm=True)

    pickle.dump(knn_f, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/knn_f'+str(first)+'.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
        X_test = X_test[:,:first]
        getmetrics(y_test, knn_f.predict(X_test))

# %% JPEG-TESTING USING CENTER COEFFICIENTS
print('\n--     GRADIENT BOOSTING\n')
for center in range(13,16):
    print(f'\n USING CENTERED {center} COEFFICIENTS:\n')
    Xc_train = betas_resampled[:,(31-center):(31+center+1)]
    yc_train = targets_resampled

    Xc_test = test_betas[:,(31-center):(31+center+1)]
    yc_test = test_targets


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

    pickle.dump(gb_f, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/gb_c'+str(center)+'.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
        X_test = X_test[:,(31-center):(31+center+1)]
        getmetrics(y_test, gb_c.predict(X_test))

print('\n--     RANDOM FOREST\n')
for center in range(13,16):
    print(f'\n USING CENTERED {center} COEFFICIENTS:\n')
    Xc_train = betas_resampled[:,(31-center):(31+center+1)]
    yc_train = targets_resampled

    Xc_test = test_betas[:,(31-center):(31+center+1)]
    yc_test = test_targets

    
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

    pickle.dump(rf_f, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/rf_c'+str(center)+'.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
        X_test = X_test[:,(31-center):(31+center+1)]
        getmetrics(y_test, rf_c.predict(X_test))

print('\n--     KNN\n')
for center in range(13,16):
    print(f'\n USING CENTERED {center} COEFFICIENTS:\n')
    Xc_train = betas_resampled[:,(31-center):(31+center+1)]
    yc_train = targets_resampled

    Xc_test = test_betas[:,(31-center):(31+center+1)]
    yc_test = test_targets

    
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

    pickle.dump(knn_f, open('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/models/jpeg_testing/knn_c'+str(center)+'.pkl', 'wb'))
    for qf in ['QF90', 'QF70', 'QF50', 'QF30']:
        print(f'\nJPEG-TESTING {qf}: ')
        dct_path=dcts_path+'_'+qf
        X_test, y_test = get_train_test(dct_path, train=False, print_size=False)
        X_test = X_test[:,(31-center):(31+center+1)]
        getmetrics(y_test, knn_c.predict(X_test))