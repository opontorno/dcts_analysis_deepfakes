from fitmodels import *
from settings import X_train, X_test, y_test, y_train, params_grids
import pickle

svc_all = fitsvc(X_train=X_train,
                 y_train=y_train,
                 X_test=X_test,
                 y_test=y_test,
                 random_state=13,
                 gridsearch=False,
                 randomsearch=True,
                 params_grid=params_grids['svc'],
                 n_iter=35,
                 cv_folds=3,
                 verbose=3,
                 plotcm=True)

pickle.dump(svc_all, open('models/all_betas/svc_all.pkl', 'wb'))