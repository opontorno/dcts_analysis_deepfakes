from dcts_analysis import dset
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler 

perc_train = 0.7
random_seed = 13
betas = dset[:,:63]
targets = dset[:,63].astype(int)

## MAKE THE DATASET BALANCED
class_counts = {label: count for label, count in zip(*np.unique(targets, return_counts=True))}
min_class_count = min(class_counts.values())
min_class_label = min(class_counts, key=class_counts.get)
undersampler = RandomUnderSampler(sampling_strategy={label: min_class_count for label in class_counts})
betas_resampled, targets_resampled = undersampler.fit_resample(betas, targets)
print(f'\nRESAMPLED BETAS: \ntype: {type(betas_resampled)} \nsize: {betas_resampled.shape}\n')
print(f'RESAMPLED TARGETS: \ntype: {type(targets_resampled)} \nsize: {targets_resampled.shape} \nunique values: {np.unique(targets_resampled, return_counts=True)}')

X_train, X_test, y_train, y_test = train_test_split(betas_resampled, targets_resampled, test_size=(1-perc_train), random_state = random_seed)
print('\nAFTER SPLITTING: ')
print(f'X_train shape:  {np.shape(X_train)}')
print(f'y_train shape:  {np.shape(y_train)}, unique values: {np.unique(y_train, return_counts=True)}')

print(f'X_test shape:  {np.shape(X_test)}')
print(f'y_test shape:  {np.shape(y_test)}, unique values: {np.unique(y_test, return_counts=True)}')

params_grids = {
    'svc' : {
    'C': [0.1, 0.5, 1, 3],                              # Regularization parameter.
    'gamma': [1, 0.1, 0.01],                   # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    'kernel': ['linear', 'poly'],     # Specifies the kernel type to be used in the algorithm.
    'degree' : [2,3]                                    # Degree of the polynomial kernel function (‘poly’).
    },
    'knn' : {
    'n_neighbors': [x for x in range(3, 11, 2)],                                    # Number of neighbors to use by default for kneighbors queries.
    'weights' : ['uniform', 'distance'],                                            # Weight function used in prediction.
    'algorithm' : ['ball_tree', 'kd_tree', 'brute'],                                # Algorithm used to compute the nearest neighbors.
    'p' : [2,3,4],                                                                  # Power parameter for the Minkowski metric.
    'leaf_size' : [int(x) for x in np.linspace(start= 5, stop= 50, num = 10)]       # Leaf size passed to BallTree or KDTree.
    },
    'rf' : {
    'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 500, num = 10)], # Number of trees in random forest
    'max_features': [ 'sqrt','log2', None],                                           # Number of features to consider at every split
    'max_depth': [int(x) for x in np.linspace(10, 30, num = 11)] + [None],            # Maximum number of levels in tree
    'min_samples_split': [2, 5, 10],                                                  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],                                                    # Minimum number of samples required at each leaf node
    'bootstrap': [True, False],                                                       # Method of selecting samples for training each tree
    'criterion':['gini', 'entropy', 'log_loss']                                       # Metrics  to measure the quality of a split.
    },
    'gb' : {
    'loss': ['log_loss','exponential'],                                        # The loss function to be optimized.
    'learning_rate':[0.01,0.1, 0.15],                                          # Learning rate shrinks the contribution of each tree by learning_rate
    'n_estimators': [int(x) for x in np.linspace( 200, 500, num = 11)],        # Number of trees in random forest
    'max_features': ['sqrt','log2',None],                                      # Number of features to consider at every split
    'max_depth':  [int(x) for x in np.linspace(10, 30, num = 11)]+[None],      # Maximum number of levels in tree
    'min_samples_split': [5, 10, 15, 20],                                      # Minimum number of samples required to split a node
    'min_samples_leaf': [3, 5, 10, 15],                                        # Minimum number of samples required at each leaf node
    'criterion':['friedman_mse','squared_error']                               # Metrics to measure the quality of a split.
    }
}