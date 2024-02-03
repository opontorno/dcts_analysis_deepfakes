#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd

plot=True

#%% Analysis of average distribution
classes = {'real':0, 'gan_generated':1, 'dm_generated':2}

def get_data(dcts_path, classes=classes, print_size=True):
    betas = []
    targets = []
    for model in os.listdir(dcts_path):
        model_path = os.path.join(dcts_path, model)
        for architecture in os.listdir(model_path):
            architecture_path = os.path.join(model_path, architecture)
            for dcts in os.listdir(architecture_path):
                img_dcts_path = os.path.join(architecture_path, dcts)
                img_dcts = pickle.load(open(img_dcts_path, 'rb'))
                betas.append(img_dcts)
                targets.append(classes[model])
    betas = np.array(betas)[:,1:] if not dcts_path.endswith('0') else np.array(betas)
    targets = np.array(targets)
    if print_size:
        print(f'BETAS: \ntype: {type(betas)} \nsize: {betas.shape}\n')
        print(f'TARGETS: \ntype: {type(targets)} \nsize: {targets.shape} \nunique values: {np.unique(targets, return_counts=True)}\n')
    return betas, targets

def get_plot(betas, targets, classes=classes):
    avg_real = betas[targets==classes['real']].mean(axis=0)
    avg_gan = betas[targets==classes['gan_generated']].mean(axis=0)
    avg_dm = betas[targets==classes['dm_generated']].mean(axis=0)

    plt.plot(avg_real, label='REAL')
    plt.plot(avg_gan, label='GAN')
    plt.plot(avg_dm, label='DIFFUSION MODEL')
    plt.legend()
    plt.show()

#%% PLOT DIFFERENCIES QF
def plot_differencies(klass, colors=['navy', 'blue', 'steelblue', 'cadetblue', 'lightblue']):
    betas, targets = get_data('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/datasets/dcts', print_size=False)
    betas90, targets90 = get_data('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/datasets/dcts_QF90', print_size=False)
    betas70, targets70 = get_data('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/datasets/dcts_QF70', print_size=False)
    betas50, targets50 = get_data('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/datasets/dcts_QF50', print_size=False)
    betas30, targets30 = get_data('/home/opontorno/data/opontorno/research_activities/dcts_analysis_deepfakes/datasets/dcts_QF30', print_size=False)
    avg_real = betas[targets==classes[klass]].mean(axis=0)
    avg_real90 = betas90[targets90==classes[klass]].mean(axis=0)
    avg_real70 = betas70[targets70==classes[klass]].mean(axis=0)
    avg_real50 = betas50[targets50==classes[klass]].mean(axis=0)
    avg_real30 = betas30[targets30==classes[klass]].mean(axis=0)
    plt.plot(avg_real, label='RAW')
    plt.plot(avg_real90, label='QF90')
    plt.plot(avg_real70,  label='QF70')
    plt.plot(avg_real50,  label='QF50')
    plt.plot(avg_real30,  label='QF30')
#    plt.title(f'Compression of {klass}')
    plt.legend()
    plt.show()

plot_differencies('real', colors=['navy', 'blue', 'steelblue', 'cadetblue', 'lightblue'])
# plot_differencies('dm_generated', colors=['darkgreen', 'green', 'forestgreen', 'lightgreen', 'yellowgreen'])
# plot_differencies('gan_generated', colors=['darkorange', 'orange', 'sandybrown', 'burlywood', 'peachpuff'])
# %%
guidance = '/home/opontorno/data/opontorno/datasets_guidance.csv'
def get_train_test(dcts_path, guidance=guidance, train=True, print_size=True):
    labels = pd.read_csv(guidance)
    df_train = labels[labels['label'].isin([0,1])]
    df_test = labels[labels['label']==2]
    betas=[]
    targets=[]
    df = df_train if train else df_test
    for _, row in df.iterrows():
        img_path = row['image_path'].split('datasets/')[1][:-4]
        img_class = classes[row['model']]
        dct_path = os.path.join(dcts_path, img_path+'-DCAC.pkl')
        img_dcts = pickle.load(open(dct_path, 'rb'))
        betas.append(img_dcts)
        targets.append(img_class)
    betas = np.array(betas)[:,1:] if not dcts_path.endswith('0') else np.array(betas)
    targets = np.array(targets)
    if print_size:
        print(f'BETAS: \ntype: {type(betas)} \nsize: {betas.shape}\n')
        print(f'TARGETS: \ntype: {type(targets)} \nsize: {targets.shape} \nunique values: {np.unique(targets, return_counts=True)}\n')
    return betas, targets
# %%
