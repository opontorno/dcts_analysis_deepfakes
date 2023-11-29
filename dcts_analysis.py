import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

dcts_path = 'dcts'
plot=True

## Analysis of average distribution

betas = []
targets = []
classes = {'real':0, 'gan_generated':1, 'dm_generated':2}
for model in os.listdir(dcts_path):
    model_path = os.path.join(dcts_path, model)
    for architecture in os.listdir(model_path):
        architecture_path = os.path.join(model_path, architecture)
        for dcts in os.listdir(architecture_path):
            img_dcts_path = os.path.join(architecture_path, dcts)
            img_dcts = pickle.load(open(img_dcts_path, 'rb'))
            betas.append(img_dcts)
            targets.append(classes[model])
betas = np.array(betas)[:,1:]
targets = np.array(targets)
dset = np.unique(np.concatenate((betas, targets.reshape(-1, 1)), axis=1), axis=0)

print(f'BETAS: \ntype: {type(betas)} \nsize: {betas.shape}\n')
print(f'TARGETS: \ntype: {type(targets)} \nsize: {targets.shape} \nunique values: {np.unique(targets, return_counts=True)}')

if plot:
    avg_real = betas[targets==classes['real']].mean(axis=0)
    avg_gan = betas[targets==classes['gan_generated']].mean(axis=0)
    avg_dm = betas[targets==classes['dm_generated']].mean(axis=0)

    plt.plot(avg_real, label='REAL')
    plt.plot(avg_gan, label='GAN')
    plt.plot(avg_dm, label='DIFFUSION MODEL')
    plt.title('Average betas distribution per class')
    plt.legend()
    plt.show()
