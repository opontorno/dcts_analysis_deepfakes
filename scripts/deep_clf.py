# %% import libaries
import numpy as np
import torch
import torch.nn as nn
import captum
from captum.attr import Lime
from wd import working_dir
import time, os, pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from IPython.display import clear_output
from torchsampler import ImbalancedDatasetSampler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, f1_score
from tqdm import tqdm

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %% define dataset class
batch_size=32

class betas_dset(Dataset):
    def __init__(self, betas_path,qf100=True):
        self.betas = betas_path
        self.files = []
        self.qf100 = qf100
        models = sorted(os.listdir(self.betas))
        for models_name in models:
            class_idx = models.index(models_name)
            model_path = os.path.join(self.betas, models_name)
            architectures = sorted(os.listdir(model_path))
            for architecture in architectures:
                architecture_path = os.path.join(model_path, architecture)
                for betas_img in os.listdir(architecture_path):
                    betas_path = os.path.join(architecture_path, betas_img)
                    self.files += [{'betas_img': betas_path, 'class': class_idx}]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        item = self.files[i]
        file_path = item['betas_img']
        class_idx = torch.tensor(item['class'])
        if self.qf100: 
            betas = torch.tensor(pickle.load(open(file_path, 'rb'))[1:])
        else:
            betas = torch.tensor(pickle.load(open(file_path, 'rb')))
        return betas, class_idx

def main():
    dset = betas_dset(working_dir+'datasets/dcts_QF30', qf100=False)
    trainset, testingset = random_split(dset, lengths=[.7,.3])
    validset, testset = random_split(testingset, lengths=[.5,.5])
    trainloader = DataLoader(
        trainset,
        shuffle=True,
        batch_size=batch_size
    )
    validloader = DataLoader(
        validset,
        shuffle=False,
        batch_size=batch_size
    )
    testloader = DataLoader(
        testset,
        shuffle=False,
        batch_size=batch_size
    )
    loaders={'train':trainloader, 'valid':validloader, 'test':testloader}

    class deep_all(nn.Module):
        def __init__(self):
            super(deep_all, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(63, 256),
                nn.ReLU()
            )
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),            
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(64, 3)
            )
        def forward(self, img):
            x = self.encoder(img)
            x = self.classifier(x)
            return x

    training = True
    model = deep_all()
    epochs=150
    plot_lc=True
    optimizer=torch.optim.Adam(model.parameters(),
                            lr=0.0001,
                            weight_decay=0.0001)
    scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.9)
    loss_fn=nn.CrossEntropyLoss()
    if training:
        model.to(dev)
        train_loader, valid_loader, test_loader = loaders['train'], loaders['valid'], loaders['test']
        losses={'train' : [], 'val' : []}
        accuracies={'train' : [], 'val' : []}
        staring_time = time.time()
        for epoch in tqdm(range(epochs), desc='epochs'):
            model.train()
            running_train_loss = 0.0
            running_train_accuracy = 0.0
            for batch, data in enumerate(train_loader):
                images, labels = data
                images = images.to(dev)
                labels = labels.squeeze_().to(dev)
                output = model(images)
                loss = loss_fn(output, labels)
                #backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #calculate performance
                preds = torch.argmax(output, 1)
                batch_accuracy = (preds == labels).sum().item()/images.size(0)
                running_train_loss += loss.item()
                running_train_accuracy += batch_accuracy
            losses['train'].append(running_train_loss / len(train_loader))
            accuracies['train'].append(running_train_accuracy / len(train_loader))
            model.eval()
            running_val_loss = 0.0
            running_val_accuracy = 0.0
            with torch.no_grad():
                for batch, data in enumerate(valid_loader):
                    images, labels = data
                    images = images.to(dev)
                    labels = labels.squeeze_().to(dev)
                    output = model(images)
                    preds = torch.argmax(output, 1)
                    loss = loss_fn(output, labels)
                    #calculate performance
                    batch_accuracy = (preds == labels).sum().item()/images.size(0)
                    running_val_loss += loss.item()
                    running_val_accuracy += batch_accuracy
            losses['val'].append(running_val_loss/len(valid_loader))
            accuracies['val'].append(running_val_accuracy/len(valid_loader))
            clear_output(wait=True)
            if plot_lc:
                # plot results
                fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))
                for i in range(2):
                    if i==0:
                        plot = losses
                        ax=ax1
                    else:
                        plot = accuracies
                        ax=ax2
                    ax.plot(plot['train'], label="Train")
                    ax.plot(plot['val'], label="Validation")
                    ax.legend()
                plt.show()
            print(f'Epoch {epoch+1}/{epochs}: ',
                f'train loss {losses["train"][-1]:.4f}, val loss: {losses["val"][-1]:.4f}, ',
                f'train acc {accuracies["train"][-1]*100:.2f}%, val acc: {accuracies["val"][-1]*100:.2f}%')
            if scheduler != None:
                scheduler.step()
        ending_time = time.time()
        esecution_time = ending_time - staring_time
        print(f'\ntotal time: {(esecution_time/60):2f} minutes.')
    else:
        model = torch.load('../data/deep_cls.pt')

    model.eval()
    plot_cm = True
    total_loss, total_correct, total_samples = 0,0,0
    y_true, y_pred = [], []
    starting_time = time.time()
    for batch, data in tqdm(enumerate(testloader), desc='batches'):
        inputs, labels = data
        inputs = inputs.to(dev)
        labels = labels.to(dev)
        with torch.no_grad():
            outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f"Test Loss: {average_loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Recall: {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Test Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Test f1-Score: {f1_score(y_true, y_pred, average='macro'):.4f}")
    if plot_cm:
        cm = confusion_matrix(y_true, y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['diffusion models', 'gans', 'real'])
        print("Confusion Matrix:")
        cm_display.plot()
        plt.show()
    ending_time = time.time()
    esecution_time = ending_time - starting_time
    print(f'\ntotal training time: {(esecution_time/60):2f} minutes.')

    # %% use XAI
    lime = Lime(model)
    def get_attributes(on_classes = [0,1,2], name_file='', save=False):
        attrs = np.zeros((1,63))
        for i in tqdm(range(len(testset))):
            if y_pred[i]==y_true[i] and y_true[i] in on_classes:
                attr = lime.attribute(testset[i][0].unsqueeze_(0).to(dev), target=int(y_true[i]), n_samples=100)
                attrs = np.append(attrs, np.array(attr.to('cpu')), axis=0)
        attrs = np.delete(attrs, (0), axis=0)
        if save: 
            np.save(name_file, attrs)
            plt.savefig(f'../data/bar-{name_file}.png')
        plt.bar(range(1,64), np.mean(attrs, axis=0))
        plt.show()
        return attrs


    attrs = get_attributes(name_file='attrs_all', save=True)

if __name__=='__main__':
    main()