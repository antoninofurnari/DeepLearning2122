import torch
import random
import numpy as np

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

NUM_EPOCHS=50

from torch.utils import data
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import itertools
from torch.utils.data import DataLoader
import torchvision
from torch.optim import SGD

from os.path import join
import numpy as np
from PIL import Image
from torch import nn
from torchvision import transforms

class EgoCart(data.Dataset):
    def __init__(self, root, mode='train', transform=None):
        self.root = join(root, f"{mode}_set", f"{mode}_RGB")
        self.transform = transform
        
        #leggiamo i nomi dei file e le etichette xy e uv
        labs = np.loadtxt(join(root, f"{mode}_set", f"{mode}_set.txt"), dtype=str)
        self.image_names = labs[:,0]
        self.xy = labs[:,2:4].astype(float)
        self.uv = labs[:,4:6].astype(float)
        
        # normalizziamo uv per assicurarci che siano dei versori
        self.uv/=np.linalg.norm(self.uv,axis=1).reshape(-1,1)
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, i):
        im = Image.open(join(self.root, self.image_names[i]))
        if self.transform is not None:
            im = self.transform(im)
        return im, self.xy[i], self.uv[i]
    
    
from torchvision.models import  mobilenet_v2
base_model = mobilenet_v2(pretrained=False)
base_model.classifier = nn.Identity()

#from torchvision.models import squeezenet1_0
#base_model = squeezenet1_0(pretrained=True)
#base_model.classifier[1]=nn.Identity()

from tqdm import tqdm
def extract_representations(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    representations, xy, uv = [], [], []
    for batch in tqdm(loader, total=len(loader)):
        x = batch[0].to(device)
        rep = model(x)
        rep = rep.detach().to('cpu').numpy()
        xy.append(batch[1])
        uv.append(batch[2])
        representations.append(rep)
    return np.concatenate(representations), np.concatenate(xy), np.concatenate(uv)

egocart_train_loader = DataLoader(EgoCart(root='egocart',mode='train', 
                        transform=transforms.Compose([
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            # utilizziamo media e deviazione standard fornite per il modello
                            # (sono quelle di imagenet)
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])), shuffle=True, num_workers=8, batch_size=32)

egocart_test_loader = DataLoader(EgoCart(root='egocart',mode='test', 
                        transform=transforms.Compose([
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])), num_workers=8, batch_size=32)


import faiss
def predict_nn(train_rep, test_rep, train_xy, train_uv):
    # inizializziamo l'oggetto index che utilizzeremo per indicizzare le rappresentazioni
    index = faiss.IndexFlatL2(train_rep.shape[1])
    # aggiungiamo le rappresentazioni di training all'indice
    index.add(train_rep.astype(np.float32))
    # effettuaimo la ricerca

    indices = np.array([index.search(x.reshape(1,-1).astype(np.float32), k=1)[1][0][0] for x in test_rep])
    # restituiamo le etichette predette
    return train_xy[indices].squeeze(), train_uv[indices].squeeze()

def evaluate_localization(pred_xy, pred_uv, gt_xy, gt_uv):
    position_error = np.sqrt(((pred_xy-gt_xy)**2).sum(1)).mean()
    orientation_error = np.rad2deg(np.arccos((gt_uv*pred_uv).sum(1))).mean()
    return position_error, orientation_error

from sklearn.metrics.pairwise import pairwise_distances

class TripletEgoCart(data.Dataset):
    def __init__(self, dataset, min_dist=0.3, min_orient=45):
        self.dataset = dataset
        
        # recuperiamo le etichette xy e uv
        xy = self.dataset.xy
        uv = self.dataset.uv

        # calcoliamo una matrice di milarità per distanze
        xy_similar = pairwise_distances(xy)<min_dist
        # matrice di similarità per orientamento
        uv_similar = np.arccos(1-pairwise_distances(uv, metric='cosine'))<np.deg2rad(min_orient)
        # mettendo in and le due matrici, otteniamo una matrice di similarità
        # in presenza di "true" abbiamo una coppia simile, mentre in presenza di "false" 
        # abbiamo una coppia dissimile
        self.similar = xy_similar & uv_similar
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        # pivot
        I_i, xy, uv = self.dataset[i]
        # campioniamo un indice simile
        j = np.random.choice(np.where(self.similar[i])[0])
        I_j,*_ = self.dataset[j]
        # campioniamo un indice dissimile
        k = np.random.choice(np.where(~self.similar[i])[0])
        I_k,*_ = self.dataset[k]
        return I_i, I_j, I_k, np.concatenate([xy, uv])
    
    
egocart_train = EgoCart(root='egocart',mode='train', 
                        transform=transforms.Compose([
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            # utilizziamo media e deviazione standard fornite per il modello
                            # (sono quelle di imagenet)
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]))

egocart_test = EgoCart(root='egocart',mode='test', 
                        transform=transforms.Compose([
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            # utilizziamo media e deviazione standard fornite per il modello
                            # (sono quelle di imagenet)
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]))


triplet_egocart_train_loader = DataLoader(TripletEgoCart(egocart_train), shuffle=True, num_workers=8, batch_size=32)

triplet_egocart_test_loader = DataLoader(TripletEgoCart(egocart_test), num_workers=8, batch_size=32)

class TripletNetworkTask(pl.LightningModule):
    def __init__(self, 
                 embedding_net, # la rete di embedding
                 lr=0.001, # il learning rate
                 momentum=0.99, # momentum
                 margin=2 # margine per la loss
                ):
        super(TripletNetworkTask, self).__init__()
        self.save_hyperparameters()
        self.embedding_net = embedding_net
        self.criterion = nn.TripletMarginLoss(margin = margin)
                    
    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        return SGD(self.embedding_net.parameters(), self.hparams.lr, momentum=self.hparams.momentum)
    
    def training_step(self, batch, batch_idx):
        I_i, I_j, I_k, *_ = batch
        #l'implementazione della rete triplet è banale quanto quella della rete siamese:
        #eseguiamo la embedding net sui tre input
        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)
        phi_k = self.embedding_net(I_k)

        #calcoliamo la loss
        loss_triplet = self.criterion(phi_i, phi_j, phi_k)
        
        loss_embedd = phi_i.norm(2) + phi_j.norm(2) + phi_k.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd
        
        self.log('train/loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        I_i, I_j, I_k, xyuv = batch
        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)
        phi_k = self.embedding_net(I_k)

        #calcoliamo la loss
        loss_triplet = self.criterion(phi_i, phi_j, phi_k)
        
        loss_embedd = phi_i.norm(2) + phi_j.norm(2) + phi_k.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd
        
        self.log('valid/loss', loss)
        return loss
        
        #if batch_idx==0:
        #    self.logger.experiment.add_embedding(phi_i, xyuv, I_i, global_step=self.global_step)
        

#egocart_train_representations_triplet, egocart_train_xy, egocart_train_uv = extract_representations(base_model, egocart_train_loader)
#egocart_test_representations_triplet, egocart_test_xy, egocart_test_uv = extract_representations(base_model, egocart_test_loader)

#egocart_pred_test_xy_triplet, egocart_pred_test_uv_triplet = predict_nn(egocart_train_representations_triplet, egocart_test_representations_triplet, #egocart_train_xy, egocart_train_uv)

#position_error, orientation_error = evaluate_localization(egocart_pred_test_xy_triplet, egocart_pred_test_uv_triplet, egocart_test_xy, egocart_test_uv)
#print(f"Position error: {position_error:0.2f}m")
#print(f"Orientation error: {orientation_error:0.2f}°")
        
        
triplet_egocart_task = TripletNetworkTask(base_model)
logger = TensorBoardLogger("metric_logs", name="egocart_triplet")
trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=50, check_val_every_n_epoch=5)

trainer.fit(triplet_egocart_task, triplet_egocart_train_loader, triplet_egocart_test_loader)

egocart_train_representations_triplet, egocart_train_xy, egocart_train_uv = extract_representations(triplet_egocart_task.embedding_net, egocart_train_loader)
egocart_test_representations_triplet, egocart_test_xy, egocart_test_uv = extract_representations(triplet_egocart_task.embedding_net, egocart_test_loader)

egocart_pred_test_xy_triplet, egocart_pred_test_uv_triplet = predict_nn(egocart_train_representations_triplet, egocart_test_representations_triplet, egocart_train_xy, egocart_train_uv)

position_error, orientation_error = evaluate_localization(egocart_pred_test_xy_triplet, egocart_pred_test_uv_triplet, egocart_test_xy, egocart_test_uv)
print(f"Position error: {position_error:0.2f}m")
print(f"Orientation error: {orientation_error:0.2f}°")
