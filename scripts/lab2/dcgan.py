import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
import pytorch_lightning as pl

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        # dimensione di partenza. Se la dimensione in output è 32 x 32, questa sarà  8 x 8
        self.init_size = img_shape[1] // 4

        # usiamo un layer lineare per proiettare il vettore latente su
        # un vettore di dimensione 128 * 8 * 8
        # useremo una view per trasformare questo vettore in una mappa 128 x 8 x 8 nel metodo forward
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            # usiamo le batch normalization
            # per accelerare il training
            nn.BatchNorm2d(128),
            # questo layer fa upsampling della mappa 8x8 e la fa diventare 16x16
            nn.Upsample(scale_factor=2),
            #convoluzione con padding=1 per mantenere la stessa dimensione 16x16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True), #usiamo le leaky relu invece delle relu
            #queste funzionano meglio con le GAN
            nn.Upsample(scale_factor=2), #16x16 -> 32x32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            #layer finale di convluzioni che fa passare da 64 canali a 3
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.l1(z)
        #la view serve per passare da un vettore a una mappa di feature
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        #definisce un "blocco del discriminatore"
        def discriminator_block(in_feat, out_feat, bn=True):
            #questo include una convoluzione, una LeakyReLU, e un dropout
            block = [nn.Conv2d(in_feat, out_feat, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                #se richiesto inseriamo la batch normalization
                block.append(nn.BatchNorm2d(out_feat, 0.8))
            return block

        #costruisce il modulo principale
        #solo il primo layer ha la batch normalization
        self.model = nn.Sequential(
            *discriminator_block(img_shape[0], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # alla fine delle convoluzioni avremo una mappa a dimensionalità ridotta
        ds_size = img_shape[1] // 2 ** 4
        #usiamo un layer lineare e un sigmoide per ottenere la classificazione
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        #la view serve per passare dalla mappa di feature al vettore linearizzato
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class DCGAN(LightningModule):

    def __init__(self,
                 latent_dim: int = 100,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64, **kwargs):
        super().__init__()
        #questo metodo conserva una copia degli iperparametri alla quale accedere in seguito
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size

        # la dimensione dell'immagine in input# networks
        img_shape = (1, 32, 32)
        #costruisce generatore e discriminatore 
        self.generator = Generator(latent_dim=self.latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

        #conserva un insieme di codici di validazione da tenere fissi
        #per monitorare l'evoluzione della rete
        self.validation_z = torch.randn(8, self.latent_dim)

        #un esempio di input del modello
        self.example_input_array = torch.zeros(2, self.latent_dim)

    def forward(self, z):
        #in fase di forward usiamo solo il generatore
        return self.generator(z)
    
    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        # per implementare la procdura di ottimizzazione alternata,
        # definiamo due ottimizzatori invece di uno
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        # restituiamo dunque una lista contenente i due ottimizzatori
        # la lista vuota indica gli scheduler, che non specifichiamo
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        # questo metodo verrà chiamato due volte, una per ogni ottimizzatore definito
        # optimizer_idx permetterà di distinguere tra i due ottimizzatori
        imgs, _ = batch #scartiamo le etichette del batch

        # campioniamo dei vettori di rumore random
        # questi ci serviranno sia per il training del generatore 
        # che per quello del discriminatore# sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        # ci assicuriamo che il "tipo" sia lo stesso delle immagini (es. entrambi sono su GPU)
        z = z.type_as(imgs)

        # primo ottimizzatore: alleniamo il generatore
        if optimizer_idx == 0:

            # generiamo le immagini dai vettori random# generate images
            self.generated_imgs = self(z)

            # prendiamo le prime 50 immagini generate e le loggiamo# log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images', grid, self.global_step)

            # definiamo le etichette delle immagini che abbiamo generato
            # durante il training del generatore, vogliamo che vengano viste come "real" (etichetta 1)
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs) #mettiamole sul device corretto cambiandone il tipo

            # loss del generatoreetichette predette per le immagini generate
            g_loss = F.binary_cross_entropy(self.discriminator(self(z)), valid)

            self.log('generator/loss', g_loss.item())

            # restituiamo la loss più qualche statistica per la visualizzazione
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # allenamento del discriminatore# train discriminator
        if optimizer_idx == 1:
            # etichette delle immagini reali in imgs
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # otteniamo le predizioni per le immagini reali
            predicted_real_labels = self.discriminator(imgs)

            # loss per classificare bene le immagini reali
            real_loss = self.adversarial_loss(predicted_real_labels, valid)

            # adesso dobbiamo ripetere il processo per i fake
            # etichette fake: tutti zeri# how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            # generiamo le predizioni per le immagini fake
            predicted_fake_labels = self(z).detach()
            # loss per fake
            fake_loss = self.adversarial_loss(
                self.discriminator(predicted_fake_labels), fake)
            
            # calcoliamo la loss del discriminatore come media fra le due
            d_loss = (real_loss + fake_loss) / 2

            # log della loss del discriminatore
            self.log('discriminator/loss', d_loss.item())
            
            # calcoliamo l'accuracy fake e real
            d_fake_acc = (predicted_fake_labels<=0.5).float().mean()
            d_real_acc = (predicted_real_labels>0.5).float().mean()
            #accuracy finale
            d_acc = (d_fake_acc + d_real_acc)/2
            
            # log
            self.log('discriminator/fake_acc', d_fake_acc.item())
            self.log('discriminator/real_acc', d_real_acc.item())
            self.log('discriminator/overall_acc', d_acc.item())

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output


    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((32, 32)), #resize a 32x32 -> la rete è stata progettata per questa dimensione
            transforms.ToTensor(),
        ])

        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size)

    def on_epoch_end(self):
        #alla fine dell'epoca, generiamo le immagini corrispondenti agli z di validazione
        z = self.validation_z.to(self.device)

        # generiamo le immagini# log sampled images
        sample_imgs = self(z)
        #facciamo log
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('validation_images', grid, self.global_step)


from pytorch_lightning.loggers import TensorBoardLogger

mnist_gan = DCGAN()
logger = TensorBoardLogger("tb_logs", name="mnist_gan")
trainer = pl.Trainer(gpus=1, logger=logger)

trainer.fit(mnist_gan)
