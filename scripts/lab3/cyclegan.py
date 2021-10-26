import torch
import random
import numpy as np

import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
import itertools
from torch.utils.data import DataLoader
import torchvision

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform #conserva la transform

        # ottieni i path delle immagini in A e B
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        #apro l'iesima immagine A (uso il modulo per evitare di sforare)
        item_A = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        #apro una immagine B a caso
        item_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB')
        
        if self.transform:
            item_A = self.transform(item_A)
            item_B = self.transform(item_B)

        return item_A, item_B

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        # Il blocco segue la struttura di un classico residual block di ResNet
        conv_block = [  nn.ReflectionPad2d(1), # Il ReflectionPad fa padding usando una versione "specchiata" del bordo dell'immagine
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features), # Instance normalization, un tipo di normalizzazione simile a batch normalization spesso usato per style transfer
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        # La forward applica il blocco e somma l'input per ottenere una connessione residua
        return x + self.conv_block(x)
    
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Blocco di convoluzioni iniziale che mappa l'input su 64 feature maps
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # ============ Encoder ============
        # Due blocchi che mappano l'input
        # da 64 a 128 e da 128 a 256 mappe
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Aggiungiamo dunque i residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # ============ Decoder =============
        # Due blocchi di convoluzione
        out_features = in_features//2
        for _ in range(2):
            # Qui usiamo la convoluzione trasposta (https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
            # che fa upsampling piuttosto che downsampling quando si impostano stride e padding
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Layer finale di output
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        # Inizializziamo l'oggetto sequential con la lista dei moduli
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # Diversi layer di convoluzione. In questo caso usiamo le LeakyReLU invece delle ReLU
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # Layer di classificazione
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

import pytorch_lightning as pl
import itertools
from torch.utils.data import DataLoader
import torchvision

class CycleGAN(pl.LightningModule):
    def __init__(self, 
                 input_nc = 3, # numero di canali in input
                 output_nc = 3, # numero di canali in output
                 image_size = 256, # la dimensione dell'immagine
                 lr=0.0002, # learning rate
                 betas=(0.5, 0.999), # valori beta di Adam
                 starting_epoch=0, # epoca di partenza (0 se non stiamo facendo resume di un training interrotto)
                 n_epochs=200, # numero totale di epoche
                 decay_epoch=100, # epoca dopo la quale iniziare a far scendere il learning rate
                 data_root = 'grumpifycat', #cartella in cui si trovano i dati
                 batch_size = 1, #batch size
                 n_cpu = 8 # numeri di thread per il dataloader
                  ):
        super(CycleGAN, self).__init__()
        self.save_hyperparameters()
        
        # Definiamo due generatori: da A a B e da B ad A
        self.netG_A2B = Generator(input_nc, output_nc)
        self.netG_B2A = Generator(output_nc, input_nc)
        
        # Definiamo due discriminatori: uno per A e uno per B
        self.netD_A = Discriminator(input_nc)
        self.netD_B = Discriminator(output_nc)
        
        # Applichiamo la normalizzazione
        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)
        
        # Definiamo le loss
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
                    
    def forward(self, x, mode='A2B'):
        if mode=='A2B':
            return netG_A2B(x)
        else:
            return netG_B2A(x)
    
    # ci servono 3 optimizer, ognuno con il suo scheduler
    def configure_optimizers(self):
        # Optimizer per il generatore
        optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                lr=self.hparams.lr, betas=self.hparams.betas)
        
        # Optimizers per i due discriminatori
        optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=self.hparams.lr, betas=self.hparams.betas)
        optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=self.hparams.lr, betas=self.hparams.betas)
        
        # Questi scheduler fanno decadere il learning rate dopo 100 epoche
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(self.hparams.n_epochs, self.hparams.starting_epoch, self.hparams.decay_epoch).step)
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(self.hparams.n_epochs, self.hparams.starting_epoch, self.hparams.decay_epoch).step)
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(self.hparams.n_epochs, self.hparams.starting_epoch, self.hparams.decay_epoch).step)
        
        # Restituiamo i tre optimizer e i tre optimizers
        return [optimizer_G, optimizer_D_A, optimizer_D_B], [lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B]
    
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A, real_B = batch
        
        target_real = torch.ones((real_A.shape[0],1)).type_as(real_A)
        target_fake = torch.zeros((real_A.shape[0],1)).type_as(real_A)
        
        # optimizer del generatore
        if optimizer_idx==0:
            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = self.netG_A2B(real_B) # passa B al generatore A2B - deve essere uguale a B
            loss_identity_B = self.criterion_identity(same_B, real_B)*5.0

            # G_B2A(A) should equal A if real A is fed
            same_A = self.netG_B2A(real_A) # passa A al generatore B2A - deve essere uguale ad A
            loss_identity_A = self.criterion_identity(same_A, real_A)*5.0

            # GAN loss
            fake_B = self.netG_A2B(real_A) # passa A ad A2B - sono le fake B
            pred_fake = self.netD_B(fake_B) # predizioni del discriminatore B
            loss_GAN_A2B = self.criterion_GAN(pred_fake, target_real) # loss GAN per A2B

            fake_A = self.netG_B2A(real_B) # passa B a B2A - sono le fake A
            pred_fake = self.netD_A(fake_A) # predizioni del discriminatore A
            loss_GAN_B2A = self.criterion_GAN(pred_fake, target_real) # loss GAN per B2A

            # Cycle consistency loss
            recovered_A = self.netG_B2A(fake_B) #passiamo le fake B a B2A - devono essere uguali a real_A
            loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A)*10.0 #cycle consistency loss

            recovered_B = self.netG_A2B(fake_A) #passiamo le fake A a A2B - devono essere uguali a real_B
            loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B)*10.0 #cycle consistency loss

            # loss globale
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            self.log('loss_G/loss_identity_A', loss_identity_A)
            self.log('loss_G/loss_identity_B', loss_identity_B)
            self.log('loss_G/loss_GAN_A2B', loss_GAN_A2B)
            self.log('loss_G/loss_GAN_B2A', loss_GAN_B2A)
            self.log('loss_G/loss_cycle_ABA', loss_cycle_ABA)
            self.log('loss_G/loss_cycle_BAB', loss_cycle_BAB)
            self.log('loss_G/overall', loss_G)
            
            # loggiamo dei campioni visivi da ispezionare durante il training ogni 100 batch
            if batch_idx % 100 ==0:
                grid_A = torchvision.utils.make_grid(real_A[:50], nrow=10, normalize=True)
                grid_A2B = torchvision.utils.make_grid(fake_B[:50], nrow=10, normalize=True)
                grid_A2B2A = torchvision.utils.make_grid(recovered_A[:50], nrow=10, normalize=True)
                
                grid_B = torchvision.utils.make_grid(real_B[:50], nrow=10, normalize=True)
                grid_B2A = torchvision.utils.make_grid(fake_A[:50], nrow=10, normalize=True)
                grid_B2A2B = torchvision.utils.make_grid(recovered_B[:50], nrow=10, normalize=True)
                
                self.logger.experiment.add_image('A/A', grid_A, self.global_step)
                self.logger.experiment.add_image('A/A2B', grid_A2B, self.global_step)
                self.logger.experiment.add_image('A/A2B2A', grid_A2B2A, self.global_step)
                
                self.logger.experiment.add_image('B/B', grid_B, self.global_step)
                self.logger.experiment.add_image('B/B2A', grid_B2A, self.global_step)
                self.logger.experiment.add_image('B/B2A2B', grid_B2A2B, self.global_step)
                
            return loss_G
        
        elif optimizer_idx==1: #discriminatore A
            # Real loss
            pred_real = self.netD_A(real_A)
            loss_D_real = self.criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = self.netG_B2A(real_B) # passa B a B2A - sono le fake A
            pred_fake = self.netD_A(fake_A.detach())
            loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

            # loss globale
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            self.log('loss_D/loss_D_A',loss_D_A)
            return loss_D_A
        
        elif optimizer_idx==2: #discriminatore B
            pred_real = self.netD_B(real_B)
            loss_D_real = self.criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = self.netG_A2B(real_A) # passa A ad A2B - sono le fake B
            pred_fake = self.netD_B(fake_B.detach())
            loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

            # loss globale
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            self.log('loss_D/loss_D_B', loss_D_B)
            return loss_D_B
    
    # Definiamo il dataloader di training
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(int(self.hparams.image_size*1.12), Image.BICUBIC), #ridimensioniamo a una dimensione più grande di quella di input
            transforms.RandomCrop(self.hparams.image_size), #random crop alla dimensione di input
            transforms.RandomHorizontalFlip(), #random flip orizzontale
            transforms.ToTensor(), #trasformiamo in tensore
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) #applichiamo la normalizzazione
        ])
        
        dataloader = DataLoader(ImageDataset(self.hparams.data_root, transform=transform), 
                        batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.n_cpu)
        return dataloader
    
    
from pytorch_lightning.loggers import TensorBoardLogger

cycle_gan = CycleGAN(data_root='grumpifycat')
logger = TensorBoardLogger("tb_logs", name="grumpifycat_cyclegan")
trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=200)

trainer.fit(cycle_gan)
