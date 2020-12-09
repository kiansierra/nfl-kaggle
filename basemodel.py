#%%
import pytorch_lightning as pl
import torch 
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from models.model_parts import *
import wandb
#%%
class NFLBaseModel(pl.LightningModule):
    def __init__(self, lr=1e-3, opt_freq = 300, dec_rate = 0.98, opt_upsteps = 3, weight_decay=0 ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr =lr
        self.opt_freq = opt_freq
        self.dec_rate = dec_rate
        self.opt_upsteps = opt_upsteps
        self.wd = weight_decay 
        self.example_input_array = torch.ones(size = (1,2,720,1280))
        self.loss = nn.BCEWithLogitsLoss()
        # self.train_accuracy = pl.metrics.Accuracy()
        # self.val_accuracy = pl.metrics.Accuracy()
        #self.train_fscore = pl.metrics.FBeta(num_classes=5)
    def log_step(self, phase, loss, x, y, y_pred, batch_idx):
        self.log(f'{phase}_loss', loss)
        #self.log(f'{phase}_acc_step', acc, prog_bar=True)
        #self.log('train_fscore', self.train_fscore(y_hat, y))
        if batch_idx % 200 == 0 and isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_images(f'{phase}_image', x, dataformats ='NCHW', global_step=self.global_step)
        elif batch_idx % 200 == 0 and isinstance(self.logger, WandbLogger):
            batch_imgs = x.data.permute(0,2,3,1).cpu().numpy()
            true_labels = y.data.permute(0,2,3,1).cpu().numpy()
            pred_labels = y_pred.data.permute(0,2,3,1).cpu().numpy()
            for num, (img, true_label, pred_label) in enumerate(zip(batch_imgs, true_labels, pred_labels)):
                self.logger.experiment.log({f'{phase}_image_{num}':[wandb.Image(img, caption=f'Image'), wandb.Image(img*(true_label + 1)/2, caption=f'True'),
                 wandb.Image(img*(pred_label + 1)/2, caption=f'Pred')]})
            # self.logger.experiment.log({f'{phase}_image':[wandb.Image(img*(label + 1)/2, caption=f'True') for img, label in zip(batch_imgs, true_labels)]})
            # self.logger.experiment.log({f'{phase}_image':[wandb.Image(img*(label + 1)/2, caption=f'Pred') for img, label in zip(batch_imgs, pred_labels)]})
            # for name, param in self.named_parameters():
            #     self.logger.experiment.log({name:[wandb.Image()]})

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        #acc = self.train_accuracy(y_hat, y)
        self.log_step('train', loss,x, y, y_hat, batch_idx)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        #acc = self.val_accuracy(y_hat, y)
        self.log_step('val', loss,x, y, y_hat, batch_idx)
        return loss
    def test_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        return {'label':torch.argmax(y_hat,axis=1)}
    # def training_epoch_end(self, outs):
    #     # log epoch metric
    #     self.log('train_acc_epoch', self.train_accuracy.compute())
    #     #self.log('train_fscore_epoch', self.train_fscore.compute())
    # def validation_epoch_end(self, outs):
    #     # log epoch metric
    #     self.log('val_acc_epoch', self.val_accuracy.compute())
    def test_epoch_end(self, outputs):
        results = torch.cat([elem['label'] for elem in outputs])
        self.test_results = results
        return results
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # schedule_fun = lambda step: (self.dec_rate**step)*(1+ step%self.opt_upsteps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [schedule_fun])
        lr_scheduler = {'scheduler':scheduler, 'interval': 'step', 'frequency':self.opt_freq,  'monitor':'val_loss'}
        return [optimizer], [lr_scheduler]
class UnetTransfomer(NFLBaseModel):
    def __init__(self, num_levels = 4, start_channels= 32,  init_channels=3, output_channels=1, **kwargs):
        super(UnetTransfomer, self).__init__(**kwargs)
        self.save_hyperparameters()
        self.example_input_array = torch.ones(size=(1,3,720,1280))
        self.unet_down = UnetDownwards(num_levels=num_levels, init_channels=init_channels, first_output=start_channels)
        self.unet_upwards = UnetUpwards(num_levels=num_levels, output_channels=start_channels*2 ,first_output=start_channels)
        self.unet_top = UnetTop(output_channels=output_channels , first_output=start_channels*2)
    def forward(self, x):
        down_levels = self.unet_down(x)
        output = self.unet_upwards(down_levels)
        output = self.unet_top(output)
        return output
#%%
class WnetTransfomer(pl.LightningModule):
    def __init__(self, num_levels = 4, start_channels= 64, init_channels=3, output_channels=1):
        super(WnetTransfomer, self).__init__()
        self.save_hyperparameters()
        self.num_levels = num_levels
        self.example_input_array = torch.ones(size=(2,3,512,512))
        self.initial_dc = DoubleConv(in_channels=init_channels, out_channels=start_channels)
        for num in range(self.num_levels):
            setattr(self, f"up-down-resblock_{num}", UpDownResBlock(start_channels))
        self.final_dc = DoubleConv(in_channels=start_channels, out_channels=output_channels)
    def forward(self, x):
        output = self.initial_dc(x)
        for num in range(self.num_levels):
            getattr(self, f"up-down-resblock_{num}")(output)
        output = self.final_dc(output)
        return output