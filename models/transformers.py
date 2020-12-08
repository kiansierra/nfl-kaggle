import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.model_parts import *

# class TransformerModule(pl.LightningModule):
#     def __init__(self, lr = 1e-3):  
#         super(TransformerNet, self).__init__()
#         self.save_hyperparameters()
#         self.lr = lr
#         self.example_input_array = torch.ones(size=(2,512,512,3))
#         self.loss = torch.nn.MSELoss()
#     def training_step(self, batch, batch_idx):
#         x, _  = batch
#         y = self(x)
#         loss = self.loss(x,y)
#         self.log('loss', loss)
#         if batch_idx % 200 == 0:
#             self.logger.experiment.add_images(f'image', x, dataformats ='NCHW', global_step=self.global_step)
#             self.logger.experiment.add_images(f'transferred', y, dataformats ='NCHW', global_step=self.global_step)
#         return loss
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         # lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3),
#         #             'interval': 'step', 'monitor':'train_loss'}
#         schedule_fun = lambda epoch: 0.99**epoch
#         lr_scheduler = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [schedule_fun]),
#                     'interval': 'step','frequency':50, 'monitor':'train_loss'}
#         return [optimizer], [lr_scheduler]

class TransformerNet(pl.LightningModule):
    def __init__(self, start_outputs=32, num_upsamples =2, num_resblocks=5, lr=1e-3):    
        super(TransformerNet, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.example_input_array = torch.ones(size=(2,3,512,512))
        self.loss = nn.MSELoss()
        self.conv_blocks = num_upsamples
        self.num_resblocks = num_resblocks
        # Initial convolution layers
        channels = start_outputs 
        self.init_conv = ConvLayer(3, channels, kernel_size=9, stride=1)
        for num in range(num_upsamples):
            setattr(self, f'conv{num}', ConvLayer(channels, channels*2, kernel_size=3, stride=2))
            channels*=2
        # ResBlocks
        for num in range(num_resblocks):
            setattr(self, f'res{num}', ResidualBlock(channels))
        # Upsampling Layers
        for num in range(num_upsamples):
            setattr(self, f'upconv{num}', UpsampleConvLayer(channels, channels//2, kernel_size=3, stride=1, upsample=2))
            channels = channels//2
        self.final_deconv = ConvLayer(channels, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.init_conv(x)
        for num in range(self.conv_blocks):
            y = self.relu(getattr(self, f'conv{num}')(y))
        for num in range(self.num_resblocks):
            y = getattr(self, f'res{num}')(y)
        for num in range(self.conv_blocks):
            y = self.relu(getattr(self, f'upconv{num}')(y))
        y = self.final_deconv(y)
        return y
    def training_step(self, batch, batch_idx):
        x, _  = batch
        y = self(x)
        loss = self.loss(x,y)
        self.log('loss', loss)
        if batch_idx % 200 == 0:
            self.logger.experiment.add_images(f'transformer_og', x, dataformats ='NCHW', global_step=self.global_step)
            self.logger.experiment.add_images(f'transformed', y, dataformats ='NCHW', global_step=self.global_step)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3),
        #             'interval': 'step', 'monitor':'train_loss'}
        schedule_fun = lambda epoch: 0.99**epoch
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [schedule_fun]),
                    'interval': 'step','frequency':50, 'monitor':'train_loss'}
        return [optimizer], [lr_scheduler]
#%%
class UnetTransfomer(pl.LightningModule):
    def __init__(self, num_levels = 4, init_channels=3, output_channels=1, start_channels= 32, lr=1e-3):
        super(UnetTransfomer, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.example_input_array = torch.ones(size=(1,3,720,1280))
        self.unet_down = UnetDownwards(num_levels=num_levels, init_channels=init_channels, first_output=start_channels)
        self.unet_upwards = UnetUpwards(num_levels=num_levels, output_channels=start_channels*2 ,first_output=start_channels)
        self.unet_top = UnetTop(output_channels=output_channels , first_output=start_channels*2)
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, x):
        down_levels = self.unet_down(x)
        output = self.unet_upwards(down_levels)
        output = self.unet_top(output)
        return output
    def training_step(self, batch, batch_idx):
        x, y  = batch
        y_hat = self(x)
        loss = self.loss(y,y_hat)
        self.log('loss', loss)
        if batch_idx % 200 == 0:
            self.logger.experiment.add_images(f'image', x, dataformats ='NCHW', global_step=self.global_step)
            self.logger.experiment.add_images(f'true', x*y.repeat(1,3,1,1), dataformats ='NCHW', global_step=self.global_step)
            self.logger.experiment.add_images(f'predictions', x*y_hat.repeat(1,3,1,1), dataformats ='NCHW', global_step=self.global_step)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y  = batch
        y_hat = self(x)
        loss = self.loss(y,y_hat)
        self.log('loss_val', loss)
        if batch_idx % 200 == 0:
            self.logger.experiment.add_images(f'image', x, dataformats ='NCHW', global_step=self.global_step)
            self.logger.experiment.add_images(f'true', x*y.repeat(1,3,1,1), dataformats ='NCHW', global_step=self.global_step)
            self.logger.experiment.add_images(f'predictions', x*y_hat.repeat(1,3,1,1), dataformats ='NCHW', global_step=self.global_step)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3),
        #             'interval': 'step', 'monitor':'train_loss'}
        schedule_fun = lambda epoch: 0.99**epoch
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [schedule_fun]),
                    'interval': 'step','frequency':50, 'monitor':'train_loss'}
        return [optimizer], [lr_scheduler]

#%%
class WnetTransfomer(pl.LightningModule):
    def __init__(self, num_levels = 4, init_channels=3, start_channels= 64, lr=1e-3):
        super(WnetTransfomer, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_levels = num_levels
        self.example_input_array = torch.ones(size=(2,3,512,512))
        self.initial_dc = DoubleConv(in_channels=init_channels, out_channels=start_channels)
        for num in range(self.num_levels):
            setattr(self, f"up-down-resblock_{num}", UpDownResBlock(start_channels))
        self.final_dc = DoubleConv(in_channels=start_channels, out_channels=init_channels)
        self.loss = nn.MSELoss()
    def forward(self, x):
        output = self.initial_dc(x)
        for num in range(self.num_levels):
            getattr(self, f"up-down-resblock_{num}")(output)
        output = self.final_dc(output)
        return output
    def training_step(self, batch, batch_idx):
        x, _  = batch
        y = self(x)
        loss = self.loss(x,y)
        self.log('loss', loss)
        if batch_idx % 200 == 0:
            self.logger.experiment.add_images(f'image', x, dataformats ='NCHW', global_step=self.global_step)
            self.logger.experiment.add_images(f'transferred', y, dataformats ='NCHW', global_step=self.global_step)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3),
        #             'interval': 'step', 'monitor':'train_loss'}
        schedule_fun = lambda epoch: 0.99**epoch
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [schedule_fun]),
                    'interval': 'step','frequency':50, 'monitor':'train_loss'}
        return [optimizer], [lr_scheduler]

# class TransformerNet(pl.LightningModule):
#     def __init__(self, lr = 1e-3):
        
#         super(TransformerNet, self).__init__()
#         self.save_hyperparameters()
#         self.lr = lr
#         # Initial convolution layers
#         self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
#         self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
#         self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
#         self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
#         self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
#         self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
#         # Residual layers
#         self.res1 = ResidualBlock(128)
#         self.res2 = ResidualBlock(128)
#         self.res3 = ResidualBlock(128)
#         self.res4 = ResidualBlock(128)
#         self.res5 = ResidualBlock(128)
#         # Upsampling Layers
#         self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
#         self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
#         self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
#         self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
#         self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
#         # Non-linearities
#         self.relu = torch.nn.ReLU()
#         self.loss = torch.nn.MSELoss()

#     def forward(self, X):
#         y = self.relu(self.in1(self.conv1(X)))
#         y = self.relu(self.in2(self.conv2(y)))
#         y = self.relu(self.in3(self.conv3(y)))
#         y = self.res1(y)
#         y = self.res2(y)
#         y = self.res3(y)
#         y = self.res4(y)
#         y = self.res5(y)
#         y = self.relu(self.in4(self.deconv1(y)))
#         y = self.relu(self.in5(self.deconv2(y)))
#         y = self.deconv3(y)
#         return y
    
#     def training_step(self, batch, batch_idx):
#         x, _  = batch
#         y = self(x)
#         loss = self.loss(x,y)
#         self.log('loss', loss)
#         if batch_idx % 200 == 0:
#             self.logger.experiment.add_images(f'image', x, dataformats ='NCHW', global_step=self.global_step)
#             self.logger.experiment.add_images(f'transferred', y, dataformats ='NCHW', global_step=self.global_step)
#         return loss


#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         # lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3),
#         #             'interval': 'step', 'monitor':'train_loss'}
#         schedule_fun = lambda epoch: 0.99**epoch
#         lr_scheduler = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [schedule_fun]),
#                     'interval': 'step','frequency':50, 'monitor':'train_loss'}
#         return [optimizer], [lr_scheduler]


# %%
# class UnetTransfomer(pl.LightningModule):
#     def __init__(self, num_levels = 4, init_channels=3, start_channels= 32, lr=1e-4 ):
#         super().__init__()
#         self.save_hyperparameters()
#         self.unet_down = UnetDownwards(num_levels=num_levels, init_channels=init_channels, first_output=start_channels)
#         self.unet_upwards = UnetUpwards(num_levels=num_levels, output_channels=start_channels, first_output=start_channels)
#         self.unet_top = UnetTop(output_channels=init_channels, first_output=start_channels)
#         self.loss = nn.MSELoss()
#         self.example_input_array = torch.ones(size=(2,512,512,3))
#         self.lr = lr
#     def forward(self, x):
#         down_levels = self.unet_down(x)
#         output = self.unet_upwards(down_levels)
#         output = self.unet_top(output)
#         return output
#     def training_step(self, batch, batch_idx):
#         x, _ = batch
#         y = self(x)
#         loss = self.loss(x, y)
#         self.log('train_loss', loss)
#         self.log('hp_metric', loss)
#         if batch_idx % 200 == 0:
#             self.logger.experiment.add_images(f'image_{batch_idx}', x, dataformats ='NCHW', global_step=self.current_epoch)
#             self.logger.experiment.add_images(f'transformed_{batch_idx}', y, dataformats ='NCHW', global_step=self.current_epoch)
#         return loss
#     def validation_step(self, batch, batch_idx):
#         x, _ = batch
#         y = self(x)
#         val_loss = self.loss(x, y)
#         self.log('val_loss', val_loss)
#         if batch_idx % 200 == 0:
#             self.logger.experiment.add_images(f'image_{batch_idx}', x, dataformats ='NCHW', global_step=self.current_epoch)
#             self.logger.experiment.add_images(f'transformed_{batch_idx}', y, dataformats ='NCHW', global_step=self.current_epoch)
#         return val_loss
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

#         schedule_fun = lambda epoch: 0.98**epoch
#         lr_scheduler = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [schedule_fun]),
#                     'interval': 'step','frequency':100 ,'monitor':'train_loss'}
#         return [optimizer], [lr_scheduler]



