from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np 

def make_classification_figure(image_tensor, y_true, y_pred):    
    batch_tensor = image_tensor.permute((0,2,3,1)).cpu().numpy()
    batch_tensor = batch_tensor/batch_tensor.max()
    batch_tensor = np.clip(batch_tensor, a_min=0.001, a_max=0.999)
    num_images = batch_tensor.shape[0]
    preds = torch.argmax(y_pred, dim=1).cpu().numpy()
    fig, axs = plt.subplots(ncols=num_images, figsize = (10, num_images*10 ))
    if num_images > 1:
        for num in range(num_images):
            axs[num].imshow(batch_tensor[num])
            axs[num].set_title(f"True: {y_true[num]} -- Pred: {preds[num]}")
    else:
        axs.imshow(batch_tensor[0])
        axs.set_title(f"True: {y_true[0]} -- Pred: {preds[0]}")
    return fig

class ClassifierTop(torch.nn.Module):
    def __init__(self, out_features=16):
        super(ClassifierTop, self).__init__()
        self.linear_1 = nn.Linear(in_features=512*7*7, out_features=500)
        self.linear_2 = nn.Linear(in_features=500, out_features=out_features)
        self.relu = nn.ReLU(inplace=True)
        # self.output_act = nn.Softmax(dim=-1)
    def forward(self,x):
        out = torch.flatten(x, start_dim=1)
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.linear_2(out)
        # out = self.output_act(out)
        return out

class SlicerClassifier(pl.LightningModule):
    def __init__(self,num_outputs=16, classifier_model = models.vgg16(pretrained=True), cutoffs =[4,9,16,23], requires_grad=True, lr=1e-4):
        super(SlicerClassifier, self).__init__()
        self.lr = lr
        classifier_features = classifier_model.features
        cutoffs = cutoffs + [len(classifier_features)]
        for num, elem in enumerate(cutoffs):
            setattr(self, f"slice_{num}", torch.nn.Sequential()) 
            start = 0 if num == 0 else cutoffs[num-1] 
            for x in range(start, elem):
                getattr(self, f"slice_{num}").add_module(str(x), classifier_features[x])
        getattr(self, f"slice_{len(cutoffs)-1}").add_module('avgpool', classifier_model.avgpool)
        getattr(self, f"slice_{len(cutoffs)-1}").add_module('top', ClassifierTop(num_outputs))
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.slices = len(cutoffs)
        self.loss = nn.CrossEntropyLoss()
        self.example_input_array = torch.ones(size=(2, 3, 512, 512))
    def forward(self, X):
        h = X
        outputs = []
        for num in range(self.slices):
            h = getattr(self, f'slice_{num}')(h)
            outputs.append(h)
        list_name = [f"out_{x}" for x in range(len(outputs))]
        vgg_outputs = namedtuple("VggOutputs", list_name)
        out = vgg_outputs(*outputs)
        return out
    def training_step(self, batch, batch_idx):
        x, y =batch
        y_out = self(x)
        loss = self.loss(y_out[-1], y)
        self.log('train_loss', loss)
        if batch_idx % 200 == 0:
            fig = make_classification_figure(x, y, y_out[-1])
            self.logger.experiment.add_figure(f'classification', fig, global_step=self.global_step)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y =batch
        y_out = self(x)
        loss = self.loss(y_out[-1], y)
        self.log('val_loss', loss)
        if batch_idx % 200 == 0:
            fig = make_classification_figure(x, y, y_out[-1])
            self.logger.experiment.add_figure(f'classification_val', fig, global_step=self.global_step)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3),
        #             'interval': 'step', 'monitor':'train_loss'}
        schedule_fun = lambda step: 0.99**step
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [schedule_fun]),
                    'interval': 'step','frequency': 20, 'monitor':'val_loss'}
        return [optimizer], [lr_scheduler]


class Vgg16(torch.nn.Module):
    def __init__(self, cutoffs =[4,9,16,23], requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        for num, elem in enumerate(cutoffs):
            setattr(self, f"slice_{num}", torch.nn.Sequential()) 
            start = 0 if num == 0 else cutoffs[num-1] 
            for x in range(start, elem):
                getattr(self, f"slice_{num}").add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, X):
        h = X
        outputs = []
        for num in range(4):
            h = getattr(self, f'slice_{num}')(h)
            outputs.append(h)
        list_name = [f"out_{x}" for x in range(4)]
        vgg_outputs = namedtuple("VggOutputs", list_name)
        out = vgg_outputs(*outputs)
        return out

class Vgg19(torch.nn.Module):
    def __init__(self, cutoffs =[4,9,18,27], requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        for num, elem in enumerate(cutoffs):
            setattr(self, f"slice_{num}", torch.nn.Sequential())
            start = 0 if num == 0 else cutoffs[num-1] 
            for x in range(start, elem):
                getattr(self, f"slice_{num}").add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, X):
        h = X
        outputs = []
        for num in range(4):
            h = getattr(self, f'slice_{num}')(h)
            outputs.append(h)
        list_name = [f"out_{x}" for x in range(4)]
        vgg_outputs = namedtuple("VggOutputs", list_name)
        out = vgg_outputs(*outputs)
        return out