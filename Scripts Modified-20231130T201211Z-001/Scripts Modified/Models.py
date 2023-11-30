import matplotlib.pyplot as plt
import torch.nn.functional as F
import utilities as utils
import torch.nn as nn
import numpy as np
import torch

from statistics import mean


class CustomModelWrapper():
    def __init__(self, model, loss_fn=F.mse_loss, loss_coeff=1.0, reg_fn=None, reg_coeff=None):
        self._model = model
        self._loss_fn = loss_fn
        self._loss_coeff = float(loss_coeff)
        self._reg_fn = reg_fn
        self._reg_coeff = float(reg_coeff)
        self._trained = 0
        
        self.freeze_encoder = False
        
        # Individual loss histories: recronstruction, structural and combined
        self.t_history = {'rec': [], 'struc': [], 'loss': []}
        self.v_history = {'rec': [], 'struc': [], 'loss': []}
        
        # reconstruction and latent representation history for train and validation sets
        self.X_history, self.Xl_history = [], []
        self.Xv_history, self.Xvl_history = [], []
        
    def train(self, optim, X_train, X_geod, epochs=64, verbose=True, X_val=None, X_geod_val=None, batch=32, p=2):
        # number of training samples, batch size and steps per epoch
        samples, batch_size = X_train.shape[0], batch
        steps_per_epoch = int(samples / batch_size)
        
        print(f'\nModel has been trained for a total of {self._trained} epochs.')
        
        epoch_s = self._trained
        self._trained += epochs
        epoch_e = self._trained
                
        for epoch in range(epoch_s, epoch_e):
            # get shuffled indexes for each epoch
            idxs = torch.randperm(samples)
            rloss, sloss, mloss = [], [], []
            
            for step in range(steps_per_epoch):
                # get indexes current batch of training samples
                batch_idxs = idxs[batch_size*step: batch_size*step + batch_size]
            
                # get current batch of training samples
                X_batch = X_train[batch_idxs]
                X_geod_batch = X_geod[batch_idxs][:, batch_idxs]
        
                # forward and backward pass
                rec_loss, struc_loss, loss = self._step(optim, X_batch, X_geod_batch, p)
            
                # append batch loss
                rloss.append(rec_loss)
                sloss.append(struc_loss)
                mloss.append(loss)
                
                if verbose:
                    print(f"""Epoch: {epoch+1: 000d} \t Step: {step+1:00d}/{steps_per_epoch:00d} \t \
Loss: {mean(mloss): .06f} \t MSE: {mean(rloss): .06f} \t \
Regularizer: {mean(sloss): .06f}""", end="\r")
                    
            self._log(rloss, sloss, mloss, X_train, X_val, X_geod_val, p)
            if verbose:
                print()
        
    def _log(self, rloss, sloss, mloss, X_train, X_val, X_geod_val, p):
        # store training losses history 
        self.t_history['rec'].append(mean(rloss))
        self.t_history['struc'].append(mean(sloss))
        self.t_history['loss'].append(mean(mloss))
    
        # get training reconstruction and latent representation after epoch
        x, x_l = self._model(X_train)
    
        # store training reconstruction and latent representation after epoch
        self.X_history.append(x.cpu().detach().numpy())
        self.Xl_history.append(x_l.cpu().detach().numpy())    
    
        if X_val is not None:
            with torch.no_grad():
                # get validation reconstruction and latent representation
                xv, xv_l = self._model(X_val)
    
            # calculate reconstruction, structural and overall losses on validation set
            rec_loss = self._loss_fn(xv, X_val, reduction='mean')
            struc_loss = self._reg_fn(X_geod_val, xv_l, p)
            loss = self._loss_coeff * rec_loss + self._reg_coeff * struc_loss
    
            # store validation losses history
            self.v_history['rec'].append(rec_loss.cpu().item())
            self.v_history['struc'].append(struc_loss.cpu().item())
            self.v_history['loss'].append(loss.cpu().item())
            
            # store validation reconstruction and latent representation after epoch
            self.Xv_history.append(xv.cpu().detach().numpy())
            self.Xvl_history.append(xv_l.cpu().detach().numpy())
    
    def _step(self, optim, X_batch, X_geod_batch, p):
        # x is reconstruction, x_l is latent representation
        x, x_l = self._model.forward(X_batch, freeze_encoder=self.freeze_encoder)
        
         
        # calculate reconstruction, structural and overall losses
        rec_loss = self._loss_fn(x, X_batch, reduction='mean')
        struc_loss = self._reg_fn(X_geod_batch, x_l, p)
        loss = self._loss_coeff * rec_loss + self._reg_coeff * struc_loss

        # calculate gradients
        loss.backward()
        
        # apply backprop with optimizer and set gradients to zero
        optim.step()
        optim.zero_grad()

        return rec_loss.cpu().item(), struc_loss.cpu().item(), loss.cpu().item()
    
    def __call__(self, X):
        return self._model(X)
    
    def save_model(self, save_path):
        torch.save(self._model.state_dict(), save_path)
        
    def save_logs(self, name):
        # Save Train History
        X_history, Xl_history = np.array(self.X_history), np.array(self.Xl_history)
        np.save(f"{name}_train_X", X_history)
        np.save(f"{name}_train_Xl", Xl_history)

        # Save Val History
        Xv_history, Xvl_history = np.array(self.Xv_history), np.array(self.Xvl_history)
        np.save(f"{name}_val_X", Xv_history)
        np.save(f"{name}_val_Xl", Xvl_history)
    
    def save_plots(self, name, save, show):
        fig = utils.train_plots(self.t_history, self._reg_coeff, epochs=self._trained, start=0)
        
        if save:
            plt.savefig(f"{name}_train_LR1.png")
        plt.close(fig)
        
        if self._trained > 128:
            fig = utils.train_plots(self.t_history, self._reg_coeff, epochs=self._trained, start=128)
            if save:
                plt.savefig(f"{name}_train_LR2.png")
            plt.close(fig)

        fig = utils.train_plots(self.v_history, self._reg_coeff, epochs=self._trained, start=0)
        if save:
            plt.savefig(f"{name}_val_LR1.png")
        plt.close(fig)
        
        if self._trained > 128:
            fig = utils.train_plots(self.v_history, self._reg_coeff, epochs=self._trained, start=128)
            if save:
                plt.savefig(f"{name}_val_LR2.png")
            plt.close(fig)


class DeepAutoEncoder (nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.device = None
        self._layers = layers
        
    def pretrain(self, X, epochs=None, lr=1e-3, device=None, verbose=False, train=True):
        encoder, decoder = None, None
        
        if epochs is None:
            epochs = [16 for i in range(1, len(self._layers))]
        
        for i in range(1, len(self._layers)):
            if i == 1:
                print('Initializing Autoencoder...')
                encoder, decoder = EncoderBase(self._layers[i-1], self._layers[i]), DecoderBase(self._layers[i], self._layers[i-1])
            else:
                if train:
                    print('\nFreezing Layers...')
                freeze_layers(encoder), freeze_layers(decoder)
                print('Adding new layers to increase Autoencoder Depth...')
                if i == len(self._layers)-1:
                    encoder, decoder = EncoderSigmoid(encoder, self._layers[i-1], self._layers[i]), DecoderReLU(decoder, self._layers[i], self._layers[i-1])
                else:
                    encoder, decoder = EncoderReLU(encoder, self._layers[i-1], self._layers[i]), DecoderReLU(decoder, self._layers[i], self._layers[i-1])

            if device is not None and train: 
                self.device = device
                encoder.to(device), decoder.to(device)
            
            if train:
                optimizer = torch.optim.RMSprop(
                    [{'params': encoder.parameters(), 'lr': lr},
                     {'params': decoder.parameters(), 'lr': lr}],
                    weight_decay=1e-5
                )
                    
                for epoch in range(1, epochs[i-1]+1):
                    self._epoch(encoder, decoder, optimizer, X, epoch, verbose=verbose)
                
        self.encoder, self.decoder = encoder, decoder
        if train:
            print('\nPretraining Complete')
        else:
            print('\nInitialization complete')
        print('Unfreeze layers to fine tune the weights')
        
    def _epoch(self, encoder, decoder, optimizer, X, epoch, batch_size=16, verbose=False):
        samples = X.shape[0]
        steps = int(samples / batch_size)
        idxs = torch.randperm(samples)
        epoch_mse = []
        for step in range(steps):
            # get current batch of samples
            batch_idxs = idxs[batch_size*step: batch_size*step + batch_size]
            X_batch = X[batch_idxs, :]
            
            mse = self._step(encoder, decoder, optimizer, X_batch)
            
            epoch_mse.append(mse)
            if verbose:
                print(f'Epoch: {epoch: 04d} -- Step: {step+1:03d}/{steps:03d} -- MSE: {mean(epoch_mse):.05f}', end='\r')
        if verbose:
            print()
            
        
    def _step(self, encoder, decoder, optimizer, X_batch):
        
        # forward propagation
        X_latent = encoder(X_batch+torch.rand(X_batch.shape, device=self.device))
        X_rec = decoder(X_latent)
        
        # calculate loss
        mse = F.mse_loss(X_rec, X_batch, reduction='mean')
        
        # calculate gradients
        mse.backward()
        
        # apply backprop with optimizer and set gradients to zero
        optimizer.step()
        optimizer.zero_grad()

        return mse.cpu().item()
    
    def forward(self, X, freeze_encoder=False,**kwargs):
        # Encoder
        if freeze_encoder:
            with torch.no_grad():
                latent = self.encoder(X)
        else:
            latent = self.encoder(X)
        
        # Decoder
        out = self.decoder(latent)
        
        return out, latent
    
class EncoderBase (nn.Module):
    def __init__(self, layer_in, layer_out):
        super().__init__()
        
        self.linear = nn.Linear(layer_in, layer_out, bias=True)
#         nn.init.kaiming_normal_(self.linear.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.linear.weight, a=0.01, nonlinearity='leaky_relu')        
        self.act = nn.LeakyReLU()
        self.reg = nn.BatchNorm1d(layer_out)
        
    def forward(self, X):
        out = self.linear(X)
        out = self.reg(out)
        out = self.act(out)
        
        return out

class EncoderReLU (nn.Module):
    def __init__(self, encoder, layer_in, layer_out):
        super().__init__()
        
        self.encoder = encoder
        self.linear = nn.Linear(layer_in, layer_out, bias=True)
#         nn.init.kaiming_normal_(self.linear.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.linear.weight, a=0.01, nonlinearity='leaky_relu')
        self.act = nn.LeakyReLU()  
        self.reg = nn.BatchNorm1d(layer_out)
        
    def forward(self, X):
        
        out = self.encoder(X)
        out = self.linear(out)
        
        out = self.reg(out)
        out = self.act(out)
        
        return out
        
class EncoderSigmoid (nn.Module):
    def __init__(self, encoder, layer_in, layer_out):
        super().__init__()
        
        self.encoder = encoder
        
        self.linear = nn.Linear(layer_in, layer_out, bias=True)
#         nn.init.xavier_normal_(self.linear.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        
        self.reg = nn.BatchNorm1d(layer_out)
        self.act = nn.Sigmoid()         
        
    def forward(self, X):
        
        out = self.encoder(X)
        out = self.linear(out)
        out = self.reg(out)
        out = self.act(out)
        
        return out
    
class DecoderBase (nn.Module):
    def __init__(self, layer_in, layer_out):
        super().__init__()
        
        self.linear = nn.Linear(layer_in, layer_out, bias=True)
#         nn.init.xavier_normal_(self.linear.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        self.reg = nn.BatchNorm1d(layer_out)
        self.act = nn.Sigmoid() 
        
    def forward(self, X):
        
        out = self.linear(X)
        out - self.reg(out)
        out = self.act(out)
        
        return out

class DecoderReLU (nn.Module):
    def __init__(self, decoder, layer_in, layer_out):
        super().__init__()
        
        self.decoder = decoder
        
        self.linear = nn.Linear(layer_in, layer_out, bias=True)
#         nn.init.kaiming_normal_(self.linear.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.linear.weight, a=0.01, nonlinearity='leaky_relu')
        self.act = nn.LeakyReLU()  
        self.reg = nn.BatchNorm1d(layer_out)
        
    def forward(self, X):
        
        out = self.linear(X)
        out = self.reg(out)
        out = self.act(out)
        out = self.decoder(out)
        
        return out
    
def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False
        
def unfreeze_layers(model, verbose=False):
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            if verbose:
                print('setting param.require')