import numpy as np
import torch
import torch.nn as nn
from .tools import to_numpy, to_tensor
 
    
class FSM_Regressor:

    def __init__(self, model, device):
        '''
        class for training a neural network to fit a regressor model.

        Attributes:
        model (torch.nn.module): the neural network model to be trained
        device (torch device): the device on which the data will be stored
        '''
        self.model = model
        self.device = device

    def train(self, input,output, lr=1e-3, epochs=1000,verbose=False):
        '''
        train the regressor model

        Parameters:
        input (numpy array): the input to the model, shape (*,D)
        output (numpy array): the output of the model, shape (*,D)
        lr (float): learning rate for the optimizer
        epochs (int): number of epochs to train the model
        verbose (bool): whether to print the loss every 100 iterations
        Returns:
        None
        '''
        input = to_tensor(input)
        output = to_tensor(output)

        input = input.to(self.device)
        output = output.to(self.device)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for itr in range(epochs):
            
            self.optimizer.zero_grad()
            pred = self.model(input)
            loss = torch.mean((output - pred)**2)
            loss.backward()
            self.optimizer.step()
            epoch_loss = loss.item()

            if verbose:
                if itr % 100 == 0:
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, epoch_loss))
    
    def predict(self, input):
        '''
        predict the output of the model

        Parameters:
        input (numpy array): the input to the model, shape (*,D)

        Returns:
        numpy array: the output of the model, shape (*,D)
        '''
        pred = self.model(to_tensor(input).to(self.device))

        return to_numpy(pred)
    
    
    def cond_predict(self, *input_parts):
        '''
        Predict the output of the model given multiple parts of the input.

        Parameters:
        *input_parts (numpy arrays): Variable number of input parts.
            Each part must be a NumPy array of shape (N, d_i) or (d_i,).
            If a part has shape (d_i,), it is reshaped to (1, d_i).
            Parts with batch size 1 will be broadcast to match the largest batch size.

        Returns:
        numpy array: The output of the model, shape (N, output_dim)
        '''

        # Ensure all parts are 2D arrays
        parts = []
        for part in input_parts:
            part = np.asarray(part)
            if part.ndim == 1:
                part = part.reshape(1, -1)
            parts.append(part)

        # Determine target batch size
        batch_sizes = [p.shape[0] for p in parts]
        max_batch_size = max(batch_sizes)

        # Broadcast all parts to the same batch size
        broadcasted_parts = []
        for p, b in zip(parts, batch_sizes):
            if b == 1 and max_batch_size > 1:
                p = np.repeat(p, max_batch_size, axis=0)
            elif b != max_batch_size:
                assert b == max_batch_size, (
                    f"Incompatible batch sizes: got {b} and expected {max_batch_size}"
                )
            broadcasted_parts.append(p)

        # Concatenate along feature axis
        full_input = np.concatenate(broadcasted_parts, axis=1)

        # Predict with model
        pred = self.model(to_tensor(full_input).to(self.device))

        return to_numpy(pred)