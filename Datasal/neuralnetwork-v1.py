# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:15:37 2020

@author: simen
base code taken from https://github.com/bgrimstad/TTK28-Courseware/blob/master/model/flow_model.ipynb
"""
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from math import sqrt

class Net(torch.nn.Module):
    """
    PyTorch offers several ways to construct neural networks.
    Here we choose to implement the network as a Module class.
    This gives us full control over the construction and clarifies our intentions.
    """
    
    def __init__(self, layers):
        """
        Constructor of neural network
        :param layers: list of layer widths. Note that len(layers) = network depth + 1 since we incl. the input layer.
        """
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert len(layers) >= 2, "At least two layers are required (incl. input and output layer)"
        self.layers = layers

        # Fully connected linear layers
        linear_layers = []

        for i in range(len(self.layers) - 1):
            n_in = self.layers[i]
            n_out = self.layers[i+1]
            layer = torch.nn.Linear(n_in, n_out)

            # Initialize weights and biases
            a = 1 if i == 0 else 2
            layer.weight.data = torch.randn((n_out, n_in)) * sqrt(a / n_in)
            layer.bias.data = torch.zeros(n_out)
            
            # Add to list
            linear_layers.append(layer)
        
        # Modules/layers must be registered to enable saving of model
        self.linear_layers = torch.nn.ModuleList(linear_layers)  

        # Non-linearity (e.g. ReLU, ELU, or SELU)
        self.act = torch.nn.ReLU(inplace=False)

    def forward(self, input):
        """
        Forward pass to evaluate network for input values
        :param input: tensor assumed to be of size (batch_size, n_inputs)
        :return: output tensor
        """
        x = input
        for l in self.linear_layers[:-1]:
            x = l(x)
            x = self.act(x)

        output_layer = self.linear_layers[-1]
        return output_layer(x)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str):
        """
        Save model state
        :param path: Path to save model state
        :return: None
        """
        torch.save({
            'model_state_dict': self.state_dict(),
        }, path)

    def load(self, path: str):
        """
        Load model state from file
        :param path: Path to saved model state
        :return: None
        """
        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.load_state_dict(checkpoint['model_state_dict'])
        
def train(
        net: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        lr: float,
        l2_reg: float,
) -> torch.nn.Module:
    """
    Train model using mini-batch SGD
    After each epoch, we evaluate the model on validation data

    :param net: initialized neural network
    :param train_loader: DataLoader containing training set
    :param n_epochs: number of epochs to train
    :param lr: learning rate (default: 0.001)
    :param l2_reg: L2 regularization factor (default: 0)
    :return: torch.nn.Module: trained model.
    """

    # Define loss and optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Train Network
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            # Zero the parameter gradients (from last iteration)
            optimizer.zero_grad()

            # Forward propagation
            outputs = net(inputs)
            
            # Compute cost function
            batch_mse = criterion(outputs, labels)
            
            reg_loss = 0
            for param in net.parameters():
                reg_loss += param.pow(2).sum()

            cost = batch_mse + l2_reg * reg_loss

            # Backward propagation to compute gradient
            cost.backward()
            
            # Update parameters using gradient
            optimizer.step()
        
        # Evaluate model on validation data
        mse_val = 0
        for inputs, labels in val_loader:
            mse_val += torch.sum(torch.pow(labels - net(inputs), 2)).item()
        mse_val /= len(val_loader.dataset)
        print(f'Epoch: {epoch + 1}: Val MSE: {mse_val}')
        
    return net

def main():
    random_seed =13371337  # This seed is also used in the pandas sample() method below
    torch.manual_seed(random_seed)
    df_unfixed = pd.read_excel(r"C:\Users\simen\Desktop\Prosjekt\NNDataSets\Afrika\87DAC810.xlsx", index_col=0)
    for col in df_unfixed:
        df_unfixed[col] = pd.to_numeric(df_unfixed[col], errors='coerce')
        
    df=df_unfixed.interpolate(method='linear',limit_direction='forward')
    df=df.dropna(axis=0)
    print("Data finished interpolating")
    # Test set (this is the period for which we must estimate QTOT)
    test_set = df.iloc[180000:199492]

    # Make a copy of the dataset and remove the test data
    train_val_set = df.copy().drop(test_set.index)

    # Sample validation data without replacement (10%)
    val_set = train_val_set.sample(frac=0.1, replace=False, random_state=random_seed)

    # The remaining data is used for training (90%)
    train_set = train_val_set.copy().drop(val_set.index)

    # Check that the numbers add up
    n_points = len(train_set) + len(val_set) + len(test_set)
    print(f'{len(df)} = {len(train_set)} + {len(val_set)} + {len(test_set)} = {n_points}')
    INPUT_COLS = ['Africa.571_TT_114',	'Africa.571_TT_124',	'Africa.601_XI_20114',	'Africa.871_CB_TR1_KW',	'Africa.871_XI_10151',	'Africa.871_XI_10207','Africa.871_XI_10259','Africa.871_XI_10302','Africa.871_XI_10303','Africa.871_XI_10304','Africa.871_XI_10306','Africa.871_XI_10312','Africa.871_XI_10315','Africa.871_XI_10363','Africa.871_XI_10409']
    OUTPUT_COLS = ['Africa.404_XI_11016']
    
    # Get input and output tensors and convert them to torch tensors
    x_train = torch.from_numpy(train_set[INPUT_COLS].values).to(torch.float)
    y_train = torch.from_numpy(train_set[OUTPUT_COLS].values).to(torch.float)

    x_val = torch.from_numpy(val_set[INPUT_COLS].values).to(torch.float)
    y_val = torch.from_numpy(val_set[OUTPUT_COLS].values).to(torch.float)

    # Create dataset loaders
    # Here we specify the batch size and if the data should be shuffled
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_set), shuffle=False)
    
    layers = [len(INPUT_COLS), 300, 300, len(OUTPUT_COLS)]
    net = Net(layers)

    print(f'Layers: {layers}')
    print(f'Number of model parameters: {net.get_num_parameters()}')
    
    n_epochs = 100
    lr = 0.002
    l2_reg = 0.0015  # 10
    net = train(net, train_loader, val_loader, n_epochs, lr, l2_reg)
    
    # Get input and output as torch tensors
    x_test = torch.from_numpy(test_set[INPUT_COLS].values).to(torch.float)
    y_test = torch.from_numpy(test_set[OUTPUT_COLS].values).to(torch.float)

    # Make prediction
    pred_test = net(x_test)

    # Compute MSE, MAE and MAPE on test data
    print('Error on test data')
    
    mse_test = torch.mean(torch.pow(pred_test - y_test, 2))
    print(f'MSE: {mse_test.item()}')
    
    mae_test = torch.mean(torch.abs(pred_test - y_test))
    print(f'MAE: {mae_test.item()}')
    
    mape_test = 100*torch.mean(torch.abs(torch.div(pred_test - y_test, y_test)))
    print(f'MAPE: {mape_test.item()} %')