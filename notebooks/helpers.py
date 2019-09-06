import time
import torch
from torch.utils.data.dataset import Dataset
import numpy as np

# this converts a multi-label column (1D tensor) into one-hot vectors (2D tensor)
def one_hot(x, class_count):
    return torch.eye(class_count)[x,:]

# see examples at https://github.com/utkuozbulak/pytorch-custom-dataset-examples
class BasicLabelledDataset(Dataset):
    def __init__(self, inputs_array, targets_array):
        self.inputs_array = inputs_array # numpy
        self.inputs_tensor = torch.tensor(self.inputs_array).float()
        self.targets_array = targets_array # numpy
        self.class_count = len(np.unique(targets_array))
        if self.class_count > 2:
            self.targets_tensor = one_hot(self.targets_array, self.class_count)

    def __getitem__(self, index):
        return (self.inputs_tensor[index], self.targets_tensor[index])

    def __len__(self):
        return len(self.targets_array)

class BasicNeuralNet(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(BasicNeuralNet, self).__init__()
        self.x_to_z = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.z_to_h = torch.nn.Sigmoid()
        self.h_to_s = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.s_to_y = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        z = self.x_to_z(x)  # Linear
        h = self.z_to_h(z)  # Sigmoid
        s = self.h_to_s(h)  # Linear
        y = self.s_to_y(s)  # SoftMax
        return y

class BasicModelTrainer():
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.criterion_record = []
        self.verbose = kwargs.get("verbose", False)

    def fit(self,
            tuple_dataset,
            epochs,
            batch_size,
            verbose=False):
        
        # this will batch the data for you (given you have a DataSet for it)
        training_loader = torch.utils.data.DataLoader(
            dataset=tuple_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # for timing
        total_data_to_fit = epochs * len(tuple_dataset)
        start_time = time.time()
        total_data_processed = 0
        last_running_loss = 0.0

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0  # will store loss for the entire dataset

            for i, data in enumerate(training_loader, 0):  # iterate on batches
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # get the inputs; data is a list of [inputs, labels]
                inputs, targets = data

                # forward prop based on inputs
                outputs = self.model(inputs)  

                # computing loss (NOTE: this is a tensor as well)
                loss = self.criterion(outputs, targets)

                # backward prop based on expected value
                loss.backward()

                # apply backward prop on parameters
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                self.criterion_record.append((epoch, i, loss.item()))
                
                # print timing
                if self.verbose:
                    total_data_processed += len(targets)
                    time_per_data = ((time.time() - start_time)) / float(total_data_processed)
                    eta = int((total_data_to_fit - total_data_processed) * time_per_data)
                    print("[epoch={}]\t epoch_loss={:2f}\t ETA: {} secs (data={}/{}, elapsed={})".format(
                        epoch,
                        last_running_loss if epoch > 0 else (running_loss / (i+1)),
                        "{:2d}:{:2d}:{:2d}".format((eta//3600), (eta % 3600)//60, eta%60),
                        total_data_processed,
                        total_data_to_fit,
                        int(time.time()-start_time)
                    ), end='\r', flush=True)
            last_running_loss = running_loss
         
        return self.model, running_loss
    
    def test_accuracy(self, tuple_dataset):
        # batch the testing data as well
        testing_loader = torch.utils.data.DataLoader(
            dataset=tuple_dataset,
            batch_size=10,
            shuffle=True
        )

        correct = 0
        total = 0

        with torch.no_grad():  # deactivate autograd during testing
            for data in testing_loader:  # iterate on batches
                # get testing data batch
                inputs, targets = data

                # apply the NN
                outputs = self.model(inputs)                 # compute output class tensor
                predicted = torch.argmax(outputs, dim=1)  # get argmax of P(y_hat|x)
                actual = torch.argmax(targets, dim=1)     # get y

                # compute score
                total += targets.size(0)
                correct += (predicted == actual).sum().item()

        return (100 * correct / total)