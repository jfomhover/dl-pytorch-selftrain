import time
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import traceback
import logging

def one_hot(x, class_count):
    """ converts a multi-label column (1D tensor) into one-hot vectors (2D tensor) """
    return torch.eye(class_count)[x,:]

# see examples at https://github.com/utkuozbulak/pytorch-custom-dataset-examples
class InputTargetPairsDataset(Dataset):
    def __init__(self, inputs_tensor, targets_tensor):
        """ A basic DIY dataset made of tuples of (input,target). """
        self.inputs_tensor = inputs_tensor
        self.targets_tensor = targets_tensor

    def __getitem__(self, index):
        return (self.inputs_tensor[index], self.targets_tensor[index])

    def __len__(self):
        return len(self.targets_tensor)


class BasicMultiClassTargetDataset(InputTargetPairsDataset):
    def __init__(self, inputs, targets, class_count):
        """
        A basic dataset made of tuples of (input,target).
        Target is transformed into a one_hot vector.
        
        Parameters
        ----------
        inputs: numpy array
        targets: numpy array
        class_count: number of classes in targets array
        """
        inputs_tensor = torch.tensor(inputs).float()
        targets_tensor = one_hot(targets, class_count)
        super(BasicMultiClassTargetDataset, self).__init__(inputs_tensor, targets_tensor)

    
class BasicNeuralNet(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        """ A neural net a la 90s, with one hidden layer only. """
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
        """
        Helper class to train a model using a batched dataset.
        
        Parameters
        ----------
        model: {torch.nn.Module}
        optimizer: {torch.optim.*}
        criterion: {torch.nn._Loss}
        
        Keyword Arguments
        -----------------
        verbose: {boolean}
            prints some ETA
            
        auto_save_path: {string}
            a string using keys epoch and loss for saving model after each epoch
            ex: "models/model-lstm-epoch{epoch}-loss{loss}.tar"
            if None, auto save is off
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.criterion_record = []
        self.verbose = kwargs.get("verbose", False)
        self.auto_save_path = kwargs.get("auto_save_path", None)
        self.logger = logging.getLogger(__file__)
        
        # testing auto save path
        if not(self.auto_save_path is None):
            try:
                self.auto_save_path.format(epoch=0, loss=0.0)
            except:
                self.logger.critical(traceback.format_exc())
                

    def epoch(self,
              tuple_dataset,
              batch_size):
        """
        Fits the model for one epoch on the data provided, batching data first.
        
        Parameters
        ----------
        tuple_dataset: {BasicLabelledDataset}
        batch_size: {int}
        verbose: {bool}
        
        Returns
        -------
        model : the trained model
        loss : the loss on whole epoch
        """
        # this will batch the data for you (given you have a DataSet for it)
        training_loader = torch.utils.data.DataLoader(
            dataset=tuple_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # for timing
        epoch_data_to_fit = len(tuple_dataset)
        start_time = time.time()
        epoch_data_processed = 0

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

            # print timing
            if self.verbose:
                epoch_data_processed += len(targets)
                time_per_data = ((time.time() - start_time)) / float(epoch_data_processed)
                eta = int((epoch_data_to_fit - i*len(targets)) * time_per_data)
                print("batch_loss={:2f}\t avg_loss={:2f}\t epoch_ETA: {} secs (data={}/{})".format(
                    loss.item(),
                    (running_loss / epoch_data_processed),
                    "{:2d}:{:2d}:{:2d}".format((eta//3600), (eta % 3600)//60, eta%60),
                    i*len(data),
                    epoch_data_to_fit
                ), end='\r', flush=True)
        
        return self.model, running_loss
        
    def fit(self,
            tuple_dataset,
            epochs,
            batch_size):
        """
        Fits the model on the data provided, for a given number of epochs, batching data first.
        
        Parameters
        ----------
        tuple_dataset: {BasicLabelledDataset}
        epochs: {int} or {list}
        batch_size: {int}
        verbose: {bool}
        
        Returns
        -------
        model : the trained model
        loss : the loss on whole epoch
        """
        if isinstance(epochs, list):
            epochs_list = epochs
        else:
            epochs_list = range(epochs)
            
        for epoch in epochs_list:  # loop over the dataset multiple times
            model, loss = self.epoch(tuple_dataset, batch_size)
            if self.verbose:
                print("[epoch={}]\t epoch_loss={:2f}".format(epoch,loss))
                self.logger.info("[epoch={}]\t epoch_loss={:2f}".format(epoch,loss))
            
            if not(self.auto_save_path is None):
                try:
                    model_file_path = self.auto_save_path.format(epoch=epoch, loss=loss)

                    torch.save(
                        {
                            'epoch': epochs,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss,
                        },
                        model_file_path
                    )

                    if self.verbose:
                        print("Model saved as {}".format(model_file_path))

                except Exception as e:
                    print(traceback.format_exc())
        
        return self.model, loss
    
    def test_accuracy(self, tuple_dataset):
        """
        Runs the model on a dataset with target classes and computes accuracy.
        
        Parameters
        ----------
        tuple_dataset: {InputTargetPairsDataset}
        
        Returns
        -------
        accuracy: {float}
        """
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