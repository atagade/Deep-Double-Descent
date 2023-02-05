from misc import launch
from misc import count_parameters
from misc.parser import default_argument_parser

from datasets import get_torch_dataloaders
from models import get_model

import torch
import torch.nn as nn

def train(model, train_loader, test_loader, num_epochs, criterion, optimizer, total_steps)
    training_losses = []
    test_losses = []
    interpolation = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(train_loader):

                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                labels = torch.nn.functional.one_hot(labels, 10).float()
                # Forward pass
                outputs = model(images)
                train_loss = criterion(outputs, labels)
                
                # Backpropagation and optimization
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                if interpolation == 0:
                    if train_loss == 0:
                        interpolation = 1
                        training_losses.append(train_loss.item())
                        
                        with torch.no_grad():
                            correct = 0
                            total = 0
                            for images, labels in test_loader:
                                images = images.reshape(-1, 28*28).to(device)
                                labels = labels.to(device)
                                labels = torch.nn.functional.one_hot(labels, 10).float()
                                outputs = model(images)
                                test_loss = criterion(outputs, labels)
                            
                        test_losses.append(test_loss.item())

                        print ('Model: FCNN-{} Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}, Test Loss: {:.4f}' 
                            .format(count_parameters(model), epoch+1, num_epochs, i+1, total_step, train_loss.item(), test_loss.item()))
                        break

                if epoch % 100 == 0 and i + 1 == len(train_loader):

                    training_losses.append(train_loss.item())
                    
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for images, labels in test_loader:
                            images = images.reshape(-1, 28*28).to(device)
                            labels = labels.to(device)
                            labels = torch.nn.functional.one_hot(labels, 10).float()
                            outputs = model(images)
                            test_loss = criterion(outputs, labels)
                    
                    test_losses.append(test_loss.item())

                    print ('Model: FCNN-{} Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}, Test Loss: {:.4f}' 
                        .format(count_parameters(model), epoch+1, num_epochs, i+1, total_steps, train_loss.item(), test_loss.item()))

def main(dataset, model, num_hidden_units, num_gpus):

    train_loader, test_loader = get_torch_dataloaders(dataset)

    model = get_model(model, num_hidden_units, dataset)
    
    print(f"Number of parameters in the model is: {count_parameters(model)}")

    train(model,
        train_loader, 
        test_loader,
        num_epochs = 6000, 
        criterion = nn.MSELoss(), 
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.95),
        total_steps = len(train_loader))


if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(main,
           dataset=args.dataset,
           model=args.model,
           num_hidden_units=args.num_hidden_units,
           num_gpus=args.num_gpus,
            )