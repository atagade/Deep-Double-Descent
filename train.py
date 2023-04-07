import torch
import torch.nn as nn
import time
import json

from misc import count_parameters
from misc import default_argument_parser
from misc import add_to_json

from datasets import get_torch_dataloaders
from models import get_model

def train(model, train_loader, test_loader, num_epochs, criterion, optimizer, total_steps):
    training_losses = []
    test_losses = []
    interpolation = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(train_loader):

                images, labels = images.to(device), labels.to(device)
                #labels = torch.nn.functional.one_hot(labels, 10).float()
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
                            for images, labels in test_loader:
                                images = images.reshape(-1, 28*28).to(device)
                                labels = labels.to(device)
                                labels = torch.nn.functional.one_hot(labels, 10).float()
                                outputs = model(images)
                                test_loss = criterion(outputs, labels)
                            
                        test_losses.append(test_loss.item())

                        print ('Model: {} Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}, Test Loss: {:.4f}' 
                            .format(count_parameters(model), epoch+1, num_epochs, i+1, total_steps, train_loss.item(), test_loss.item()))
                        break

                if epoch % 100 == 0 and i + 1 == len(train_loader):

                    training_losses.append(train_loss.item())
                    
                    with torch.no_grad():
                        for images, labels in test_loader:
                            images = images.to(device)
                            labels = labels.to(device)
                            outputs = model(images)
                            test_loss = criterion(outputs, labels)
                    
                    test_losses.append(test_loss.item())

                    print ('Model: {} Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}, Test Loss: {:.4f}' 
                        .format(count_parameters(model), epoch+1, num_epochs, i+1, total_steps, train_loss.item(), test_loss.item()))
                    
    return train_loss.item(), test_loss.item()

def generate_logs(dataset, model, width, num_gpus=1):

    train_loader, test_loader = get_torch_dataloaders(dataset, batch_size=int(128*16*2.5))

    torch_model = get_model(model=model, dataset=dataset, width=width)
    
    print(f"Number of parameters in the model is: {count_parameters(torch_model)}")

    start_time = time.time()

    final_train_loss, final_test_loss = train(torch_model,
        train_loader, 
        test_loader,
        num_epochs = 4000, 
        criterion = nn.CrossEntropyLoss(), 
        optimizer = torch.optim.Adam(torch_model.parameters(), lr = 0.0004),
        total_steps = len(train_loader))
    
    end_time = time.time()

    print(f'Time taken to train: {(end_time - start_time)/60} minutes')
    
    add_to_json(f'results/{model}-{dataset}.json',count_parameters(torch_model), final_train_loss, final_test_loss)

    print('Logs dumped')

if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    generate_logs(
           dataset=args.dataset,
           model=args.model,
           width=args.width,
           num_gpus=args.num_gpus,
            )