import torch
import torchvision
import torchvision.transforms as transforms

def get_torch_dataloaders(dataset, batch_size = 100):

    if dataset == "MNIST":

        train_dataset = torchvision.datasets.MNIST(root='/data/mnist', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

        test_dataset = torchvision.datasets.MNIST(root='/data/mnist', 
                                          train=False, 
                                          transform=transforms.ToTensor())

        train_4000 = torch.utils.data.Subset(train_dataset, range(0, 4000))


        train_loader = torch.utils.data.DataLoader(dataset=train_4000, 
                                           batch_size=batch_size, 
                                           shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

        return train_loader, test_loader