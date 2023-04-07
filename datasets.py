import torch
import torchvision
import torchvision.transforms as transforms

def get_torch_dataloaders(dataset, batch_size = 128):

    if dataset.lower() == "mnist":

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

        print('Dataset Loaded')
        return train_loader, test_loader
    
    if dataset.lower() == "cifar10":

        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                              transforms.RandomHorizontalFlip(), 
                                              transforms.ToTensor(), 
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        
        train_dataset = torchvision.datasets.CIFAR10(root = '/data/cifar10',
                                                     train = True,
                                                     download = True,
                                                     transform=transform_train)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size*100/128, shuffle=False, num_workers=2)

        print(f'Created dataloaders with batch size {batch_size}')
        return train_loader, test_loader
    
    if dataset.lower() == "cifar100":

        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                              transforms.RandomHorizontalFlip(), 
                                              transforms.ToTensor(), 
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        
        train_dataset = torchvision.datasets.CIFAR100(root = '/data/cifar100',
                                                     train = True,
                                                     download = True,
                                                     transform=transform_train)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        test_dataset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(batch_size*100/128), shuffle=False, num_workers=2)

        print(f'Created dataloaders with batch size {batch_size}')
        return train_loader, test_loader