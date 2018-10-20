'''
To get the mean and std of the selected dataset
'''

import torch
import torchvision
import torchvision.transforms as transforms

datasets = 'CIFAR100'
transform = transforms.Compose(
    [
        # transforms.Pad(4, padding_mode='reflect'),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32),
        transforms.ToTensor()
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

trainset = torchvision.datasets.__dict__[datasets](root='./data/{}'.format(datasets), train=True,
                                        download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.__dict__[datasets](root='./data/{}'.format(datasets), train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

RGB_mean_tmp = torch.zeros((3,32,32))
RGB_std_tmp = torch.zeros((3,32,32))
RGB_std_sub = torch.ones((3,32,32))


for i in range(len(trainset)):
    RGB_mean_tmp += trainset[i][0]
RGB_mean = RGB_mean_tmp.mean(1).mean(1) / len(trainset)

print(RGB_mean.numpy())

for i in range(3):
    RGB_std_sub[i,:,:] *= RGB_mean[i]

for i in range(len(trainset)):
    RGB_std_tmp += (trainset[i][0]-RGB_std_sub)**2
RGB_std = RGB_std_tmp.mean(1).mean(1) / len(trainset)
RGB_std = torch.sqrt(RGB_std)
print(RGB_std.numpy())

torch.save({'mean':RGB_mean,'std':RGB_std}, './data/{}/stats.pkl'.format(datasets))
