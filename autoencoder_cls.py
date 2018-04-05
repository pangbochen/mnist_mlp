# use auto encoder to enhance the model performance
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from config import get_args, ModelConfiger, ModelConfigerAutoencoder
from collections import OrderedDict
from utils import model_save
import time

#get args
args = get_args()

batch_size = args.batch_size
test_batch_size = args.test_batch_size

# load dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True)

# plot one example
print(train_loader.dataset.train_data.size())
print(train_loader.dataset.train_labels.size())

def showImg(im_np_data):
    plt.imshow(im_np_data, cmap='gray')
    plt.show()


class AutoEncoder(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_middle, type_act, dropout=0.2):
        super(AutoEncoder, self).__init__()

        self.input_dims = input_dims
        self.n_hiddens = n_hiddens
        self.n_class = n_middle
        self.dropout = dropout
        tmp_dim = input_dims


        encoder_layers = OrderedDict()
        decoder_layers = OrderedDict()
        # for activation function
        dic_act = {'relu': nn.ReLU(), 'sig': nn.Sigmoid(), 'tanh': nn.Tanh()}

        for idx, n_hidden in enumerate(n_hiddens):
            encoder_layers['fc_{}'.format(idx + 1)] = nn.Linear(tmp_dim, n_hidden)
            encoder_layers['{}_{}'.format(type_act, idx + 1)] = dic_act[type_act]
            encoder_layers['drop_{}'.format(idx + 1)] = nn.Dropout(dropout)
            tmp_dim = n_hidden

        encoder_layers['output'] = nn.Linear(tmp_dim, n_middle)

        self.encoder = nn.Sequential(encoder_layers)

        tmp_dim = n_middle

        for idx, n_hidden in enumerate(n_hiddens[::-1]):
            decoder_layers['fc_{}'.format(idx + 1)] = nn.Linear(tmp_dim, n_hidden)
            decoder_layers['{}_{}'.format(type_act, idx + 1)] = dic_act[type_act]
            decoder_layers['drop_{}'.format(idx + 1)] = nn.Dropout(dropout)
            tmp_dim = n_hidden

        decoder_layers['output'] = nn.Linear(tmp_dim, input_dims)
        decoder_layers['sigmoid']=nn.Sigmoid()
        self.decoder = nn.Sequential(decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# get model config
parameter = ModelConfigerAutoencoder()

n_hiddens = parameter.n_hiddens
type_act = parameter.type_act
rate_dropout = parameter.rate_dropout
n_middle = parameter.n_middle

# load autoencoder model
autoencoder = AutoEncoder(input_dims=784, n_hiddens=n_hiddens, n_middle=n_middle, type_act=type_act, dropout=rate_dropout)

autoencoder.load_state_dict(torch.load('auto_best_model.pth'))

# set optimizer
#optimizer = optim.SGD(autoencoder.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adagrad(autoencoder.parameters(), lr=args.lr)
# now for classification training

# acc, for report use
best_acc = 0


# time
start_time = time.time()


# train
for epoch in range(args.epochs):
    autoencoder.train()

    # judge for decreasing learning rate
    for batch_index, (data, target) in enumerate(train_loader):
        index_target = target.clone()

        # Variable
        data = data.view(-1, 784)

        data, target = Variable(data), Variable(target)

        # zero_grad
        optimizer.zero_grad()
        # get output from mlp model
        encoded, decoded = autoencoder(data)
        # use cross_entropy
        loss = F.cross_entropy(encoded, target)
        loss.backward()
        optimizer.step()

        if batch_index % args.log_interval == 0 and batch_index > 0:
            pred = encoded.data.max(1)[1]
            correct = pred.eq(index_target).sum()
            acc = correct * 1.0 / len(data)
            print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                loss.data[0], acc, optimizer.param_groups[0]['lr']))

    # cahce model
    model_save(autoencoder, 'auto_latest.pth')

    # then test after training epoch
    if (epoch % args.test_interval == 0 and epoch > 0) or epoch == args.epochs-1:
        autoencoder.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            index_target = target.clone()
            # if args.cuda:
            #     data, target = data.cuda(), target.cuda()
            data = data.view(-1, 784)
            data, target = Variable(data), Variable(target)
            output, decoded = autoencoder(data)
            test_loss += F.cross_entropy(output, target).data[0]
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(index_target).sum()
        # print test loss
        test_loss = test_loss / len(test_loader)
        acc = 100. * correct / len(test_loader.dataset)
        print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
            test_loss, correct, len(test_loader.dataset), acc))

        # if to update model
        if acc>best_acc:
            model_save(autoencoder, 'auto_best_model.pth')
            best_acc = acc
print('Best test acc: {:.3f}'.format(best_acc))

#
end_time = time.time()


print('Time: {:.2f}s'.format(end_time-start_time))
