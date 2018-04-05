from config import get_args, ModelConfiger
import torch
from torchvision import datasets, transforms
from model import MLP
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from utils import model_save
import time
from tensorboardX import SummaryWriter

# for tensorboard pytorch
tb_writer = SummaryWriter()
tb_interval = 100

# for cmd
args = get_args()
args.cuda = False

# for model parameters
parameters = ModelConfiger()

print('Training parameters')
print(args)

print('Model parameters')
print(parameters.__dict__)

# set random seed
torch.manual_seed(args.seed)



print('loading data')
# load data
# use pytorch DataLoader
# torchvision MNIST
# the url is http://yann.lecun.com/exdb/mnist
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

# model



n_hiddens = parameters.n_hiddens
type_act = parameters.type_act
rate_dropout = parameters.rate_dropout

mlp_model = MLP(input_dims=784, n_hiddens=n_hiddens, n_class=10, type_act=type_act, dropout=rate_dropout)

# SGD, Adam
#optimizer = optim.SGD(mlp_model.parameters(), lr=args.lr, momentum=args.momentum)

# optimizer = optim.Adam(mlp_model.parameters(), lr=args.lr)

optimizer = optim.Adagrad(mlp_model.parameters(),lr=args.lr)

# acc, for report use
best_acc = 0

# count_time
start_time = time.time()

# train
for epoch in range(args.epochs):
    mlp_model.train()

    # judge for decreasing learning rate
    for batch_index, (data, target) in enumerate(train_loader):
        index_target = target.clone()
        # cuda
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # Variable
        data, target = Variable(data), Variable(target)

        # zero_grad
        optimizer.zero_grad()
        # get output from mlp model
        output = mlp_model(data)
        # use cross_entropy
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if (batch_index % args.log_interval == 0 and batch_index > 0):
            pred = output.data.max(1)[1]
            correct = pred.eq(index_target).sum()
            acc = correct * 1.0 / len(data)
            print('Train Epoch: {} [{}/{}] Loss: {:.6f} Accuracy: {:.4f} lr: {:.2e}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                loss.data[0], acc, optimizer.param_groups[0]['lr']))
        if batch_index % tb_interval == 0 and batch_index > 0:
            pred = output.data.max(1)[1]
            correct = pred.eq(index_target).sum()
            acc = correct * 1.0 / len(data)
            # add tb_writer
            tb_writer.add_scalar('tblog/acc',acc,60000*epoch+batch_index*len(data))
            tb_writer.add_scalar('tblog/loss', loss.data[0], 60000 * epoch + batch_index * len(data))

    # cahce model
    model_save(mlp_model, 'latest.pth')

    # then test after training epoch
    if (epoch % args.test_interval == 0 and epoch > 0) or epoch == args.epochs-1:
        mlp_model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            index_target = target.clone()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = mlp_model(data)
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
            model_save(mlp_model, 'best_model.pth')
            best_acc = acc
print('Best test acc: {:.3f}'.format(best_acc))

# count time
end_time = time.time()

print('Time: {:.2f}s'.format(end_time-start_time))


tb_writer.export_scalars_to_json('./tblog/all_scalars.json')
tb_writer.close()