# %%
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np

# %%
IMAGE_SIZE = 16
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=composed)
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=composed)

# %%
# class train(Dataset):
#     def __init__(self):
#         df = pd.read_csv('mnist_train.csv')
#         self.x = torch.tensor(df.drop(['label'], axis=1).values)
#         self.x = self.x.reshape(60000, 28, 28)
#         self.y = torch.tensor(df['label'])
#         self.samples = df.shape[0]
        
#     def __getitem__(self, index):
#         x = self.x[index].reshape(28, 28)
#         return x, self.y[index]
    
#     def __len__(self):
#         return self.samples
    
# train_dataset = train()

# train_dataloader = DataLoader(dataset=train_dataset, shuffle=True)

# %%
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(512,10)
        
    def forward(self,x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# %%
model = CNN()

# %%
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)
epochs = 1
def train( model, optimizer, criterion, epochs):
    cost = []
    i = 0
    total = 0
    for epoch in range(epochs):
        for x,y in train_loader:
            i += 1
            x = x.float()
            yhat = model(x)
            loss = criterion(yhat, y.long())
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            total += loss.item()
            if i%1000 == 0:
                print(i, loss)
        cost.append(total)
        print("loss", loss)
        print("cost", cost)
    return cost

# %%
# saving a model 
PATH = './cifar_net.pth'
dict = model.state_dict()
torch.save(dict, PATH)
# model.load_state_dict(torch.load(PATH)) 
# can be used to load that model again

# %%
cost = train( model, optimizer, criterion, epochs)
print(cost)

# %%
# class test(Dataset):
#     def __init__(self):
#         dataset = pd.read_csv('mnist_test.csv', dtype=float)
#         self.x = torch.tensor(dataset.drop(['label'], axis=1).values)
#         self.y = torch.tensor(dataset['label'].values)
#         self.smaples = dataset.shape[0]
        
#     def __len__(self):
#         return self.smaples
    
#     def __getitem__(self,index):
#         x = self.x[index].reshape(28, 28)
#         return x, self.y[index]
    
# test_dataset = test()

# test_dataloader = DataLoader(dataset=test_dataset, shuffle=True)

# %%
def test():
    accuracy_list=[]
    N_test=len(validation_dataset)
    correct=0
    #perform a prediction on the validation  data  
    for x_test, y_test in validation_loader:
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()
    accuracy = correct / N_test
    accuracy_list.append(accuracy)
    print(accuracy)
test()


