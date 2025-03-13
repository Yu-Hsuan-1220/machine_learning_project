import torch
from torch import nn
from preprocess import *
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from torchvision import models
from torch.optim.lr_scheduler import MultiStepLR
device = "cuda:3" if torch.cuda.is_available() else "cpu"

class Conv(nn.Module):
    def __init__(self, intput_dim, output_dim, diff = True):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(intput_dim, output_dim, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )
        self.downSample = nn.Sequential(
            nn.Conv2d(intput_dim, output_dim, 3, 1, 1),
            nn.BatchNorm2d(output_dim)
        )
        self.diff = diff
        
    def forward(self, x):
        identify = x
        output = self.model(x)
        if(self.diff):
            identify = self.downSample(identify)
        return output + identify


class Classifier(nn.Module):

    def __init__(self):
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        super().__init__()
        
        self.cnn = nn.Sequential(
            Conv(3, 64),         # (64, 128, 128)
            Conv(64, 64, False), # (64, 128, 128)
            nn.MaxPool2d(2, 2),  # (64, 64, 64)
            Conv(64, 128),
            Conv(128, 128, False),
            nn.MaxPool2d(2, 2),
            Conv(128, 256),
            Conv(256, 256, False),
            nn.MaxPool2d(2, 2),  
            Conv(256, 512),
            Conv(512, 512, False),
            nn.MaxPool2d(2, 2),     
            Conv(512, 512, False),
            nn.MaxPool2d(2, 2),   #(512, 4, 4)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11),
        )
    
    def forward(self, x):
        out = self.cnn(x)
        
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

batch_size = 64
n_epoch = 50
model = Classifier().to(device)

# model = models.swin_t(weights='DEFAULT')
# model.head = nn.Linear(model.head.in_features, 11)
# model.to(device)


optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
scheduler = MultiStepLR(optimizer, milestones=[int(n_epoch * 0.5), int(n_epoch * 0.8)], gamma=0.5)



train_set = FoodDataset('./ml2023spring-hw3/train', tfm= train_tfm)
original_set = FoodDataset('./ml2023spring-hw3/train', tfm = test_tfm)
valid_set = FoodDataset('./ml2023spring-hw3/valid', tfm = test_tfm)
train_set = ConcatDataset([train_set, original_set])


train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)


best_acc = 0
for epoch in range(n_epoch):
    model.train()
    train_loss = []
    train_accs = []

    for batch, (x, y) in enumerate(tqdm(train_dataloader)):
        

        logits = model(x.to(device))
        loss = criterion(logits, y.to(device))
        optimizer.zero_grad()
        
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        acc = (logits.argmax(dim=-1) == y.to(device)).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print(f"[ Train | {epoch + 1:03d}/{n_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    scheduler.step()

    model.eval()
    valid_loss = []
    valid_accs = []
    # Iterate the validation set by batches.
    for batch in tqdm(valid_dataloader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), 'model_weights.pth')
        print(f"best_acc:{best_acc}, saving model")

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")