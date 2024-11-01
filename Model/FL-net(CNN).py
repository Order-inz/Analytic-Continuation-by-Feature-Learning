import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# training dataset
G = np.genfromtxt("/8_G.csv", delimiter=',')
A = np.genfromtxt("/8_A.csv", delimiter=',')
A = A / np.sum(A, axis=1, keepdims=True)

I_G = torch.tensor(G).float().to(device)
I_A = torch.tensor(A).float().to(device)

# the dimensionality of latent feature
DIM = 256


class EncoderA(nn.Module):
    def __init__(self):
        super(EncoderA, self).__init__()
        self.fc1 = nn.Linear(in_features=256, out_features=256)

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1)
        self.ln1 = nn.LayerNorm(128)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm(32)
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=DIM)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = x.unsqueeze(1)
        x = F.elu(self.ln1(self.conv1(x)))
        x = self.pool1(x)
        x = F.elu(self.ln2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.elu(self.fc2(x))
        h = F.elu(self.fc3(x))
        return h


class DecoderA(nn.Module):
    def __init__(self):
        super(DecoderA, self).__init__()
        self.fc1 = nn.Linear(in_features=DIM, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=256)

    def forward(self, h):
        x = F.elu(self.fc1(h))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x


class Autoencoder(nn.Module):
    def __init__(self, encoder_A, decoder_A):
        super(Autoencoder, self).__init__()
        self.EncoderA = encoder_A
        self.DecoderA = decoder_A

    def forward(self, x):
        h = self.EncoderA(x)
        A = self.DecoderA(h)
        return A


class EncoderG(nn.Module):
    def __init__(self):
        super(EncoderG, self).__init__()
        self.fc1 = nn.Linear(in_features=128, out_features=128)

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1)
        self.ln1 = nn.LayerNorm(64)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm(16)
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=DIM)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = x.unsqueeze(1)
        x = F.elu(self.ln1(self.conv1(x)))
        x = self.pool1(x)
        x = F.elu(self.ln2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.elu(self.fc2(x))
        h = F.elu(self.fc3(x))
        return h


# Split the data into training and validation sets
I_G_train, I_G_val, I_A_train, I_A_val = train_test_split(I_G, I_A, test_size=0.1, random_state=42)

# Initialize model and optimizer
EncoderA = EncoderA().to(device)
DecoderA = DecoderA().to(device)

Autoencoder = Autoencoder(EncoderA, DecoderA).to(device)
EncoderG = EncoderG().to(device)

batch_size = 160

learning_rate1 = 0.0001
learning_rate2 = 0.00008

num_epochs = 1500

A_var = torch.var(I_A_val)

optimizer_A = torch.optim.Adam(Autoencoder.parameters(), lr=learning_rate1)
optimizer_G = torch.optim.Adam(EncoderG.parameters(), lr=learning_rate2)
scheduler_A = StepLR(optimizer_A, step_size=450, gamma=0.5)
scheduler_G = StepLR(optimizer_G, step_size=350, gamma=0.5)

# Generate a shuffled indices for this epoch
shuffled_indices = torch.randperm(len(I_A_train)).to(device)

# A-h-A'
for epoch in range(num_epochs):
    Autoencoder.train()

    epoch_loss = 0
    # Mini-batch loop
    for i in range(0, len(I_A_train), batch_size):
        # Get mini-batch
        indices = shuffled_indices[i:i + batch_size].to(device)
        batchA = I_A_train[indices]

        # Forward pass
        A1 = Autoencoder(batchA)
        A1_loss = F.mse_loss(A1, batchA)

        # Backward pass
        optimizer_A.zero_grad()
        A1_loss.backward()
        optimizer_A.step()

        epoch_loss += A1_loss.item()

    epoch_loss /= len(I_A_train) // batch_size

    # Validation
    Autoencoder.eval()
    with torch.no_grad():
        A1_val = Autoencoder(I_A_val)
        A1_val_loss = F.mse_loss(A1_val, I_A_val)
        A1_re = A1_val_loss / (A_var + 1e-8).item()

    scheduler_A.step()
    current_lr = optimizer_A.param_groups[0]['lr']
    print(f"Epoch {epoch},Learning Rate: {current_lr}")
    print(f"Train Loss: {epoch_loss}, Val Loss: {A1_val_loss}")
    print(f"A_re: {A1_re}")


# G-h'-A'
for epoch in range(num_epochs):
    EncoderG.train()

    epoch_loss = 0
    # Mini-batch loop
    for i in range(0, len(I_G_train), batch_size):
        # Get mini-batch
        indices = shuffled_indices[i:i + batch_size].to(device)
        batchG = I_G_train[indices]
        batchA = I_A_train[indices]

        H1 = Autoencoder.EncoderA(batchA)
        H2 = EncoderG(batchG)
        H_loss = F.mse_loss(H2, H1)

        optimizer_G.zero_grad()
        H_loss.backward()
        optimizer_G.step()

        epoch_loss += H_loss.item()

    epoch_loss /= len(I_G_train) // batch_size

    # Validation
    EncoderG.eval()
    Autoencoder.DecoderA.eval()
    with torch.no_grad():
        H2_val = EncoderG(I_G_val)
        A2_val = Autoencoder.DecoderA(H2_val)
        A2_val_loss = F.mse_loss(A2_val, I_A_val).item()
        G_re = A2_val_loss / (A_var + 1e-8).item()

    scheduler_G.step()
    current_lr = optimizer_G.param_groups[0]['lr']
    print(f"Epoch {epoch},Learning Rate: {current_lr}")
    print(f"H Loss: {epoch_loss}")
    print(f"Val Loss: {A2_val_loss}")
    print(f"G_re: {G_re}")


# h‘-A
for epoch in range(num_epochs):
    Autoencoder.DecoderA.train()

    epoch_loss = 0
    # Mini-batch loop
    for i in range(0, len(I_A_train), batch_size):
        # Get mini-batch
        indices = shuffled_indices[i:i + batch_size].to(device)
        batchG = I_G_train[indices]
        batchA = I_A_train[indices]

        H_g = EncoderG(batchG)

        # Forward pass
        A3 = Autoencoder.DecoderA(H_g)
        A3_loss = F.mse_loss(A3, batchA)

        # Backward pass
        optimizer_A.zero_grad()
        A3_loss.backward()
        optimizer_A.step()

        epoch_loss += A3_loss.item()

    epoch_loss /= len(I_A_train) // batch_size

    # Validation
    EncoderG.eval()
    Autoencoder.DecoderA.eval()
    with torch.no_grad():
        H3_val = EncoderG(I_G_val)
        A3_val = Autoencoder.DecoderA(H3_val)
        A3_val_loss = F.mse_loss(A3_val, I_A_val).item()
        P_re = A3_val_loss / (A_var + 1e-8).item()

    print(f"Epoch {epoch}")
    print(f"Train Loss: {epoch_loss}, Val Loss: {A3_val_loss}")
    print(f"P_re: {P_re}")
