# libraries importing
import pandas as pd
import matplotlib.pyplot as plt
import os
# additional modules
import sys
from utils.evaluating import evaluating_change_point
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.utils.data as data
from torch.optim import Adam
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import shutil

"Environment Settings"
cudnn.deterministic = True
cudnn.benchmark = False
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
num_workers = 0
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

"Path for results"
result_dir = "results"
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
run_name = "ConAE-MTCE5"
result_dir = os.path.join(result_dir, run_name)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

backup_dir = os.path.join(result_dir, "backup")
if not os.path.exists(backup_dir):
    os.mkdir(backup_dir)
shutil.copyfile("ConvAE-MTCE-PyTorch.py", os.path.join(backup_dir, "ConvAE-MTCE-PyTorch.py.bk"))


log_file = "exp_log"
log_file = os.path.join(result_dir, log_file)
log_file = open(log_file, "w+")

all_files = []
for root, dirs, files in os.walk("./data/"):
    for file in files:
        if file.endswith(".csv"):
             all_files.append(os.path.join(root, file))

all_files.sort()

# datasets with anomalies loading
list_of_df = [pd.read_csv(file,
                          sep=';',
                          index_col='datetime',
                          parse_dates=True) for file in all_files if 'anomaly-free' not in file]
# anomaly-free df loading
anomaly_free_df = pd.read_csv([file for file in all_files if 'anomaly-free' in file][0],
                            sep=';',
                            index_col='datetime',
                            parse_dates=True)


# dataset characteristics printing
print(f'A number of datasets in the SkAB v1.0: {len(list_of_df)}\n')
print(f'Shape of the random dataset: {list_of_df[0].shape}\n')
n_cp = sum([len(df[df.changepoint==1.]) for df in list_of_df])
n_outlier = sum([len(df[df.anomaly==1.]) for df in list_of_df])
print(f'A number of changepoints in the SkAB v1.0: {n_cp}\n')
print(f'A number of outliers in the SkAB v1.0: {n_outlier}\n')
print(f'Head of the random dataset:')
print(list_of_df[0].head())


# hyperparameters selection
N_STEPS = 64
Q = 0.999  # quantile for upper control limit (UCL) selection
batch_size = 32
lr = 1e-3
num_epochs = 100


# Generated training sequences for use in the model.
def create_sequences(values, time_steps=N_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


class Wrapper(data.Dataset):
    def __init__(self, X):
        super(Wrapper, self).__init__()
        self.X = X

    def __getitem__(self, item):
        return self.X[item, :, :]

    def __len__(self):
        return self.X.shape[0]


class MTCE(nn.Module):
    def __init__(self, channels=[32, 16], ratio=0.5):
        super(MTCE, self).__init__()
        ich = channels[-1]
        self.blocks = []
        self.ratio = ratio
        for ci, channel in enumerate(channels):
            block_name = "Branch_{}".format(ci + 1)
            self.add_module(block_name, nn.Sequential(
                nn.Conv1d(channel, ich, 1),
                nn.ReLU()
            ))
            self.blocks.append(block_name)
        self.out_trans = nn.Sequential(
            nn.Conv1d(ich, channels[-1], 1),
            nn.ReLU()
        )

    def reconstruct(self, x_c, mu):
        # x_c: B x C x N, mu: B x C x K
        # similarity
        z_d = torch.bmm(x_c.permute(0, 2, 1), mu)  # B x N x K
        # normalization
        z_d = F.softmax(z_d, dim=2)
        z_d = z_d / (1e-6 + z_d.sum(dim=1, keepdim=True))
        z_d = z_d.permute(0, 2, 1)  # B x K x N
        z_d = torch.bmm(mu, z_d)
        z_d = self.out_trans(z_d)
        return z_d

    def forward(self, x):
        xs = []
        for xi, xt in enumerate(x):
            xs.append(self.__getattr__(self.blocks[xi])(xt))
        x_rr = xs[-1].clone()
        xs = torch.cat(xs, dim=2)
        sim = torch.bmm(xs.permute(0, 2, 1), xs)
        sim = F.softmax(sim, dim=2)
        ssim, tidxs = torch.sort(torch.sum(sim, dim=1), dim=1, descending=True)
        tidxs = tidxs[:, : int(self.ratio * xs.shape[2])]
        basis = torch.cat([torch.index_select(a, 1, i).unsqueeze(0) for a, i in zip(xs, tidxs)])
        x_r = self.reconstruct(x_rr, basis)
        return x_r + x_rr


class ConvAE_MTCE(nn.Module):
    def __init__(self, channels=8):
        super(ConvAE_MTCE, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3, stride=2),
            nn.ReLU(True),
            nn.Dropout(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=7, padding=3, stride=2),
            nn.ReLU(True),

        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(16, 16, kernel_size=7, padding=3, stride=2, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose1d(16, 32, kernel_size=7, padding=3, stride=2, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 8, kernel_size=7, padding=3),
        )
        self.mtce = MTCE()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x1_r = self.mtce([x0, x1])
        x2 = self.conv2(x1)
        x2_r = self.conv2(x1_r)
        return torch.transpose(x2, 1, 2), torch.transpose(x2_r, 1, 2)


model = ConvAE_MTCE(8).to(device)
net_paramn = sum(p.numel() for p in model.parameters())
msg = "param num: {}".format(net_paramn)
log_file.write(msg + "\n")
print(msg)


# inference
predicted_outlier, predicted_cp = [], []
for di, df in enumerate(list_of_df):
    X_train = df[:400].drop(['anomaly', 'changepoint'], axis=1)

    # scaler init and fitting
    StSc = StandardScaler()
    StSc.fit(X_train)

    # convert into input/output
    X = create_sequences(StSc.transform(X_train), N_STEPS)

    # model defining and fitting
    train_dataset = Wrapper(X)
    train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True)
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    for ei in range(num_epochs):
        error_sum = 0
        for ti, item in enumerate(train_dataloader):
            item = item.to(device).float()
            pred, pred_r = model(item)
            loss = F.mse_loss(pred, item) + 2 * F.mse_loss(pred_r, item)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            error_sum += loss.detach().cpu().item()
        # print("epoch {}, loss {}".format(ei + 1, error_sum))

    torch.save(model.state_dict(), os.path.join(result_dir, "model_{}.pth".format(di + 1)))

    model.eval()
    # results predicting
    with torch.no_grad():
        preds, preds_r = model(torch.tensor(X, device=device).float())
        preds = preds.detach().cpu().numpy()
        preds_r = preds_r.detach().cpu().numpy()
    residuals = pd.Series(np.sum(np.mean(np.abs(X - preds) + np.abs(X - preds_r), axis=1), axis=1))
    UCL = residuals.quantile(Q)

    # results predicting
    X = create_sequences(StSc.transform(df.drop(['anomaly', 'changepoint'], axis=1)), N_STEPS)
    with torch.no_grad():
        preds, preds_r = model(torch.tensor(X, device=device).float())
        preds = preds.detach().cpu().numpy()
        preds_r = preds_r.detach().cpu().numpy()
    cnn_residuals = pd.Series(np.sum(np.mean(np.abs(X - preds) + np.abs(X - preds_r), axis=1), axis=1))

    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data = cnn_residuals > (3 / 2 * UCL)
    anomalous_data_indices = []
    for data_idx in range(N_STEPS - 1, len(X) - N_STEPS + 1):
        if np.all(anomalous_data[data_idx - N_STEPS + 1: data_idx]):
            anomalous_data_indices.append(data_idx)

    prediction = pd.Series(data=0, index=df.index)
    prediction.iloc[anomalous_data_indices] = 1

    # predicted outliers saving
    predicted_outlier.append(prediction)

    # predicted CPs saving
    prediction_cp = abs(prediction.diff())
    prediction_cp[0] = prediction[0]
    predicted_cp.append(prediction_cp)

    gt = df.anomaly
    f1, FAR, MAR = evaluating_change_point(gt, prediction, metric='binary', numenta_time='30 sec')
    msg = "processed dataset {}, f1 score {}, FAR {}, MAR {}".format(di, f1, FAR, MAR)
    print(msg)
    log_file.write(msg + "\n")

true_outlier = [df.anomaly for df in list_of_df]
f1, FAR, MAR = evaluating_change_point(true_outlier, predicted_outlier, metric='binary', numenta_time='30 sec')
torch.save([true_outlier, predicted_outlier, f1, FAR, MAR], os.path.join(result_dir, "predictions.pth"))
msg = "final f1 score {}, FAR {}, MAR {}".format(f1, FAR, MAR)
log_file.write(msg + "\n")
print(msg)
print("end of test")
