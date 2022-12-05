from main import load_dataset
from models import DNMC, myloss, train_model
import numpy as np
import pandas as pd
from generate_data import generate_semi_synthetic, generate_synth_censoring, onehot
import torch
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

x, y, t = load_dataset(360)
y = y["ASD Dx"]
t = t["First ASD Code Date"]

N_BINS = 10
BIN_LENGTH = (np.amax(t) * 1.0001 - np.amin(t)) / N_BINS
t_idx = (t - np.amin(t)) // BIN_LENGTH
t = onehot(t_idx, ncategories=10)

df = pd.concat([pd.DataFrame(t).reset_index(drop=True), x.reset_index(drop=True)], axis=1)
y = y.reset_index(drop=True)
oversampler_train = RandomOverSampler(sampling_strategy={1:7000})
oversampler_val = RandomOverSampler(sampling_strategy={1:1500})

df_train, df_val, df_test = df[:30000], df[30000:37000], df[37000:]
y_train, y_val, y_test = y[:30000], y[30000:37000], y[37000:]

df_train_over, y_train_over = oversampler_train.fit_sample(df_train, y_train)
df_val_over, y_val_over = oversampler_val.fit_sample(df_val, y_val)

x_train_over = df_train_over.iloc[:, 10:]
t_train_over = df_train_over.iloc[:, 0:10]

x_val_over = df_val_over.iloc[:, 10:]
t_val_over = df_val_over.iloc[:, 0:10]

idx_train = np.random.permutation(y_train_over.index)
x_train = x_train_over.reindex(idx_train)
y_train = y_train_over.reindex(idx_train)
t_train = t_train_over.reindex(idx_train)

idx_val = np.random.permutation(y_val_over.index)
x_val = x_val_over.reindex(idx_val)
y_val = y_val_over.reindex(idx_val)
t_val = t_val_over.reindex(idx_val)

x_test = df_test.iloc[:, 10:]
t_test = df_test.iloc[:, 0:10]

x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
t_train = torch.tensor(t_train.to_numpy(), dtype=torch.float32)

x_val = torch.tensor(x_val.to_numpy(), dtype=torch.float32)
y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
t_val = torch.tensor(t_val.to_numpy(), dtype=torch.float32)

def discrete_ci(st, tt, tp):

    s_true = np.array(st).copy()
    t_true = np.array(tt).copy()
    t_pred = np.array(tp).copy()

    t_true_idx = np.argmax(t_true, axis=1)
    t_pred_cdf = np.cumsum(t_pred, axis=1)

    concordant = 0
    total = 0
    
lr_list = np.logspace(-2, 1, 10)#[1e-2, 0.1, 1] #val loss will be inf np.logspace(-5, -1, 10)
ld_list = [0, 0.1, 1, 10]
ci = []
auc = []
for lr in lr_list:
    for ld in ld_list:
        x_train_copy = x_train
        t_train_copy = t_train
        y_train_copy = y_train
        x_val_copy = x_val
        t_val_copy = t_val
        y_val_copy = y_val

        model_iter = DNMC(n_bins=N_BINS, lr=lr, ld=ld)
        criterion = myloss(lr=lr, ld=ld)
        train_model(
            model_iter, (x_train_copy, t_train_copy, y_train_copy), (x_val_copy, t_val_copy, y_val_copy), criterion=criterion, n_epochs=40, learning_rate=1e-4,
            batch_size=64)
        e_pred_iter, t_pred_iter, c_pred_iter = model_iter.forward_pass(torch.tensor(x_test.to_numpy(), dtype=torch.float32))
        concordance_ci = discrete_ci(y_test, t_test, t_pred_iter.detach().numpy() * e_pred_iter.detach().numpy()[:, None])
        # concordance index at 6 years concordance_index_censored(y_test, actual_times_test, np.sum(t_pred_iter.detach().numpy()[:, :6]) * e_pred_iter.detach().numpy())
        auc_score = roc_auc_score(y_test, e_pred_iter.detach().numpy())
        ci.append(concordance_ci)
        auc.append(auc_score)


    N = len(s_true)
    idx = np.arange(N)

    for i in range(N):

        if s_true[i] == 0:
            continue

        # time bucket of observation for i, then for all but i
        tti_idx = t_true_idx[i]
        tt_idx = t_true_idx[idx != i]

        # calculate predicted risk for i at the time of their event
        tpi = t_pred_cdf[i, tti_idx]

        # predicted risk at that time for all but i
        tp = t_pred_cdf[idx != i, tti_idx]

        total += np.sum(tti_idx < tt_idx) # observed in i first
        concordant += np.sum((tti_idx < tt_idx) * (tpi > tp)) # and i predicted as higher risk

    return concordant / total
