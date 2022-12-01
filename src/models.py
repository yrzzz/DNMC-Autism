import numpy as np
import torch
from torch import nn
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.generate_data import generate_semi_synthetic, generate_synth_censoring, onehot
import matplotlib.pyplot as plt
class DNMC(nn.Module):
    def __init__(
        self,
        phi_layer_sizes=[256,],
        psi_layer_sizes=[256,],
        omega_layer_sizes=[256,],
        e_layer_sizes=[256,],
        t_layer_sizes=[256,],
        c_layer_sizes=[256,],
        importance_weights=[1.0, 1.0],
        include_censoring_density=True,
        dependent_censoring=False,
        include_psi=True,
        mtlr_head=False,
        n_bins=50,
        activation="ReLU",
        rep_activation="ReLU",
        ld=1e-3,
        lr=1e-3,
        l_mtlr=0.0,
        tol=1e-8,
    ):
        super().__init__()

        self.ld = ld
        self.lr = lr
        self.l_mtlr = l_mtlr
        self.w0 = torch.tensor(importance_weights[0], dtype=torch.float32)
        self.w1 = torch.tensor(importance_weights[1], dtype=torch.float32)
        self.include_censoring_density = include_censoring_density
        self.dependent_censoring = dependent_censoring
        self.include_psi = include_psi
        self.mtlr_head = mtlr_head
        self.n_bins = n_bins
        self.activation = activation
        self.tol = tol

        self.losses = torch.tensor([])
        self.phi_layers = [
            layer
            for layers in (
                self.dense(ls, activation=rep_activation) for ls in phi_layer_sizes
            )
            for layer in layers
        ]

        if include_psi:
            self.psi_layers = [
                layer
                for layers in (
                    self.dense(ls, activation=rep_activation) for ls in psi_layer_sizes
                )
                for layer in layers
            ]

        self.omega_layers = [
            layer
            for layers in (
                self.dense(ls, activation=rep_activation) for ls in omega_layer_sizes
            )
            for layer in layers
        ]

        self.e_layers = [
            layer
            for layers in (self.dense(ls) for ls in e_layer_sizes)
            for layer in layers
        ] + [layer for layer in (self.dense(1, activation="Sigmoid"))]

        if mtlr_head:
            self.t_layers = [
                layer
                for layers in (self.dense(ls) for ls in t_layer_sizes)
                for layer in layers
            ] + [layer for layer in (self.dense(n_bins - 1, activation="Sigmoid"))]
        else:
            self.t_layers = [
                layer
                for layers in (self.dense(ls) for ls in t_layer_sizes)
                for layer in layers
            ] + [layer for layer in (self.dense(n_bins, activation="Softmax"))]

        self.phi_model = nn.Sequential(*self.phi_layers)
        if include_psi:
            self.psi_model = nn.Sequential(*self.psi_layers)
        self.omega_model = nn.Sequential(*self.omega_layers)

        self.e_model = nn.Sequential(*self.e_layers)
        self.t_model = nn.Sequential(*self.t_layers)

        if include_censoring_density:

            if dependent_censoring:
                self.c_layers = [
                    [
                        layer
                        for layers in (self.dense(ls) for ls in c_layer_sizes)
                        for layer in layers
                    ]
                    + [layer for layer in (self.dense(n_bins, activation="Softmax"))]
                    for i in range(2)
                ]
                self.c_model = [nn.Sequential(*cl) for cl in self.c_layers]

            else:

                if mtlr_head:
                    self.t_layers = [ # why not c_layers?
                        layer
                        for layers in (self.dense(ls) for ls in c_layer_sizes)
                        for layer in layers
                    ] + [
                        layer
                        for layer in (self.dense(n_bins - 1, activation="Sigmoid"))
                    ]
                else:
                    self.c_layers = [
                        layer
                        for layers in (self.dense(ls) for ls in c_layer_sizes)
                        for layer in layers
                    ] + [layer for layer in (self.dense(n_bins, activation="Softmax"))]
                self.c_model = nn.Sequential(*self.c_layers)

    def dense(self, layer_size, activation=None):
        if activation is None:
            activation = self.activation

        layer = nn.LazyLinear(layer_size)
        activation_function = getattr(nn, activation)()
        layers = [layer, activation_function]
        return layers

    def forward_pass(self, x):
        if self.include_psi:
            self.psi = self.psi_model(x)
        self.phi = self.phi_model(x)
        self.omega = self.omega_model(x)

        if self.include_psi:
            self.e_pred = torch.squeeze(self.e_model(torch.cat([self.phi, self.psi], dim=-1)), dim=1)
            self.t_pred = self.t_model(torch.cat([self.psi, self.omega], dim=-1))
        else:
            self.e_pred = torch.squeeze(self.e_model(self.phi), dim=1)
            self.t_pred = self.t_model(self.omega)

        if self.include_censoring_density:
            if self.dependent_censoring:
                if self.include_psi:
                    self.c_pred = [
                        cm(torch.cat([self.psi, self.omega], dim=-1))
                        for cm in self.c_model
                    ]
                else:
                    self.c_pred = [
                        cm(self.omega)
                        for cm in self.c_model
                    ]
                return self.e_pred, self.t_pred, self.c_pred
            else:
                if self.include_psi:
                    self.c_pred = self.c_model(torch.cat([self.psi, self.omega], dim=-1))
                else:
                    self.c_pred = self.c_model(self.omega)
                return self.e_pred, self.t_pred, self.c_pred

        else:
            return self.e_pred, self.t_pred

    def call(self, x):
        return self.forward_pass(x)

    def iweights(self, s):
        return s*self.w1 + (1-s)*self.w0

    def loss(self, x, y, s):

        nll = torch.mean(self.iweights(s) * self.nll(x, y, s))

        # MMD Term
        # l = nll + self.ld * tf.cast(self.mmd(x, s), dtype=tf.float32)
        l = nll + self.ld * self.mmd(self.omega_model(x), s)

        # Global L2 regularizer
        for param in self.parameters():
            l += self.lr * param.norm(2)

        # MTLR L2 regularizer
        if self.mtlr_head:
            l += self.l_mtlr * mtlr_sum_of_squares(self.t_layers[-1])
            if self.include_censoring_density:
                l += self.l_mtlr * mtlr_sum_of_squares(self.c_layers[-1])

        return l, nll

    def nll(self, x, y, s):

        # yt = torch.tensor(y, dtype=torch.float32)

        if self.include_censoring_density:

            e_pred, t_pred, c_pred = self.forward_pass(x)

            if self.dependent_censoring:

                fc = torch.sum(y * c_pred[0], dim=1) + self.tol
                Fc = torch.sum(y * self._survival_from_density(c_pred[1]), dim=1) + self.tol

            else:

                fc = torch.sum(y * c_pred, dim=1) + self.tol
                Fc = torch.sum(y * self._survival_from_density(c_pred), dim=1) + self.tol

        else:

            e_pred, t_pred = self.forward_pass(x)

            fc = 1.
            Fc = 1.

        ft = torch.sum(y * t_pred, dim=1) + self.tol
        Ft = torch.sum(y * self._survival_from_density(t_pred), dim=1) + self.tol

        ll1 = torch.log(e_pred) + torch.log(ft) + torch.log(Fc)
        ll2 = torch.log(1 - e_pred * (1 - Ft)) + torch.log(fc)

        ll = s * ll1
        ll += (1 - s) * ll2

        return -1 * ll

    def mmd(self, x, s, beta=None):

        if beta is None:
            beta = get_median(torch.sum((x[:, None, :] - x[None, :, :]) ** 2, dim=-1))

        x0 = x[s == 0]
        x1 = x[s == 1]

        x0x0 = self._gaussian_kernel(x0, x0, beta)
        x0x1 = self._gaussian_kernel(x0, x1, beta)
        x1x1 = self._gaussian_kernel(x1, x1, beta)

        return torch.mean(x0x0) - 2. * torch.mean(x0x1) + torch.mean(x1x1)

    def _gaussian_kernel(self, x1, x2, beta=1.):
        return torch.exp(-1. * beta * torch.sum((x1[:, None, :] - x2[None, :, :]) ** 2, dim=-1))

    def _survival_from_density(self, f):
        return torch.flip(torch.cumsum(torch.flip(f, [1]), 1), [1])


def train_model(
        model, train_data, val_data, n_epochs,
        batch_size=50, learning_rate=1e-3, early_stopping_criterion=2,
        overwrite_output=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = np.inf
    no_decrease = 0

    for epoch_idx in range(n_epochs):

        train_losses = []
        train_nlls = []

        for batch_idx, (xt, yt, st) in enumerate(get_batches(*train_data, batch_size=batch_size)):
            optimizer.zero_grad()
            train_loss, train_nll = model.loss(xt, yt, st)

            # for param in model.parameters():
            #     train_loss += model.lr*param.norm(2)

            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss)
            train_nlls.append(train_nll)
        # Display metrics at the end of each epoch.

        val_losses = []
        val_nlls = []

        # Run a validation loop at the end of each epoch.
        for batch_idx, (xv, yv, sv) in enumerate(get_batches(*val_data, batch_size=batch_size)):
            val_loss, val_nll = model.loss(xv, yv, sv)

            val_losses.append(val_loss)
            val_nlls.append(val_nll)

        new_val_loss = torch.mean(torch.stack(val_losses))
        print(new_val_loss)
        if overwrite_output:
            print(
                'Epoch %2i | Train Loss: %.4f | Train NLL: %.4f | Val Loss: %.4f | Val NLL: %.4f'
                % (epoch_idx, torch.mean(torch.stack(train_losses)), torch.mean(torch.stack(train_nlls)), torch.mean(torch.stack(val_losses)), torch.mean(torch.stack(val_nlls))),
                end='\r'
            )

        else:
            print(
                'Epoch %2i | Train Loss: %.4f | Train NLL: %.4f | Val Loss: %.4f | Val NLL: %.4f'
                % (epoch_idx, torch.mean(torch.stack(train_losses)), torch.mean(torch.stack(train_nlls)), torch.mean(torch.stack(val_losses)), torch.mean(torch.stack(val_nlls)))
            )

        if new_val_loss > best_val_loss:
            no_decrease += 1
        else:
            no_decrease = 0
            best_val_loss = new_val_loss

        if no_decrease == early_stopping_criterion:
            break

    if overwrite_output:
        print(
            'Epoch %2i | Train Loss: %.4f | Train NLL: %.4f | Val Loss: %.4f | Val NLL: %.4f'
            % (epoch_idx, torch.mean(torch.stack(train_losses)), torch.mean(torch.stack(train_nlls)), torch.mean(torch.stack(val_losses)), torch.mean(torch.stack(val_nlls)))
        )
        print('')





def get_median(v):
    v = torch.reshape(v, (-1,))
    m = v.shape[0]//2
    return torch.min(torch.topk(v, m, sorted=False).values)

def mtlr_sum_of_squares(layer):
    kernel = layer.get_weights()[0]
    return torch.sum((kernel[:, 1:] - kernel[:, :-1]) ** 2)

def get_batches(*arrs, batch_size=1):
    l = len(arrs[0])
    for ndx in range(0, l, batch_size):
        yield (arr[ndx:min(ndx + batch_size, l)] for arr in arrs)
