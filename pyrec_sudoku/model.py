import torch
import torch.nn as nn


class SudokuSolver(nn.Module):
    def __init__(self, constraint_mask, n=9, hidden1=512):
        super(SudokuSolver, self).__init__()
        self.constraint_mask = constraint_mask.view(1, n * n, 3, n * n, 1)
        self.n = n
        self.hidden1 = hidden1

        # Feature vector is the 3 constraints
        self.input_size = 3 * n

        self.l1 = nn.Linear(self.input_size,self.hidden1)

        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(self.hidden1,self.hidden1)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(self.hidden1,self.hidden1)
        self.a3 = nn.ReLU()
        self.l4 = nn.Linear(self.hidden1,self.hidden1)
        self.a4 = nn.ReLU()
        self.l5 = nn.Linear(self.hidden1,self.hidden1)
        self.a5 = nn.ReLU()
        self.l6 = nn.Linear(self.hidden1,self.hidden1)
        self.a6 = nn.ReLU()
        self.l7 = nn.Linear(self.hidden1,self.hidden1)
        self.a7 = nn.ReLU()
        self.l8 = nn.Linear(self.hidden1,self.hidden1)
        self.a8 = nn.ReLU()
        self.l9 = nn.Linear(self.hidden1, self.hidden1)
        self.a9 = nn.ReLU()

        self.l10 = nn.Linear(self.hidden1,n)


        self.softmax = nn.Softmax(dim=1)

    # X is a (batch, n^2, n) tensor
    def forward(self, x):
        n = self.n
        bts = x.shape[0]
        c = self.constraint_mask
        min_empty = (x.sum(dim=2) == 0).sum(dim=1).max()
        x_pred = x.clone()
        for a in range(min_empty):
            # Score empty numbers
            constraints = (x.view(bts, 1, 1, n * n, n) * c).sum(dim=3)
            # Empty cells
            empty_mask = (x.sum(dim=2) == 0)

            f = constraints.reshape(bts, n * n, 3 * n)
            y_ = self.a2(self.l2(self.a1(self.l1(f[empty_mask]))))
            y_= self.a4(self.l4(self.a3(self.l3(y_))))
            y_= self.a6(self.l6(self.a5(self.l5(y_))))
            y_= self.a8(self.l8(self.a7(self.l7(y_))))
            y_= self.l10(self.a9(self.l9(y_)))
            s_ = self.softmax(y_)

            # Score the rows
            x_pred[empty_mask] = s_

            s = torch.zeros_like(x_pred).cuda()
            s[empty_mask] = s_
            # Find most probable guess
            score, score_pos = s.max(dim=2)
            mmax = score.max(dim=1)[1]
            # Fill it in
            nz = empty_mask.sum(dim=1).nonzero().view(-1)
            mmax_ = mmax[nz]
            ones = torch.ones(nz.shape[0]).cuda()
            x.index_put_((nz, mmax_, score_pos[nz, mmax_]), ones)
        return x_pred, x

