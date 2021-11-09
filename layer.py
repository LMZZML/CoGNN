from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
import numpy as np


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()

class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho


class dy_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)
        self.lin2 = linear(c_in,c_in)


    def forward(self,x):
        #adj = adj + torch.eye(adj.size(0)).to(x.device)
        #d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj0)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)


        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2


class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        # self.kernel_set = [2,3,6]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x


# 第一次矩阵分割尝试
class graph_constructor(nn.Module):
    def __init__(self, l_matrix, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = sum(l_matrix)
        self.nmatrix = len(l_matrix)**2
        self.l_matrix = l_matrix

        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.lin1 = nn.ModuleList()
            self.lin2 = nn.ModuleList()
            self.emb1 = nn.ModuleList()
            self.emb2 = nn.ModuleList()
            for n_sub1 in l_matrix:
                for n_sub2 in l_matrix:
                    self.emb1.append(nn.Embedding(n_sub1, dim))
                    self.emb2.append(nn.Embedding(n_sub2, dim))
                    self.lin1.append(nn.Linear(dim,dim))      # 对应paper中的theta1
                    self.lin2.append(nn.Linear(dim,dim))

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        nodevec1 = []
        nodevec2 = []
        idx_list = []
        n_mod = len(self.l_matrix)
        for num in self.l_matrix:
            idx_list.append(torch.arange(num).to(self.device))
        if self.static_feat is None:
            for i in range(n_mod):
                for j in range(n_mod):
                    index = i*n_mod+j
                    idx1 = idx_list[i]
                    nodevec1.append(self.emb1[index](idx1))
                    idx2 = idx_list[j]
                    nodevec2.append(self.emb2[index](idx2))
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        adj_list = []
        for i in range(self.nmatrix):
            nodevec1[i] = torch.tanh(self.alpha*self.lin1[i](nodevec1[i]))
            nodevec2[i] = torch.tanh(self.alpha*self.lin2[i](nodevec2[i]))
            t1 = torch.mm(nodevec1[i], nodevec2[i].transpose(1,0))
            t2 = torch.mm(nodevec2[i], nodevec1[i].transpose(1,0))
            a = t1-t2.transpose(1,0)
            adj_list.append(F.relu(torch.tanh(self.alpha*a)))
        adj_row_list = []
        for i in range(n_mod):
            adj_row_list.append(torch.cat(adj_list[i*n_mod:(i+1)*n_mod], 1))
        adj = torch.cat(adj_row_list, 0)

        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


    def fullA(self, idx):
        nodevec1 = []
        nodevec2 = []
        idx_list = []
        n_mod = len(self.l_matrix)
        for num in self.l_matrix:
            idx_list.append(torch.arange(num).to(self.device))
        if self.static_feat is None:
            for i in range(n_mod):
                for j in range(n_mod):
                    index = i*n_mod+j
                    idx1 = idx_list[i]
                    nodevec1.append(self.emb1[index](idx1))
                    idx2 = idx_list[j]
                    nodevec2.append(self.emb2[index](idx2))
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        adj_list = []
        for i in range(self.nmatrix):
            nodevec1[i] = torch.tanh(self.alpha*self.lin1[i](nodevec1[i]))
            nodevec2[i] = torch.tanh(self.alpha*self.lin2[i](nodevec2[i]))
            t1 = torch.mm(nodevec1[i], nodevec2[i].transpose(1,0))
            t2 = torch.mm(nodevec2[i], nodevec1[i].transpose(1,0))
            a = t1-t2.transpose(1,0)
            adj_list.append(F.relu(torch.tanh(self.alpha*a)))
        # 矩阵拼接
        adj_row_list = []
        for i in range(n_mod):
            adj_row_list.append(torch.cat(adj_list[i*n_mod:(i+1)*n_mod], 1))
        adj = torch.cat(adj_row_list, 0)

        return adj


class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class graph_undirected_sep(nn.Module):
    def __init__(self, l_matrix, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected_sep, self).__init__()
        self.nnodes = sum(l_matrix)
        self.nmatrix = len(l_matrix)**2
        self.l_matrix = l_matrix
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.lin1 = nn.ModuleList()
            self.emb1 = nn.ModuleList()
            for n_sub1 in l_matrix:
                for n_sub2 in l_matrix:
                    self.emb1.append(nn.Embedding(n_sub1, dim))
                    self.lin1.append(nn.Linear(dim,dim))

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        nodevec1 = []
        nodevec2 = []
        idx_list = []
        n_mod = len(self.l_matrix)
        for num in self.l_matrix:
            idx_list.append(torch.arange(num).to(self.device))
        if self.static_feat is None:
            for i in range(n_mod):
                for j in range(n_mod):
                    index1 = i*n_mod+j
                    index2 = j*n_mod+i
                    idx1 = idx_list[i]
                    nodevec1.append(self.emb1[index1](idx1))
                    idx2 = idx_list[j]
                    nodevec2.append(self.emb1[index2](idx2))
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        adj_list = []
        for i in range(self.nmatrix):
            nodevec1[i] = torch.tanh(self.alpha*self.lin1[i](nodevec1[i]))
            nodevec2[i] = torch.tanh(self.alpha*self.lin1[i](nodevec2[i]))
            a = torch.mm(nodevec1[i], nodevec2[i].transpose(1,0))
            adj_list.append(F.relu(torch.tanh(self.alpha*a)))
        adj_row_list = []
        for i in range(n_mod):
            adj_row_list.append(torch.cat(adj_list[i*n_mod:(i+1)*n_mod], 1))
        adj = torch.cat(adj_row_list, 0)
        # k邻域
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class multigraph_undirected_sep(nn.Module):
    def __init__(self, pre_adj_list, k, dim, device, alpha=3, static_feat=None):
        super(multigraph_undirected_sep, self).__init__()
        self.pre_adj_list = pre_adj_list
        self.l_matrix = []
        for pre_adj in pre_adj_list:
            self.l_matrix.append(pre_adj.shape[0])
        self.nnodes = sum(self.l_matrix)
        self.nmatrix = len(self.l_matrix)**2
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.lin1 = nn.ModuleList()
            self.emb1 = nn.ModuleList()
            for n_sub1 in self.l_matrix:
                for n_sub2 in self.l_matrix:
                    self.emb1.append(nn.Embedding(n_sub1, dim))
                    self.lin1.append(nn.Linear(dim,dim))

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

        self.W = nn.ModuleList()
        for pre_adj in pre_adj_list:
            self.W.append(nn.Linear(pre_adj.shape[0], pre_adj.shape[0]))

    def forward(self, idx):
        nodevec1 = []
        nodevec2 = []
        idx_list = []
        n_mod = len(self.l_matrix)
        for num in self.l_matrix:
            idx_list.append(torch.arange(num).to(self.device))
        if self.static_feat is None:
            for i in range(n_mod):
                for j in range(n_mod):
                    index1 = i*n_mod+j
                    index2 = j*n_mod+i
                    idx1 = idx_list[i]
                    nodevec1.append(self.emb1[index1](idx1))
                    idx2 = idx_list[j]
                    nodevec2.append(self.emb1[index2](idx2))
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        adj_list = []
        for i in range(self.nmatrix):
            nodevec1[i] = torch.tanh(self.alpha*self.lin1[i](nodevec1[i]))
            nodevec2[i] = torch.tanh(self.alpha*self.lin1[i](nodevec2[i]))
            a = torch.mm(nodevec1[i], nodevec2[i].transpose(1,0))
            if a.shape[0] == a.shape[1]:
                index = int(i/len(self.pre_adj_list))
                a = a + self.W[index](self.pre_adj_list[index])
            adj_list.append(F.relu(torch.tanh(self.alpha*a)))
        adj_row_list = []
        for i in range(n_mod):
            adj_row_list.append(torch.cat(adj_list[i*n_mod:(i+1)*n_mod], 1))
        adj = torch.cat(adj_row_list, 0)
        # k邻域
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class graph_directed_sep(nn.Module):
    def __init__(self, l_matrix, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed_sep, self).__init__()
        self.nnodes = sum(l_matrix)
        self.nmatrix = len(l_matrix)**2
        self.l_matrix = l_matrix
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.lin1 = nn.ModuleList()
            self.lin2 = nn.ModuleList()
            self.emb1 = nn.ModuleList()
            self.emb2 = nn.ModuleList()
            for n_sub1 in l_matrix:
                for n_sub2 in l_matrix:
                    self.emb1.append(nn.Embedding(n_sub1, dim))
                    self.emb2.append(nn.Embedding(n_sub2, dim))
                    self.lin1.append(nn.Linear(dim,dim))
                    self.lin2.append(nn.Linear(dim,dim))

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        nodevec1 = []
        nodevec2 = []
        idx_list = []
        n_mod = len(self.l_matrix)
        for num in self.l_matrix:
            idx_list.append(torch.arange(num).to(self.device))
        if self.static_feat is None:
            for i in range(n_mod):
                for j in range(n_mod):
                    index = i*n_mod+j
                    idx1 = idx_list[i]
                    nodevec1.append(self.emb1[index](idx1))
                    idx2 = idx_list[j]
                    nodevec2.append(self.emb2[index](idx2))
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        adj_list = []
        for i in range(self.nmatrix):
            nodevec1[i] = torch.tanh(self.alpha*self.lin1[i](nodevec1[i]))
            nodevec2[i] = torch.tanh(self.alpha*self.lin2[i](nodevec2[i]))
            a = torch.mm(nodevec1[i], nodevec2[i].transpose(1,0))
            adj_list.append(F.relu(torch.tanh(self.alpha*a)))
        adj_row_list = []
        for i in range(n_mod):
            adj_row_list.append(torch.cat(adj_list[i*n_mod:(i+1)*n_mod], 1))
        adj = torch.cat(adj_row_list, 0)
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class graph_directed_sep_init(nn.Module):
    def __init__(self, l_matrix, k, dim, device, init_adj, alpha=3, static_feat=None):
        super(graph_directed_sep_init, self).__init__()
        self.nnodes = sum(l_matrix)
        self.nmatrix = len(l_matrix)**2
        self.l_matrix = l_matrix
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.lin1 = nn.ModuleList()
            self.lin2 = nn.ModuleList()
            self.emb1 = nn.ModuleList()
            self.emb2 = nn.ModuleList()

            cur_i = 0
            initemb1, initemb2 = list(), list()
            for num in l_matrix:
                m, p, n = torch.svd(init_adj[cur_i:cur_i+num,cur_i:cur_i+num])
                cur_i = cur_i + num
                initemb1.append(torch.mm(m[:, :dim], torch.diag(p[:dim] ** 0.5)))
                initemb2.append(torch.mm(torch.diag(p[:dim] ** 0.5), n[:, :dim].t()).t())

            for i, n_sub1 in enumerate(l_matrix):
                for j, n_sub2 in enumerate(l_matrix):
                    emb1_ = nn.Embedding(n_sub1, dim)
                    emb1_.weight.data.copy_(initemb1[i])
                    emb2_ = nn.Embedding(n_sub2, dim)
                    emb2_.weight.data.copy_(initemb2[j])
                    self.emb1.append(emb1_)
                    self.emb2.append(emb2_)
                    lin1_ = nn.Linear(dim,dim)
                    lin1_.weight.data = torch.eye(dim)
                    lin1_.bias.data = torch.zeros(dim)
                    lin2_ = nn.Linear(dim,dim)
                    lin2_.weight.data = torch.eye(dim)
                    lin2_.bias.data = torch.zeros(dim)
                    self.lin1.append(lin1_)
                    self.lin2.append(lin2_)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        nodevec1 = []
        nodevec2 = []
        idx_list = []
        n_mod = len(self.l_matrix)
        for num in self.l_matrix:
            idx_list.append(torch.arange(num).to(self.device))
        if self.static_feat is None:
            # embedding的初始化
            for i in range(n_mod):
                for j in range(n_mod):
                    index = i*n_mod+j
                    idx1 = idx_list[i]
                    nodevec1.append(self.emb1[index](idx1))
                    idx2 = idx_list[j]
                    nodevec2.append(self.emb2[index](idx2))
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        adj_list = []
        for i in range(self.nmatrix):
            nodevec1[i] = self.lin1[i](nodevec1[i])
            nodevec2[i] = self.lin2[i](nodevec2[i])
            a = torch.mm(nodevec1[i], nodevec2[i].transpose(1,0))
            adj_list.append(a)
        adj_row_list = []
        for i in range(n_mod):
            adj_row_list.append(torch.cat(adj_list[i*n_mod:(i+1)*n_mod], 1))
        adj = torch.cat(adj_row_list, 0)
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class graph_sep_init(nn.Module):
    def __init__(self, l_matrix, k, dim, device, init_adj, alpha=3, static_feat=None):
        super(graph_sep_init, self).__init__()
        self.nnodes = sum(l_matrix)
        self.nmatrix = len(l_matrix)**2
        self.l_matrix = l_matrix
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.w1 = nn.Linear(xd, dim)
            self.w2 = nn.Linear(xd, dim)
        else:
            self.w1 = nn.ParameterList()
            self.w2 = nn.ParameterList()
            self.b1 = nn.ParameterList()
            self.b2 = nn.ParameterList()
            self.emb1 = nn.ParameterList()
            self.emb2 = nn.ParameterList()

            cur_i = 0
            initemb1, initemb2 = list(), list()
            for num in l_matrix:
                m, p, n = torch.svd(init_adj[cur_i:cur_i+num,cur_i:cur_i+num])
                cur_i = cur_i + num
                initemb1.append(torch.mm(m[:, :dim], torch.diag(p[:dim] ** 0.5)))
                initemb2.append(torch.mm(torch.diag(p[:dim] ** 0.5), n[:, :dim].t()).t())

            for i, n_sub1 in enumerate(l_matrix):
                for j, n_sub2 in enumerate(l_matrix):
                    self.emb1.append(nn.Parameter(initemb1[i], requires_grad=True))
                    self.emb2.append(nn.Parameter(initemb2[j], requires_grad=True))
                    self.w1.append(nn.Parameter(torch.eye(dim), requires_grad=True))
                    self.w2.append(nn.Parameter(torch.eye(dim), requires_grad=True))
                    self.b1.append(nn.Parameter(torch.zeros(dim), requires_grad=True))
                    self.b2.append(nn.Parameter(torch.zeros(dim), requires_grad=True))

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.emb_test = nn.Embedding(l_matrix[0],dim)
        self.emb_test.weight.data.copy_(initemb1[0])

    def forward(self, idx):
        nodevec1 = []
        nodevec2 = []
        idx_list = []
        n_mod = len(self.l_matrix)
        for num in self.l_matrix:
            idx_list.append(torch.arange(num).to(self.device))
        if self.static_feat is None:
            for i in range(n_mod):
                for j in range(n_mod):
                    index = i*n_mod+j
                    nodevec1.append(self.emb1[index].float())
                    nodevec2.append(self.emb2[index].float())
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        adj_list = []
        for i in range(self.nmatrix):
            nodevec1[i] = nodevec1[i].mm(self.w1[i]) + self.b1[i].repeat(nodevec1[i].size(0), 1)
            nodevec2[i] = nodevec2[i].mm(self.w2[i]) + self.b2[i].repeat(nodevec2[i].size(0), 1)

            a = torch.mm(nodevec1[i], nodevec2[i].transpose(1,0))
            adj_list.append(F.relu(torch.tanh(self.alpha*a)))
        adj_row_list = []
        for i in range(n_mod):
            adj_row_list.append(torch.cat(adj_list[i*n_mod:(i+1)*n_mod], 1))
        adj = torch.cat(adj_row_list, 0)
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


