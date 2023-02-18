import sys
import numpy as np
import json
import multiprocessing
import time
from scipy import linalg
import matplotlib.pyplot as plt

import torch # テンソル計算など
import torch.nn as nn  # ネットワーク構築用
import torch.optim as optim  # 最適化関数


class TSSC(nn.Module):
    def __init__(self, M, N, W, md, ml, K, mod, Niter, AorF, v_com):
        super(TSSC, self).__init__()
        #SSC
        if v_com == 1:
            non0_c_ = -20*torch.ones([2*md, M*md])
        else:
            non0_c_ = -20*torch.ones([1*md, M*md])
        non0_c_[0, 0::2] = -2.2
        if md >= 2:
            for idx_md in range(1, md):
                non0_c_[idx_md, 1::2] = -2.2
        self.non0_c = nn.Parameter(non0_c_)

        self.M = M
        self.N = N
        self.W = W
        self.md = md
        self.ml = ml
        self.K = K

        #GaBP
        self.MM = 2*md*M
        if AorF == 1:
            self.NN = 2*N
        elif AorF == 0:
            self.NN = 2*N*M
        self.mod = mod

        v=4
        temp = -0.5 * np.ones(self.NN)
        for idx in range(0, int(self.NN/v)):
            temp[idx*v]=0.5

        temp_ = np.zeros((self.NN,Niter))
        for idx in range(0, Niter):
            temp_[:,idx] = np.roll(temp,idx)
        self.eta = nn.Parameter(torch.from_numpy(temp_))
        self.mu = nn.Parameter(torch.from_numpy(2*(np.arange(0,Niter,1)+1)/Niter))
#        self.eta = torch.from_numpy(temp_)
#        self.mu = torch.from_numpy(2*(np.arange(0,Niter,1)+1)/Niter)

    #SSC
    def sp_tanh(self, ele):
        return 10*torch.tanh(0.1*ele)
    
    def v_sigmoid(self,sp_ele):
#        return 1 + 1/(1 + torch.exp(sp_ele))
        return 10/(1 + torch.exp(-1.0*sp_ele))

    def SSC_code(self, x_com, non0_c, x, codebook, v_com, N0_, AorF):
        if v_com == 1:
            for idx_x in range(self.M):
                if idx_x == 0:
                    c_mtx_re = self.v_sigmoid(non0_c[:self.md, self.md*idx_x:self.md*(idx_x+1)])
                    c_mtx_im = self.v_sigmoid(non0_c[self.md:, self.md*idx_x:self.md*(idx_x+1)])

                    if AorF == 1:   # fading
                        fad_re = torch.randn(self.M, self.N)/torch.sqrt(2*torch.ones(self.M, self.N))#*torch.sqrt([1/2])
                        fad_im = torch.randn(self.M, self.N)/torch.sqrt(2*torch.ones(self.M, self.N))#*torch.sqrt([1/2])
                        fading_re = torch.diag(fad_re[0, :])
                        fading_im = torch.diag(fad_im[0, :])
                else:
                    c_mtx_re = torch.block_diag(c_mtx_re, self.v_sigmoid(non0_c[:self.md, self.md*idx_x:self.md*(idx_x+1)]))
                    c_mtx_im = torch.block_diag(c_mtx_im, self.v_sigmoid(non0_c[self.md:, self.md*idx_x:self.md*(idx_x+1)]))

                    if AorF == 1:   # fading
                        fad_i_re = torch.diag(fad_re[idx_x, :])
                        fad_i_im = torch.diag(fad_im[idx_x, :])
                        fading_re = torch.cat((fading_re, fad_i_re), 1)
                        fading_im = torch.cat((fading_im, fad_i_im), 1)

            c_mtx_row1 = torch.cat((c_mtx_re, -1*c_mtx_im), 1)
            c_mtx_row2 = torch.cat((c_mtx_im, c_mtx_re), 1)
            c_mtx = torch.cat((c_mtx_row1, c_mtx_row2), 0)
            codebook_ac = torch.mm(codebook, c_mtx)

            if AorF == 1:   # fading
                fading_row1 = torch.cat((fading_re, -1*fading_im), 1)
                fading_row2 = torch.cat((fading_im, fading_re), 1)
                fading = torch.cat((fading_row1, fading_row2), 0)         
        else:
            for idx_x in range(self.M):
                if idx_x == 0:
                    c_mtx_re = self.v_sigmoid(non0_c[:, self.md*idx_x:self.md*(idx_x+1)])

                    if AorF == 1:   # fading
                        fad_re = torch.randn(self.M, self.N)/torch.sqrt(2*torch.ones(self.M, self.N))#*torch.sqrt([1/2])
                        fad_im = torch.randn(self.M, self.N)/torch.sqrt(2*torch.ones(self.M, self.N))#*torch.sqrt([1/2])
                        fading_re = torch.diag(fad_re[0, :])
                        fading_im = torch.diag(fad_im[0, :])

                else:
                    c_mtx_re = torch.block_diag(c_mtx_re, self.v_sigmoid(non0_c[:, self.md*idx_x:self.md*(idx_x+1)]))

                    if AorF == 1:   # fading
                        fad_i_re = torch.diag(fad_re[idx_x, :])
                        fad_i_im = torch.diag(fad_im[idx_x, :])
                        fading_re = torch.cat((fading_re, fad_i_re), 1)
                        fading_im = torch.cat((fading_im, fad_i_im), 1)
                        
            c_mtx = torch.block_diag(c_mtx_re, c_mtx_re)
            codebook_ac = torch.mm(codebook, c_mtx)

            if AorF == 1:   # fading
                fading_row1 = torch.cat((fading_re, -1*fading_im), 1)
                fading_row2 = torch.cat((fading_im, fading_re), 1)
                fading = torch.cat((fading_row1, fading_row2), 0)
            
        x_code_ = torch.mm(codebook_ac, x)
        norm = torch.sqrt((x_code_.norm(dim=1)**2).sum()/self.K)
        x_code = x_code_/norm
        codebook = codebook_ac/norm

        x_ene = torch.sum(torch.sum(torch.abs(x_code)**2, 0)) / (self.K * self.M * self.md *self.ml)

        N0 = N0_ * x_ene.float()
        if AorF == 1:
            z_re = torch.randn(self.N, self.K)*torch.sqrt(N0/2)
            z_im = torch.randn(self.N, self.K)*torch.sqrt(N0/2)
        elif AorF == 0:
            z_re = torch.randn(self.N*self.M, self.K)*torch.sqrt(N0/2)
            z_im = torch.randn(self.N*self.M, self.K)*torch.sqrt(N0/2)

        # equivalent channel
        H = torch.mm(fading.float(), codebook.float())

        z = torch.cat((z_re, z_im), 0)

        y = torch.mm(H, x) + z

        return x.T.float(), y.T.float(), H.float(), N0

    def RX(self, x_code, codebook, x_ene, x, N0_, AorF):
        N0 = N0_ * x_ene

        if AorF == 1:   # fading
            fad_re = torch.randn(self.M, self.N)/torch.sqrt(2*torch.ones(self.M, self.N))#*torch.sqrt([1/2])
            fad_im = torch.randn(self.M, self.N)/torch.sqrt(2*torch.ones(self.M, self.N))#*torch.sqrt([1/2])
            fading_re = torch.diag(fad_re[0, :])
            fading_im = torch.diag(fad_im[0, :])
            for fad_i in range(1, self.M):
                fad_i_re = torch.diag(fad_re[fad_i, :])
                fad_i_im = torch.diag(fad_im[fad_i, :])
                fading_re = torch.cat((fading_re, fad_i_re), 1)
                fading_im = torch.cat((fading_im, fad_i_im), 1)
            fading_row1 = torch.cat((fading_re, -1*fading_im), 1)
            fading_row2 = torch.cat((fading_im, fading_re), 1)
            fading = torch.cat((fading_row1, fading_row2), 0) 

            z_re = torch.randn(self.N, self.K)*torch.sqrt(N0/2)
            z_im = torch.randn(self.N, self.K)*torch.sqrt(N0/2)

        elif AorF == 0: # AWGN
            fading = torch.eye(2*self.N*self.M)

            z_re = torch.randn(self.N*self.M, self.K)*torch.sqrt(N0/2)
            z_im = torch.randn(self.N*self.M, self.K)*torch.sqrt(N0/2)
        
        # equivalent channel
        H = torch.mm(fading.float(), codebook)
#        N0 = N0_ * x_ene
#        z_re = torch.randn(self.N, self.K)*torch.sqrt(N0/2)
#        z_im = torch.randn(self.N, self.K)*torch.sqrt(N0/2)
        z = torch.cat((z_re, z_im), 0)

        y = torch.mm(H, x) + z
#        y = torch.mm(fading.float(), x_code) + z

        return x.T.float(), y.T.float(), H.float(), N0  #, zz.float()

    #GaBP
    def sigmoid(self,eta):
        return 1/(1 + torch.exp(-4.0*eta))

    def SC(self, y, H, SR_mat, ER_mat):  # Soft Canceller
        # SC
        Reconstruct_matrix = H.T * SR_mat   #Reconstruct_matrix = c.mm(H.T, SR_mat)
        y_tilde = y - torch.sum(Reconstruct_matrix, axis=0) + Reconstruct_matrix
        delta = ER_mat - SR_mat ** 2
        return y_tilde, delta

    def TBG(self, H, HH, N0, y_tilde, delta, uu, vv, eta, x, ER_mat):  # BG
        element = HH * delta

        psi = (torch.sum(element, axis=0).reshape(1, -1) - element) + N0 / 2.0
        if self.ml == 1:
            u = 2 * torch.sqrt(ER_mat) * H.T * y_tilde / psi
            v = 2 * torch.sqrt(ER_mat) * HH / psi
        else:
            u = H.T * y_tilde / psi
            v = HH / psi

        uu = self.sigmoid(eta) * u + (1 - self.sigmoid(eta)) * uu

        s = (torch.sum(uu, axis=1) - uu.transpose(0, 1)).transpose(0, 1)

#        v = 2 * torch.sqrt(ER_mat) * HH / psi
        vv = self.sigmoid(eta) * v + (1 - self.sigmoid(eta)) * vv

        omega = (torch.sum(vv, axis=1) - vv.transpose(0, 1)).transpose(0, 1)

        if self.ml == 1:
            gamma = s / (omega * torch.sqrt(ER_mat))
        else:
            gamma = s / omega
        gamma_post = torch.sum(u, axis=1) / torch.sum(v, axis=1)
        return gamma, gamma_post, uu, vv    #, u_

    def BG(self, H, HH, N0, y_tilde, delta, uu, vv, eta, x, ER_mat, ns_num):  # BG
        element = HH * delta

        psi = (torch.sum(element, axis=0).reshape(1, -1) - element) + N0 / 2.0
        if self.ml == 1:
            u = 2 * torch.sqrt(ER_mat) * H.T * y_tilde / psi
            v = 2 * torch.sqrt(ER_mat) * HH / psi
        else:
            u = H.T * y_tilde / psi
            v = HH / psi

#        uu = self.sigmoid(eta) * u + (1 - self.sigmoid(eta)) * uu
        uu[:, ns_num::4] = u[:, ns_num::4]

        s = (torch.sum(uu, axis=1) - uu.transpose(0, 1)).transpose(0, 1)
#        s = eta_n * s_ + (1 - eta_n) * s

#        v = 2 * torch.sqrt(ER_mat) * HH / psi
#        vv = self.sigmoid(eta) * v + (1 - self.sigmoid(eta)) * vv
        vv[:, ns_num::4] = v[:, ns_num::4]

        omega = (torch.sum(vv, axis=1) - vv.transpose(0, 1)).transpose(0, 1)
#        omega = eta_n * omega_ + (1 - eta_n) * omega

        if self.ml == 1:
            gamma = s / (omega * torch.sqrt(ER_mat))
        else:
            gamma = s / (omega + 1e-30)
        gamma_post = torch.sum(u, axis=1) / torch.sum(v, axis=1)
        return gamma, gamma_post, uu, vv    #, u_

    def RG(self, gamma, mu, ER_mat):  # RG
        if self.ml == 1:
            ER_mat_1 = torch.sqrt(ER_mat)
#        ER_mat_1[((self.MM+2-1)//2):, :] = 0 # ((self.M+2-1)//2)(self.M//2)
            ER_mat_1[(self.M*self.md):, :] = 0
            SR_mat = torch.tanh(mu * gamma) * ER_mat_1
        elif self.ml == 2:
            SR_mat = torch.tanh(mu * gamma) * self.mod.norm
            ER_mat = torch.ones((self.MM, self.NN)).float() / 2.0
        else:
            SR_mat = torch.zeros((self.MM, self.NN))
            ER_mat = torch.zeros((self.MM, self.NN))
            for gamma_ in self.mod.lay:
                temp = mu * (gamma - gamma_) / self.mod.norm
                SR_mat += torch.tanh(temp)
                ER_mat += gamma_ * torch.tanh(temp)
            SR_mat *= self.mod.norm
            ER_mat *= 2 * self.mod.norm
            ER_mat += self.mod.Esmax / 2
        return SR_mat, ER_mat

    def SD(self, gamma_post, mu, ER_mat):
        if self.ml == 1:
#        gamma_post[((self.MM+2-1)//2):] = 0
            gamma_post[(self.M*self.md):] = 0
#       SD_mat = gamma_post
            SD_mat = torch.tanh(mu * gamma_post) # * torch.t(ER_mat)
        elif self.ml == 2:
            SD_mat = torch.tanh(mu * gamma_post) * self.mod.norm
        else:
            SD_mat = torch.zeros(self.MM)
            for gamma_ in self.mod.lay:
                temp = mu*(gamma_post - gamma_) / self.mod.norm
                SD_mat += torch.tanh(temp)
            SD_mat *= self.mod.norm
        return SD_mat

    def forward(self, x_com, x, N0_, mod, Niter, AorF, TBPorBP, codebook, v_com): #x, y, H, N0, sp_mtx = model(dictionary_mtx, x_0, N0_)

        x_1, y, H, N0 = TSSC.SSC_code(self, x_com, self.non0_c, x, codebook, v_com, N0_, AorF)

        K, NN = y.shape
        HH = (H * H).T

        x_ = torch.zeros((K, self.MM))
        x_hat = torch.zeros((K, self.MM))
#        llr_ = np.zeros((K, self.MM*mod.ml))
#        x = x.float()
#        y = y.float()
#        H = H.float()

        lam = torch.zeros((K, self.MM))

        for idx_sym in range(0, K):
            SR_mat = torch.zeros((self.MM, self.NN)).float()
#            SR_mat = x.repeat(1, self.NN)
            if self.ml == 1:
                ER_mat = torch.ones((self.MM, self.NN)).float()
            else:
                ER_mat = torch.ones((self.MM, self.NN)).float() / 2.0

            # # Perfect priori
            # SR_mat = np.tile(x[idx_sym, :], (N, 1)).T
            # ER_mat = SR_mat ** 2
            uu = torch.zeros((self.MM, self.NN))
            vv = torch.zeros((self.MM, self.NN))
            if TBPorBP == 1:
                for idx_iter in range(0, Niter):
                    # SC
                    y_tilde, delta = TSSC.SC(self, y[idx_sym, :], H, SR_mat, ER_mat)
                    # BG
                    gamma, gamma_post, uu, vv = TSSC.TBG(self, H, HH, N0, y_tilde, delta, uu, vv, self.eta[:, idx_iter], x[:, idx_sym], ER_mat)
                    # RG
                    SR_mat, ER_mat = TSSC.RG(self, gamma, self.mu[idx_iter], ER_mat)
            else:
                for idx_iter in range(0, Niter):
                    for ns_num in range(0, 4):
                        # SC
                        y_tilde, delta = TSSC.SC(self, y[idx_sym, :], H, SR_mat, ER_mat)
                        # BG
                        gamma, gamma_post, uu, vv = TSSC.BG(self, H, HH, N0, y_tilde, delta, uu, vv, self.eta[:, idx_iter], x[:, idx_sym], ER_mat, ns_num)
                        # RG
                        SR_mat, ER_mat = TSSC.RG(self, gamma, self.mu[idx_iter], ER_mat)
            # Output
            x_[idx_sym, :] = TSSC.SD(self, gamma_post, self.mu[idx_iter], ER_mat)
            x_hat[idx_sym, :] = gamma_post

            lam[idx_sym, :] = gamma_post

#        x_ = TSSC.GaBP_main(self, x, y, H, N0, mod, Niter)
        return x_, lam

class Customloss(nn.Module):
    def __init__(self):
        super(Customloss, self).__init__()
        self.k1 = 36/(8*np.sqrt(3)-9)
        self.k2 = 24/(16*np.sqrt(3)-27)

    def forward(self, x, x_hat, x_tch, loss_w):
        mse = torch.mean((x_tch - x) ** 2)
        
        x_bar = (x_hat - torch.mean(x_hat)) / torch.std(x_hat, unbiased=False)
        x1 = torch.mean(x_bar * torch.exp(-(x_bar**2)/2))
        x2 = torch.mean(torch.exp(-(x_bar**2)/2)) - 1/np.sqrt(2)

        neg = self.k1 * x1 ** 2 + self.k2 * x2 ** 2

        loss = loss_w * neg + (1 - loss_w) * mse

        return loss


class MOD():
    def __init__(self, ml, md):
        self.md = md
        self.ml = ml
        self.nsym = 2 ** ml

    def demodulation(self, y):
        b_tmp = np.empty((y.shape[1]*self.md, y.shape[0]), int)
        for idx_k in range(0, y.shape[0]):
            for idx_m in range(0, y.shape[1]):
                b_tmp[idx_m, idx_k] = np.signbit(y[idx_k, idx_m])
        return b_tmp


def gen_minibatch(M, md, K, mod, ml):
    if ml == 1:
        b = torch.randint(0, 2, (M * md, K))
        x_com = 2.0 * b - 1.0
        x = torch.cat((x_com, torch.zeros([M * md, K])), 0)
    elif ml == 2:
        b = torch.randint(0, 2, (M * md * ml, K))
        x_com = 2.0 * b - 1.0
        x =  (2.0 * b - 1.0) / np.sqrt(2)
    else:
        b = np.random.randint(0, 2, (M * md * mod.ml, K))
        a = np.dot(np.kron(mod.lv, np.eye(M * md, dtype=int)), b)
        # TX symbol
        x_com = np.array(mod.val[mod.amap[a]])
        x = torch.from_numpy(np.concatenate([x.real, x.imag])).transpose(0, 1)
    return b.T, x_com, x

def Dictionary_hada(M, N, W, md):
    dic_mtx = np.zeros([N, M*md])
    hada_ = (1/np.sqrt(md))*linalg.hadamard(W*md)
    hada = hada_[1:, :]

    dic_idx_ = range(W*md-1)
    dic_idx_1_ = range(md, W*md)
    for idx_sec in range(M):
        dic_idx = np.random.permutation(dic_idx_)
        dic_idx_1 = np.random.permutation(dic_idx_1_)
        for idx_sec_2 in range(N):
            if idx_sec_2 == 0:
                dic_mtx_ = hada[dic_idx[idx_sec_2], dic_idx_1[idx_sec]-md:dic_idx_1[idx_sec]]
            else:
                dic_mtx_i = hada[dic_idx[idx_sec_2], dic_idx_1[idx_sec]-md:dic_idx_1[idx_sec]]
                dic_mtx_ = np.append(dic_mtx_, dic_mtx_i)
        dic_mtx[:, md*idx_sec:md*(idx_sec+1)] = dic_mtx_.reshape(N, md)
    return dic_mtx
#    return torch.from_numpy(dic_mtx)

    
def Dictionary_rand(M, N, W, md):
    dic_mtx_re = np.zeros([N*M, W*md])
    dic_mtx_im = np.zeros([N*M, W*md])
    for idx_sec in range(M):
        dic_mtx_re[idx_sec*N:(idx_sec+1)*N, :] = (np.random.randn((N, W*md)) + 1j * np.random.randn((N, W*md)))/np.sqrt(2*2)
    return linalg.block_diag(dic_mtx_re, dic_mtx_im)


def codebook_mtx_gen(M, N, md, dic_ri, codebook_):
    if dic_ri != 0:
        codebook_re__ = codebook_[:N, :]
        codebook_im__ = codebook_[N:, :]
    for idx_x in range(M):
        if idx_x == 0:
            if dic_ri == 0:
                codebook_re_ = codebook_[:, md*idx_x:md*(idx_x+1)]
            else:
                codebook_re_ = codebook_re__[:, md*idx_x:md*(idx_x+1)]
                codebook_im_ = codebook_im__[:, md*idx_x:md*(idx_x+1)]
        else:
            if dic_ri == 0:
                codebook_re_ = torch.block_diag(codebook_re_, codebook_[:, md*idx_x:md*(idx_x+1)])
            else:
                codebook_re_ = torch.block_diag(codebook_re_, codebook_re__[:, md*idx_x:md*(idx_x+1)])
                codebook_im_ = torch.block_diag(codebook_im_, codebook_im__[:, md*idx_x:md*(idx_x+1)])
    if dic_ri == 0:
        codebook_re = torch.block_diag(codebook_re_, codebook_re_)
    else:
        codebook_re_row1 = torch.cat((codebook_re_, -1*codebook_im_), 1)
        codebook_re_row2 = torch.cat((codebook_im_, codebook_re_), 1)
        codebook_re = torch.cat((codebook_re_row1, codebook_re_row2), 0)
    return codebook_re


def position_non0(B_, k, num_hot):
    if num_hot == 1:
        pos = []
        while len(pos) < k:
            n = np.random.randint(0, B_)
            if not n in pos:
                pos.append(n)
    else:
        pos = np.random.randint(0, B_, (num_hot, 1))
        while len(pos[0, :]) < k:
            n = np.random.randint(0, B_, (num_hot, 1))
            if not n in pos:
                pos = np.concatenate([pos, n], 1)
    return pos

def spread_mtx_gen(B_, M_, md, pos, num_hot):
    spread_mtx_re_ = np.zeros([B_*md, M_*md])
    for spm_row in range(M_*md):
        if num_hot == 1:
            spread_mtx_re_[B_*(spm_row % 2) + pos[spm_row], spm_row] = 1
        else:
            for spm_col in range(num_hot):
                spread_mtx_re_[B_*(spm_row % 2) + pos[spm_col, spm_row], spm_row] = 1
    return spread_mtx_re_

def train(params):
    rng = np.random.RandomState(params[0])
    torch.manual_seed(params[0])

#    Niter = 4  # GaBPの反復回数
    adam_lr = 0.005  # Adamの学習率

    loss_w = 0.6    # loss weight

    # train_or_test tx rx dic_size mbs md ml loop num_core En_st delta En_en
    method = params[1][1]
    EsN0 = range(int(params[1][10]),int(params[1][12])+int(params[1][11]),int(params[1][11]))
    M_ = int(params[1][2])
    N_ = int(params[1][3])
    B_ = int(params[1][4])
    K  = int(params[1][5])
    md = int(params[1][6])
    ml = int(params[1][7])
    wloop = int(params[1][8])
    nloop = int(2*(10**wloop))
    AorF = int(params[1][13])
    TBPorBP = int(params[1][14]) # 1:TGaBP or 0:GaBP
    dic_i_h = 1 # 0:iid 1:hada
    num_hot = 1
    v_com = 0    # 0:real 1:complex
    dic_ri = 1  # dic 0:real only 1:complex

    if TBPorBP == 1:
        Niter = 32  # 反復回数
    else:
        Niter = 8

    mod = MOD(ml, md)
    model = TSSC(M_, N_, B_, md, ml, K, mod, Niter, AorF, v_com)   # model = TGaBP(M_, N_, mod, Niter)
    opt   = optim.Adam(model.parameters(), lr=adam_lr)
    # MSE
    loss_func = nn.MSELoss()
    # neg + MSE
#    loss_func = Customloss()

    N0_ = 10.0 ** (-EsN0[params[0]] / 10.0)
    # dictionary matrix
    if dic_ri == 0:
        if dic_i_h == 1:
            dic_mtx_1 = Dictionary_hada(M_, N_, B_, md)
        else:
            dic_mtx_1 = Dictionary_rand(M_, N_, B_, md)
        codebook_ = torch.from_numpy(dic_mtx_1).float()
    else:
        for idx_d in range(2):
            if dic_i_h == 1:
                dic_mtx_2_ = Dictionary_hada(M_, N_, B_, md)
            else:
                dic_mtx_2_ = Dictionary_rand(M_, N_, B_, md)
            if idx_d == 0:
                dic_mtx_1 = dic_mtx_2_
            else:
                dic_mtx_1 = np.concatenate([dic_mtx_1, dic_mtx_2_], 0)
        codebook_ = torch.from_numpy(dic_mtx_1).float()

    fn_dic = 'DATA/dic_'
    fn_dic += params[1][2] + '_'
    fn_dic += params[1][3] + '_'
    fn_dic += params[1][4] + '_'
    fn_dic += params[1][6] + '_'
    fn_dic += str(EsN0[params[0]])
    np.save(fn_dic, dic_mtx_1)
    fn_dic_csv = fn_dic + '.csv'
    np.savetxt(fn_dic_csv, dic_mtx_1, delimiter=',')

    codebook = codebook_mtx_gen(M_, N_, md, dic_ri, codebook_)

    for idx_loop in range(0, nloop):
        b, x_com, x = gen_minibatch(M_, md, K, mod, ml)
        opt.zero_grad()
        x_, lam = model(x_com, x, N0_, mod, Niter, AorF, TBPorBP, codebook, v_com)

        # MSE
        loss  = loss_func(x_, x.T)
        loss.backward()  # 誤差逆伝播法(後ろ向き計算の実行)
#        opt.step()       # 学習可能パラメータの更新
        # print(gen, loss.item())
        if ((idx_loop % 100) == 0):
            print(idx_loop, EsN0[params[0]], loss.item())
            
            SIM_dict_loss = {'EsN0':EsN0[params[0]], 'loop':0, 'loss':1.0}
            SIM_dict_loss['loop'] = idx_loop
            SIM_dict_loss['loss'] = loss.item()
            fn_loss = 'DATA/TSSC_sp_loss_'
            if AorF == 1:
                fn_loss += 'fade_'
            else:
                fn_loss += 'AWGN_' 
            if TBPorBP == 1:
                fn_loss += 'TGaBP_'
            else:
                fn_loss += 'GaBP_'
            fn_loss += str(nloop) + '.json'
            f_out_loss = open(fn_loss, 'a')
            json.dump(SIM_dict_loss, f_out_loss)
            f_out_loss.write("\n")
            f_out_loss.close()
#        if ((idx_loop % 200) == 0):
#            fn_TSSC = 'DATA/TSSC_sp_mtx'
#            fn_TSSC += str(EsN0[params[0]])
#            fn_TSSC += str(idx_loop) + '.csv'
#            np.savetxt(fn_TSSC, sp_mtx, delimiter=',')
        del loss
        torch.cuda.empty_cache()
        opt.step()       # 学習可能パラメータの更新

    model_path = 'SSC_model/TSSC_sp_'
    model_path += params[1][2] + '_'    # M
    model_path += params[1][3] + '_'    # N
    model_path += params[1][4] + '_'    # B
#    model_path += params[1][5] + '_'    # K
    model_path += params[1][6] + '_'    # md
#    model_path += params[1][8] + '_'    # wloop
    model_path += params[1][13] + '_'    # 1:fading 0:AWGN
    model_path += params[1][14] + '_'    # 1:TGaBP 0:GaBP
    model_path += str(EsN0[params[0]]) + '.pth'
    torch.save(model.to('cpu').state_dict(), model_path)


    fn_tssc = 'DATA/c_mtx_'
    fn_tssc += params[1][2] + '_'
    fn_tssc += params[1][2] + '_'
    fn_tssc += params[1][3] + '_'
    fn_tssc += params[1][4] + '_'
    fn_tssc += params[1][6] + '_'
    fn_tssc += str(EsN0[params[0]]) + '.csv'
    np.savetxt(fn_tssc, model.v_sigmoid(model.non0_c).detach().numpy(), delimiter=',')
    c = plt.pcolor(model.v_sigmoid(model.non0_c).detach().numpy(), cmap='RdBu')

    plt.xlabel('TX')
    plt.ylabel(r'$section\,size$')
    plt.colorbar(c)

    plt.savefig('FIG/TSSC_c_mtx_' + str(EsN0[params[0]]) + '_' + str(params[1][2]) + '_' + str(params[1][3]) + '_' + str(params[1][4]) + '_' + str(params[1][6]) + '_' + str(params[1][13]) + '.eps')


def main_task(params):
    rng = np.random.RandomState(params[0])
    torch.manual_seed(params[0])
    
    method = params[1][1]
    EsN0 = range(int(params[1][10]),int(params[1][12])+int(params[1][11]),int(params[1][11]))
    M_ = int(params[1][2])
    N_ = int(params[1][3])
    B_ = int(params[1][4])
    K  = int(params[1][5])
    md = int(params[1][6])
    ml = int(params[1][7])
    wloop = int(params[1][8])
    nproc = int(params[1][9])
    nloop = int(np.ceil((5*(10**wloop))/nproc))
    AorF = int(params[1][13])
    TBPorBP = int(params[1][14]) # 1:TGaBP or 0:GaBP
    dic_i_h = 1 # 0:iid 1:hada
    noe   = np.zeros((2,len(EsN0)),dtype = int)
    num_hot = 1
    v_com = 0    # 0:real 1:complex
    dic_ri = 1  # dic 0:real only 1:complex

    if TBPorBP == 1:
        Niter = 32  # 反復回数
    else:
        Niter = 8

    # load ssc codebook
    fn_dic = 'DATA/dic_'
    fn_dic += params[1][2] + '_'
    fn_dic += params[1][3] + '_'
    fn_dic += params[1][4] + '_'
    fn_dic += params[1][6] + '_'
    fn_dic += str(16) + '.npy'
    codebook_av = np.load(file=fn_dic)
    codebook_ = torch.from_numpy(codebook_av).float()
    codebook = codebook_mtx_gen(M_, N_, md, dic_ri, codebook_)


    mod = MOD(ml, md)
    model = TSSC(M_, N_, B_, md, ml, K, mod, Niter, AorF, v_com)   # model = TGaBP(M_, N_, mod, Niter)

    model_path = 'SSC_model/TSSC_sp_'
    model_path += params[1][2] + '_'
    model_path += params[1][3] + '_'
    model_path += params[1][4] + '_'
#    model_path += params[1][5] + '_'
    model_path += params[1][6] + '_'
#    model_path += params[1][8] + '_'
    model_path += params[1][13] + '_'    # 1:fading 0:AWGN
    model_path += params[1][14] + '_'    # 1:TGaBP 0:GaBP
    model_path += str(16) + '.pth'
    model.load_state_dict(torch.load(model_path))


    with torch.no_grad():
        for idx_En in range (0, len(EsN0)):
            N0_ = 10.0 ** (-EsN0[idx_En] / 10.0)
            for idx_loop in range (0, nloop):
                # Fading generation
                b, x_com, x = gen_minibatch(M_, md, K, mod, ml)
#                if dic_i_h == 1:
#                    dic_hada = dic_hada = Dictionary_mtx(M_, N_, B_, md)
#                else:
#                    dic_hada = 1
                x_, lam = model(x_com, x, N0_, mod, Niter, AorF, TBPorBP, codebook, v_com)
                if ml == 1:
                    b_1 = x_[:, :M_*md] > 0
                elif ml == 2:
                    b_1 = x_ > 0
                b_r = b_1.to('cpu').detach().numpy().copy()
                b_t = b.to('cpu').detach().numpy().copy()
                tmp_ = np.abs(b_t - b_r)
                tmp = tmp_.sum()

                noe[0,idx_En] += tmp.sum()
                noe[1,idx_En] += (md*M_*ml*K)
                if noe[0,idx_En] > (md*ml*M_*K*nloop)*0.01:
                    break
            print(params[0], EsN0[idx_En], noe[0,idx_En]/noe[1,idx_En])
    return noe


def resut2f(argvs,BER):
    EsN0 = range(int(argvs[10]),int(argvs[12])+int(argvs[11]),int(argvs[11]))
    for idx_En in range (0, len(EsN0)):
        SIM_dict = {'Method':argvs[1], 'M':argvs[2], 'N':argvs[3], 'B':argvs[4], 'K':argvs[5],'md':argvs[6], 'ml':argvs[7], 'wloop':argvs[8], 'AorF':argvs[13], 'TBPorBP':argvs[14], 'EbN0':0.0,'BER':0.0}
        SIM_dict['EbN0'] = EsN0[idx_En]
        SIM_dict['BER'] = BER[idx_En]
        fn = 'DATA/TSSC.json'
        f_out = open(fn, 'a')
        json.dump(SIM_dict, f_out)
        f_out.write("\n")
        f_out.close()


if __name__ == '__main__':
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    argc = len(argvs) # 引数の個数
    if (argc != 15):  # 引数が足りない場合
        print('Usage: # TSSC.py Method M N B K md ml wloop nproc En1 delta En2 0AWGNor1Fade TGaBP')
        quit()        # プログラムの終了

    start = time.time()
    params = [(i, argvs) for i in range (0, int(argvs[9]))]

    if int(argvs[1]) == 0:
        # Training
        if int(argvs[9]) == 1:
            train((0, argvs))
        else:
            pool = multiprocessing.Pool(processes=int(argvs[9]))
            res_ = pool.map(train, params)
            pool.close()
    else:
        # Test
        if int(argvs[9])==1:
            res = main_task((0,argvs))
        else:
            pool = multiprocessing.Pool(processes=int(argvs[9]))
            res_ = pool.map(main_task, params)
            pool.close()
            res = sum(res_)
        BER = res[0]/res[1]

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    if int(argvs[1]) != 0:
        resut2f(argvs,BER)

        EsN0 = range(int(argvs[10]),int(argvs[12])+int(argvs[11]),int(argvs[11]))
        fig = plt.plot(EsN0, BER, 'bo', EsN0, BER, 'k')
        plt.axis([int(argvs[10]), int(argvs[12]), 1e-5, 1])
        plt.xticks(np.arange(int(argvs[10]), int(argvs[12]), 4))
        plt.xscale('linear')
        plt.yscale('log')
        plt.xlabel(r'$E_b/N_0$ [dB]')
        plt.ylabel('BER')
        plt.title('M='+ str(argvs[2]) + ', N=' + str(argvs[3])+ ', B=' + str(argvs[4]))
        plt.grid(True)
        plt.savefig('TSSC_' + argvs[1] + '_'+ str(argvs[2]) + '_' + str(argvs[3])+ '_' + str(argvs[4]) + '_' + str(argvs[5]) + '_' + str(argvs[6]) + '_' + str(argvs[13]) + '_' + str(argvs[14]) + '.eps', bbox_inches="tight", pad_inches=0.05)
        plt.show()

        exit()

    exit()
