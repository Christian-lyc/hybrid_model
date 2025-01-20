import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import random

class SIR(nn.Module):
    def __init__(self,input_sz: int, hidden_sz1: int, hidden_sz2: int,out_features:int, criterion,dim,Max,Min):
        super(SIR, self).__init__()
        self.input_size = input_sz
        self.hidden_size1 = hidden_sz1
        self.hidden_size2 = hidden_sz2
        self.out_features=out_features
        self.loss = criterion
        #self.l1_loss=nn.L1Loss(reduction='sum')
        self.Max=Max
        self.Min=Min
        stdv_hi = 1.0 / math.sqrt(self.hidden_size1)
        #stdv_in = 1.0 / math.sqrt(self.input_size)


        self.lstm = nn.LSTMCell(input_sz,hidden_sz1)
        self.lstm1 = nn.LSTMCell(hidden_sz1, hidden_sz2)
        self.lstm2 = nn.LSTMCell(hidden_sz2, hidden_sz2)
        self.lstm3 = nn.LSTMCell(hidden_sz2, hidden_sz2)
        #self.lstm = LSTM(input_sz, hidden_sz)
        #self.lstm0=LSTM(hidden_sz, hidden_sz)
        #self.lstm1 = LSTM(hidden_sz, hidden_sz)

        self.linear = nn.Linear(hidden_sz2, out_features)
        #self.linear.weight.data.uniform_(-stdv_hi, stdv_hi)
    def smape(self,outputs, answers):
        #outputs=outputs/100
        #answers=answers/100
        return 1/len(outputs)*torch.sum(2 * torch.abs(outputs - answers) / (torch.abs(answers) + torch.abs(outputs)+1e-10)) ##1/len(outputs)*

    def SIR(self,S_hidden0,E_hidden0,I_hidden0,R_hidden0,beta0,N,t,beta1,S_hidden1,E_hidden1,I_hidden1,R_hidden1):
        gamma=1
        omiga=0.7

        b_I0 = beta0 * S_hidden0 #I_hidden* /N
        g_I0 = gamma * I_hidden0
        e_E0 = omiga * E_hidden0
        new_cases= e_E0
        #if t==0 and i==0:
        S_t1 = S_hidden0 - b_I0
        E_t1 = E_hidden0 + b_I0 - e_E0
        I_t1 = I_hidden0 + e_E0 - g_I0
        #I_t1 = I_hidden0 + b_I0 - g_I0
        R_t1 = R_hidden0 + g_I0

        return S_t1,E_t1,I_t1,R_t1,new_cases
    def forward(self,x,case_data,SEIR0,SEIR1,N,args,epoch=0,init_states=None,train=0,I_new=0,S_ratio=0,I_ratio=0):  #beta0_w
        sample,seq_sz, dim = x.shape  #sample,
        SEIR_seq=torch.zeros(case_data.shape[0],case_data.shape[1] - 1,4)
        if train==1:
            SEIR_seq1 = torch.zeros(case_data.shape[0], case_data.shape[1] - 314+0, 4) #+5
        else:
            SEIR_seq1 =torch.zeros(case_data.shape[0],case_data.shape[1] ,4)

        if train==1:
            SIR_seq = torch.zeros( case_data.shape[0],case_data.shape[1] - 1).cuda()  #case_data.shape[0],
            beta = torch.zeros( sample,seq_sz- 1).cuda()#sample,
        else:
            SIR_seq = torch.zeros(case_data.shape[0], case_data.shape[1]).cuda() #case_data.shape[0],
            beta = torch.zeros(sample, seq_sz).cuda()#sample,
        (S_0,E_0,I_0,R_0)=SEIR0
#        (S_1,E_1,I_1,R_1)=SEIR1


        if train==1:
            IR = case_data[:,1:] #:,
            covariates=x[:,1:,:]#:,
        else:
            IR=case_data
            covariates= x


        if init_states is None:
            h_t, c_t,h_t1,c_t1,h_t2,c_t2,h_t3,c_t3 = (
                torch.zeros( x.shape[0], self.hidden_size1).to(x.device),
                torch.zeros( x.shape[0], self.hidden_size1).to(x.device),
                torch.zeros(x.shape[0], self.hidden_size2).to(x.device),
                torch.zeros(x.shape[0], self.hidden_size2).to(x.device),
                torch.zeros(x.shape[0], self.hidden_size2).to(x.device),
                torch.zeros(x.shape[0], self.hidden_size2).to(x.device),
                torch.zeros(x.shape[0],self.hidden_size2).to(x.device),
                torch.zeros(x.shape[0],self.hidden_size2).to(x.device)
                #torch.zeros(self.hidden_size).to(x.device),
                #torch.zeros(self.hidden_size).to(x.device),
                #torch.zeros(self.hidden_size).to(x.device),
                #torch.zeros(self.hidden_size).to(x.device)
            )

        else:
            h_t, c_t, h_t1,c_t1,h_t2,c_t2,h_t3,c_t3 = init_states


        #

        if train==1:

            S_0 =S_0-I_0
        #else:
        #    S_hidden0_w = S_hidden0

        #I_hidden2 = I_hidden0
        S_hidden, E_hidden ,I_hidden, R_hidden=0,0,0,0
        #S_hidden2_w, E_hidden2_w, I_hidden2_w, R_hidden2_w = 0, 0, 0, 0
        beta1_w, S_hidden1_w, E_hidden1_w, I_hidden1_w, R_hidden1_w=0,0,0,0,0
        beta1, S_hidden1, E_hidden1, I_hidden1, R_hidden1 = 0, 0, 0, 0, 0


        S_hidden0=S_0
        I_hidden0 = I_0
        E_hidden0= E_0
        R_hidden0=R_0
 #       S_hidden0_1 = S_1
 #       I_hidden0_1 = I_1
 #       E_hidden0_1 = E_1
 #       R_hidden0_1 = R_1
        new_cases_1=torch.zeros(S_hidden0.shape).cuda()
        for t in range(covariates.shape[1]): #covariates.shape[1]
            l=0
            if  t>=52-l and t<104-l:   #53 #2012
                N_new=N[:,0]*args.growth_rate[1] #:,
                I_new1=I_ratio*N_new
                if t==52-l:
                    S_hidden0 = S_ratio * N_new*(1-args.immunity)

            elif t>=104-l and t <156-l:  #2013
                N_new=N[:,0]*args.growth_rate[2] # :,
                I_new1 = I_ratio * N_new
                if t==104-l:
                    S_hidden0 = S_ratio * N_new*(1-args.immunity)
            elif t>=156-l and t<208-l:   #2014
                N_new=N[:,0]*args.growth_rate[3] # :,
                I_new1 = I_ratio * N_new
                if t==156-l:
                    S_hidden0 = S_ratio * N_new*(1-args.immunity)
            elif t>=208-l and t<260-l:   #2015
                N_new = N[:,0]*args.growth_rate[4]  # :,
                I_new1 = I_ratio * N_new
                if t==208-l:
                    S_hidden0 = S_ratio * N_new*(1-args.immunity)
            elif t>=260-l and t<313-l:   #2016
                N_new = N[:,0]*args.growth_rate[5]  # :,
                I_new1 = I_ratio * N_new
                if t==260-l:
                    S_hidden0 = S_ratio * N_new*(1-args.immunity)
            elif t>=313-l and t<365-l:   #2017 no immune after 2017
                N_new = N[:,0]*args.growth_rate[6]  # :,
                I_new1 = I_ratio * N_new

#                if t==313-l:
#                    S_hidden0 = S_ratio * N_new*(1-args.immunity)
###strain shift
                if t==313-l:
                    S_hidden0=S_ratio * N_new
                    R_hidden0=0
            elif t>=365-l and t<417-l:   #2018
                N_new = N[:,0]*args.growth_rate[7]  # :,
                I_new1 = I_ratio * N_new

#                if t==365-l:
#                    S_hidden0 = S_ratio * N_new*(1-args.immunity)
###new strain
                if t==365-l:
                    S_hidden0=S_hidden0*1.00329
            else:
                if t>=417-l or train==0:  #2019 
                    N_new=N[:,0]*args.growth_rate[8] # :,
#                    if t==417-l: #or train==0 and t==0:
#                        S_hidden0 = S_ratio * N_new*(1-args.immunity)


###new strain
                    if t==417-l: #or train==0 and t==0:
                        S_hidden0=S_hidden0*1.00309
                else:
                    N_new = N[:,0] # :, #2010


            ### teacher network
            xt=covariates[:,t,:]  #:,
            if t == 0 and train == 1:
                    # x_t[-1] = torch.tensor((I_hidden0 - self.Min) / (self.Max - self.Min),requires_grad=True)
                xt = torch.cat((xt, ((I_0.unsqueeze(1) - self.Min) / (self.Max - self.Min))), dim=1) #  , dim=1
            elif t==0 and train==0:
                xt = torch.cat((xt, ((I_new.unsqueeze(1) - self.Min) / (self.Max - self.Min))), dim=1) #.unsqueeze(1)  , dim=1
            else:
                 # x_t = torch.tensor((beta0 * S_hidden0/N - self.Min) / (self.Max - self.Min))
                 if t==52-l or t==104-l or t ==156-l or t==208-l or t==260-l or t==313-l or t==365-l or t==417-l:
                     xt = torch.cat((xt, ((I_new1.unsqueeze(1) - self.Min) / (self.Max - self.Min))),dim=1) #,dim=1   .unsqueeze(1)

                 else:
                     xt = torch.cat((xt, ((SIR_seq[:,t-1].unsqueeze(1)- self.Min) / (self.Max - self.Min))),dim=1)  #.unsqueeze(1)  :,  , dim=1

            h_t, c_t = self.lstm(xt, (h_t, c_t))
            h_t1, c_t1 = self.lstm1(h_t, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            ###old
            #h_t1, c_t1 = self.lstm0(h_t, h_t1, c_t1)
            #h_t2, c_t2 = self.lstm1(h_t1, h_t2, c_t2)
            #h_t3 = self.linear(h_t1)

            h_t4 = self.linear(h_t3)
            h_th = 0.005 * torch.sigmoid(h_t4) #0.005
            beta0 = h_th.squeeze()
            beta[:,t]=beta0 
            #    if t==0 and i==0:
            #        predict_list.append([S_hidden0, E_hidden0, I_hidden0, R_hidden0, beta0])
            #    else:
            #        predict_list.append([S_hidden2, E_hidden2, I_hidden2, R_hidden2, beta0])
            #        S_hidden0, E_hidden0, I_hidden0, R_hidden0, beta0 = predict_list[(t + 1) * (i + 1) - 2]
            #        S_hidden1, E_hidden1, I_hidden1, R_hidden1, beta1 = predict_list[(t + 1) * (i + 1) - 1]
            # beta1=h_t2
            
            S_hidden2, E_hidden2, I_hidden2, R_hidden2,new_cases = self.SIR(S_hidden0, E_hidden0, I_hidden0, R_hidden0, beta0, N_new,  #  :, N[0]
                                                                   t, beta1,S_hidden1, E_hidden1, I_hidden1, R_hidden1)

###new strain
#            if t>=313-l or train==0:
#                S_hidden2_1, E_hidden2_1, I_hidden2_1, R_hidden2_1, new_cases_1 = self.SIR(S_hidden0_1, E_hidden0_1, I_hidden0_1,
#                                                                                 R_hidden0_1, beta0, N_new,  # :, N[0]
#                                                                                 t, beta1, S_hidden1, E_hidden1,
#                                                                                 I_hidden1, R_hidden1)

            SIR_seq[:,t]=(new_cases)  #+new_cases_1
            if train==1:
                SEIR_seq[:,t,0]=S_hidden2
                SEIR_seq[:, t, 1]=E_hidden2
                SEIR_seq[:, t, 2] = I_hidden2
                SEIR_seq[:, t, 3]=R_hidden2
###new strain
#            if t>=313-l:
#                SEIR_seq1[:, t-313+l, 0] = S_hidden2_1
#                SEIR_seq1[:, t-313+l, 1] = E_hidden2_1
#                SEIR_seq1[:, t-313+l, 2] = I_hidden2_1
#                SEIR_seq1[:, t-313+l, 3] = R_hidden2_1
            S_hidden0=S_hidden2.clone()
            I_hidden0 = I_hidden2.clone()
            E_hidden0=E_hidden2.clone()
            R_hidden0=R_hidden2.clone()
#            if t>=313-l or train==0:
#                S_hidden0_1=S_hidden2_1.clone()
#                I_hidden0_1 = I_hidden2_1.clone()
#                E_hidden0_1 = E_hidden2_1.clone()
#                R_hidden0_1 = R_hidden2_1.clone()

        ######
        #hidden_seq=0 ##debug

        loss_total=self.smape(SIR_seq,IR)
        MAE=torch.mean(torch.abs(SIR_seq-IR))
        SEIR_0=(S_hidden0,E_hidden0,I_hidden0,R_hidden0)
#        SEIR_1=(S_hidden0_1,E_hidden0_1,I_hidden0_1,R_hidden0_1)
        SEIR_1=0 #add

        return SEIR_seq1, SEIR_0,SEIR_1,SIR_seq,loss_total,beta,MAE,SEIR_seq,(h_t, c_t, h_t1, c_t1, h_t2, c_t2, h_t3, c_t3)



class LSTM(nn.Module):
    def __init__(self,input_sz: int, hidden_sz: int):
        super(LSTM, self).__init__()

        self.input_size = input_sz
        self.hidden_size = hidden_sz
        # i_t
        self.W_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        # f_t
        self.W_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        # c_t
        self.W_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        # o_t
        self.W_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights(hidden_sz)
    def init_weights(self,hidden_size):
        stdv = 1.0 / math.sqrt(hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self,x,h_t,c_t):

        i_t = torch.sigmoid(x @ self.W_i + h_t @ self.U_i + self.b_i)#
        f_t = torch.sigmoid(x @ self.W_f + h_t @ self.U_f + self.b_f)#
        g_t = torch.tanh(x @ self.W_c + h_t @ self.U_c + self.b_c)#
        o_t = torch.sigmoid(x @ self.W_o + h_t @ self.U_o + self.b_o)#
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t) # tanh
        return h_t,c_t

