import torch
import numpy as np
import torch.nn as nn
from dengue_model import LSTM
#from model_sird import SIRD
from dengue_model import SIR
import argparse
import pandas as pd
import matplotlib.pyplot as plt
#import data_preprocess as dp
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
#from torchviz import make_dot

#plt.get_backend()
#plt.switch_backend('TkAgg')
parser = argparse.ArgumentParser(description='PyTorch Infectious disease model Training')
parser.add_argument('--epochs', default=4500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--immunity',default=0,type=float)
#parser.add_argument('--forcing_decay',default=1000,type=float,help='decay rate')
#parser.add_argument('--forcing_ratio',default=0,type=float,help='initial forcing rate')
#parser.add_argument('--forcing_decay_type',default='sigmoid')

path='../data_extraction/'
data_path='/home/yichao/Desktop/dengue_data/forecast/data/'
path_pop='../data_aggregate/population.csv'

district_list=['Colombo','Gampaha','Kalutara','Kandy','Matale','Nuwara_Eliya','Galle','Hambantota',
'Matara','Jaffna','Kilinochchi','Mannar','Vavuniya','Mullaitivu','Batticaloa','Ampara','Trincomalee',
'Kurunegala','Puttalam','Anuradhapura','Polonnaruwa','Badulla','Monaragala','Ratnapura','Kegalle']
district_list=np.array(district_list)
#district_list=np.delete(district_list,[5,10,11,12,13,15,20,22],axis=0)

best_prec1=torch.inf
def main():
    args = parser.parse_args()
    print("args", args)
    torch.cuda.set_device('cuda:2')
    criterion=nn.MSELoss(reduction='mean')
    cov_set=np.load('../data_aggregate/covariates_lag.npy')#covariates_lag
    cov_set=np.delete(cov_set,[0,2,4,5,6,7,8,9,10,11,14,16],axis=2) #3,12

    cov_set1 = torch.tensor(cov_set, dtype=torch.float32)  # old ,[:, 105:, :]
    #cov_set0=torch.tensor(cov_set,dtype=torch.float32)# 1.1 [:,105:,:]

    #cov_set_lag=cov_set1[:,13:,:]
#    print(cov_set1.shape)
    cov_set_lag= np.delete(cov_set1[:,13:,:],[],axis=2)
#    cov_set_lag= np.delete(cov_set1[:,13:,:],[2,3,4,5,7,9],axis=2) #cov_set1[:,13:,] #[:,:,np.newaxis] #np.delete(cov_set1[:,13:,:10],[6,8,9],axis=2)
#    cov_set_lag[:,:,8]=(cov_set1[:,1:-12,8]+cov_set1[:,2:-11,8]+cov_set1[:,3:-10,8])/3
###worst
#    cov_set_lag[:,:,8]=(cov_set1[:,10:-3,8]+cov_set1[:,11:-2,8]+cov_set1[:,12:-1,8])/3
#    cov_set_lag[:,:,7]=(cov_set1[:,4:-9,7]+cov_set1[:,5:-8,7]+cov_set1[:,6:-7,7])/3
#    cov_set_lag[:,:,6]=(cov_set1[:,10:-3,6]+cov_set1[:,11:-2,6]+cov_set1[:,12:-1,6])/3
#    cov_set_lag[:,:,5]=(cov_set1[:,4:-9,5]+cov_set1[:,5:-8,5]+cov_set1[:,6:-7,5])/3

    ##cov_set_lag[:,:,8]=(cov_set1[:,7:-6,8]+cov_set1[:,8:-5,8]+cov_set1[:,9:-4,8])/3 #humidity
    #cov_set_lag[:,:,3]=(cov_set1[:,7:-6,3]+cov_set1[:,8:-5,3]+cov_set1[:,9:-4,3])/3 #precipitation7
    cov_set_lag[:,:,2]=(cov_set1[:,7:-6,2]+cov_set1[:,8:-5,2]+cov_set1[:,9:-4,2])/3
    cov_set_lag[:,:,3]=(cov_set1[:,10:-3,3]+cov_set1[:,11:-2,3]+cov_set1[:,12:-1,3])/3 #meanT6
    #cov_set_lag[:,:,2]=(cov_set1[:,10:-3,2]+cov_set1[:,11:-2,2]+cov_set1[:,12:-1,2])/3
#    cov_set_lag[:,:,5]=(cov_set1[:,10:-3,5]+cov_set1[:,11:-2,5]+cov_set1[:,12:-1,5])/3 #DTR
    cov_set_lag[:,:,4]=(cov_set1[:,7:-6,4]+cov_set1[:,8:-5,4]+cov_set1[:,9:-4,4])/3

    cov_set_lag= np.delete(cov_set_lag[:,:,:],[1,2],axis=2)

    #cov_set0[:, :, 11] = cov_set0[:, :, 11] * 1.01
    val, _ = torch.max(torch.reshape(cov_set_lag, (-1, 3)), 0) #11
    #cov_set0 = cov_set0 / val#remember to change to cov0 during SA(dataset)
    cov_set_lag = cov_set_lag / val
    #print(torch.sum(cov_set0-cov_set1))
    #print(torch.var(cov_set1[:, :, 11]))

    #cov_set=dp.data_loader(data_path,district_list)

    case_data=np.load('../data_aggregate/case_data.npy')
    case_data=torch.tensor(case_data,dtype=torch.float32)*11 #105,[:,105:]
    #case_data = dp.casedata_reader(path)
    
    population=pd.read_csv(path_pop)
    df=pd.DataFrame(population)
    population=df[['2011','2012','2013','2014','2015','2016','2017','2018','2019']].values #
    
    ##normalization

    Max = torch.max(case_data)
    Min = torch.min(case_data)
    stepn=0
    train_set=cov_set_lag[:,stepn:-52+stepn,:] #[:,45:-52+45,:]
    test_set=cov_set_lag[:,-52+stepn:,:] #


    train_label=case_data[:,stepn:-52+stepn]
    test_label=case_data[:,-52+stepn:]


    input_size=4
    hidden_size1=5
    hidden_size2=3
    out_features=1
    #model=LSTM(input_size,hidden_size,criterion)
    model=SIR(input_size,hidden_size1,hidden_size2,out_features,criterion,train_label.shape[0],Max,Min)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)#
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.01)#
    #scheduler = 1
    model = model.cuda()
    #model.load_state_dict(torch.load('/home/yichao/Desktop/dengue_data/forecast_lstm/immunity/model8y_mobility.pkl'))
    args.immunity=torch.tensor([0.58,0.37,0.37,0.37,0.365,0.25,0.37,0.353,0.37,0.37,0.345,0.35,0.36,0.365,0.37,0.37]).cuda()
    args.growth_rate=torch.tensor([1.00951,1.01510,1.02052,1.02545,1.02986,1.03398,1.03767,1.04109,1.04430]).cuda()
    S_init=torch.tensor(population[:,0],dtype = torch.float32).cuda()*(1-args.immunity)
    E_init=torch.zeros(S_init.shape).cuda()
    I_init=train_label[:,0].cuda()
    R_init=torch.zeros(S_init.shape).cuda()+torch.tensor(population[:,0],dtype = torch.float32).cuda()*args.immunity
    SEIR0=(S_init,E_init,I_init,R_init)
    N = torch.tensor(population[:,:],dtype = torch.float32).cuda()
    I_ratio =I_init/(N[:,0])
    S_ratio=(N[:,0]-I_init)/N[:,0]
    S_init1 = N[:, 6] - 10
    E_init1 = torch.zeros(S_init.shape).cuda()
    I_init1 = 10 * torch.ones(S_init.shape).cuda()
    R_init1 = torch.zeros(S_init.shape).cuda()
    SEIR1 = (S_init1, E_init1, I_init1, R_init1)

    L_out=[]
    SIR_Loss=[]
    beta_M=0
    SIR_seq=0
    for epoch in range(args.epochs):

        SIR_loss,SEIR_0,SEIR_1,SIR_seq,init_states=train(
                        train_set, train_label,SEIR0,SEIR1,N, model, scheduler, optimizer, epoch, args,S_ratio,I_ratio)

#        L_out.append(l_out.cpu().detach().numpy())
        SIR_Loss.append(SIR_loss.cpu().detach().numpy())



        if epoch%500==0:
            print('train_err:' + str(SIR_loss))
        SEIR_test0=SEIR_0
        SEIR_test1=SEIR_1
        validate(test_set, test_label,SEIR_test0,SEIR_test1,beta_M,N, model, SIR_seq, args,S_ratio,I_ratio,init_states)
        #print('test_err:' + str(test_loss))
    #SEIR_test0=SEIR_0
    #SEIR_test1=SEIR_1
    #validate(test_set, test_label, SEIR_test0,SEIR_test1, beta_M, N, model, SIR_seq, args,S_ratio,I_ratio,init_states)
    np.savetxt('Loss.csv',SIR_Loss,delimiter=',')
    #np.savetxt('w_out.csv',W_out,delimiter=',')
#    np.savetxt('L_out.csv', L_out, delimiter=',')
    np.save('L_out.npy',L_out)

    #plt.plot(Loss)
    #plt.show()
    #plt.plot(Var_loss)
    #plt.title('var')
    #plt.show()
    #plt.plot(SIRD_Loss)
    #plt.title('SIRD')
    #plt.show()

def train(train_set,train_label,SEIR0,SEIR1,N,model,scheduler,optimizer,epoch,args,S_ratio,I_ratio):
    model.train()
    SEIR_seq1,SEIR_0,SEIR_1,SIR_pre,loss,beta,_,SEIR_seq,init_states=model(train_set.cuda(),
                                        train_label.cuda(),SEIR0,SEIR1,N,args,S_ratio=S_ratio,I_ratio=I_ratio,train=1)#beta0_w=beta0_w
    #np.save('train_maxNDVI8.npy', SIR_pre.cpu().detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(),5)
    optimizer.step()

    scheduler.step()

    #var_loss=criterion(w_out,l_out)

    #torch.save(model.state_dict(), 'model.pkl')
    #SIR_loss=criterion(SIR_pre,train_label[:,1:])
    #print(np.array(Weight.cpu())[:, 4, 0])
    #print(l_out)
    if epoch==args.epochs-1:
        #np.save('beta.npy',beta.cpu().detach().numpy())
        np.save('./SEIR_8.npy',SEIR_seq.cpu().detach().numpy())
        np.save('./train_pre.npy',SIR_pre.cpu().detach().numpy())
        np.save('./train_label.npy',train_label[:,1:].cpu().detach().numpy()) #:,
        np.save('./SEIR_17.npy',SEIR_seq1.cpu().detach().numpy())
        print('train_err:'+str(loss))
        #torch.save(model.state_dict(),'./model8y.pkl')
        #torch.save(model.state_dict(), 'model_wo_pre.pkl')
        #plt.plot(SIR_pre.cpu().detach().numpy()*100,label='fitting')
        #plt.plot(range(len(train_label[1:])),train_label[1:].cpu().detach().numpy()*100,label='label')
        #plt.title('confirm cases')
        #plt.legend()
        #plt.show()
         #print(SIRD_pre[:,0])
         #print(l_out)
    return loss,SEIR_0,SEIR_1,SIR_pre,init_states

def validate(test_set,test_label,SEIR0,SEIR1,beta_M,N,model,SIR_pre,args,S_ratio,I_ratio,init_states):
    global best_prec1
    model.eval()
    I_new=SIR_pre[:,-1] # :,
    _, _,_, SIR_seq, loss,beta,MAE,_,_ = model(test_set.cuda(), test_label.cuda(),SEIR0,SEIR1,N,args,
                                              I_new=I_new,S_ratio=S_ratio,I_ratio=I_ratio,init_states=init_states)

    #np.save('SEIR.npy',np.array([SEIR[0].cpu().detach().numpy(),SEIR[1].cpu().detach().numpy(),
    #                    SEIR[2].cpu().detach().numpy(),SEIR[3].cpu().detach().numpy()]))
    #var_loss = criterion(w_out, l_out)
    #SIR_loss = criterion(SIR_pre, test_label)

    #print('test_err:'+str(loss_total))
    #print('MAE:'+str(MAE))
    
    is_best= MAE<best_prec1
    best_prec1=min(MAE,best_prec1)
    if is_best:
        print('test_err:'+str(loss))
        print('MAE:'+str(MAE))
        np.save('./test_pre.npy', SIR_seq.cpu().detach().numpy())
        np.save('./test_label.npy', test_label.cpu().detach().numpy())
        np.save('./beta_t.npy',beta.cpu().detach().numpy())
        torch.save(model.state_dict(),'./model8y.pkl')

    #plt.plot(range(len(test_label[1:])),test_label[1:].cpu().detach().numpy()*100,label='label')
    #plt.title('confirm cases_test')
    #plt.legend()
    #plt.show()



if __name__ == '__main__':
    main()
