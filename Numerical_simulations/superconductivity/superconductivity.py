import numpy as np 
import matplotlib.pyplot as plt
import os
import pickle
from numpy import genfromtxt
import copy


def gen_orth(p,d):   ##generate W
     a=np.random.normal(0,1,size=(d,p))
     q, _= np.linalg.qr(a)
     W=q.T
     W=W*(2*np.random.binomial(1,0.5,size=(p,1))-1)   ## guarantee Haar
     return W



def predict(X_test,X,W,Y,lam):
    n=X.shape[0]
    p=W.shape[0]
    B=X@W.T
    beta=np.linalg.inv(B.T@B/n+lam*np.eye(p))@B.T@Y/n
    train_pred=B@beta
    test_pred=X_test@W.T@beta
    return train_pred,test_pred

def normalize(X,y):
    X,y=(X-x_train_mean)/x_train_std, (y-y_train_mean)/y_train_std
    return X,y


cur_dir = os.path.dirname(__file__) 

rel_path='user_record'
rel_path1='record'
rel_path2='user_figures'

my_data=np.genfromtxt(os.path.join(cur_dir,'data','train.csv'),delimiter=',')[1:,:]
n,d=my_data.shape
np.random.seed(15)
perm=np.random.permutation(n)
np.random.seed()
my_data=my_data[perm,:]

testset=my_data[int(0.9*n):,:]
trainset=my_data[:int(0.9*n),:]
train_size=trainset.shape[0]
test_size=testset.shape[0]

X_train,y_train,X_test,y_test=trainset[:,:-1],trainset[:,-1:],testset[:,:-1],testset[:,-1:]

x_train_mean=X_train.mean(0)
y_train_mean=y_train.mean(0)
x_train_std=X_train.std(0)
y_train_std=y_train.std(0)

X_train,y_train=normalize(X_train,y_train)
X_test,y_test=normalize(X_test,y_test)





def compute(num_sample,num_X,num_init,pi,lam,sep):  ## calculate and record the test loss
    test_loss=np.zeros((num_X,num_init))     ##used for determining the optimal lambda
    width=int(pi*81)    
    record_test_pred=np.zeros((num_X,num_init,test_size))
    
    if not sep:    
        permute_index=[np.random.permutation(X_train.shape[0])[:num_sample] for ii in range(num_X)]
    else:     
        permute_index = np.split(np.random.permutation(len(trainset)-len(trainset)%num_X),num_X)
        for ii in range(len(permute_index)):
            permute_index[ii]=permute_index[ii][:num_sample]
    init1=[gen_orth(width,81) for ii in range(num_init)]  ## generate initializations

    for i in range(num_X):
        subset_X = X_train[permute_index[i]]
        subset_y = y_train[permute_index[i]]
        
        
        for j in range(num_init):
            train_pred,test_pred=predict(X_test,subset_X,init1[j],subset_y,lam)
            train_loss=((train_pred-subset_y)**2).mean()
            test_loss[i,j]=((test_pred-y_test)**2).mean()

            print('pi,i,j',pi,i,j,'train_loss: {:.6f}, test loss: {:.6f}'.format(train_loss, test_loss[i,j]))
            record_test_pred[i,j,:]=test_pred.reshape(-1)

    return record_test_pred   



def calculate_bvm_sobol(num_sample,num_X,num_init,pi,lam,sep): 

    
    outputs=compute(num_sample,num_X,num_init,pi,lam,sep)

    Ewf=np.zeros((num_X,test_size)) 
    Exf=np.zeros((num_init,test_size)) 
    Ef=np.zeros((test_size))
    Ef2=np.zeros((test_size))

   
    for i in range(num_X):
        for j in range(num_init):
            Ewf[i,:]=Ewf[i,:]+outputs[i,j,:]
            Exf[j,:]=Exf[j,:]+outputs[i,j,:]
            Ef=Ef+outputs[i,j,:]
            Ef2=Ef2+outputs[i,j,:]**2  
    Ewf=Ewf/num_init
    Exf=Exf/num_X  
    num=num_X*num_init
    Ef=Ef/num
    Ef2=Ef2/num
   
    var=(Ef2-Ef**2).mean()
    bias=((Ef-y_test.reshape(-1))**2).mean()
    mse=var+bias
    
    v_s=((Ewf-Ef)**2).mean()
    v_i=((Exf-Ef)**2).mean()
    v_si=var-v_s-v_i

    del Ewf,Exf,Ef,Ef2

    return mse,var,bias,v_s,v_i,v_si





##perform simulations and save results in "user_record" file.
def simulation(lam,sam_list,all_pi,sep,num_X=10,num_init=10,repetitions=10):
    mse,bias,var,v_s,v_i,v_si=[np.zeros((len(all_pi),repetitions)) for ii in range(6)]
    for num_sample in sam_list:
        for i in range(len(all_pi)):
            for j in range(repetitions):
                mse[i,j],var[i,j],bias[i,j],v_s[i,j],v_i[i,j],v_si[i,j]=calculate_bvm_sobol(num_sample,num_X,num_init,all_pi[i],lam,sep)

        with open(os.path.join(cur_dir,'user_record','sample_{}_lam_{}_repetitions_{}_sep_{}.txt'.format(num_sample,lam,repetitions,sep)),'wb') as file:
            print(np.stack([mse,var,bias,v_s,v_i,v_si],0).shape)
            pickle.dump(np.stack([mse,var,bias,v_s,v_i,v_si],0),file)



lam=0.01
num_X=10     ## number of X
num_init=10  ## number of initializations
repetitions=10
all_pi=[0.1*(ii+1) for ii in range(10)] 
     
sep=False  ##whether separate the trainset into num_X pieces 
           ##so that the randomly generated X-s would not contain same samples.

sam_list=[5,10,15,20,25,30,35,40,45,50,60,70,80,\
    100,120,150,200,400,500,1000,2000,5000]


#simulation(lam,sam_list,all_pi,sep,num_X,num_init,repetitions)




def plot_fig_10(filename='record'):   ##plot figure 10
    samplenum=1000
    with open(os.path.join(cur_dir,filename,'sample_{}_lam_{}_repetitions_{}_sep_{}.txt'.format(samplenum,lam,repetitions,sep)),'rb') as file:
        all_record=pickle.load(file)
    plt.errorbar(all_pi,all_record[0,:,:].mean(1),all_record[0,:,:].std(1),label='MSE',fmt='-',marker='.',markersize=5,capsize=1,)
    plt.errorbar(all_pi,all_record[2,:,:].mean(1),all_record[2,:,:].std(1),label=r'$Bias^2$',fmt='-',marker='.',markersize=5,capsize=1.5)
    plt.errorbar(all_pi,all_record[1,:,:].mean(1),all_record[1,:,:].std(1),label='Var',fmt='-',marker='.',markersize=5,capsize=1,c='C2')
    plt.legend(fontsize=14)
    plt.xlabel(r'$\pi$')
    plt.grid(linestyle='dotted')
    #plt.show()

    plt.savefig(os.path.join(cur_dir,rel_path2,'sample_{}_lam_{}_repetitions_{}_sep_{}.png'.format(samplenum,lam,repetitions,sep)))
    plt.close() 

#plot_fig_10('record')   ## or 'user_record'





##plot bias,var,mse,v_s,v_i as functions of n given \pi     Figure 8,9
def plot_fig_8_9(pi_list,sam_list,kind,filename='record'):
    samplenum=sam_list
    all_record=np.zeros((6,len(samplenum),repetitions))
    for i in range(len(pi_list)):
       
        for j in range(len(samplenum)):   
            with open(os.path.join(cur_dir,filename,'sample_{}_lam_{}_repetitions_{}_sep_{}.txt'.format(samplenum[j],lam,repetitions,sep)),'rb') as file:

                record=pickle.load(file)
            all_record[:,j,:]=record[:,int(pi_list[i]*10-0.5),:]

        s0=all_record[1,:,:]-(all_record[3,:,:]+all_record[4,:,:])  ##rest:=var-v_s-v_i
        s1=all_record[5,:,:]+all_record[3,:,:]   #\Sigma^s_{si}
        s2=all_record[5,:,:]+all_record[4,:,:]   #\Sigma^i_{is}

        
        
        if kind=='fig_8':
            plt.errorbar(samplenum,all_record[0,:,:].mean(1),all_record[0,:,:].std(1),label='MSE',fmt='-',marker='.',markersize=5,capsize=1)
            plt.errorbar(samplenum,all_record[2,:,:].mean(1),all_record[2,:,:].std(1),label=r'$Bias^2$',fmt='-',marker='.',markersize=5,capsize=1.5)
            plt.errorbar(samplenum,all_record[1,:,:].mean(1),all_record[1,:,:].std(1),label='Var',fmt='-',marker='.',markersize=5,capsize=1,c='C2')
        
        elif kind=='fig_9_left':
            plt.errorbar(samplenum,all_record[3,:,:].mean(1),all_record[3,:,:].std(1),label=r'$V_s$',fmt='-',marker='.',markersize=5,capsize=1)
            plt.errorbar(samplenum,all_record[4,:,:].mean(1),all_record[4,:,:].std(1),label=r'$V_i$',fmt='-',marker='.',markersize=5,capsize=1.5)
            plt.errorbar(samplenum,s0.mean(1),s0.std(1),label='Rest',fmt='-',marker='.',markersize=5,capsize=1.5)
        
        elif kind=='fig_9_middle':
            plt.errorbar(samplenum,all_record[1,:,:].mean(1),all_record[1,:,:].std(1),label='Var',fmt='-',marker='.',markersize=5,capsize=1,c='C2')
            plt.errorbar(samplenum,s1.mean(1),s1.std(1),label=r'$\Sigma^{s}_{si}$',fmt='-',marker='.',markersize=5,capsize=1.5,c='C3')
            plt.errorbar(samplenum,all_record[4,:,:].mean(1),all_record[4,:,:].std(1),label=r'$\Sigma^{i}_{si}$',fmt='-',marker='.',markersize=5,capsize=1.5,c='C4')
        
        elif kind=='fig_9_right': 
            plt.errorbar(samplenum,all_record[1,:,:].mean(1),all_record[1,:,:].std(1),label='Var',fmt='-',marker='.',markersize=5,capsize=1,c='C2')
            plt.errorbar(samplenum,all_record[3,:,:].mean(1),all_record[3,:,:].std(1),label=r'$\Sigma^{s}_{is}$',fmt='-',marker='.',markersize=5,capsize=1.5,c='C3')
            plt.errorbar(samplenum,s2.mean(1),s2.std(1),label=r'$\Sigma^{i}_{is}$',fmt='-',marker='.',markersize=5,capsize=1.5,c='C4')
        

        plt.legend(fontsize=23)
        plt.xlabel('n')
        plt.xlim(0,210)
        plt.grid(linestyle='dotted')
        #plt.show()
        plt.savefig(os.path.join(cur_dir,rel_path2,'pi_{}_lam_{}_repetitions_{}_sep_{}_{}.png'.format(round(pi_list[i],2),lam,repetitions,sep,kind)))
        plt.close()




#plot_fig_8_9([0.2,0.9],sam_list,'fig_8',filename='record') ## or 'user_record'

#for kind in ['fig_9_left','fig_9_middle','fig_9_right']:
    #plot_fig_8_9([0.2],sam_list,kind,filename='record')     ## or 'user_record'






