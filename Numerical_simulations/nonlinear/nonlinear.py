import numpy as np 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pickle




'''
This file is used to plot Figure 7(right). Here we assume the nonlinear 
activation is ReLU(x)-1/(2*np.pi)**0.5 (see Figure 7). The results still
hold for other zero mean activation (e.g. tanh).
'''


def theta_1(ga,lam):
    a=-lam+ga-1
    return (a+(a**2+4*ga*lam)**0.5)/(2*lam*ga)
def theta_2(ga,lam):
    return 1/(2*ga*lam**2)*(ga-1+((ga+1)*lam+(ga-1)**2)/((-lam+ga-1)**2+4*lam*ga)**0.5)

def solve(f,x,W,X,eps,theta,lam):
    n,d=X.shape
    p=W.shape[0]
    B=f(X@W.T)
    hf=f(x@W.T)@np.linalg.inv(B.T@B/n+lam*np.eye(p))@B.T@(X@theta+eps)/n
    return hf

def bvm_nl(pi,delt,kind='mse',v=1,mu=1,alpha=1,sig=0.3,lam='opt'):
    if lam =='opt':      ##optimal lambda

        lam=v**2/mu**2*(delt*(1-pi+(sig/alpha)**2)+(v-mu**2)/v*pi*delt)
    ga=pi*delt
    th_1=theta_1(ga,lam/v)
    th_2=theta_2(ga,lam/v)

    if kind=='bias':
        f=alpha**2*(1-pi*mu**2/v*(1-lam/v*th_1))**2
    elif kind=='mse':
        f=alpha**2*(1-pi+delt*pi*(1-pi)*th_1+lam*pi/v*(lam*mu**2/v**2-delt*(1-pi))*th_2\
            +pi*(v-mu**2)/v*(ga*th_1+1-lam*ga/v*th_2))+sig**2*ga*(th_1-lam/v*th_2)
    elif kind=='var':
        f1=alpha**2*(1-pi+delt*pi*(1-pi)*th_1+lam*pi/v*(lam*mu**2/v**2-delt*(1-pi))*th_2\
            +pi*(v-mu**2)/v*(ga*th_1+1-lam*ga/v*th_2))+sig**2*ga*(th_1-lam/v*th_2)
        f2=alpha**2*(1-pi*mu**2/v*(1-lam/v*th_1))**2
        f=f1-f2
    return f


def gen_orth(p,d):   ##generate W
     a=np.random.normal(0,1,size=(d,p))
     q, _= np.linalg.qr(a)
     W=q.T
     W=W*(2*np.random.binomial(1,0.5,size=(p,1))-1)   ## guarantee Haar
     return W


def num_mse_nl(p,d,n,lam,kind='mse',alpha=1,sig=0.3,k=100):
    mse=0
    def act(X):    ##ReLU(x)-1/(2*np.pi)**0.5
        return (np.abs(X)+X)/2-1/(2*np.pi)**0.5

    for i in range(k):
        X=np.random.normal(0,1,(n,d))
        W=gen_orth(p,d)
        theta=np.random.normal(0,alpha/d**0.5,(d,1))
        eps=np.random.normal(0,sig,(n,1))
        x=np.random.normal(0,1,(1,d))
        
        hf=solve(act,x,W,X,eps,theta,lam)
        mse=mse+(hf-x@theta)**2
    mse=mse/k
    return mse













def simulation(ga,kind='mse',alpha=1,sig=0.3,lam='opt',axis=2,n=200,k=10):
    ## Simulations for section 4.1. Figure 7 (right)
    ## This function is used to perform simulations and record the results (but not plot the figures).
    ## This function will store all simulation results in the './user_record/' file.
    if axis==1:
        pi=np.linspace(0.1,1,19)
        delt=ga
        d=np.round(n*delt)*np.ones(shape=pi.shape)
        p=np.round(d*pi)
    elif axis==2:
        pi=ga
        delt=np.linspace(0.1,5,25)    ##(0.5,1,3)
        d=np.round(n*delt)
        p=np.round(d*pi)
        pi=pi*np.ones(shape=p.shape)
    

    v=0.5-1/2/np.pi   ## variance of  ReLU-EReLU
    mu=1/2            ## mean of ReLU-EReLU

    if lam =='opt':
        lam=v**2/mu**2*(delt*(1-pi+(sig/alpha)**2)+(v-mu**2)*pi*delt/v)
    else:
        lam=lam*np.ones(shape=p.shape[-1])
    times=p.shape[-1]   #length
    record=np.zeros((k,times))
    for i in range(times):
        
        d0=int(d[i])
        p0=int(p[i])

        for j in range(k):
            record[j,i]=num_mse_nl(p0,d0,n,lam[i],kind,alpha,sig,400)
            theory=bvm_nl(pi,delt,kind,v,mu,alpha,sig,lam) 

        print('setting '+str(i+1)+' finished ('+ str(times)+' settings in total).')       
    

    sd=np.std(record,0)
    mean=np.mean(record,0)
   
    
    
    if axis==1:
        x=pi
    else: 
        x=delt 

    '''
    ## used for illustration
    plt.errorbar(x,mean,sd,fmt='-',label='simulation')       ## should be delete
    plt.plot(x,theory,label='theory')
   
    #plt.ylim(0,1)
    plt.grid(linestyle='dotted')
    plt.xlabel(r'$\gamma_{}$'.replace('{}','{'+str(axis)+'}'))
    plt.ylabel(r'${}$'.replace('{}',kind))
    plt.legend()
    plt.show()'''


    with open('./user_record/record_'+str(kind)+'_lam_0.01_nl.txt','wb') as file:   
        pickle.dump([x,mean,sd,theory,record],file)

#simulation(0.8,'mse',alpha=1,sig=0.3,lam=0.01,axis=2,n=150,k=5)




def plot_emse(filename='record'):   

    ## plot Figure 7(right)

    with open('./'+filename+'/record_mse_lam_0.01_nl.txt','rb') as file:
        x,mean,sd,theory,record=pickle.load(file)
    length=x.shape[0]
    ind=np.arange(0,length,2)
    x,mean,sd,theory,record=x[ind],mean[ind],sd[ind],theory[ind],record[:,ind]
    plt.errorbar(x,mean,sd,label='nMSE',linewidth=2)
    plt.plot(x,theory,label='MSE',linewidth=2) 
    plt.grid(linestyle='dotted')
    plt.xlabel(r'$\mathbb{\delta}$',fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=15)
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.show()
    
    plt.savefig('./user_figures/numerical_mse_lam_0.01_nl.png') 
    plt.close()

#plot_emse('record') ##user_record
       


