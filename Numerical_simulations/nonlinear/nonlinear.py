import numpy as np 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pickle




'''
This file is used to plot Figure 6(right). Here we assume the nonlinear 
activation is ReLU(x)-1/(2*np.pi)**0.5 (see Figure 6). The results still
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
            +pi*(v-mu**2)/v*th_2/th_1**2)+sig**2*ga*(th_1-lam/v*th_2)

    elif kind=='var':
        f1=alpha**2*(1-pi+delt*pi*(1-pi)*th_1+lam*pi/v*(lam*mu**2/v**2-delt*(1-pi))*th_2\
            +pi*(v-mu**2)/v*th_2/th_1**2)+sig**2*ga*(th_1-lam/v*th_2)
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
    ## Simulations for section 4.2. Figure 6 (right)
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

    ## plot Figure 6(right)

    with open('./'+filename+'/record_mse_lam_0.01_nl.txt','rb') as file:
        x,mean,sd,theory,record=pickle.load(file)
    length=x.shape[0]
    ind=np.arange(0,length,2)
    x,mean,sd,theory,record=x[ind],mean[ind],sd[ind],theory[ind],record[:,ind]
    plt.errorbar(x,mean,sd,label='nMSE')
    plt.plot(x,theory,label='MSE') 
    plt.grid(linestyle='dotted')
    plt.xlabel(r'$\mathbb{\delta}$')
    plt.legend(fontsize=14)

    #plt.show()
    
    plt.savefig('./user_figures/numerical_mse_lam_0.01_nl.png') 
    plt.close()

#plot_emse('record') ##user_record
       














######################################################################


##some other functions (not used in the paper).






def plot_one_dim(kind='var',alpha=1,sig=0.1,lam='opt',plot=['1',0.9]):   ##plot 1-dim figures
    dic={'var':'Var','bias':'Bias^2','mse':'MSE'}
    num=2000
    pi=np.linspace(0,1.0,num)
    delt=np.linspace(0.25,20,num)              
    for x in plot[1:]:
        if plot[0]=='1':       ##1d plot, fix the first dim
            pi=np.ones(num)*x
        elif plot[0]=='2':     ##1d plot, fix the second dim
            delt=np.ones(num)*x
        v=0.5-1/2/np.pi
        mu=1/2
        f=bvm_nl(pi,delt,kind,v,mu,alpha,sig,lam)
        if plot[0]=='1':
            plt.plot(1/delt,f,label=str(r'$\pi=$')+str(x))
        else:
            plt.plot(pi,f,label=str(r'$\mathbb{\delta}=$')+str(x))
            plt.set_xlabel(r'$\pi$')
    if plot[0]=='1':
        plt.xlabel(r'$1/\mathbb{\delta}$')
    else:
        plt.xlabel(r'$\pi$')
    plt.grid(linestyle='dotted')
    plt.ylabel(r'${}$'.replace('{}',dic[kind]))
    plt.legend()
   
    #plt.show()
    
    plt.savefig('./user_figures/'+kind+'_one_dim_nl.png')
    plt.close()


#plot_one_dim(kind='mse',lam=0.01,plot=['1',0.3,0.5,0.7,0.9,1])











def plot_decomp(kind='var',alpha=1,sig=0.3,lam='opt',plot='2d'):   

    ##plot 2,3-dim figures

    dic={'var':'Var','bias':'Bias^2','mse':'MSE'}
    pi=np.linspace(0.0,1.0,1500)
    delt=np.linspace(0.1,5,1500)              
    pi,delt=np.meshgrid(pi,delt)
    v=0.5-1/2/np.pi
    mu=1/2
    if kind in ['var','bias','mse']:
        f=bvm_nl(pi,delt,kind,v,mu,alpha,sig,lam)

    fig = plt.figure(figsize=(6,6))
    if plot=='3d':
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(pi,delt,f)
        ax.set_xlabel(r'$\pi$',fontsize=15)
        ax.set_ylabel(r'$\mathbb{\delta}$',fontsize=15)
        ax.set_zlabel(r'${}$'.replace('{}',dic[kind]),fontsize=15)  
        ax.tick_params(labelsize=15)
        #plt.show()
        plt.savefig('./user_figures/'+kind+'_3dim.png')
        plt.close()
    elif plot=='2d':
        ax = fig.add_subplot()
        plt.gcf().subplots_adjust(bottom=0.15)

        ax.set_xlabel(r'${\pi}$',fontsize=20)
        ax.set_ylabel(r'$\mathbb{\delta}$',fontsize=20)
        plt.contourf(pi,delt,f,levels=20)
        ax2=plt.colorbar()
        ax2.set_label(label=r'${}$'.replace('{}',dic[kind]),size=20)
        ax2.ax.tick_params(labelsize=15)
        ax.tick_params(labelsize=15)
        #plt.show()

        if lam=='opt':
            n1='opt'
        else:
            n1='fix'
        #plt.show()

        plt.savefig('./user_figures/'+kind+'_'+n1+'_nl.png')  
        plt.close()



#for kind in ['var','bias','mse']:   ##heat maps and wireframe
    #plot_decomp(kind,alpha=1,sig=0.3,lam='opt',plot='2d')
    #plot_decomp(kind,alpha=1,sig=0.3,lam=0.01,plot='2d')
    #plot_decomp(kind,alpha=1,sig=0.3,lam='opt',plot='3d')











