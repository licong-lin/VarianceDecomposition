import numpy as np 
from matplotlib import pyplot as plt
import pickle




'''
This file is used to perform numerical simulations in section 4.1, 4.2 (Figure 6(left), 7).
We assume $X$ has i.i.d. N(0,1) entries for simplicity. The same results still hold
for non-gaussian $X$ (with i.i.d. zero-mean, one-variance, finite $8+\eta$ moment entries).

'''





def theta_1(ga,lam):
    a=-lam+ga-1
    return (a+(a**2+4*ga*lam)**0.5)/(2*lam*ga)
def theta_2(ga,lam):
    return 1/(2*ga*lam**2)*(ga-1+((ga+1)*lam+(ga-1)**2)/((-lam+ga-1)**2+4*lam*ga)**0.5)

def bias_var_mse(pi,delt,kind='var',alpha=1,sig=0.3,lam='opt'):
    if lam =='opt':      ##optimal lambda
        lam=delt*(1-pi+(sig/alpha)**2)
    ga=pi*delt
    th_1=theta_1(ga,lam)
    th_2=theta_2(ga,lam)
   
    if kind=='bias':
        f=alpha**2*(1-pi+lam*pi*th_1)**2
    elif kind=='var':
        f=alpha**2*pi*(1-pi+(pi-1)*(2*lam-delt)*th_1-pi*lam**2*\
            th_1**2+lam*(lam-delt+ga)*th_2)+sig**2*ga*(th_1-lam*th_2)
    elif kind=='mse':
        c=1-pi+(sig/alpha)**2
        f=alpha**2*(1-pi+ga*c*th_1+(lam-delt*c)*lam*pi*th_2)
    return f

def anova_indices(pi,delt,kind='var',alpha=1,sig=0.3,lam='opt'):
    if lam =='opt':      ##optimal lambda
        lam=delt*(1-pi+(sig/alpha)**2)
    ga=pi*delt
    th_1=((-lam+ga-1)+((-lam+ga-1)**2+4*lam*ga)**0.5)/(2*lam*ga)
    th_2=(ga-1)/(2*lam**2*ga)+((ga+1)*lam+(ga-1)**2)/(2*lam**2*ga*((-lam+ga-1)**2+4*lam*ga)**0.5)
   

    b0=pi
    b1=-((1+pi)*lam+(1-pi)*(1-ga))
    b2=lam*(lam+(1-pi)*(1-delt))
    tlam=(-b1+(b1**2-4*b0*b2)**0.5)/2/b0
    tth_1=theta_1(delt,tlam)
    tth_2=theta_2(delt,tlam)

    if kind=='v_s':
        f=alpha**2*(1-2*tlam*tth_1+tlam**2*tth_2-pi**2\
        *(1-lam*th_1)**2)
    elif kind=='v_i':
        f=alpha**2*pi*(1-pi)*(1-lam*th_1)**2
    elif kind=='v_sl':
        f=sig**2*delt*(tth_1-tlam*tth_2)
    elif kind=='v_si':
        f=alpha**2*(pi*(1-2*lam*th_1+lam**2*th_2+\
            (1-pi)*delt*(th_1-lam*th_2)-(1-pi)*(1-lam*th_1)**2)\
        -1+2*tlam*tth_1-tlam**2*tth_2)
    elif kind=='v_sli':
        f=sig**2*delt*(pi*(th_1-lam*th_2)-(tth_1-tlam*tth_2))
    else:
        f=0*pi
    return f

def gen_orth(p,d):   ##generate W
     a=np.random.normal(0,1,size=(d,p))
     q, _= np.linalg.qr(a)
     W=q.T
     W=W*(2*np.random.binomial(1,0.5,size=(p,1))-1)   ## guarantee Haar
     return W
def numerical_M(W,X,lam):
    p,d=W.shape
    n=X.shape[0]
    B=X@W.T
    tM=W.T@np.linalg.inv(B.T@B/n+lam*np.eye(p))@B.T/n
    return tM,tM@X

def num_bias_var_mse(p,d,n,lam,kind='bias',alpha=1,sig=0.3,k=100):
    EtM=0
    EM=0
    EtrMMT=0
    EtrtMtMT=0
    for i in range(k):
        X=np.random.normal(0,1,(n,d)) 

        ##another sample distribution
        #X=(3/14)**0.5*np.random.choice(np.array([-2,-1,3]),(n,d)) 

        W=gen_orth(p,d)
        tM,M=numerical_M(W,X,lam)
        EtM=EtM+tM
        EM=EM+M
        EtrMMT=EtrMMT+np.linalg.norm(M)**2
        EtrtMtMT=EtrtMtMT+np.linalg.norm(tM)**2
    EtM,EM,EtrMMT,EtrtMtMT=EtM/k,EM/k,EtrMMT/k,EtrtMtMT/k
    if kind=='bias':
        return alpha**2/d*np.linalg.norm(EM-np.eye(d))**2
    elif kind=='var':
        return sig**2*EtrtMtMT+alpha**2/d*(EtrMMT-np.linalg.norm(EM)**2)
    elif kind=='mse':
        return sig**2*EtrtMtMT+alpha**2/d*(EtrMMT+d-2*np.trace(EM))


def num_mse2(p,d,n,lam,kind='mse',alpha=1,sig=0.3,k=100):  ##estimate the mse directly
    mse=0
    for i in range(k):
        X=np.random.normal(0,1,(n,d))

        ##another sample distribution
        #X=(3/14)**0.5*np.random.choice(np.array([-2,-1,3]),(n,d)) 

        W=gen_orth(p,d)
        theta=np.random.normal(0,alpha/d**0.5,(d,1))
        eps=np.random.normal(0,sig,(n,1))
        x=np.random.normal(0,1,(1,d))
        
        ##another sample distribution
        #x=(3/14)**0.5*np.random.choice(np.array([-2,-1,3]),(1,d)) 


        Y=X@theta+eps
        hf=x@numerical_M(W,X,lam)[0]@Y
        mse=mse+(hf-x@theta)**2
    mse=mse/k
    return mse



def numerical_anova(p,d,n,lam,kind='V_s',alpha=1,sig=0.3,k=20):
    
        
    allW=[0 for i in range(k)]
    allX=[0 for i in range(k)]
    E_wM=[0 for i in range(k)]
    E_XM=[0 for i in range(k)]
    E_wtM=[0 for i in range(k)]
    E_XtM=[0 for i in range(k)]
    for i in range(k):
        allW[i]=gen_orth(p,d)
        allX[i]=np.random.normal(0,1,(n,d))

        ## another sample distribution
        #allX[i]=(3/14)**0.5*np.random.choice(np.array([-2,-1,3]),(n,d))
    
    EtM=0
    EM=0
    EtrMMT=0
    EtrtMtMT=0
    for i in range(k):
        for j in range(k):
            tM,M=numerical_M(allW[i],allX[j],lam)
            E_XM[i]=E_XM[i]+M
            E_XtM[i]=E_XtM[i]+tM
            E_wM[j]=E_wM[j]+M
            E_wtM[j]=E_wtM[j]+tM
            EM=EM+M
            EtM=EtM+tM
            EtrMMT=EtrMMT+np.linalg.norm(M)**2
            EtrtMtMT=EtrtMtMT+np.linalg.norm(tM)**2
        E_XM[i]=E_XM[i]/k
        E_XtM[i]=E_XtM[i]/k
    for j in range(k):
        E_wM[j]=E_wM[j]/k
        E_wtM[j]=E_wtM[j]/k
    EM,EtM,EtrMMT,EtrtMtMT=EM/k**2,EtM/k**2,EtrMMT/k**2,EtrtMtMT/k**2
    s=0
    if kind=='v_s':   
        for i in range(k):
            s=s+np.linalg.norm(E_wM[i]-EM)**2
        s=alpha**2/d*s/k
    elif kind=='v_i':
        for i in range(k):
            s=s+np.linalg.norm(E_XM[i]-EM)**2
        s=alpha**2/d*s/k
            
    elif kind=='v_sl':
        for i in range(k):
            s=s+np.linalg.norm(E_wtM[i])**2
        s=sig**2*s/k
    elif kind=='v_si':
        for i in range(k):
            s=s+np.linalg.norm(E_XM[i]-EM)**2+np.linalg.norm(E_wM[i]-EM)**2
        s=s/k
        s=alpha**2/d*(-np.linalg.norm(EM)**2-s+EtrMMT)
    elif kind=='v_sli':
        for i in range(k):
            s=s+np.linalg.norm(E_XtM[i]-EtM)**2+np.linalg.norm(E_wtM[i]-EtM)**2
        s=s/k
        s=sig**2*(-s+EtrtMtMT-np.linalg.norm(EtM)**2)
    return s





#############################################################################





def simulation(ga,kind='mse',alpha=1,sig=0.3,lam='opt',axis=2,n=150,k=10):
    ## Simulations for section 4.1, 4.2. (Figure 6 (left), 7) 
    ## This function is used to perform simulations and record the results (but not plot the figures).
    ## This function will store all simulation results in the './user_record/' file.

    if axis==1:
        pi=np.linspace(0.1,1,19)
        delt=ga
        d=np.round(n*delt)*np.ones(shape=pi.shape)
        p=np.round(d*pi)
    elif axis==2:
        pi=ga
        delt=np.linspace(0.1,5,50)    ##(0.5,1,3)   
        d=np.round(n*delt)
        p=np.round(d*pi)
        pi=pi*np.ones(shape=p.shape)
    if lam =='opt':
        lam=delt*(1-pi+(sig/alpha)**2)
    else:
        lam=lam*np.ones(shape=p.shape[-1])
    times=p.shape[-1]   #length
    record=np.zeros((k,times))
    for i in range(times):
        d0=int(d[i])
        p0=int(p[i])
        for j in range(k):
            if kind in ['mse','bias','var']:   
                record[j,i]=num_bias_var_mse(p0,d0,n,lam[i],kind,alpha,sig,100)
                theory=bias_var_mse(pi,delt,kind,alpha,sig,lam)
            elif kind in ['mse2']:    ## direct simulation (section 4.1)
                record[j,i]=num_mse2(p0,d0,n,lam[i],kind[:-1],alpha,sig,400)
                theory=bias_var_mse(pi,delt,kind[:-1],alpha,sig,lam)
            else:
                record[j,i]=numerical_anova(p0,d0,n,lam[i],kind,alpha,sig,20)
                theory=anova_indices(pi,delt,kind,alpha,sig,lam)
        print('setting '+str(i+1)+' finished ('+ str(times)+' settings in total).')
    sd=np.std(record,0)
    mean=np.mean(record,0)
    
    if axis==1:
        x=pi
    else: 
        x=delt 

    ##(used for illustration)
    plt.errorbar(x,mean,sd,fmt='-',label='simulation')       ## should be delete
    plt.plot(x,theory,label='theory')
   
    plt.ylim(0,1)
    plt.grid(linestyle='dotted')
    plt.xlabel(r'$\gamma_{}$'.replace('{}','{'+str(axis)+'}'))
    plt.ylabel(r'${}$'.replace('{}',kind))
    plt.legend()
    plt.show()

    with open('./user_record/record_'+str(kind)+'_opt_lam.txt','wb') as file:   
        pickle.dump([x,mean,sd,theory,record],file)


#for kind in ['mse','bias','var','v_s','v_i','v_sl','v_si','v_sli']:   ## simulations for Figure 7
    #simulation(0.8,kind,alpha=1,sig=0.3,lam='opt',axis=2,n=150,k=5)  

#for kind in ['mse2']: ## simulations for Figure 6(left)
    #simulation(0.8,kind,alpha=1,sig=0.3,lam='opt',axis=2,n=150,k=20)   


def plot_numerical_bvm_anova(anova=False,filename='record'):  
    ##Plot Figure 7 (using simulation resuts in 'record(user_record)' file).
    if not anova:
        items=['mse','bias','var']
        names=['MSE','Bias^2','Var']
    else:
        items=['v_s','v_i','v_si','v_sl','v_sli']
        names=['V_{s}','V_{i}','V_{si}','V_{sl}','V_{sli}']
    for t in range(2):
        for k in range(len(items)):
            with open('./'+filename+'/record_'+items[k]+'_opt_lam.txt','rb') as file:    
                x,mean,sd,theory,record=pickle.load(file)
            if t==1:
                plt.errorbar(x,mean,sd,linestyle='--',linewidth=1,marker='.',markersize=7,
                label='n'+r'${}$'.replace('{}',names[k]))
            else:
                plt.plot(x,theory,label=r'${}$'.replace('{}',names[k]),linewidth=3)
    plt.grid(linestyle='dotted')
    if not anova:
        plt.ylim(0,1.2)    
    else:
        plt.ylim(0,0.2)  
    plt.xlabel(r'$\mathbb{\delta}$',fontsize=20)
    plt.legend(ncol=2,fontsize=14)
    plt.tick_params(labelsize=15)
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.show()

    if anova:
        kind='anova'
    else:
        kind='bvm'

    plt.savefig('./user_figures/numerical_'+kind+'_opt.png')
    plt.close()


##anova=True: plot variance components (anova) (Figure 7 right).
##anova=False:  plot Bias, Var and MSE (Figure 7 left).
#plot_numerical_bvm_anova(anova=True,filename='record')  ##filename='user_record'




def plot_emse(filename='record'):   
    ##plot Figure 6 (left)
    with open('./'+filename+'/record_mse2_opt_lam.txt','rb') as file:    
        x,mean,sd,theory,record=pickle.load(file)
    length=x.shape[0]
    ind=np.arange(0,length,2)
    x,mean,sd,theory,record=x[ind],mean[ind],sd[ind],theory[ind],record[:,ind]
    plt.errorbar(x,mean,sd,
    label='nMSE',linewidth=2)
    plt.plot(x,theory,label='MSE',linewidth=2) 
    plt.grid(linestyle='dotted')
    plt.xlabel(r'$\mathbb{\delta}$',fontsize=20)
    plt.legend(fontsize=20)
    plt.tick_params(labelsize=17)
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.show()
    
    plt.savefig('./user_figures/numerical_mse_opt.png')  
    plt.close() 


##plot Figure 6(left)
##plot_emse('record')  ##'user_record'
        


















