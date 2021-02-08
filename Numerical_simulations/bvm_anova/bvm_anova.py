import numpy as np 
from matplotlib import pyplot as plt
import pickle




'''
This file is used to perform numerical simulations in section 4.1, 4.2, 4.3 (Figure 1(left), 7(left), 8,9).
We assume $X$ has i.i.d. N(0,1) entries for simplicity. The results also hold
for non-gaussian $X$ (with i.i.d. zero-mean, one-variance, finite $8+\eta$ moment entries).

'''





def theta_1(ga,lam):
    a=-lam+ga-1
    return (a+(a**2+4*ga*lam)**0.5)/(2*lam*ga)
def theta_2(ga,lam):
    return 1/(2*ga*lam**2)*(ga-1+((ga+1)*lam+(ga-1)**2)/((-lam+ga-1)**2+4*lam*ga)**0.5)

def bias_var_mse(pi,delt,terms,alpha=1,sig=0.3,lam='opt'):
    if lam =='opt':      ##optimal lambda
        lam=delt*(1-pi+(sig/alpha)**2)
    ga=pi*delt
    th_1=theta_1(ga,lam)
    th_2=theta_2(ga,lam)

    value=np.zeros((len(ga),terms))

    c=1-pi+(sig/alpha)**2
    if terms==1:   ##mse2
        value[:,0]=alpha**2*(1-pi+ga*c*th_1+(lam-delt*c)*lam*pi*th_2)    ##mse
        return value

    value[:,0]=alpha**2*(1-pi+lam*pi*th_1)**2    ##bias
    value[:,1]=alpha**2*pi*(1-pi+(pi-1)*(2*lam-delt)*th_1-pi*lam**2*\
            th_1**2+lam*(lam-delt+ga)*th_2)+sig**2*ga*(th_1-lam*th_2)   ##variance
    value[:,2]=alpha**2*(1-pi+ga*c*th_1+(lam-delt*c)*lam*pi*th_2)    ##mse
    return value

def anova_indices(pi,delt,alpha=1,sig=0.3,lam='opt'):
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

   
    kind=['v_s','v_i','v_si','v_sl','v_sli']
    value=np.zeros((len(ga),5))
    for i0 in range(5):
        if kind[i0]=='v_s':
            f=alpha**2*(1-2*tlam*tth_1+tlam**2*tth_2-pi**2\
            *(1-lam*th_1)**2)
        elif kind[i0]=='v_i':
            f=alpha**2*pi*(1-pi)*(1-lam*th_1)**2
        elif kind[i0]=='v_si':
            f=alpha**2*(pi*(1-2*lam*th_1+lam**2*th_2+\
                (1-pi)*delt*(th_1-lam*th_2)-(1-pi)*(1-lam*th_1)**2)\
            -1+2*tlam*tth_1-tlam**2*tth_2)
        elif kind[i0]=='v_sl':
            f=sig**2*delt*(tth_1-tlam*tth_2)
        elif kind[i0]=='v_sli':
            f=sig**2*delt*(pi*(th_1-lam*th_2)-(tth_1-tlam*tth_2))
        value[:,i0]=f
    return value

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

def ar1_cov(r,d):     ##AR-1 covariance matrix
    if r=='iso':
        return False
    a0=np.zeros((d,d))
    for t1 in range(d):
        for t2 in range(d):
            a0[t1,t2]=r**np.abs(t2-t1)
    return a0




def num_bias_var_mse(p,d,n,lam,alpha=1,sig=0.3,k=100):
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
    s=np.zeros(3)   ##bias,var mse
    s[0]= alpha**2/d*np.linalg.norm(EM-np.eye(d))**2  ##bias
    s[1]= sig**2*EtrtMtMT+alpha**2/d*(EtrMMT-np.linalg.norm(EM)**2)   ##var
    s[2]=sig**2*EtrtMtMT+alpha**2/d*(EtrMMT+d-2*np.trace(EM))    ##mse

    return s



def get_trace(M,cov):
    return np.trace(M@M.T@cov)
def num_bias_var_mse_ar1(p,d,n,lam,alpha=1,sig=0.3,k=100,cov=0.9):
    EtM=0
    EM=0
    EMMT=np.zeros((d,d))
    EtrMMT=0
    EtMtMT=np.zeros((d,d))
    EtrtMtMT=0
    covariance=ar1_cov(cov,d)

    for i in range(k):        
        X=np.random.multivariate_normal(np.zeros(d), covariance, n)   ##general covariance

        ##another sample distribution
        #X=(3/14)**0.5*np.random.choice(np.array([-2,-1,3]),(n,d)) 

        W=gen_orth(p,d)
        tM,M=numerical_M(W,X,lam)
        EtM=EtM+tM
        EM=EM+M
        EMMT=EMMT+M@M.T
        EtMtMT=EtMtMT+tM@tM.T

    EtrMMT=np.trace(EMMT@covariance)
    EtrtMtMT=np.trace(EtMtMT@covariance)


    EtM,EM,EtrMMT,EtrtMtMT=EtM/k,EM/k,EtrMMT/k,EtrtMtMT/k
    s=np.zeros(3)   ##bias,var mse
    s[0]= alpha**2/d*get_trace(EM-np.eye(d),covariance) ##bias

    s[1]= sig**2*EtrtMtMT+alpha**2/d*(EtrMMT-get_trace(EM,covariance))   ##var
    s[2]=sig**2*EtrtMtMT+alpha**2/d*(EtrMMT+np.trace((np.eye(d)-2*EM)@covariance))    ##mse

    return s


def num_mse2(p,d,n,lam,alpha=1,sig=0.3,k=100):  ##estimate the mse directly
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



def num_mse2_ar1(p,d,n,lam,alpha=1,sig=0.3,k=100,cov=0.9):  ##estimate the mse directly
    mse=0
    covariance=ar1_cov(cov,d)

    for i in range(k):
        X=np.random.multivariate_normal(np.zeros(d), covariance, n)


        ##another sample distribution
        #X=(3/14)**0.5*np.random.choice(np.array([-2,-1,3]),(n,d)) 

        W=gen_orth(p,d)
        theta=np.random.normal(0,alpha/d**0.5,(d,1))
        eps=np.random.normal(0,sig,(n,1))

        x=np.random.multivariate_normal(np.zeros(d), covariance, 1)

        
        ##another sample distribution
        #x=(3/14)**0.5*np.random.choice(np.array([-2,-1,3]),(1,d)) 


        Y=X@theta+eps
        hf=x@numerical_M(W,X,lam)[0]@Y
        mse=mse+(hf-x@theta)**2
    mse=mse/k
    return mse    



def numerical_anova(p,d,n,lam,alpha=1,sig=0.3,k=20):
    
        
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

    
    kind=['v_s','v_i','v_si','v_sl','v_sli']

    value=np.zeros(len(kind))

    for i0 in range(len(kind)):    
        s=0
        if kind[i0]=='v_s':   
            for i in range(k):
                s=s+np.linalg.norm(E_wM[i]-EM)**2
            s=alpha**2/d*s/k
        elif kind[i0]=='v_i':
            for i in range(k):
                s=s+np.linalg.norm(E_XM[i]-EM)**2
            s=alpha**2/d*s/k
        elif kind[i0]=='v_si':
            for i in range(k):
                s=s+np.linalg.norm(E_XM[i]-EM)**2+np.linalg.norm(E_wM[i]-EM)**2
            s=s/k
            s=alpha**2/d*(-np.linalg.norm(EM)**2-s+EtrMMT)        
        elif kind[i0]=='v_sl':
            for i in range(k):
                s=s+np.linalg.norm(E_wtM[i]-EtM)**2
            s=sig**2*s/k
        elif kind[i0]=='v_sli':
            for i in range(k):
                s=s+np.linalg.norm(E_XtM[i]-EtM)**2+np.linalg.norm(E_wtM[i]-EtM)**2
            s=s/k
            s=sig**2*(-s+EtrtMtMT-np.linalg.norm(EtM)**2)

        value[i0]=s

    
    return value



def numerical_anova_ar1(p,d,n,lam,alpha=1,sig=0.3,k=20,cov=0.9):
    
        
    allW=[0 for i in range(k)]
    allX=[0 for i in range(k)]
    E_wM=[0 for i in range(k)]
    E_XM=[0 for i in range(k)]
    E_wtM=[0 for i in range(k)]
    E_XtM=[0 for i in range(k)]

    covariance=ar1_cov(cov,d)

    for i in range(k):
        allW[i]=gen_orth(p,d)
       

        allX[i]=np.random.multivariate_normal(np.zeros(d), covariance, n)   ##general covariance

    
    EtM=0
    EM=0
    EMMT=np.zeros((d,d))
    EtMtMT=np.zeros((d,d))
    for i in range(k):
        for j in range(k):
            tM,M=numerical_M(allW[i],allX[j],lam)
            E_XM[i]=E_XM[i]+M
            E_XtM[i]=E_XtM[i]+tM
            E_wM[j]=E_wM[j]+M
            E_wtM[j]=E_wtM[j]+tM
            EM=EM+M
            EtM=EtM+tM
            EMMT=EMMT+M@M.T
            EtMtMT=EtMtMT+tM@tM.T
        E_XM[i]=E_XM[i]/k
        E_XtM[i]=E_XtM[i]/k
    for j in range(k):
        E_wM[j]=E_wM[j]/k
        E_wtM[j]=E_wtM[j]/k
    EtrMMT=np.trace(EMMT@covariance)
    EtrtMtMT=np.trace(EtMtMT@covariance)
    EM,EtM,EtrMMT,EtrtMtMT=EM/k**2,EtM/k**2,EtrMMT/k**2,EtrtMtMT/k**2

    
    kind=['v_s','v_i','v_si','v_sl','v_sli','v_l','v_li']
    
    value=np.zeros(len(kind))

    for i0 in range(len(kind)):    
        s=np.zeros((d,d))
        if kind[i0]=='v_s':   
            for i in range(k):
                s=s+(E_wM[i]-EM)@(E_wM[i]-EM).T
            s=alpha**2/d*np.trace(s@covariance)/k
        elif kind[i0]=='v_i':
            for i in range(k):
                s=s+(E_XM[i]-EM)@(E_XM[i]-EM).T
            s=alpha**2/d*np.trace(s@covariance)/k
        elif kind[i0]=='v_si':
            s=alpha**2/d*(-get_trace(EM,covariance)+EtrMMT)-value[0]-value[1]        
        elif kind[i0]=='v_sl':
            for i in range(k):
                s=s+(E_wtM[i]-EtM)@(E_wtM[i]-EtM).T
            s=sig**2*np.trace(s@covariance)/k

        elif kind[i0]=='v_sli':
            for i in range(k):
                s=s+(E_XtM[i]-EtM)@(E_XtM[i]-EtM).T
            s=np.trace(s@covariance)/k
            s=sig**2*(-s+EtrtMtMT-get_trace(EtM,covariance))-value[3]


        elif kind[i0]=='v_l':         ##general covariance
            s=get_trace(EtM,covariance)*sig**2
        elif kind[i0]=='v_li':        ##general covariance
            for i in range(k):
                s=s+(E_XtM[i]-EtM)@(E_XtM[i]-EtM).T
            s=np.trace(s@covariance)*sig**2/k
        value[i0]=s

    
    return value




#############################################################################





def simulation(ga,kind='mse',alpha=1,sig=0.3,lam='opt',axis=2,n=150,k=10,cov='iso'):    
    ## Simulations for section 4.1, 4.2, 4.3. (Figure 1(left), 7(left), 8,9) 
    ## This function is used to perform simulations and record the results (but not plot the figures).
    ## This function will store all simulation results in the './user_record/' file.
    lam_r=lam  
    if axis==1:
        pi=np.linspace(0.1,1,19)
        delt=ga
        d=np.round(n*delt)*np.ones(shape=pi.shape)
        p=np.round(d*pi)
    elif axis==2:
        pi=ga
        delt=np.linspace(0.1,5,50)    
        d=np.round(n*delt)
        p=np.round(d*pi)
        pi=pi*np.ones(shape=p.shape)
    if lam =='opt':
        lam=delt*(1-pi+(sig/alpha)**2)
    else:
        lam=lam*np.ones(shape=p.shape[-1])

    times=p.shape[-1]   #number of settings


    terms={'bvm':3,'mse2':1,'anova':5}
    if cov!='iso':
        terms['anova']=7

    record=np.zeros((k,times,terms[kind]))
    for i in range(times):
        d0=int(d[i])
        p0=int(p[i])
        if cov=='iso':
            for j in range(k):
                if kind =='bvm':   
                    record[j,i,:]=num_bias_var_mse(p0,d0,n,lam[i],alpha,sig,100)
                    theory=bias_var_mse(pi,delt,3,alpha,sig,lam)
                elif kind =='mse2':    ## direct simulation (section 4.1)
                    record[j,i,:]=num_mse2(p0,d0,n,lam[i],alpha,sig,400)
                    theory=bias_var_mse(pi,delt,1,alpha,sig,lam)
                elif kind=='anova':
                    record[j,i]=numerical_anova(p0,d0,n,lam[i],alpha,sig,20)  ## 50 for Figure 1(left)  
                    theory=anova_indices(pi,delt,alpha,sig,lam)
            print('setting '+str(i+1)+' finished ('+ str(times)+' settings in total).')

        else:     ##general covariance
            for j in range(k):
                if kind =='bvm':   
                    record[j,i,:]=num_bias_var_mse_ar1(p0,d0,n,lam[i],alpha,sig,100,cov)
                elif kind =='mse2':    ## direct simulation (section 4.1)
                    record[j,i,:]=num_mse2_ar1(p0,d0,n,lam[i],alpha,sig,400,cov)
                elif kind=='anova':
                    record[j,i,:]=numerical_anova_ar1(p0,d0,n,lam[i],alpha,sig,20,cov)
            print('setting '+str(i+1)+' finished ('+ str(times)+' settings in total).')

    sd=np.std(record,0)
    mean=np.mean(record,0)
    
    if axis==1:
        x=pi
    else: 
        x=delt 

    
    ##(used for illustration)
    '''
    for i in range(terms[kind]):
        plt.errorbar(x,mean[:,i],sd[:,i],fmt='-',label='simulation')      
        if cov=='iso':
            plt.plot(x,theory[:,i],label='theory')
       
        plt.ylim(0,1)
        plt.grid(linestyle='dotted')
        plt.xlabel(r'$\gamma_{}$'.replace('{}','{'+str(axis)+'}'))
        plt.ylabel(r'${}$'.replace('{}',kind))
        plt.legend()
        plt.show()
    '''

    if cov=='iso':

        with open('./user_record/record_'+kind+'_{}_lam.txt'.format(lam_r),'wb') as file:   ##record data
            pickle.dump([x,mean,sd,theory,record],file)
    else:   ##general covariance
        with open('./user_record/record_general_cov_'+kind+'_lam_{}_cov_{}'.format(lam[0],cov)+'.txt','wb') as file:   ##record data
            pickle.dump([x,mean,sd,record],file)









#simulation(0.8,'anova',alpha=1,sig=0.3,lam=0.01,axis=2,n=150,k=5)  ## simulations for Figure 1(right)


#simulation(0.8,'bvm',alpha=1,sig=0.3,lam='opt',axis=2,n=150,k=5)     ## simulations for Figure 8
#simulation(0.8,'anova',alpha=1,sig=0.3,lam='opt',axis=2,n=150,k=5)   ## simulations for Figure 8
#simulation(0.8,'mse2',alpha=1,sig=0.3,lam='opt',axis=2,n=150,k=20)   ## simulations for Figure 7(left)



## general covariance
#simulation(0.8,'bvm',alpha=1,sig=0.3,lam=0.001,axis=2,n=150,k=5,cov=0.9)      ##simulations for Figure 9
#simulation(0.8,'anova',alpha=1,sig=0.3,lam=0.001,axis=2,n=150,k=5,cov=0.9)    ##simulations for Figure 9
































def plot_numerical_bvm_anova(kind,filename='record',cov='iso',lam='opt'):  
    ##Plot Figure 8 (using simulation resuts in 'record'' or 'user_record' file).
    if kind=='bvm':
        items=['mse','bias','var']
        names=['MSE','Bias^2','Var']
    elif kind=='anova':
        items=['v_s','v_i','v_si','v_sl','v_sli','v_l','v_li']
        names=['V_{s}','V_{i}','V_{si}','V_{sl}','V_{sli}','V_{l}','V_{li}']
    
    if cov=='iso':
        with open('./'+filename+'/record_'+kind+'_{}_lam.txt'.format(lam),'rb') as file:    
            x,mean,sd,theory,record=pickle.load(file)
    else:
        with open('./'+filename+'/record_general_cov_'+kind+'_lam_{}_cov_{}'.format(lam,cov)+'.txt','rb') as file:  
            x,mean,sd,record=pickle.load(file)
    
    if kind=='bvm':
        seq=[2,0,1]
    elif kind=='anova' and cov=='iso':
        seq=[0,1,2,3,4]
    else:
        seq=[0,1,2,3,4,5,6]
    if cov=='iso':
        for k in range(len(seq)):
            plt.plot(x,theory[:,seq[k]],label=r'${}$'.replace('{}',names[k]),linewidth=3)
        
    
    for k in range(len(seq)):   
        plt.errorbar(x,mean[:,seq[k]],sd[:,seq[k]],linestyle='--',linewidth=1,marker='.',markersize=7,
        label='n'+r'${}$'.replace('{}',names[k]))

      
      

    plt.grid(linestyle='dotted')

    if kind=='bvm' and cov=='iso':
        if lam=='opt':
            plt.ylim(0,1.2) 
        else:
            plt.ylim(0,1.5)   
    elif kind=='anova' and cov=='iso':
        if lam=='opt':
            plt.ylim(0,0.2) 
        else:
            plt.ylim(0,0.5)
    elif kind=='bvm':
        plt.ylim(0,0.8)
    elif kind=='anova':
        plt.ylim(0,0.3)
    plt.xlabel(r'$\mathbb{\delta}$',fontsize=20)
    plt.legend(ncol=2,fontsize=14)
    plt.tick_params(labelsize=15)
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.show()

    if cov=='iso':
        plt.savefig('./user_figures/numerical_'+kind+'_{}.png'.format(lam))
    else:
        plt.savefig('./user_figures/record_general_cov_'+kind+'_lam_{}_cov_{}'.format(lam,cov)+'.png')
    plt.close()


##'bvm':  plot Bias, Var and MSE (Figure 8 left).
##'anova': plot variance components (anova) (Figure 8 right).

#plot_numerical_bvm_anova('anova',filename='record',cov=0.9,lam=0.001)  ##filename='user_record'
#plot_numerical_bvm_anova('bvm',filename='record',cov=0.9,lam=0.001)  ##filename='user_record', general covariance   


#plot_numerical_bvm_anova('anova',filename='record',cov='iso')  ##filename='user_record'
#plot_numerical_bvm_anova('bvm',filename='record',cov='iso')  ##filename='user_record' 

#plot_numerical_bvm_anova('anova',filename='record',cov='iso',lam=0.01)  ##filename='user_record'  (Figure 1 (right))




def plot_emse(filename='record'):   
    ##plot Figure 6 (left)
    with open('./'+filename+'/record_mse2_opt_lam.txt','rb') as file:    
        x,mean,sd,theory,record=pickle.load(file)
    length=x.shape[0]
    ind=np.arange(0,length,2)
    x,mean,sd,theory,record=x[ind],mean[ind,0],sd[ind,0],theory[ind,0],record[:,ind,0]
    plt.errorbar(x,mean,sd,
    label='nMSE',linewidth=2)
    plt.plot(x,theory,label='MSE',linewidth=2) 
    plt.grid(linestyle='dotted')
    plt.xlabel(r'$\mathbb{\delta}$',fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=15)
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.show()
    
    plt.savefig('./user_figures/numerical_mse2_opt.png')  
    plt.close() 


##plot Figure 7(left)
#plot_emse('record')  ##'user_record'
        


















