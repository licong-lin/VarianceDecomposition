import numpy as np 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


'''
This file aims to plot figure 1--5 in the paper.
'''


def theta_1(ga,lam):  
    a=-lam+ga-1
    return (a+(a**2+4*ga*lam)**0.5)/(2*lam*ga)
def theta_2(ga,lam):
    return 1/(2*ga*lam**2)*(ga-1+((ga+1)*lam+(ga-1)**2)/((-lam+ga-1)**2+4*lam*ga)**0.5)


def bias_var_mse(pi,delt,kind='var',alpha=1,sig=0.3,lam='opt'):

    if lam=='opt':      ##optimal lambda
        lam=delt*(1-pi+(sig/alpha)**2)
    ga=pi*delt
    th_1=((-lam+ga-1)+((-lam+ga-1)**2+4*lam*ga)**0.5)/(2*lam*ga)
    th_2=(ga-1)/(2*lam**2*ga)+((ga+1)*lam+(ga-1)**2)/(2*lam**2*ga*((-lam+ga-1)**2+4*lam*ga)**0.5)

    if kind=='bias':
        f=alpha**2*(1-pi+lam*pi*th_1)**2
    elif kind=='var':
        f=alpha**2*pi*(1-pi+(pi-1)*(2*lam-delt)*th_1-pi*lam**2*\
            th_1**2+lam*(lam-delt+ga)*th_2)+sig**2*ga*(th_1-lam*th_2)
    elif kind=='mse':
        c=1-pi+(sig/alpha)**2
        f=alpha**2*(1-pi+ga*c*th_1+(lam-delt*c)*lam*pi*th_2)
    return f


def sobol_indices(pi,delt,kind='v_s',alpha=1,sig=0.3,lam='opt'):
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





##########################################################################









##plot figures



def plot_variances(pi,lam,alpha=1,sig=0.3,seq='sil'):  
    
    ##plot Figure 1
    
    delt=np.concatenate([np.linspace(0.01,0.2,500),np.linspace(0.2,10,1000)])
    pi=np.ones(1500)*pi
    bias=bias_var_mse(pi,delt,'bias',alpha,sig,lam)
    v_l=sobol_indices(pi,delt,'v_l',alpha,sig,lam)
    v_s=sobol_indices(pi,delt,'v_s',alpha,sig,lam)
    v_i=sobol_indices(pi,delt,'v_i',alpha,sig,lam)
    v_li=sobol_indices(pi,delt,'v_li',alpha,sig,lam)
    v_sl=sobol_indices(pi,delt,'v_sl',alpha,sig,lam)
    v_si=sobol_indices(pi,delt,'v_si',alpha,sig,lam)
    v_sli=sobol_indices(pi,delt,'v_sli',alpha,sig,lam)
    var=v_s+v_i+v_sl+v_si+v_sli
    mse=var+bias
    if len(seq)==3:
       
        a_3=eval('v_'+seq[2])
        try:
            a_2=eval('v_'+seq[1:])+eval('v_'+seq[1])
        except:
            a_2=eval('v_'+seq[2:0:-1])+eval('v_'+seq[1])
        a_1=var-a_2-a_3

        lw=2.5 ##set the linewidth


        plt.gcf().subplots_adjust(bottom=0.15)
        plt.plot(delt,mse,label=r'$MSE$',linestyle='-',linewidth=lw)
        plt.plot(delt,bias,label=r'$Bias^2$',linestyle='--',linewidth=lw)
        

        allcolor=[0,0,0]
        alllinestyle=[0,0,0]
        dic2={'l':('red','-.'),'s':('green',(0,(3,1,1,1,1,1))),'i':('purple',(0,(1,1)))}

        plt.plot(delt,a_1,label=r'$\Sigma_{}^{}$'.replace('_{}^{}','_{'+seq+'}^{'+seq[0]+'}'),color=dic2[seq[0]][0],linestyle=dic2[seq[0]][1],linewidth=lw)
        plt.plot(delt,a_2,label=r'$\Sigma_{}^{}$'.replace('_{}^{}','_{'+seq+'}^{'+seq[1]+'}'),color=dic2[seq[1]][0],linestyle=dic2[seq[1]][1],linewidth=lw) 
        plt.plot(delt,a_3,label=r'$\Sigma_{}^{}$'.replace('_{}^{}','_{'+seq+'}^{'+seq[2]+'}'),color=dic2[seq[2]][0],linestyle=dic2[seq[2]][1],linewidth=lw)  
        #plt.plot(delt,a_1,label=r'$\Sigma_{label}$',color='red',linestyle='-.')
        #plt.plot(delt,a_2,label=r'$\Sigma_{sample}$',color='green',linestyle=(0,(3,1,1,1,1,1)))
        #plt.plot(delt,a_3,label=r'$\Sigma_{init}$',color='purple',linestyle=(0,(1,1)))
    else:
        plt.plot(delt,v_s,label=r'$V_s$')
        plt.plot(delt,v_i,label=r'$V_i$')
        plt.plot(delt,v_sl,label=r'$V_{sl}$')
        plt.plot(delt,v_si,label=r'$V_{si}$')
        plt.plot(delt,v_sli,label=r'$V_{sil}$')
    plt.legend(fontsize=17,ncol=2)
    plt.xlabel(r'$\mathbb{\delta}$',fontsize=20)
    plt.grid(linestyle='dotted')
    plt.tick_params(labelsize=17)
    #plt.show()
    plt.savefig('./user_figures/vars_'+seq+'.png')
    plt.close()
#plot_variances(0.8,0.01,alpha=1,sig=0.3,seq='lsi')   ##'lis' , 'sli'  











def plot_decomp(kind='var',alpha=1,sig=0.3,lam='opt',plot='2d'):   
    
    ##plot 2,3-dim figures
    ##plot Figure 2, 3(left), 4, 5  

    dic={'var':'Var','bias':'Bias^2','mse':'MSE',
    'v_s':'V_s','v_i':'V_i','v_sl':'V_{sl}','v_si':'V_{si}','v_sli':'V_{sli}'}
    pi=np.linspace(0.0,1.0,1500)
    delt=np.linspace(0.1,5,1500)              
    pi,delt=np.meshgrid(pi,delt)
    if kind in ['var','bias','mse']:
        f=bias_var_mse(pi,delt,kind,alpha,sig,lam)
    else:
        f=sobol_indices(pi,delt,kind,alpha,sig,lam)
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
        plt.savefig('./user_figures/'+kind+'_'+n1+'.png')  
        plt.close()


#for kind in ['var','bias','mse','v_s','v_i','v_sl','v_si','v_sli']:   ##heat maps (Figure 3(left), 4, 5)
    #plot_decomp(kind,alpha=1,sig=0.3,lam='opt',plot='2d')
    #plot_decomp(kind,alpha=1,sig=0.3,lam=0.01,plot='2d')

#for kind in ['var','bias','mse']:   ## 3-dim wireframe (Figure 2)
    #plot_decomp(kind,alpha=1,sig=0.3,lam='opt',plot='3d')








def plot_one_dim(kind='mse',alpha=1,sig=0.1,lam='opt',plot=['1',0.9]):  
    ##plot 1-dim figures. (Figure 3(right))

    dic={'var':'Var','bias':'Bias^2','mse':'MSE',
    'v_s':'V_s','v_i':'V_i','v_sl':'V_{sl}','v_si':'V_{si}','v_sli':'V_{sli}'}
    linestyle=['-','--','-.',(0,(3,1,1,1,1,1)),(0,(1,1))]
    i=0
    num=2000
    pi=np.linspace(0,1.0,num)
    delt=np.linspace(0.25,20,num)              
    for x in plot[1:]:

        if plot[0]=='1':       ##1d plot, fix the first dim
            pi=np.ones(num)*x
        elif plot[0]=='2':     ##1d plot, fix the second dim
            delt=np.ones(num)*x
        f=bias_var_mse(pi,delt,kind,alpha,sig,lam)
        if plot[0]=='1':
            plt.plot(1/delt,f,label=str(r'$\pi=$')+str(x),linestyle=linestyle[i])
            i+=1
        else:
            plt.plot(pi,f,label=str(r'$\mathbb{\delta}=$')+str(x))
            plt.set_xlabel(r'$\pi$')
    if plot[0]=='1':
        plt.xlabel(r'$1/\mathbb{\delta}$')
    else:
        plt.xlabel(r'$\pi$')
    plt.grid(linestyle='dotted')
    plt.ylabel(r'${}$'.replace('{}',dic[kind]))
    plt.legend(fontsize=20)
    #plt.show()
    plt.savefig('./user_figures/'+kind+'_one_dim.png')
    plt.close()

#plot_one_dim(kind='mse',lam=0.01,plot=['1',0.3,0.5,0.7,0.9,0.95])




























