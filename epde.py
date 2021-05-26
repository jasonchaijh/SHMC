import math
import torch
import numpy as np
import matplotlib.pyplot as plt




device=torch.device("cpu")
dtype=torch.float64
a=torch.tensor([1,2,3])
a.device


class ePDE():
    def __init__(self,d,x,y,sigma2_y,l,nsim,stepsize):
        self.theta=torch.zeros([d,1],dtype=torch.float64,requires_grad=True)
        self.npt=y.shape[0]
        self.d=d
        self.x=x
        self.y=y
        self.l=l
        self.sig2y=sigma2_y
        self.nsim=nsim    #number of simulation per round 
        self.ss=stepsize
        self.acprat=0
        self.thetalist=np.zeros([6,0],dtype=np.float64)
        #thetalist is numpy array, x,y,theta are all torch tensors
        
        ev1=self.calc_eigv(0)
        ev2=np.sqrt(self.calc_eigv(0)*self.calc_eigv(1))
        ev3=self.calc_eigv(1)
        ev4=np.sqrt(self.calc_eigv(0)*self.calc_eigv(2))
        self.eigv=torch.tensor([[ev1],[ev2],[ev2],[ev3],[ev4],[ev4]],dtype=torch.float64)
        
    @staticmethod
    def hermite(x,degree):
        n=x.shape[0]
        retvar=torch.zeros([n,(degree+1)],dtype=torch.float64)
        retvar[:,0]=1.0
        if degree>0:
            retvar[:,[1]]=x
            for ii in range(1, degree):
                retvar[:,[(ii+1)]]=x*retvar[:,[ii]]-retvar[:,[(ii-1)]]*ii
        return retvar[:,[degree]]
    
    @staticmethod
    def hermite_mat(x,degree):
        n1=x.shape[0]
        n2=x.shape[1]
        retvar=torch.zeros([n1,(degree+1)*n2],dtype=torch.float64)
        retvar[:,0:n2]=1.0
        if degree>0:
            retvar[:,n2:2*n2]=x
            for ii in range(1, degree):
                retvar[:,(ii+1)*n2:(ii+2)*n2]=x*retvar[:,ii*n2:(ii+1)*n2]-retvar[:,(ii-1)*n2:ii*n2]*ii
        return retvar[:,[degree*n2,(degree+1)*n2]]
    
    def calc_eigv(self,n):
        #calc eigenvalues with degrees <= n
        alpha=1
        temp=(1+2/self.l**2/alpha**2)
        lamb=alpha/2**n/self.l**(2*n)/(alpha**2/2*(1+np.sqrt(temp))+1/2/self.l**2)**(n+1/2)
        return lamb
        
    def calc_eigf(self,x,n):
        #calc eigenfunctions with degree = n
        alpha=1
        temp=(1+2/self.l**2/alpha**2)
        #lamb=alpha/2**n/self.l**(2*n)/(alpha**2/2*(1+np.sqrt(temp))+1/2/self.l**2)**(n+1/2)
        phi=(temp)**(1/8)/np.sqrt(2**n*np.math.factorial(n))*torch.exp(-(np.sqrt(temp)-1)*alpha**2*x**2/2)*                self.hermite((temp)**(1/4)*alpha*x,n)
        if n>0:
            dphi=-phi*(np.sqrt(temp)-1)*alpha**2*x + (temp)**(1/8)/np.sqrt(2**n*np.math.factorial(n))*                torch.exp(-(np.sqrt(temp)-1)*alpha**2*x**2/2)*n*(temp)**(1/4)*alpha*                        self.hermite((temp)**(1/4)*alpha*x,n-1)
        else:
            dphi=-phi*(np.sqrt(temp)-1)*alpha**2*x
        return phi,dphi
        
               
        
    def calc_c(self,x):
        phi=torch.cat((self.calc_eigf(x,0)[0],self.calc_eigf(x,1)[0],self.calc_eigf(x,2)[0]),1)
        dphi=torch.cat((self.calc_eigf(x,0)[1],self.calc_eigf(x,1)[1],self.calc_eigf(x,2)[1]),1)
        val=torch.cat((phi[0,[0]]*phi[1,0],phi[0,[0]]*phi[1,1],phi[0,[1]]*phi[1,0],phi[0,[1]]*phi[1,1]                     ,phi[0,[0]]*phi[1,2],phi[1,[0]]*phi[0,2]),0).reshape([-1,1])
        dval1=torch.cat((dphi[0,[0]]*phi[1,0],dphi[0,[0]]*phi[1,1],dphi[0,[1]]*phi[1,0],dphi[0,[1]]*phi[1,1]                     ,dphi[0,[0]]*phi[1,2],phi[1,[0]]*dphi[0,2]),0).reshape([-1,1])
        dval2=torch.cat((phi[0,[0]]*dphi[1,0],phi[0,[0]]*dphi[1,1],phi[0,[1]]*dphi[1,0],phi[0,[1]]*dphi[1,1]                     ,phi[0,[0]]*dphi[1,2],dphi[1,[0]]*phi[0,2]),0).reshape([-1,1])
        c=torch.exp(self.theta.T@(val*self.eigv))
        dc=torch.tensor([[c*self.theta.T@(dval1*self.eigv)],[c*self.theta.T@(dval2*self.eigv)]],dtype=torch.float64)
        return c,dc
    
    def simU(self):
        sim_u=torch.zeros([self.npt,1],dtype=torch.float64)
        sim_du=torch.zeros([6,self.npt],dtype=torch.float64)
        for i in range(self.npt):
            usum=0
            #dusum=torch.zeros([6,1],dtype=torch.float64)
            for k in range(self.nsim):
                xt=self.x[:,[i]]
                tag=0
                counter=0
                while True:
                    counter+=1
                    c,dc=self.calc_c(xt)
                    #torch.manual_seed(1)
                    xn=xt+self.ss*dc+torch.sqrt(2*c)*np.sqrt(self.ss)*torch.normal(0,1,[2,1])
                    if xn[0]<=0:
                        #print(xt[0],xn[0])
                        ut=(xt[0]*xn[1]-xn[0]*xt[1])/(xt[0]-xn[0])
                        #print(ut)
                        tag=1
                    elif xn[0]>=1:
                        #print(xt[0]-1,xn[0]-1)
                        ut=1-((xt[0]-1)*xn[1]-(xn[0]-1)*xt[1])/(xt[0]-xn[0])
                        #print(ut)
                        tag=1
                    elif xn[1]<=0:
                        #print(xt[1],xn[1])
                        ut=(xt[1]*xn[0]-xn[1]*xt[0])/(xt[1]-xn[1])
                        #print(ut)
                        tag=1
                    elif xn[1]>=1:
                        #print(xt[1]-1,xn[1]-1)
                        ut=1-((xt[1]-1)*xn[0]-(xn[1]-1)*xt[0])/(xt[1]-xn[1])
                        #print(ut)
                        tag=1
                    if tag==1:
                        #print(counter)
                        usum+=ut
                        #self.theta.grad=None
                        #ut.backward()
                        #dusum+=self.theta.grad
                        break
                    xt=xn
            self.theta.grad=None
            usum.backward()
            dusum=self.theta.grad
            sim_u[i,0]=usum.item()/self.nsim
            sim_du[:,[i]]=dusum/self.nsim
        U=self.theta.T@self.theta/2+(self.y-sim_u).T@(self.y-sim_u)/(2*self.sig2y)
        dU=self.theta+sim_du@(self.y-sim_u)/self.sig2y
        return U,dU
        
        
    def hmc(self,L,epi):
        theta=self.theta.detach().clone()
        r=torch.normal(0,1,size=(self.d,1))
        rh=r
        U1,dU1=self.simU()
        rh=rh-epi/2*dU1
        print(U1)
        #print(dU1)
        for i in range(L):
            with torch.no_grad():
                #print(rh)
                #print(self.theta)
                self.theta=self.theta+epi*rh
            self.theta.requires_grad=True
            rh=rh-epi*self.simU()[1]
        with torch.no_grad():
            self.theta=self.theta+epi*rh
        self.theta.requires_grad=True
        U2,dU2=self.simU()
        rh=rh-epi/2*dU2  
        #print(['rh',rh])
        #print(['r',r])
        acpt=torch.exp(-U2+U1-rh.T@rh/2+r.T@r/2).item()
        print(acpt)
        #print(self.theta.requires_grad)
        if np.random.uniform()>acpt:
            self.theta=theta
            self.theta.requires_grad=True
        else:
            self.acprat+=1
        self.thetalist=np.c_[self.thetalist,self.theta.detach().clone().numpy()]
        return


