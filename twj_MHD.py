from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pdb as pdb
from math import pi
import pickle as pickle

def vecMHD(rho,vx,vy,vz,Bx,By,Bz,P):
    Bsq=Bx*Bx+By*By+Bz*Bz
    vsq=vx*vx+vy*vy+vz*vz
    E=P/(gam-1)+.5*rho*vsq+Bsq*.5
    Bdv=Bx*vx+By*vy+Bz*vz
    Pst=P+Bsq/2
    r=np.array([rho,rho*vx,rho*vy,rho*vz,E,By,Bz])
    F=np.array([rho*vx,rho*vx*vx+P+.5*Bsq-Bx*Bx,rho*vx*vy-Bx*By,rho*vx*vz-Bx*Bz,(E+Pst)*vx-Bdv*Bx,By*vx-Bx*vy,Bz*vx-Bx*vz])
    return r, F
def comp(r):
    rho=np.copy(r[0,:])
    vx,vy,vz=r[1,:]/rho,r[2,:]/rho,r[3,:]/rho
    By=r[5,:]
    Bz=r[6,:]
    Bsq=Bx*Bx+By*By+Bz*Bz
    vsq=vx*vx+vy*vy+vz*vz
    P=(r[4,:]-.5*rho*vsq-Bsq*.5)*(gam-1)
    return np.array([rho,vx,vy,vz,E,By,Bz,P])

def minmod(x,y,z):
    return .25*np.abs(np.sign(x)+np.sign(y))*(np.sign(x)+np.sign(z))*np.amin([np.abs(x),np.abs(y),np.abs(z)],axis=0)

def cmethod(r):
    cr=np.zeros([r.shape[0],(N+1)*2])
    thet=1.5
    cr[:,::2]=r[:,1:N+2]+0.5*minmod(thet*(r[:,1:N+2]-r[:,:N+1]),0.5*(r[:,2:N+3]-r[:,:N+1]),thet*(r[:,2:N+3]-r[:,1:N+2]))
    cr[:,1::2]=r[:,2:N+3]-0.5*minmod(thet*(r[:,2:N+3]-r[:,1:N+2]),0.5*(r[:,3:N+4]-r[:,1:N+2]),thet*(r[:,3:N+4]-r[:,2:N+3]))
    return cr

def LU(r,F,delx,tmpt=None):
    cr=comp(r)
    vx=cr[1,:]
    bx=Bx/np.sqrt(cr[0,:])
    Csq=(gam*cr[6,:]/cr[0,:])
    #vsq=cr[1,:]*cr[1,:]+cr[2,:]*cr[2,:]+cr[3,:]*cr[3,:]
    bsq=(Bx*Bx+cr[5,:]*cr[5,:]+cr[4,:]*cr[4,:])/cr[0,:]
    #H=(cr[4,:]+cr[6,:]+bsq*.5)/cr[0,:]
    #a=(gam-1)*(H-vsq*.5-bsq/cr[0,:])
    Ca=np.sqrt(Bx*Bx/(vx*cr[0,:])) # CHECK TO SEE IF ITS VXNOT
    Cf=np.sqrt(.5*(Csq+bsq+np.sqrt((Csq+bsq)**2-4*Csq*bx*bx)))
    Cs=np.sqrt(.5*(Csq+bsq-np.sqrt((Csq+bsq)**2-4*Csq*bx*bx)))
    lambp=np.array([vx+Cs,vx+Ca,vx+Cf])
    lambm=np.array([vx-Cs,vx-Ca,vx-Cf])
    alphp=np.amax([np.repeat(0,N+1),lambp[0,::2],lambp[1,::2],lambp[2,::2],lambp[0,1::2],lambp[1,1::2],lambp[2,1::2]],axis=0)
    alphm=np.amax([np.repeat(0,N+1),-1*lambm[0,::2],-1*lambm[1,::2],-1*lambm[2,::2],-1*lambm[0,1::2],-1*lambm[1,1::2],-1*lambm[2,1::2]],axis=0)
    flag=0
    if tmpt == None:
        flag=1
        tmpt=delx*.5/max(np.append(alphp,alphm))
    Fip=((alphp[1:]*F[:,2::2]+alphm[1:]*F[:,3::2]-alphp[1:]*alphm[1:]*(r[:,3::2]-r[:,2::2]))/(alphp[1:]+alphm[1:]))
    Fim=((alphp[:-1]*F[:,:-2:2]+alphm[:-1]*F[:,1:-1:2]-alphp[:-1]*alphm[:-1]*(r[:,1:-1:2]-r[:,:-2:2]))/(alphp[:-1]+alphm[:-1]))
    if flag:
        return tmpt*(Fip-Fim)/delx, tmpt
    else:
        return tmpt*(Fip-Fim)/delx
    
def hydroHO(rho,vx,vy,vz,E,Bx,By,Bz,P,xgrid,ttot,tf):
    N=xgrid.size
    delx=(xgrid[-1]-xgrid[0])/N
    r,F=vec(rho,vx,vy,vz,E,Bx,By,Bz,P)
    jq=[rho,vx,vy,vz,E,Bx,By,Bz,P]
    r1=None
    r2=None
    jdelt=None
    for j in range(3):
        crho,cvx,cvy,cvz,cE,cBx,cBy,cBz,cP=cmethod(jq)
        jr,jF=vec(crho,cvx,cvy,cvz,cE,cBx,cBy,cBz,cP)
        if j == 0:
            Utmp, jdelt=LU(jr,jF,delx)
            if ttot+jdelt > tf:
                jdelt=tf-ttot
                Utmp=LU(jr,jF,delx,tmpt=jdelt)
        else:
            Utmp=LU(jr,jF,delx,tmpt=jdelt)
        if j == 0:
            r1=np.copy(r)
            r1[:,2:-2]=r[:,2:-2]-Utmp
            jrho,jv,jP=comp(r1)
        if j == 1:
            r2=np.copy(r)
            r2[:,2:-2]=.75*r[:,2:-2]+.25*r1[:,2:-2]-.25*Utmp
            jrho,jv,jP=comp(r2)
        if j == 2:
            r[:,2:-2]=(1/3)*r[:,2:-2]+(2/3)*r2[:,2:-2]-(2/3)*Utmp
    if np.any(np.isnan(r)):
        pdb.set_trace()
    rho,v,P=comp(r)
    return rho,v,P,jdelt
