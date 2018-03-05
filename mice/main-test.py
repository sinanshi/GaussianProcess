import warnings
warnings.filterwarnings("ignore")

import sys
import os
import numpy as np
import time

import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy

import random
import math

from Gp import *

def main(argv=None):
    if argv==None:
        argv=sys.argv
    else:
        for argument in argv:
            pass

    adir=argv[0] #e.g. 'results/'
 
    if not os.path.exists(adir):
    	os.makedirs(adir)

    def initdesign():
    	return None;

    print 'GP example: BRANIN-2D'
   
    sequential=True;

    # Problem
    p_init=7;
    a=np.zeros(p_init);b=np.ones(p_init);
    myfun=lambda var: piston7D(var);

    #p=20;
    #a=np.zeros(p);b=np.ones(p);
    #myfun=lambda var: test20D16(var);

    #p=30;
    #a=np.zeros(p);b=np.ones(p);
    #myfun=lambda var: morris30D(var);

    #p=9;
    #a=np.zeros(p);b=np.ones(p);
    #myfun=lambda var: piston7D(var);

    #p=20;
    #a=np.zeros(p);b=np.ones(p);
    #myfun=lambda var: test20D8(var);

    #p=2;
    #a=np.zeros(p);b=np.ones(p);
    #myfun=lambda var: branin2D(var);

    #p=5;
    #a=np.zeros(p);b=np.ones(p);
    #myfun=lambda var: test5D(var);

    p=p_init;

    myfun_active=lambda var: myfun(var);

    # Training data
    M_D=2;

    ## LHD
    rs_D=2;
    m=M_D;
    #X=lhd(p,m,a,b,rs_D);
    maximin_num=10000;
    X=maximinLHD(p,m,a,b,maximin_num,rs_D);

    #def initdesign():
    #	m=21;
    #	nx,ny=(m,m)
    #	x = np.linspace(a[0],b[0],nx); 
    #	y = np.linspace(a[1],b[1],ny);
    #	Xinit=np.zeros((nx*ny,p));
    #	k=0;
    #	for i in range(nx):
    #		for j in range(ny):
    #			Xinit[k,0]=x[i];
    #			Xinit[k,1]=y[j];
    #			k=k+1;
    #	return Xinit;
    #Xinit=initdesign();
    #ind=np.array(range(0,Xinit.shape[0]))
    #np.random.seed(2);
    #np.random.shuffle(ind)
    #X=Xinit[ind[0:M_D],:].reshape((M_D,p));
    
    Y=myfun(X);

    if not sequential:
	XY=np.hstack([X,Y]);
        np.savetxt("".join([adir,"X.txt"]),X,fmt='%.16f')
        np.savetxt("".join([adir,"Y.txt"]),Y,fmt='%.16f')
        np.savetxt("".join([adir,"XY.txt"]),XY,fmt='%.16f')
    	return 0;

    # Screening
    seqmorris_no=0;
    xmorris=None;
    deltamorris=10;
    screening=False;
    s_tot=3;
    s=1;

    # GP Settings
    tau2_min=-12;tau2_max=-12;
    a_MLE=np.hstack([0.3*np.ones(p),tau2_min]);
    b_MLE=np.hstack([10.0*np.ones(p),tau2_max]);
    default_corr=np.hstack([0.75*np.ones(p),tau2_max]);

    # Setup GP
    GP=Gp();
    GP.set_domain(a,b);

    corrtype='matern52';
    GP.set_corr(corrtype);

    regfun_1=lambda x: 1.0;
    regfun_2=lambda x: x[0];
    regfun_3=lambda x: x[1];
    regfun_4=lambda x: x[2];
    regfun_5=lambda x: x[3];
    regfun_6=lambda x: x[4];

    coeff=np.zeros((p_init));
    regfun_linear=lambda x: np.dot(coeff,x);
    #regfun_bool=[1,2,3,4,5];
    #regfun=lambda x: scipy.delete(np.array([regfun_1(x),regfun_2(x),regfun_3(x),regfun_4(x),regfun_5(x),regfun_6(x)]),regfun_bool,0);
    regfun=lambda x: np.array([regfun_1(x)]);

    GP.set_reg('function',regfun);

    GP.set_data(X,Y);

    # Fit
    fit_corr_bool=True;
    ng=20;m=60;cr=0.9;mr=1.0/p;
    if not fit_corr_bool:
    	    corr=default_corr;
    else:
	    #print 'MLE-fit...',
	    #sys.stdout.flush();
	    #corr=GP.fit(a_MLE,b_MLE,M_gen,M_pop,cr,mr)
    	    #print 'Completed.'

	    print 'MLE-fit...',
	    sys.stdout.flush();

            timenow=time.time();
	    corr=GP.fit(a_MLE,b_MLE,ng,m,cr,mr);
	    timeelap=time.time()-timenow;
    	    print 'Completed.',timeelap

    # Build
    print 'Build GP...',
    sys.stdout.flush();
    GP.build(corr);

    print 'Completed.'

    X_D=np.zeros(X.shape);
    Y_D=np.zeros(Y.shape);
    X_D[:]=X;
    Y_D[:]=Y;

    print 'Initialize MICE'
    maxiter=20;
    if not initdesign()==None:
    	maxiter=initdesign().shape[0]-X_D.shape[0];
    M_cand=100;
    M_Mm=200;
    rs_Mm=777;
    tau2_MICE=0.0;
    iter_fit=range(0,100);
  

    def generate_cand(j):
    	rs_Mm=j;
	
	if initdesign()==None:
		X_cand_j=maximinLHD(p,M_cand,a,b,M_Mm,rs_Mm);	
	else:
		return Xinit;

	return X_cand_j;

    def generate_cand_morris(j,l,a,b,xmorris,deltamorris):
    	rs_Mm=j;
	if deltamorris==1:
		deltas=0.5;
	else:
		deltas=np.linspace(a[l],b[l],deltamorris);
	X_cand_l=np.zeros((deltamorris,a.shape[0]));
	for s in range(deltamorris):
		X_cand_l[s,:]=xmorris.copy()
		if deltamorris==1:
			X_cand_l[s,l]=X_cand_l[s,l]+deltas;
		else:
			X_cand_l[s,l]=deltas[s];
		if X_cand_l[s,l]>1.0:
			if deltamorris==1:
				X_cand_l[s,l]=X_cand_l[s,l]-2.0*deltas;
	return X_cand_l;


    # Out of sample
    sys.stdout.flush();
    M_trial=1000;
    rs_trial=4;
    Xtrial=lhd(p,M_trial,a,b,rs_trial);
    Ytrial=myfun(Xtrial);
    XtrialGP=Xtrial;

    # MI algorithm
    print 'MICE algorithm:'
    fast=False;
    l2_vec=[]
    nrmspe_vec=[]
    design_size_vec=[]
    GPD=GP;

    EE_i=np.zeros((maxiter,p));
    mu_i=np.zeros((maxiter,p));
    mustar_i=np.zeros((maxiter,p));
    sigma_i=np.zeros((maxiter,p));
    I_screen=range(0,p);
    corr_vec=[];
    pts_rmv=0;

    for j in range(maxiter):
    	print 'Iter:',j+1

	print 'the_shape',X_D.shape
        ## MI step

    	GPDc=Gp();
    	GPDc.set_domain(a,b);
    	GPDc.set_corr(GPD.corr_type);
	MICE_param=GPD.corr_param.copy()
        MICE_param[-1]=tau2_MICE;
        
	crit_best=-777;
	ibest=0;

	if seqmorris_no==0:
		X_cand_j=generate_cand(j);
		numcand=X_cand_j.shape[0]
	else:
		X_cand_j=generate_cand_morris(j,I_screen[seqmorris_no-1],a,b,xmorris,deltamorris);
		numcand=deltamorris;

	# remove candidates already in design
	rmv_list=[]
	for i in range(X_cand_j.shape[0]):
		x_i=X_cand_j[i,:]
		for t in range(X_D.shape[0]):
			x_t=X_D[t,:]
			if np.linalg.norm(x_i-x_t)<1E-5:
				rmv_list.append(i);

	X_cand_j=np.delete(X_cand_j,rmv_list,0)

	Y_cand_j=np.ones((X_cand_j.shape[0],1)); #dummy

	XC_MICE=np.zeros((numcand,X_cand_j.shape[1]+1));

	now=time.time();
	if fast:
		X_Dc=X_cand_j;
		Y_Dc=Y_cand_j;
    		GPDc.set_reg('function',regfun);
		GPDc.set_data(X_Dc,Y_Dc);
		#GPDc.meanX=GPD.meanX;
		#GPDc.stdX=GPD.stdX;
		#GPDc.Xs=np.divide((X_Dc-np.ones((X_cand_j.shape[0],1))*GPDc.meanX),np.ones((X_cand_j.shape[0],1))*GPDc.stdX);
		GPDc.build(MICE_param.copy());
		GPDc.sigma2=GPD.sigma2;

	for i in range(0,numcand):
		x=X_cand_j[i]

		if fast:
			invCC=np.zeros(GPDc.invCC.shape);invCC[:]=GPDc.invCC;
			R=np.zeros(GPDc.R.shape);R[:]=GPDc.R;
			MICEcrit=lambda s: MI_fast(X_Dc,GPD,GPDc,invCC,R,s);		
		else:
			X_Dc=np.delete(X_cand_j,(i),0)
			Y_Dc=np.delete(Y_cand_j,(i),0)
    			GPDc.set_reg('function',regfun);
			GPDc.set_data(X_Dc,Y_Dc);
			#GPDc.meanX=GPD.meanX;
			#GPDc.stdX=GPD.stdX;
			#GPDc.Xs=np.divide((X_Dc-np.ones((X_cand_j.shape[0]-1,1))*GPDc.meanX),np.ones((X_cand_j.shape[0]-1,1))*GPDc.stdX);
			GPDc.build(MICE_param.copy());
			GPDc.sigma2=GPD.sigma2;

			MICEcrit=lambda s: GPD.var(s)/GPDc.var(s);

		crit_i=MICEcrit(x);
		XC_MICE[i,:]=np.hstack([x,crit_i]);
		
		if crit_i>crit_best:
			xbest=x;
			crit_best=crit_i;
			ibest=i;

	print 'elap time MICE iter',time.time()-now;

        # Evaluate underlying function

	xeval=xbest;
	yeval=myfun_active(xbest);

        print 'selected, x:',xeval,'y(x):',yeval

	if seqmorris_no==0:
		xmorris=xbest;

        # Save data

	XC_MICE[:,p]=XC_MICE[:,p]/max(XC_MICE[:,p]);
	if p<3:
		np.savetxt("".join([adir,'XC_',str(X_D.shape[0]),'.txt']),XC_MICE,fmt='%.16f')
		np.savetxt("".join([adir,'Xd_',str(X_D.shape[0]),'.txt']),X_D,fmt='%.16f')

	X_D=np.vstack([X_D,xeval])
	Y_D=np.vstack([Y_D,yeval]);
	XY_D=np.hstack([X_D,Y_D]);

	if screening:	
		seqmorris_no=seqmorris_no+1;	

	# sequential Morris method
	if seqmorris_no==(len(I_screen)+1):
		#s=10;
		delta=0.5#s/(2.0*(s-1.0)); #we consider the limit as s goes to infinity
		gamma=((0.025*(max(Y_D)-min(Y_D)))/2.0)**2.0;
		sigma0=np.sqrt(stats.chi2.ppf(0.99,j)*(2.0*gamma/(delta**2.0))/j);

		I_rmv=[]
		r_tot=len(I_screen)
		for r in range(0,r_tot):
			x_orig=X_D[-1-r_tot,:]
			x_r=X_D[-r_tot+r,:]
			y_orig=Y_D[-1-r_tot]
			y_r=Y_D[-r_tot+r]
			delta=abs(x_r[I_screen[r]]-x_orig[I_screen[r]]);

			EE_i[s,I_screen[r]]=(y_orig-y_r)/(x_orig[I_screen[r]]-x_r[I_screen[r]]);
			mu_i[s,I_screen[r]]=(1.0/(s+1.0))*np.sum(EE_i[0:(s+1),I_screen[r]])
			mustar_i[s,I_screen[r]]=(1.0/(s+1.0))*np.sum(np.abs(EE_i[0:(s+1),I_screen[r]]))
			if s>0:
				sigma_i[s,I_screen[r]]=np.sqrt((1.0/s)*np.sum((np.abs(EE_i[0:(s+1),I_screen[r]])-mu_i[s,I_screen[r]])**2.0))
				if sigma_i[s,I_screen[r]]>sigma0:
					I_rmv.append(I_screen[r]);
			print 'j',s,'i',I_screen[r],'EE_i',EE_i[s,I_screen[r]],'mu_i',mu_i[s,I_screen[r]],'sigma_i',sigma_i[s,I_screen[r]]

		for r in I_rmv:
			I_screen.remove(r)
			print 'Removed variable:',r

		if s==s_tot:
			I_rmv=range(0,p);
			I_rmv=[x for x in I_rmv if x not in I_screen];
			for r in I_screen:
				if sigma_i[s,r]<1E-3:
					print 'identified variable',r,'to be noise of size',sigma_i[s,r]
					#I_rmv.append(r);
			for r in I_screen:
				if sigma_i[s,r]>=1E-3:
					print 'identified variable',r,'to be linear with sigma',sigma_i[s,r]
					I_rmv.append(r);
					#coeff[r]=sum(EE_i[:,r])/EE_i.shape[0]
			regfun_linear=lambda x: np.dot(coeff,x);
			print 'Removed remaining variables:',I_screen	

			X_D=X_D[:,I_rmv].reshape((X_D.shape[0],len(I_rmv)));
			XtrialGP=Xtrial[:,I_rmv].reshape((Xtrial.shape[0],len(I_rmv)));

			# remove multiple entries in design
			unique_list=[]
			unique_list.append(0);
			for i in range(X_D.shape[0]):
				x_i=X_D[i,:]
				if i>0:
					unique=True;
					for t in range(0,len(unique_list)):
						x_t=X_D[unique_list[t],:];
						if np.linalg.norm(x_i-x_t)<1E-5:
							unique=False;
					if unique==True:
						unique_list.append(i);
			pts_rmv=X_D.shape[0]-len(unique_list);
			X_D=X_D.copy()[unique_list,:].reshape((len(unique_list),len(I_rmv)))
			Y_D=Y_D.copy()[unique_list,:].reshape((len(unique_list),1))

			p=len(I_rmv);
			def myfun_red(var):
				x=np.zeros(p_init);
				x[I_rmv]=var;
				return myfun(x);
			myfun_active=lambda var: myfun_red(var);
			a=a[I_rmv]
			b=b[I_rmv]

			I_rmv.append(-1)
			a_MLE=a_MLE[I_rmv]
			b_MLE=b_MLE[I_rmv]
			corr=corr[I_rmv]
			GPD.corr_param=GPD.corr_param[I_rmv]
			I_rmv=I_rmv[0:p]

    			GPD=Gp();
    			GPD.set_domain(a,b);
    			GPD.set_corr(corrtype);
    			GPD.set_reg('function',regfun);

			I_screen=[]
			screening=False;


		s=s+1;
		seqmorris_no=0;
	
        np.savetxt("".join([adir,"X.txt"]),X_D,fmt='%.16f')
        np.savetxt("".join([adir,"Y.txt"]),Y_D,fmt='%.16f')
        np.savetxt("".join([adir,"XY.txt"]),XY_D,fmt='%.16f')

	
        # Update
        GPD.set_data(X_D,Y_D);
	if X_D.shape[0]<100:
	    print 'MLE-fit...',
	    sys.stdout.flush();
	    xi=corr[0:p].reshape((1,p));
            timenow=time.time()
	    corr=GPD.fit(a_MLE,b_MLE,ng,m,cr,mr,xi)

    	    print 'Completed.'
	    print 'Corr. param:',corr
        GPD.build(corr);

	# Error estimate for out-of-sample
    	print 'Prediction for out-of-sample:',
        Ym=np.zeros(M_trial);Yv=np.zeros(M_trial);
        e_i=[];    
        for i in range(Xtrial.shape[0]): 
		x_i=Xtrial[i];
		xGP_i=XtrialGP[i];
		Ym[i]=regfun_linear(x_i)+GPD.mean(xGP_i)
		Yv[i]=GPD.var(xGP_i)
        	e_i.append(abs(Ym[i]-Ytrial[i]))
        l2_error=np.sqrt(sum(map(lambda x:x**2.0, e_i))/Xtrial.shape[0])[0]
        nrmspe=(l2_error/(max(Ytrial)-min(Ytrial)))[0]

	design_size_vec.append(X_D.shape[0]+pts_rmv);
   	l2_vec.append(l2_error)
   	nrmspe_vec.append(nrmspe)
	corr_vec.append(corr.copy());

        np.savetxt("".join([adir,"l2.txt"]),np.vstack([np.array(design_size_vec),np.array(l2_vec)]).transpose(),fmt='%.16f')
        np.savetxt("".join([adir,"nrmspe.txt"]),np.vstack([np.array(design_size_vec),np.array(nrmspe_vec)]).transpose(),fmt='%.16f')
        np.savetxt("".join([adir,"corr.txt"]),np.hstack([np.array(design_size_vec).reshape((len(design_size_vec),1)),np.array(corr_vec)]),fmt='%.16f')

        print 'L2 Error:',l2_error,', NRMSPE:',nrmspe

    print 'End MICE'


# latin hypercube design (LHD)
def lhd(N,M,arange,brange,theseed=None):
	if not theseed==None: 
		np.random.seed(theseed);
	D=np.empty((M,N),dtype=np.float64)
	D_ind=np.empty((M,N),dtype=np.float64)
	D_l=np.arange(M)
	for l in range(N):
		np.random.shuffle(D_l)
		D_ind[:,l]=D_l
	D_norm=np.random.uniform(low=0.0,high=1.0,size=(M,N))
	for j in range(M):
		for l in range(N):
			a=arange[l];b=brange[l];
			D[j,l]=a+(b-a)*(D_ind[j,l]+D_norm[D_ind[j,l],l])/M; #improve possible: matrix computation
	return D

# Maximin LHD
def maximinLHD(N,M,arange,brange,maximin_num,theseed=None):
    	maximinD=0.0;
    	maximinX=None;
	if theseed==None:
		np.random.seed();
	else:
		np.random.seed(theseed);

        barange=np.zeros(arange.shape);
	for l in range(arange.shape[0]):
		barange[l]=brange[l]-arange[l];

    	for j in range(0,maximin_num):
		X=lhd(N,M,arange,brange,None);
		for k1 in range(X.shape[0]):
			for k2 in range(X.shape[1]):
				X[k1,k2]=arange[k2]+(brange[k2]-arange[k2])*X[k1,k2];
   		MX=X.shape[0];
		D=666*np.ones((MX,MX));		
		for i in range(MX):
			for k in range(MX):
				if not i==k:
					D[i,k]=sum(abs(X[i,:]/barange-X[k,:]/barange)**2.0)
				else:
					D[i,k]=666.66 # dummy
			if maximinD>D.min():
				break;
		minD=D.min()	
		if minD>maximinD:
			maximinD=minD;
			maximinX=X;
	return maximinX

# test function

def piston7D(D):
	if D.ndim==1:
		dlen=1;
	else:
		dlen=D.shape[0];
	g=np.zeros((dlen,1));

	for j in range(dlen):
		if D.ndim==1:
			x=D;
		else:
			x=D[j,:];
		M=30.0+(60.0-30.0)*x[0] # piston weight (kg)
		S=0.005+(0.020-0.005)*x[1] # piston surface area (m^2)
		V0=0.002+(0.010-0.002)*x[2] # initial gas volume (m
		k=1000.0+(5000.0-1000.0)*x[3] # spring coefficient (N/m)
		P0=90000.0+(110000.0-90000.0)*x[4] # atomospheric pressure (N/m^2)
		Ta=290.0+(296.0-290.0)*x[5] # ambient temperature (K)
		T0=340.0+(360.0-340.0)*x[6] # filling gas temperature (K)
	
		A=P0*S+19.62*M-k*V0/S;
		V=(S/(2.0*k))*(math.sqrt(A*A+4.0*k*(P0*V0/T0)*Ta-A));
		g[j,0]=2.0*math.pi*math.sqrt(M/(k+S*S*P0*V0*Ta/(T0*V*V))); #C(x), cycle time, time to complete one cycle, in seconds.
	return g

def piston5D(D):
	if D.ndim==1:
		dlen=1;
	else:
		dlen=D.shape[0];
	g=np.zeros((dlen,1));

	for j in range(dlen):
		if D.ndim==1:
			x=D;
		else:
			x=D[j,:];
		M=30.0+(60.0-30.0)*x[0] # piston weight (kg)
		S=0.005+(0.020-0.005)*x[1] # piston surface area (m^2)
		V0=0.002+(0.010-0.002)*x[2] # initial gas volume (m
		k=1000.0+(5000.0-1000.0)*x[3] # spring coefficient (N/m)
		P0=90000.0+(110000.0-90000.0)*x[4] # atomospheric pressure (N/m^2)
		Ta=290.0+(296.0-290.0)*0.5 # ambient temperature (K)
		T0=340.0+(360.0-340.0)*0.5 # filling gas temperature (K)
	
		A=P0*S+19.62*M-k*V0/S;
		V=(S/(2.0*k))*(math.sqrt(A*A+4.0*k*(P0*V0/T0)*Ta-A));
		g[j,0]=2.0*math.pi*math.sqrt(M/(k+S*S*P0*V0*Ta/(T0*V*V))); #C(x), cycle time, time to complete one cycle, in seconds.
	return g


def branin2D(D):
	if D.ndim==1:
		dlen=1;
	else:
		dlen=D.shape[0];
	g=np.zeros((dlen,1));

	for j in range(dlen):
		if D.ndim==1:
			x=D;
		else:
			x=D[j,:];

    		fval=((0.0+x[0]*15.0)-5.1*((-5.0+x[0]*15.0)**2.0)/(4.0*(math.pi**2.0))+(5.0/math.pi)*(-5.0+x[0]*15.0)-6.0)**2.0+10.0*(1.0-1.0/(8.0*math.pi))*math.cos((-5.0+x[0]*15.0))+10.0;
    		g[j,0]=fval;
        return g;

def test5D(D):
	if D.ndim==1:
		dlen=1;
	else:
		dlen=D.shape[0];
	g=np.zeros((dlen,1));

	for j in range(dlen):
		if D.ndim==1:
			x=D;
		else:
			x=D[j,:];

    		fval=np.cos(x[2]/5.0)*(x[1]+0.5)**4.0/((x[0]+0.5)**2.0)+x[4];
    		g[j,0]=fval;
        return g;

def test10D4(D):
	if D.ndim==1:
		dlen=1;
	else:
		dlen=D.shape[0];
	g=np.zeros((dlen,1));
	c=np.array([1.0,0.5,0.4,0.3,0.2,0.0,0.0,0.0,0.0,0.0,0.0]);
	for j in range(dlen):
		if D.ndim==1:
			x=D;
		else:
			x=D[j,:];

    		fval=1.0/(c[0]+c[1]*x[0]+c[2]*x[1]+c[3]*x[2]+c[4]*x[3]+c[5]*x[4]+c[6]*x[5]+c[7]*x[6]+c[8]*x[7]+c[9]*x[8]+c[10]*x[9]);
    		g[j,0]=fval;
        return g;

def test10D8(D):
	if D.ndim==1:
		dlen=1;
	else:
		dlen=D.shape[0];
	g=np.zeros((dlen,1));
	c=np.array([1.0,0.5,0.5,0.4,0.4,0.3,0.3,0.2,0.2,0.0,0.0]);
	for j in range(dlen):
		if D.ndim==1:
			x=D;
		else:
			x=D[j,:];

    		fval=1.0/(c[0]+c[1]*x[0]+c[2]*x[1]+c[3]*x[2]+c[4]*x[3]+c[5]*x[4]+c[6]*x[5]+c[7]*x[6]+c[8]*x[7]+c[9]*x[8]+c[10]*x[9]);
    		g[j,0]=fval;
        return g;

def morris30D(D):
	if D.ndim==1:
		dlen=1;
	else:
		dlen=D.shape[0];
	k_1=8;
	g=np.zeros((dlen,1));
	alpha=np.sqrt(12.0)-6.0*np.sqrt(0.1*(k_1-1));
	beta=12.0*np.sqrt(0.1*(k_1-1));
	for j in range(dlen):
		if D.ndim==1:
			x=D;
		else:
			x=D[j,:];
		fval=0.0;
		for s in range(0,k_1):
    			fval=fval+alpha*x[s]
			if s>0:
				for t in range(s,k_1):
					fval=fval+alpha*beta*x[s]*x[t];
    		g[j,0]=fval;
        return g;

def test20D8(D):
	if D.ndim==1:
		dlen=1;
	else:
		dlen=D.shape[0];
	g=np.zeros((dlen,1));
	c=np.array([1.0,0.8,0.8,0.6,0.6,0.4,0.4,0.2,0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]);
	for j in range(dlen):
		if D.ndim==1:
			x=D;
		else:
			x=D[j,:];

    		fval=1.0/(c[0]+c[1]*x[0]+c[2]*x[1]+c[3]*x[2]+c[4]*x[3]+c[5]*x[4]+c[6]*x[5]+c[7]*x[6]+c[8]*x[7]+c[9]*x[8]+c[10]*x[9]+c[11]*x[10]+c[12]*x[11]++c[13]*x[12]+c[14]*x[13]+c[15]*x[14]+c[16]*x[15]+c[17]*x[16]+c[18]*x[17]+c[19]*x[18]+c[20]*x[19]);
    		g[j,0]=fval;
        return g;

def test20D16(D):
	if D.ndim==1:
		dlen=1;
	else:
		dlen=D.shape[0];
	g=np.zeros((dlen,1));
	c=np.array([1.0,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.4,0.4,0.4,0.4,0.2,0.2,0.2,0.2,0.0,0.0,0.0,0.0]);
	for j in range(dlen):
		if D.ndim==1:
			x=D;
		else:
			x=D[j,:];

    		fval=1.0/(c[0]+c[1]*x[0]+c[2]*x[1]+c[3]*x[2]+c[4]*x[3]+c[5]*x[4]+c[6]*x[5]+c[7]*x[6]+c[8]*x[7]+c[9]*x[8]+c[10]*x[9]+c[11]*x[10]+c[12]*x[11]++c[13]*x[12]+c[14]*x[13]+c[15]*x[14]+c[16]*x[15]+c[17]*x[16]+c[18]*x[17]+c[19]*x[18]+c[20]*x[19]);
    		g[j,0]=fval;
        return g;

def test30D8(D):
	if D.ndim==1:
		dlen=1;
	else:
		dlen=D.shape[0];
	g=np.zeros((dlen,1));
	c=np.array([1.0,0.8,0.8,0.6,0.6,0.4,0.4,0.2,0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]);
	for j in range(dlen):
		if D.ndim==1:
			x=D;
		else:
			x=D[j,:];

    		fval=1.0/(c[0]+c[1]*x[0]+c[2]*x[1]+c[3]*x[2]+c[4]*x[3]+c[5]*x[4]+c[6]*x[5]+c[7]*x[6]+c[8]*x[7]+c[9]*x[8]+c[10]*x[9]+c[11]*x[10]+c[12]*x[11]+c[13]*x[12]+c[14]*x[13]+c[15]*x[14]+c[16]*x[15]+c[17]*x[16]+c[18]*x[17]+c[19]*x[18]+c[20]*x[19]);
    		g[j,0]=fval;
        return g;

def test30D16(D):
	if D.ndim==1:
		dlen=1;
	else:
		dlen=D.shape[0];
	g=np.zeros((dlen,1));
	c=np.array([1.0,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.4,0.4,0.4,0.4,0.2,0.2,0.2,0.2,0.0,0.0,0.0,0.0]);
	for j in range(dlen):
		if D.ndim==1:
			x=D;
		else:
			x=D[j,:];

    		fval=1.0/(c[0]+c[1]*x[0]+c[2]*x[1]+c[3]*x[2]+c[4]*x[3]+c[5]*x[4]+c[6]*x[5]+c[7]*x[6]+c[8]*x[7]+c[9]*x[8]+c[10]*x[9]+c[11]*x[10]+c[12]*x[11]+c[13]*x[12]+c[14]*x[13]+c[15]*x[14]+c[16]*x[15]+c[17]*x[16]+c[18]*x[17]+c[19]*x[18]+c[20]*x[19]);
    		g[j,0]=fval;
        return g;

def test30D24(D):
	if D.ndim==1:
		dlen=1;
	else:
		dlen=D.shape[0];
	g=np.zeros((dlen,1));
	c=np.array([1.0,0.8,0.8,0.8,0.8,0.8,0.8,0.6,0.6,0.6,0.6,0.6,0.6,0.4,0.4,0.4,0.4,0.4,0.4,0.2,0.2,0.2,0.2,0.2,0.2,0.0,0.0,0.0,0.0,0.0,0.0]);
	for j in range(dlen):
		if D.ndim==1:
			x=D;
		else:
			x=D[j,:];

    		fval=1.0/(c[0]+c[1]*x[0]+c[2]*x[1]+c[3]*x[2]+c[4]*x[3]+c[5]*x[4]+c[6]*x[5]+c[7]*x[6]+c[8]*x[7]+c[9]*x[8]+c[10]*x[9]+c[11]*x[10]+c[12]*x[11]+c[13]*x[12]+c[14]*x[13]+c[15]*x[14]+c[16]*x[15]+c[17]*x[16]+c[18]*x[17]+c[19]*x[18]+c[20]*x[19]+c[21]*x[20]+c[22]*x[21]+c[23]*x[22]+c[24]*x[23]+c[25]*x[24]+c[26]*x[25]+c[27]*x[26]+c[28]*x[27]+c[29]*x[28]+c[30]*x[29]);
    		g[j,0]=fval;
        return g;

def welsh20D(D):
	if D.ndim==1:
		dlen=1;
	else:
		dlen=D.shape[0];
	g=np.zeros((dlen,1));

	for j in range(dlen):
		if D.ndim==1:
			x=D;
		else:
			x=D[j,:];

    		fval=5.0*(x[11]-0.5)/(1.0+x[0]-0.5)+5.0*(x[3]-x[19])**2.0+(x[4]-0.5)+40.0*((x[18]-0.5)**3.0)-5.0*(x[18]-0.5)+0.05*(x[1]-0.5)+0.08*(x[2]-0.5)-0.03*(x[5]-0.5)+0.03*(x[6]-0.5)+0.09*(x[8]-0.5)-0.01*(x[9]-0.5)-0.07*(x[10]-0.5)+0.25*((x[12]-0.5)**2.0)-0.04*(x[13]-0.5)+0.06*(x[14]-0.5)-0.01*(x[16]-0.5)-0.03*(x[17]-0.5);
    		g[j,0]=fval;
        return g;

def var_MI_fast(x,GPDc,Xs,F,invCC_MI):
		
	m=Xs.shape[0]
	meanX=GPDc.meanX;
	stdX=GPDc.stdX;
	param=GPDc.corr_param;
	sigma2=GPDc.sigma2;

	if x.ndim==2:
		x=x[0]

	x=(x-meanX)/stdX;

	r=np.zeros((m,1));
	for j in range(m):
		r[j]=GPDc.corr_model(x,Xs[j,:],param);

	sigma2_return=sigma2*(GPDc.corr_model(x,x,param)-r.transpose()*invCC_MI*r+(1.0-F.transpose()*invCC_MI*r).transpose()*(1.0/(F.transpose()*invCC_MI*F))*(1.0-F.transpose()*invCC_MI*r))
	sigma2_return=np.asarray(sigma2_return).reshape((1));

	return sigma2_return

def MI_fast(X_Dc,GPD,GPDc,invCC,R,x):

	p=X_Dc.shape[1]
	xnorm=np.linalg.norm(x);
	for s in range(X_Dc.shape[0]):
		if np.linalg.norm(X_Dc[s,0:p]-x)<1E-10:
			j=s;
			break;

	allidx=np.arange(0,X_Dc.shape[0]).astype(int)
	remainingidxlist_j=np.setdiff1d(allidx,np.array([j])).tolist();
	X_Dc_j=X_Dc[remainingidxlist_j,0:p];

	Xs_MI,F_MI,invCC_MI,R_MI=calc_MI_fast(j,X_Dc_j,GPDc.meanX,GPDc.stdX,invCC,R,GPDc.Xs);

	objfun=lambda s: GPD.var(s)*((1.0/var_MI_fast(s,GPDc,Xs_MI,F_MI,invCC_MI)));
	y=objfun(x)

	return y;

def calc_MI_fast(j,X,meanX,stdX,invCC,R,Xs):
	if not X.shape[0]==invCC.shape[0]:
		np.seterr(invalid='ignore')

		allidx=np.arange(0,X.shape[0]+1).astype(int)
		idxlist=np.setdiff1d(allidx,np.array([j])).tolist()

		M_11=np.zeros((len(idxlist),len(idxlist)));
		R_MI=np.zeros((len(idxlist),len(idxlist)));
		Xs_MI=np.divide((X-np.ones((X.shape[0],1))*meanX),np.ones((X.shape[0],1))*stdX);
		i1n=0;i2n=0
		for i1 in idxlist:
			for i2 in idxlist:
				M_11[i1n,i2n]=invCC[i1,i2];
				R_MI[i1n,i2n]=R[i1,i2];
				i2n=i2n+1
			i1n=i1n+1;
			i2n=0;
		b=np.zeros((len(idxlist),1));
		b=R[idxlist,j];
		M_12=invCC[idxlist,j].reshape((len(idxlist),1));
		M_21=invCC[j,idxlist].reshape((1,len(idxlist)));
		M_22=invCC[j,j];

		invCC_MI=M_11-M_22*np.dot((M_12/M_22),(M_21/M_22));
		invCC_MI=np.asmatrix(invCC_MI)

		F_MI=np.ones((Xs_MI.shape[0],1));

	else:
		Xs_MI=Xs;
		F_MI=np.ones((R.shape[0],1));
		invCC_MI=np.matrix(invCC)
		R_MI=R;

	return Xs_MI,F_MI,invCC_MI,R_MI

if __name__=='__main__':
     main(sys.argv[1:])
