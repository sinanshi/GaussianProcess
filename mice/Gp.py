import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import math
import time
from GA import GA
import scipy.optimize as op

class Gp:
	m=None
	X=None
	Y=None
	meanX=None
	meanY=None
	stdX=None
	stdY=None
	Xs=None
	Ys=None

	corr_param=None
	Yt=None
	F=None
	Ft=None
	FtT=None
	mu=None
	R=None
	Q=None
	G=None
	C=None
	Ct=None

	Cinv=None
	Ctinv=None
	invCC=None
	invCC_MI=None
	sigma2=None
	diff=None
	regfun=None
	regtype=None

	a=None
	b=None


	def gen_reg(self):
		regtype=self.regtype;

		if regtype=='zero':
			p=1;
			F=np.zeros((self.X.shape[0],p));
			self.regfun=lambda x: 0.0; 
		elif regtype=='constant':
			p=1;
			F=np.ones((self.X.shape[0],p));
			self.regfun=lambda x: 1.0; 
		elif regtype=='function':
			p=self.regfun(self.Xs[0,:]).shape[0];
			F=np.zeros((self.Xs.shape[0],p));
			for j in range(self.X.shape[0]):
				F[j,:]=self.regfun(self.Xs[j,:]);
		return F;	

	def set_reg(self,regtype=None,regfun=None):
		self.regtype=regtype;
		self.regfun=regfun;	

	def build(self,param):
		self.corr_param=param
		N=self.X.shape[1];

		try:
			self.R=self.get_corr_matrix(self.corr_param);
			self.Ct=np.linalg.cholesky(self.R);
			self.C=self.Ct.transpose();
		except:
			print 'Failed: Matrix is not positive definite for x=',corr_param
			return np.array([100.0]).reshape((1,1));

		if self.regtype==None:
			self.set_reg('constant');
		self.F=self.gen_reg();

		self.C=self.Ct.transpose();
		self.Cinv=np.linalg.inv(self.C);
		self.Ctinv=np.linalg.inv(self.C.transpose());
		self.invCC=np.dot(self.Cinv,self.Ctinv);

		self.invCC_MI=self.invCC

		self.Yt=np.linalg.solve(self.Ct,self.Ys);

		if np.count_nonzero(self.F)>0:
			self.Ft=np.linalg.solve(self.Ct,self.F);
			self.FtT=self.Ft.transpose()
			self.Q,self.G=np.linalg.qr(self.Ft);
			self.mu=np.linalg.solve(self.G,np.dot(self.Q.transpose(),self.Yt));
			self.diff=self.Yt-np.dot(self.Ft,self.mu);
		else:	
			self.diff=self.Yt;
			self.mu=np.array([0.0]);

		
		self.sigma2=(self.stdY**2)*sum(self.diff**2)/self.m;

	def mean(self,x):
		x=(x-self.meanX)/self.stdX;

		if x.ndim==2:
			x=x[0]

		r=np.zeros((self.m,1));
		for j in range(self.m):
			r[j]=self.corr_model(x,self.Xs[j,:],self.corr_param);

		y_scaled=np.dot(self.mu.transpose(),self.regfun(x))+np.dot(r.transpose(),np.dot(self.Cinv,self.diff));
		y=self.meanY+np.dot(y_scaled,self.stdY);

		return y[0]

	def var(self,x):
		sigma2=self.sigma2;
		x=(x-self.meanX)/self.stdX;

		if x.ndim==2:
			x=x[0]
		r=np.zeros((self.m,1));
		for j in range(self.m):
			r[j]=self.corr_model(x,self.Xs[j,:],self.corr_param);
		
		rt=np.linalg.solve(self.Ct,r);
		if np.count_nonzero(self.F)>0:
			u=np.linalg.solve(self.G,np.dot(self.FtT,rt)-1.0);
			varest=sigma2*(1.0+sum(u**2.0)-sum(rt**2.0));
		else:
			varest=sigma2*(1.0-sum(rt**2.0));

		return varest;

	def cov(self,x1,x2):
		sigma2=self.sigma2;

		x1=(x1-self.meanX)/self.stdX;
		x2=(x2-self.meanX)/self.stdX;
		if x1.ndim==2:
			x1=x1[0]
		if x2.ndim==2:
			x2=x2[0]
		r1=np.zeros((self.m,1));
		for j in range(self.m):
			r1[j]=self.corr_model(x1,self.Xs[j,:],self.corr_param);
		r2=np.zeros((self.m,1));
		for j in range(self.m):
			r2[j]=self.corr_model(x2,self.Xs[j,:],self.corr_param);
		
		rt1=np.linalg.solve(self.Ct,r1);
		rt2=np.linalg.solve(self.Ct,r2);
		u1=np.linalg.solve(self.G,np.dot(self.FtT,rt1)-1.0);
		u2=np.linalg.solve(self.G,np.dot(self.FtT,rt2)-1.0);
		sigma2_return=sigma2*(self.corr_model(x1,x2,self.corr_param)+sum(u1*u2)-sum(rt1*rt2));

		return sigma2_return.reshape((1,1));

	def get_corr_matrix(self,param=corr_param):
		Xs=self.Xs;
		N=Xs.shape[1];
		R=np.zeros((Xs.shape[0],Xs.shape[0]));
		for i in range(R.shape[0]):
			for j in range(i,R.shape[1]):
				x_i=Xs[i,:];x_j=Xs[j,:];

				R[i,j]=self.corr_model(x_i,x_j,param);
				R[j,i]=R[i,j];	#symmetric

		for i in xrange(R.shape[0]):
			R[i,i]=R[i,i]+10.0**(param[-1])

		return R

	def corr_model(self,x_i,x_j,param):
		N=x_i.shape[0];
		r=np.abs(x_i-x_j);

		# standard GP model
		if self.corr_type=='sqe':
			val=np.exp(-np.sum(r**2.0/(param[0:N]**2.0)))

		# v=3/2
		if self.corr_type=='matern32':
					
			val=np.prod((1.0+np.sqrt(3.0)*r/param[0:N])*np.exp(-np.sqrt(3.0)*r/param[0:N]));

		# v=5/2
		if self.corr_type=='matern52':
			val=np.prod((1.0+np.sqrt(5.0)*r/param[0:N]+5.0*(r**2.0)/(3.0*param[0:N]**2.0))*np.exp(-np.sqrt(5.0)*r/param[0:N]));
		
		return val

	def fit(self,a_MLE,b_MLE,ng,m,cr,mr,Xi=None):
		N=a_MLE.shape[0]-1;
		corr=np.zeros(a_MLE.shape);
		nugget_min=int(a_MLE[-1])
		nugget_max=int(b_MLE[-1])
		nuggets=np.arange(nugget_min,nugget_max+1);
		ymax=-999999999.9;

		for nugget in nuggets:
	    		objfun=lambda var: -1.0*self.MLElnL(var,nugget);
			ga=GA(objfun,a_MLE[0:N],b_MLE[0:N])

			#P0=ga.create_rnd_pop(m);
		        P0=ga.create_rnd_pop_cand(m,Xi);
			ga.sort(P0);
			ga.config(ng,cr,mr);
			Q0=ga.create_offspring_pop(P0,m);
			R0=ga.union_pop(P0,Q0)
			R=R0
			ga.sort(R)
			print 'GA iter',
			for i in range(ng):
				print i,
				R_order=R.rank.astype(int)
				P=ga.create_empty_pop()

				P.X=R.X[R_order[0:m],:]
				P.Y=R.Y[R_order[0:m]]
				P.rank=R.rank[R_order[0:m]]

				Q=ga.create_offspring_pop(P,m)
				R=ga.union_pop(P,Q)
				ga.sort(R)

				if R.Y[0,:]>ymax:
					ymax=R.Y[0,:]
					corr[0:N]=R.X[0,:]
					corr[N]=nugget;
		print ''
		print 'GA summary: nfev',ng*m,'corr param:',corr,'MLE value:',ymax
		objfun=lambda var: self.MLElnL(var,corr[N]);
		bd=np.vstack([a_MLE[0:N],b_MLE[0:N]]).transpose()
		result=op.minimize(objfun,x0=corr[0:N].tolist(),bounds=bd.tolist(),options={'maxiter': 3000},jac=False,method='TNC')
		corr[0:N]=result['x']

		print 'Scipy TNC summary: nfev',result['nfev'],'corr param:',corr,'MLE value:', -1.0*result['fun']

		return corr;	

	def MLElnL(self,x,nugget):
		# avoid warning message
		np.seterr(invalid='ignore')

		N=x.shape[0]
		param=np.zeros((N+1));
		param[0:N]=x;
		param[N]=nugget;

		try:
			R=self.get_corr_matrix(param);
			Ct=np.linalg.cholesky(R);
		except:
			print 'Failed: Matrix is not positive definite for x=',param
			return np.array([100.0]).reshape((1,1));

		if self.regtype==None:
			self.set_reg('constant');
		F=self.gen_reg();
		Yt=np.linalg.solve(Ct,self.Ys);

		if np.count_nonzero(F)>0:
			Ft=np.linalg.solve(Ct,F);
			Q,G=np.linalg.qr(Ft);
			mu=np.linalg.solve(G,np.dot(Q.transpose(),Yt));
			diff=Yt-np.dot(Ft,mu);
		else:	
			diff=Yt;
			mu=np.array([0.0]);
		sigma2=(self.stdY**2)*sum(diff**2)/self.m;
		diagC = np.diag(Ct)
		detR=np.prod(diagC**(2.0/self.m));

		try:
			val=self.m*np.log(sigma2)+self.m*np.log(detR)
		except:
			print 'Failed: math domain error for x=',param
			return np.array([100.0]).reshape((1,1));
		return val;

	def set_domain(self,a,b):
		self.a=a;
		self.b=b;

	def set_corr(self,corr_type='sqe'):
		self.corr_type=corr_type;

	def set_data(self,X,Y,normalized=False):
		self.m=X.shape[0];

		if X.ndim==1:
			X=X.reshape((X.shape[0],1));
		if Y.ndim==1:
			Y=Y.reshape((Y.shape[0],1));

		self.X=X;
		self.Y=Y;

		b=np.array([list(c) for c in set(tuple(c) for c in X)])
		if not b.shape[0]==X.shape[0]:
			print 'multiple design sites!'
		if normalized:
			self.meanX=np.mean(X,axis=0,dtype=np.float64);self.meanY=np.mean(Y,axis=0,dtype=np.float64);
			self.stdX=np.std(X,axis=0,dtype=np.float64,ddof=1);
			self.stdY=np.std(Y,axis=0,dtype=np.float64,ddof=1); #ddof=1, that is, unbias estimator of the variance of the infinite population		
		else:
			self.meanX=0.0*np.mean(X,axis=0,dtype=np.float64);self.meanY=0.0*np.mean(Y,axis=0,dtype=np.float64);
			self.stdX=np.ones(np.std(X,axis=0,dtype=np.float64,ddof=1).shape);
			self.stdY=np.ones(np.std(Y,axis=0,dtype=np.float64,ddof=1).shape); #ddof=1, that is, unbias estimator of the variance of the infinite population
		self.Xs=np.divide((X-np.ones((X.shape[0],1))*self.meanX),np.ones((X.shape[0],1))*self.stdX);
		self.Ys=np.divide((Y-np.ones((Y.shape[0],1))*self.meanY),np.ones((Y.shape[0],1))*self.stdY);
