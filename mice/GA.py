import numpy as np

import sys
import time
import copy

class Population:
    X=None
    Y=None
    rank=None

class GA: 
	objfun=None	# objective function
        a=None		# a: lower bounds
	b=None  	# b: upper bounds

	M=None		#population size
	ng=None		#number of generations
	cr=None 	#crossover rate
	mr=None		#mutation rate
	ne=None

    	def __init__(self,f,a,b):
	    self.objfun=f
	    self.a=a
	    self.b=b

	    self.ng=100;
	    self.cr=0.9;
	    N=len(self.a)
	    self.mr=1.0/N;

	def config(self,ng,cr,mr):
		self.ng=ng;
		self.cr=cr;
		self.mr=mr;

	def create_empty_pop(self):
		return Population();

	def create_rnd_pop(self,M):
		a=self.a
		b=self.b

		np.random.seed(8);
		X=a+(b-a)*np.random.uniform(0,1,(M,len(a)));
		Y=self.eval_objfun(X);

		P=Population()
		P.X=X;P.Y=Y;

		return P

	def create_rnd_pop_cand(self,M,Xi=None):
		a=self.a
		b=self.b

		np.random.seed(8);
		X=np.random.uniform(0,1,(M,len(a)));
		X=a+(b-a)*X;

 		if not Xi==None:
			X[0:Xi.shape[0],:]=Xi;

		Y=self.eval_objfun(X);
		
		P=Population()
		P.X=X;P.Y=Y;

		return P

	def sort(self,P):
		I=P.Y[:,0].argsort(axis=0)[::-1]
		P.Y=P.Y[I]
		P.X=P.X[I,:]
		k=P.Y.shape[0]
		rank=np.linspace(0,k-1,k).astype(int);
		P.rank=np.zeros((k),dtype=int)
		P.rank=rank

	def cr_op(self,x,crfun=None):
		if crfun==None:
			crfun=self.crossover_sbx;
		return crfun(x);

	def mut_op(self,x,mutfun=None):
		if mutfun==None:
			mutfun=self.mutation_poly;
		return mutfun(x);

	def create_offspring_pop(self,P,M):
		a=self.a
		b=self.b
		N=len(a);

		Q=Population()
		cr=self.cr;
		mr=self.mr;
		ne=M/2;

		Q=Population()
		Q.X=np.zeros((M,N));

		ind_sel=self.bin_tourn(P,ne);

		for j in xrange(0,M,2):
			new=False
			while not new:
				equal=True;
				while equal:
					ind_par=np.random.randint(0,len(ind_sel),2);
					if ind_par[0]==ind_par[1]:
						equal=True;
					else:
						equal=False;	

				x1=P.X[ind_sel[ind_par[0]],:]
				x2=P.X[ind_sel[ind_par[1]],:]

				x=np.vstack([x1,x2]);

				parents=self.cr_op(x)
				childs=self.mut_op(parents);

				if np.equal(x,childs).all():
					new=False;
				else:
					new=True;

			Q.X[j:j+2,:]=childs;

		Q.Y=self.eval_objfun(Q.X);

		return Q

	def union_pop(self,P1,P2): # not used now
		P=Population()		
		P.X=np.vstack([P1.X,P2.X]);
		P.Y=np.vstack([P1.Y,P2.Y]);

		# check unique
		I=[]
		Irmv=[]
		for j in range(P.X.shape[0]):
			if not j in Irmv:
				xj=P.X[j,:];
				I.append(j);
				for i in range(j+1,P.X.shape[0]):
					xi=P.X[i,:];
					if np.linalg.norm(xj-xi)<1E-10:
						Irmv.append(i);

		P.X=P.X[I,:]
		P.Y=P.Y[I,:]

		return P

        ## Deb, Kalyanmoy, and Mayank Goyal. "A combined genetic adaptive search (GeneAS) for engineering design." Computer Science and Informatics 26 (1996): 30-45.
	def mutation_poly(self,x_in,nu=100.0):
		a=self.a
		b=self.b
		mr=self.mr

		# loop over input vectors
		x_out=np.empty((x_in.shape));
		for j in range(x_in.shape[0]):
			x_j=x_in[j,:];
			delta_1=(x_j-a)/(b-a);
			delta_2=(b-x_j)/(b-a);
			delta_bd=np.min([x_j-a,b-x_j])/(b-a);

			u=np.random.rand(x_in.shape[1]);
			ind=np.nonzero((mr-u).clip(0))[0];

			r=np.random.rand(x_in.shape[1]);
			delta=np.zeros(x_in.shape[1]);
			for i in ind:
				if r[i]<=0.5:			
					delta[i]=(2.0*r[i]+(1.0-2.0*r[i])*((1.0-delta_bd[i])**(nu+1.0)))**(1.0/(nu+1))-1.0;
				else:
					delta[i]=1.0-(2.0*(1.0-r[i])+2.0*(r[i]-0.5)*((1.0-delta_bd[i])**(nu+1)))**(1.0/(nu+1));

				x_j[i]=x_j[i]+delta[i]*(b[i]-a[i])

				if x_j[i]<a[i]:
					x_j[i]=a[i];
				elif x_j[i]>b[i]:
					x_j[i]=b[i];
			x_out[j,:]=x_j;

		return x_out;

	## Simulated binary crossover, SBX (Deb, K. Agrawal, R.B., 1995)
	def crossover_sbx(self,x_in,nu=5.0):
		a=self.a
		b=self.b

		x1=x_in[0,:];x2=x_in[1,:];
		ind=np.arange(len(x1))
		np.random.shuffle(ind);
		x1=x1[ind];x2=x2[ind];

		i=np.random.randint(0,len(x1));
		u=np.random.rand(i)

		beta=np.zeros(u.shape);
		for l in range(len(u)):
			if u[l]<=0.5:
				beta[l]=(2.0*u[l])**(1.0/(nu+1.0));
			else:
				beta[l]=(1.0/(2*(1.0-u[l])))**(1.0/(nu+1.0));

		# crossover (SBX)
		newx=np.vstack([x1,x2]);
		for j in range(i):
			newx[0,j]=0.5*((1.0+beta[j])*x1[j]+(1.0-beta[j])*x2[j])
			newx[1,j]=0.5*((1.0-beta[j])*x1[j]+(1.0+beta[j])*x2[j])	

			# respect bounds
			if newx[0,j]<a[ind[j]]:
				newx[0,j]=a[ind[j]];
			elif newx[0,j]>b[ind[j]]:
				newx[0,j]=b[ind[j]];
			if newx[1,j]<a[ind[j]]:
				newx[1,j]=a[ind[j]];
			elif newx[1,j]>b[ind[j]]:
				newx[1,j]=b[ind[j]];

		# return to original positions before shuffle
		x=x_in.copy();
		x[:,ind]=newx;

		return x

	def bin_tourn(self,P,ne):

		#ne = elite population size (need to be <= len(P))
		ind=np.arange(2*ne)
		np.random.shuffle(ind);
		ind_1=ind[0:ne]
		ind_2=ind[ne:]

		mp_ind=[] #mp=mating pool
		for j in range(ne):
			i_1=ind_1[j];
			i_2=ind_2[j];
			if np.where(P.rank==i_1)>np.where(P.rank==i_2):
				i=i_2;
			else:
				i=i_1;
			mp_ind.append(i)

		return mp_ind

	def dominates(self,p_1,p_2):

		if ((p_1>=p_2).all() and (p_1>p_2).any()):
			return True;
		else:
			return False;

	def eval_objfun(self,X):
		Y=None;
		numruns=X.shape[0]
		Y=np.zeros((numruns,1));
                for j in range(numruns):
			#st=time.time()
                        Y[j,:]=self.objfun(X[j,:]);
			#print time.time()-st
               
		return Y;
