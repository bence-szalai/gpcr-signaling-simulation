import numpy as np
import pandas as pd
from scipy.integrate import ode as ode
from matplotlib import pyplot as plt

def remove_newline(string):
    if string[-1]=='\n':
        return string[:-1]
    else:
        return string

class VecModel(object):
    """Main class of kinetic models"""
    def __init__(self,dt):
        self.dt=dt
        self.t=[0.0] #time points of simulation
        self.ccs=None #concentrations, (n_mol+1) x 1 matrix (more columns for increased time)
        self.is_constant=None #indicator, if==1 dm/dt is allways 0, (n_mol+1) x 1 vector
        self.constants=None #reaction constants, (n_reac+1) x 1 vector
        self.reactions=None #reaction matrix, (n_reac+1) x 1 matrix (can have more columns, for more complex reactions)
        self.reactions_from=None #reaction matrix, (n_mol+1) x 1 matrix (can have more columns, when a molecule participates in more reactions)
        self.reactions_to=None #reaction matrix, (n_mol+1) x 1 matrix (can have more columns, when a molecule participates in more reactions)
    def read(self,fname):
        """reads the initial concentrations and reactions from a model file"""
        #create variables to read data
        ccs=[1.0]
        is_constant=[1]
        k=[0.0]
        r=[[]]
        r_from=[[]]
        r_to=[[]]
        s1=0 #dimension 2 of readctions matrix
        s2=0 #dimension 2 of reactions_from matrix
        s3=0 #dimension 2 of reactions_to matrix
        #just processing the input file
        molecule=False
        reaction=False
        fin=open(fname)
        flines=fin.readlines()
        fin.close()
        for i in range(len(flines)):
            line=remove_newline(flines[i])
            if line=='#molecules':
                molecule=True
            if line=='#reactions':
                molecule=False
                reaction=True
                for i in range(len(ccs)):
                    r_from.append([])
                    r_to.append([])
            if line=='#end':
                break
            if line[0]!='#':
                if molecule:
                    v1,v2,v3=line.split(',')
                    is_constant.append(int(v2))
                    ccs.append(float(v3))
                if reaction:
                    v1,v2,v3,v4=line.split(',')
                    r.append(list(np.array(v2.split()).astype(int)))
                    for m in np.array(v2.split()).astype(int):
                        r_from[m].append(int(v1))
                    for m in np.array(v3.split()).astype(int):
                        r_to[m].append(int(v1))   
                    k.append(float(v4))
        for x in r:
            if len(x)>s1:
                s1=len(x)
        for x in r_from:
            if len(x)>s2:
                s2=len(x)
        for x in r_to:
            if len(x)>s3:
                s3=len(x)       
        #create model variables
        self.ccs=np.array(ccs).reshape((-1,1))
        self.is_constant=np.array(is_constant).reshape((-1,1))
        self.constants=np.array(k).reshape((-1,1))
        self.reactions=np.zeros((len(self.constants),s1),int)
        for i in range(len(r)):
            for j in range(len(r[i])):
                self.reactions[i,j]=r[i][j]        
        self.reactions_from=np.zeros((len(self.ccs),s2),int)
        for i in range(len(r_from)):
            for j in range(len(r_from[i])):
                self.reactions_from[i,j]=r_from[i][j]
        self.reactions_to=np.zeros((len(self.ccs),s3),int)
        for i in range(len(r_to)):
            for j in range(len(r_to[i])):
                self.reactions_to[i,j]=r_to[i][j]
        self.ccs=self.ccs.T
        self.is_constant=self.is_constant[:,0]
        self.constants=self.constants[:,0]
        self.reactions=self.reactions.T
        self.reactions_from=self.reactions_from.T
        self.reactions_to=self.reactions_to.T
    def calculate_dydt(self,t,y):
        """calculates dmdt based m"""
        flux=np.product(y[self.reactions],0)*self.constants
        dmdt=(np.sum(flux[self.reactions_to],0)-np.sum(flux[self.reactions_from],0))*(1-self.is_constant)
        return dmdt
    def simulate(self,time):
        t0=self.t[-1]
        t=self.t[-1]
        while t<t0+time:
            r=ode(self.calculate_dydt)
            y0=self.ccs[-1,:]
            r.set_initial_value(y0,t)
            y=r.integrate(t+self.dt)
            t+=self.dt
            self.t.append(t)
            self.ccs=np.concatenate([self.ccs,y.reshape((1,-1))],0)
    def write(self,fname):
        data=pd.DataFrame(self.ccs,index=self.t,columns=range(self.ccs.shape[1]))
        data.to_csv(fname,sep=',')
    def plot(self,molecules,ax=False):
        if ax:
            ax.plot(self.t,np.sum(self.ccs[:,molecules],1))
        else:
            plt.plot(self.t,np.sum(self.ccs[:,molecules],1))
    def change_constant(self,constants,values):
        self.constants[constants]=values
    def change_ccs(self,molecules,values):
        self.ccs[-1][molecules]=values
            

  
                                             

         
        
            
                    
                
                
        
        
        


