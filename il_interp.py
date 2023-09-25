import numpy as n
import matplotlib.pyplot as plt
import h5py

class ilint:
    def __init__(self,fname="ion_line_interpolate.h5"):
        h=h5py.File(fname,"r")
        self.S=h["S"][()]   # 3d array molecular to atomic fraction x te/ti x ti
        self.te_ti_ratio=h["te_ti_ratios"][()]
        self.dte_ti_ratio=n.diff(self.te_ti_ratio)[0]
        self.te_ti_ratio0=n.min(self.te_ti_ratio)
        self.te_ti_ratio1=n.max(self.te_ti_ratio)
        self.te_ti_ratioN=len(self.te_ti_ratio)
        
        self.mol_fracs=h["mol_fracs"][()]
        self.dmol_fracs=n.diff(self.mol_fracs)[0]
        self.mol_fracs0=n.min(self.mol_fracs)
        self.mol_fracs1=n.max(self.mol_fracs)
        self.mol_fracsN=len(self.mol_fracs)        
        
        self.tis=h["tis"][()]
        self.dtis=n.diff(self.tis)[0]
        self.tis0=n.min(self.tis)
        self.tis1=n.max(self.tis)
        self.tisN=len(self.tis)        
        
        self.radar_freq=h["freq"][()]
        self.angular_doppler=h["om"][()]
        self.ne0=h["ne"][()]
        h.close()

        # calculate acfs
        for fr_i in range(self.mol_fracsN):
            for teti_i in range(self.te_ti_ratioN):
                for ti_i in range(self.tisN):
                    print(self.angular_doppler)
                    spec=self.S[fr_i,teti_i,ti_i,:]
                    plt.plot(spec)
                    plt.show()
                    acf=n.fft.ifft(n.fft.fftshift(self.S[fr_i,teti_i,ti_i,:]))
                    plt.plot(acf.real)
                    plt.plot(acf.imag)                    
                    plt.show()
        


    def getspec(self,
                ne=n.array([1e11]),
                te=n.array([600]),
                ti=n.array([300]),
                mol_frac=n.array([1.0]),
                vi=n.array([0.0]),
                debug=False):
        te_ti_ratio_idx=(te/ti - self.te_ti_ratio0)/self.dte_ti_ratio
        # edge cases
        te_ti_ratio_idx[te_ti_ratio_idx<=0]=1e-4
        te_ti_ratio_idx[te_ti_ratio_idx>=(self.te_ti_ratioN-1) ]=self.te_ti_ratioN-1-1e-4

        mol_frac_idx=(mol_frac - self.mol_fracs0)/self.dmol_fracs
        # edge cases
        mol_frac_idx[mol_frac_idx<=0]=1e-4
        mol_frac_idx[mol_frac_idx>=(self.mol_fracsN-1) ]=self.mol_fracsN-1-1e-4

        ti_idx=(ti - self.tis0)/self.dtis
        # edge cases
        ti_idx[ti_idx<=0]=1e-4
        ti_idx[ti_idx>=(self.tisN-1) ]=self.tisN-1-1e-4


        # p_rx(ne,te,ti) = ne/(1+te/ti)
        # p_rx(ne,te,ti)/p_rx(ne0,te,ti) = ne/ne0
        pwr_scaling_factor=ne/self.ne0

        # and now linearly interpolate this thing.
        # there are three dimensions, so we need to look at eight corners

        specs=[]
        S=n.zeros([len(ne),self.S.shape[3]],dtype=n.float32)
        
        # for all parameter triplets
        for i in range(len(ne)):
            w00=1.0-(mol_frac_idx[i]-n.floor(mol_frac_idx[i]))  # how close to floor [0,1]
            w01=1.0-(n.ceil(mol_frac_idx[i])-mol_frac_idx[i])   # how close to ceil

            w10=1.0-(te_ti_ratio_idx[i]-n.floor(te_ti_ratio_idx[i]))
            w11=1.0-(n.ceil(te_ti_ratio_idx[i])-te_ti_ratio_idx[i])

            w20=1.0-(ti_idx[i]-n.floor(ti_idx[i]))
            w21=1.0-(n.ceil(ti_idx[i])-ti_idx[i])

            if debug:
                print("weights %1.2f,%1.2f-%d,%d %1.2f,%1.2f-%d,%d %1.2f,%1.2f - %d,%d"%(w00,w01,n.floor(mol_frac_idx[i]),n.ceil(mol_frac_idx[i]),
                                                                                         w10,w11,n.floor(te_ti_ratio_idx[i]),n.ceil(te_ti_ratio_idx[i]),
                                                                                         w20,w21,n.floor(ti_idx[i]),n.ceil(ti_idx[i])))
            # 000
            w0=w00*w10*w20
            S[i,:] += w0*self.S[int(n.floor(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]

            # 001
            w1=w00*w10*w21
            S[i,:] += w1*self.S[int(n.floor(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]

            # 010
            w2=w00*w11*w20
            S[i,:] += w2*self.S[int(n.floor(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]

            # 011
            w3=w00*w11*w21
            S[i,:] += w3*self.S[int(n.floor(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]
            
            # 100
            w4=w01*w10*w20
            S[i,:] += w4*self.S[int(n.ceil(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]

            # 101
            w5=w01*w10*w21
            S[i,:] += w5*self.S[int(n.ceil(mol_frac_idx[i])), int(n.floor(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]

            # 110
            w6=w01*w11*w20
            S[i,:] += w6*self.S[int(n.ceil(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.floor(ti_idx[i])), :]
            
            # 111
            w7=w01*w11*w21
            S[i,:] += w7*self.S[int(n.ceil(mol_frac_idx[i])), int(n.ceil(te_ti_ratio_idx[i])), int(n.ceil(ti_idx[i])), :]
            
            S[i,:]=pwr_scaling_factor[i]*S[i,:]/(w0+w1+w2+w3+w4+w5+w6+w7)
            
        return(S)
        

def testne():
    il=ilint()
    ne=n.array([1e10,1e11,1e12])
    S=il.getspec(ne=ne,
                 te=n.array([1000,1000,1000]),
                 ti=n.array([900,900,900]),
                 mol_frac=n.array([1,1,1]),
                 vi=n.array([0,0,0])
                 )
#self.angular_doppler
    for i in range(S.shape[0]):
        plt.semilogy(il.angular_doppler/2/n.pi,S[i,:],label="ne=%1.2g"%(ne[i]),color="black",alpha=0.3)
#    plt.legend()
    plt.show()

def testte():
    il=ilint()
    ne=n.repeat(1e12,400)
    te=n.linspace(500,4000,num=400)
    ti=n.repeat(500,400)
    mol_frac=n.repeat(0.1,400)
    vi=n.repeat(0,400)
    for i in range(100):
        print(i)
        S=il.getspec(ne=ne,
                     te=te,
                     ti=ti,
                     mol_frac=mol_frac,
                     vi=vi)

    
    plt.pcolormesh(S)
    plt.colorbar()
    plt.show()
    for i in range(S.shape[0]):
        plt.plot(il.angular_doppler/2/n.pi,S[i,:],color="black",alpha=0.3)

#    plt.legend()
    plt.show()


def testti():
    il=ilint()
    ne=n.repeat(1e12,100)
    ti=n.linspace(500,2000,num=100)
    te=n.copy(ti)
    mol_frac=n.repeat(0.423,100)
    vi=n.repeat(0,100)
    S=il.getspec(ne=ne,
                 te=te,
                 ti=ti,
                 mol_frac=mol_frac,
                 vi=vi)

    plt.pcolormesh(S)
    plt.colorbar()
    plt.show()
    for i in range(S.shape[0]):
        plt.plot(S[i,:],label="ti=%1.2g"%(ti[i]),color="black",alpha=0.3)
    plt.show()
    

if __name__ == "__main__":
    testti()    
    testte()
    testne()
