#!/usr/bin/env python3

import numpy as n
import matplotlib.pyplot as plt

def ion_fraction(h, h0=150e3, H=20e3):
    """
    Ion-fraction for a Chapman-type exponential scale height behaviour
    h0 = transition height
    H = width of the transition
    """
    return(1-n.tanh( n.exp(-(h-h0)/H) ))

def mh_ion_fraction(h):
    h=h/1e3
    zz1=-(h-120.0)/40.0
    
    zz1[zz1>50.0]=50.0
    
    H=10.0 - 6.0*n.exp(zz1)
#    plt.plot(H)
 #   plt.show()
    zz2=-(h-180)/H
    zz2[zz2>50]=50

    fr=1-2.0/(1+n.sqrt(1+8.0*n.exp(zz2)))
    fr[h<120]=1.0
#    plt.plot(fr)
 #   plt.show()
    
    return(fr)

if __name__ == "__main__":
    h=n.linspace(0,1000e3,num=1000)
    mhfr=mh_ion_fraction(h)
    
    plt.plot(ion_fraction(h),h/1e3,label="O$^{+}$")
    plt.plot(1-ion_fraction(h),h/1e3,label="O$_2^{+}$")

    plt.plot(mhfr,h/1e3,label="MH O$_2^{+}$")
    plt.plot(1-mhfr,h/1e3,label="MH O$^{+}$")    
    
    plt.xlabel("Fraction")
    plt.ylabel("Height (km)")
    plt.legend()
    plt.show()
            
