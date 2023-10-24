import h5py
import numpy as n
import matplotlib.pyplot as plt
import re
import glob

# 31 and 16
fl=glob.glob("ion_line_interpolate_31_16_??.h5")
fl.sort()
h=h5py.File("ion_line_interpolate_31_16.h5","a")

S=h["S"][()]
for f in fl:
    print(f)
    idx=int(re.search(".*_(..).h5",f).group(1))
    print(idx)
    h1=h5py.File(f,"r")
    S1=h1["S"][()]

    S[idx,:,:,:]=S1[idx,:,:,:]
    print(n.sum(n.isnan(S1[idx,:,:,:])))
    
del(h["S"])
h["S"]=S
del(h["sample_rate"])
h["sample_rate"]=200e3
h.close()


# 16 and 1
fl=glob.glob("ion_line_interpolate_16_1_??.h5")
fl.sort()
h=h5py.File("ion_line_interpolate_16_1.h5","a")

S=h["S"][()]
for f in fl:
    print(f)
    idx=int(re.search(".*_(..).h5",f).group(1))
    print(idx)
    h1=h5py.File(f,"r")
    S1=h1["S"][()]

    S[idx,:,:,:]=S1[idx,:,:,:]
    print(n.sum(n.isnan(S1[idx,:,:,:])))
    
del(h["S"])
h["S"]=S
del(h["sample_rate"])
h["sample_rate"]=200e3
h.close()
