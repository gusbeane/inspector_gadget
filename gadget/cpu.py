import numpy as np

def load_cpu(filename):
    f = open(filename,"r")
    header = f.readline().strip().split(',')[:-1]
    
    dat = np.atleast_2d(np.loadtxt(f,delimiter=",",usecols=np.arange(0,len(header))))
    f.close()

    data = {}

    for i in np.arange(0,len(header)):
        data[header[i].strip()] = dat[:,i]

    return data



