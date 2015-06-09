import numpy as np
from mpi4py import MPI

def map(func):
    def map_wrapper(self, *args, **kwargs):
        self._collected = False

        for i in np.arange(len(self._myitems)):
            self._currdict = self._dicts[i]
            func(self, self._myitems[i], *args, **kwargs)
            
        self._mapped = True
        
    return map_wrapper
    
def reduce(func):
    def reduce_wrapper(self, bcast=False, *args, **kwargs):    
        if self._collected == False:
            self._collect()
            
        ret = None
        
        if self._thistask == 0:
            ret = func(self, *args, **kwargs)
            
        if bcast==True:
            ret = self._comm.bcast(ret)
            
        return ret
        
    return reduce_wrapper        
    
class MapReduce(object):
    """
        *num* : Number of work items
        *order* : This parameter sets the mapping of work items to MPI tasks, default is 'sequential', 'stride' is possible as well   
        """
    def __init__(self, num=None, order=None):
        self._comm = MPI.COMM_WORLD
        self._thistask = self._comm.rank
        self._ntasks = self._comm.size
        
        if num == None:
            num = self._ntasks
        self._num = num
        
        items = int(np.floor(num/self._ntasks))
        rem = num - items*self._ntasks
        
        if order == None or order == "sequential":
            if self._thistask < rem:
                self._myitems = np.arange(self._thistask*(items+1), (self._thistask+1)*(items+1))
            else:
                self._myitems = np.arange(rem*(items+1)+(self._thistask-rem)*items,rem*(items+1)+(self._thistask-rem+1)*items )
        elif order == "stride":
            self._myitems = np.arange(0,self._num-self._thistask,self._ntasks)+self._thistask
        else:
            raise Exception("unknown order argument: %s"%order)
            

        self._dicts = []
        for i in np.arange(len(self._myitems)):
            self._dicts.append({})
            
        if len(self._myitems) > 0:
            self._currdict = self._dicts[0]
        else:
            self._currdict = None
            
        self._mapped = False
        self._collected = False

        
    def __setitem__(self,item, value):
        if type(value) == int:
            value = np.int32(value)
        elif type(value) == long:
            value = np.int64(value)
        elif type(value) == float:
            value = np.float64(value)
        
        self._currdict[item] = value
    
    def __getitem__(self, item):
        return self._currdict[item]
        
    def _collect(self):
        if self._mapped == False:
            raise Exception("Call a map function before collect")
            
        self._colldict = {}
        
        for key in self._dicts[0]:
            cval = self._dicts[0][key]
            
            if self._thistask == 0:
                shape = np.append(self._num, cval.shape)
                
            if len(self._myitems) > 1:
                app = []
                for i in np.arange(1, len(self._myitems)):
                    app.append(self._dicts[i][key])
                cval = np.append(cval,app)

            result = None
            if self._thistask == 0:
                result = np.zeros(shape,dtype=cval.dtype )

            self._comm.Gatherv(cval, result)
            
            if self._thistask == 0:
                self._colldict[key] = result
                               
        del self._currdict
        del self._dicts
        
        self._currdict = self._colldict
        self._collected = True
            
            
            
            
    
    
            
    


