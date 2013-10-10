import numpy as np
from mpi4py import MPI

def map(func):
    def map_wrapper(self, *args, **kwargs):
        self.__collected__ = False

        for i in np.arange(len(self.__myitems__)):
            self.__currdict__ = self.__dicts__[i]
            func(self, self.__myitems__[i], *args, **kwargs)
            
        self.__mapped__ = True
        
    return map_wrapper
    
def reduce(func):
    def reduce_wrapper(self, bcast=False, *args, **kwargs):    
        if self.__collected__ == False:
            self.__collect__()
            
        ret = None
        
        if self.__thistask__ == 0:
            ret = func(self, *args, **kwargs)
            
        if bcast==True:
            ret = self.__comm__.bcast(ret)
            
        return ret
        
    return reduce_wrapper        
    
class MapReduce(object):
    """
        *num* : Number of work items
        *order* : This parameter sets the mapping of work items to MPI tasks, default is 'sequential', 'stride' is possible as well   
        """
    def __init__(self, num=None, order=None):
        self.__comm__ = MPI.COMM_WORLD
        self.__thistask__ = self.__comm__.rank
        self.__ntasks__ = self.__comm__.size
        
        if num == None:
            num = self.__ntasks__
        self.__num__ = num
        
        items = int(np.floor(num/self.__ntasks__))
        rem = num - items*self.__ntasks__
        
        if order == None or order == "sequential":
            if self.__thistask__ < rem:
                self.__myitems__ = np.arange(self.__thistask__*(items+1), (self.__thistask__+1)*(items+1))
            else:
                self.__myitems__ = np.arange(rem*(items+1)+(self.__thistask__-rem)*items,rem*(items+1)+(self.__thistask__-rem+1)*items )
        elif order == "stride":
            self.__myitems__ = np.arange(0,self.__num__-self.__thistask__,self.__ntasks__)+self.__thistask__
        else:
            raise Exception("unknown order argument: %s"%order)
            

        self.__dicts__ = []
        for i in np.arange(len(self.__myitems__)):
            self.__dicts__.append({})
            
        self.__currdict__ = self.__dicts__[0]
        self.__mapped__ = False
        self.__collected__ = False

        
    def __setitem__(self,item, value):
        if type(value) == int:
            value = np.int32(value)
        elif type(value) == long:
            value = np.int64(value)
        elif type(value) == float:
            value = np.float64(value)
        
        self.__currdict__[item] = value
    
    def __getitem__(self, item):
        return self.__currdict__[item]
        
    def __collect__(self):
        if self.__mapped__ == False:
            raise Exception("Call a map function before collect")
            
        self.__colldict__ = {}
        
        for key in self.__dicts__[0]:
            cval = self.__dicts__[0][key]
            
            if self.__thistask__ == 0:
                shape = np.append(self.__num__, cval.shape)
                
            if len(self.__myitems__) > 1:
                app = []
                for i in np.arange(1, len(self.__myitems__)):
                    app.append(self.__dicts__[i][key])
                cval = np.append(cval,app)

            result = None
            if self.__thistask__ == 0:
                result = np.zeros(shape,dtype=cval.dtype )

            self.__comm__.Gatherv(cval, result)
            
            if self.__thistask__ == 0:
                self.__colldict__[key] = result
                               
        del self.__currdict__
        del self.__dicts__
        
        self.__currdict__ = self.__colldict__
        self.__collected__ = True
            
            
            
            
    
    
            
    


