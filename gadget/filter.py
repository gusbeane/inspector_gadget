import numpy as np

class Filter(object):
    def __init__(self):
        self.requieredFields = []
    def getIndices(self, data):
        return []
    

    
    
class Rectangle(Filter):
    def __init__(self, center, boxsize):
        center = np.array(center)
        boxsize = np.array(boxsize)
        
        self.lower = center-boxsize/2.
        self.upper = center+boxsize/2.
        
        self.requieredFields = ['pos']
        self.parttype = [0,1,2,3,4,5]
    
        
    def getIndices(self, data):
        ind = np.where( (np.all(data['pos'][:,:]>=self.lower,axis=1)) & (np.all(data['pos'][:,:]<=self.upper,axis=1)) )[0]

        return ind
        
        
class Sphere(Filter):
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
        
        self.requieredFields = ['pos']
        self.parttype = [0,1,2,3,4,5]
    
        
    def getIndices(self, data):
        ind = np.where( np.sum( (data['pos'][:,:]-self.center)**2,axis=1) <= self.radius*self.radius)[0]

        return ind
        
        
class Halo(Filter):
    def __init__(self, catalog, halo=None, subhalo=None):
        self.catalog = catalog
        self.halo = halo
        self.subhalo = subhalo
        self.requieredFields=[]
        
    def setHalo(self,halo=None, subhalo=None):
        self.halo = halo
        self.subhalo = subhalo
        
        
    def getIndices(self, data):
        pass
    
class Stars(Filter):
    def __init__(self):
        self.requieredFields=['GFM_StellarFormationTime']
        self.parttype = [5]

        
    def getIndices(self, data):
        ind = np.where( data['GFM_StellarFormationTime'] > 0.)[0]
        return ind
    