import numpy as np

class Filter(object):
    def __init__(self):
        self.requieredFields = []
        self.parttype = []
    def getIndices(self, data):
        return []
    
    def reset(self):
        pass

    
    
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
        self.cat = catalog
        self.halo = halo
        self.subhalo = subhalo
        self.requieredFields=[]
        self.parttype = [0,1,2,3,4,5]
        
    def setHalo(self,halo=None, subhalo=None):
        self.halo = halo
        self.subhalo = subhalo
        
        
    def getIndices(self, data):
        ind = None
        gr = data['group']
        if self.offset[gr] + self.len[gr] > self.sn_offset[gr] and self.offset[gr] < self.sn_offset[gr]+data['NumPart_ThisFile']:
              start = np.max([np.int64(0), self.offset[gr] - self.sn_offset[gr]])
              stop = np.min([(self.offset[gr]+self.len[gr])-self.sn_offset[gr], data['NumPart_ThisFile']])
              ind = slice(start,stop)
        else:
            ind = slice(0,0)
    
        self.sn_offset[gr] += data['NumPart_ThisFile']
        
        return ind
    
    def reset(self):
        self.sn_offset = np.zeros(6, dtype=np.int64)
        
        self.halo_offset = np.zeros((self.cat.npart_loaded[0],6), dtype=np.int64)
        
        self.halo_offset[1:,:] = np.cumsum(self.cat.group.GroupLenType[:-1,:], axis=0, dtype=np.int64)
        
        if self.halo != None:
            self.offset = self.halo_offset[self.halo,:]
            self.len = self.cat.GroupLenType[self.halo,:]
            return
        
        if self.subhalo != None:
            halo = self.cat.subhalo.SubhaloGrNr[self.subhalo]
            self.offset = self.halo_offset[halo,:]   

            first =  np.int64(self.cat.group.GroupFirstSub[halo])
            if self.subhalo - first > 0:
                self.offset += np.sum(self.cat.subhalo.SubhaloLenType[first:self.subhalo,:], axis=0, dtype=np.int64)

            self.len = self.cat.subhalo.SubhaloLenType[self.subhalo,:]  
                
        
        
        
    
class Stars(Filter):
    def __init__(self):
        self.requieredFields=['GFM_StellarFormationTime']
        self.parttype = [4]

        
    def getIndices(self, data):
        ind = np.where( data['GFM_StellarFormationTime'] > 0.)[0]
        return ind
    
