import numpy as np
import re
import functools

import gadget.fields as flds

class Loader(object):
       
    def __init__(self,filename, format=None, fields=None, parttype=None, combineFiles=False, toDouble=False, onlyHeader=False, verbose=False, **param):     
        #detect backend
        if format==None:
            if filename.endswith('.hdf5') or filename.endswith('.h5'):
                format = 3
            else:
                format = 2

        self.filename = filename

        self.__format__ = format
        
        self.__fields__ = fields
        self.__normalizeFields__()
        
        self.__writeable__ = True
        self.__verbose__ = verbose
        self.__onlyHeader__ = onlyHeader
        self.__combineFiles__ = combineFiles
        self.__toDouble__ = toDouble
        
        if parttype == None:
            parttype = [0,1,2,3,4,5]
        self.__parttype__ = np.array(parttype)


        if format==3:
            import format3
            self.__backend__=format3.Format3(self, **param)

        if format==2:              
            import format2
            self.__backend__=format2.Format2(self, **param)
                
        self.data = {}
            
        
    def __normalizeFields__(self):
        if self.__fields__ == None:
            return

        for i in np.arange(len(self.__fields__)):
            self.__fields__[i] = self.__normalizeName__(self.__fields__[i])
                    
    def __getattr__(self,attr):       
        attr = self.__normalizeName__(attr)

        if attr in self.data:
            return self.data[attr]
        else:
            raise AttributeError
        
    def __dir__(self):
        dir = self.__dict__.keys()
        
        if hasattr(self, "data"):
            for i in self.data.keys():
                dir.append(i)
                if flds.shortnames.has_key(i):
                    dir.append(flds.shortnames[i])
        
        return dir 

    def __getitem__(self, item):
        item = self.__normalizeName__(item)
        return self.data[item]
    
    
    def __normalizeName__(self, name):
        name = flds.hdf5toformat2.get(name,name)
        name = flds.rev_shortnames.get(name,name)
        return name
        
        
    def __isPresent__(self, name):
        try:
            pres = self.__present__[name]
        except KeyError:
            raise Exception("Unkonwn array shape for field %s"%name)
    
        #filter for loaded particle types
        tmp = np.zeros(6,dtype=np.longlong)
        tmp[self.__parttype__] = pres[self.__parttype__]
        return tmp

    def __learnPresent__(self, name, gr=None, shape=1):
        if not hasattr(self,"__present__"):
            self.__present__ = flds.present
            
        if gr !=None:
            pres = np.zeros(6,dtype=np.int64)
            pres[gr] = shape
            if name == 'mass':
                pres[gr] = (0 if self.masses[gr]>0  else 1)
            old = self.__present__.get(name,np.zeros(6,dtype=np.longlong))
            pres = np.maximum(old,pres)
            self.__present__[name] = pres
        else:
            shape = np.array(shape)
            if shape.shape!=(6,):
               raise Exception("invalide present array")
            pres = shape
            self.__present__[name] = pres
    
        #filter for loaded particle types
        tmp = np.zeros(6,dtype=np.longlong)
        tmp[self.__parttype__] = pres[self.__parttype__]
        return tmp
    
    def __str__(self):
        tmp = self.header.__str__()
        
        if not self.__onlyHeader__:
            for i in np.arange(0,6):
                if self.nparticlesall[i] > 0:
                    tmp += re.sub("[^\n]*\n","\n",self.groups[i].__str__(),count=1)

        return tmp
    
    def __repr__(self):
        return self.header.__repr__()

    def addField(self, name, pres=None, dtype=None):
        if not self.__writeable__:
            raise Exception("This snapshot can not be modified")
        
        name = flds.hdf5toformat2.get(name,name)

        if pres != None:
            pres = self.__learnPresent__(name,shape=pres)
        else:
            pres = self.__isPresent__(name)
            
        num = np.where(pres>0,self.nparticles,0)
        
        if dtype==None:
            if name=='id':
                if self.__longids__:
                    dtype = np.uint64
                else:
                    dtype = np.uint32
            elif self.__precision__ != None:
                dtype = self.__precision__
            else:
                dtype = np.float32
             
        if np.max(pres)>1:        
            f = np.zeros([num.sum(),np.max(pres)], dtype=dtype)
        else:
            f = np.zeros([num.sum(),], dtype=dtype)
            
        self.data[name] = f
        
    def write(self, filename=None, format=None):
        if not self.__writeable__:
            raise Exception("This snapshot can not be written")
        
        if filename==None:
            filename = self.filename
            
        if format==None:
            format = self.__format__
            
        if format == self.__format__:
            if format==2:
                raise Exception( "Creating format2 snapshots is not supported yet")
            
            self.__backend__.write(filename=filename)
        else:
            if format==3:
                import format3
                tmp_backend = format3.Format3(self)
                tmp_backend.write(filename=filename)
            elif format==2:
                raise Exception( "Creating format2 snapshots is not supported yet")
    
    def nextFile(self, num=None):
        if self.currFile == None:
            return False
        if num == None:
            if self.currFile < self.num_files-1:
                num = self.currFile+1
            else:
                if self.__verbose__:
                    print "last chunk reached"
                return False
            
        else:
            if num >= self.num_files or num < 0:
                if self.__verbose__:
                    print "invalide file number %d"%num
                return False
            
        if self.currFile == num:
            return True
        
        self.close()


        #open next file
        self.__backend__.load(num)
        
        if self.__onlyHeader__: 
            if  isinstance(self, Snapshot):       
                self.part0 = PartGroup(self,0)
                self.part1 = PartGroup(self,1)
                self.part2 = PartGroup(self,2)
                self.part3 = PartGroup(self,3)
                self.part4 = PartGroup(self,4)
                self.part5 = PartGroup(self,5)
                self.groups = [self.part0, self.part1, self.part2, self.part3, self.part4, self.part5]
            else:
                self.group = PartGroup(self,0)
                self.subhalo = PartGroup(self,1)
                self.groups = [self.group, self.subhalo]   
        
        self.__onlyHeader__ = False

        return True

    def iterFiles(self):
        if self.currFile == None:
            yield self
            return

        for i in np.arange(self.num_files):
            self.nextFile(i)
            yield self

    def close(self):
        self.__backend__.close()
        
        self.__rmconvenience__()
        
        #TODO keep groups, only clean data
        if isinstance(self, Snapshot):    
            del self.part0
            del self.part1
            del self.part2
            del self.part3
            del self.part4
            del self.part5
        else:
            del self.group
            del self.subhalo
            
        del self.groups
        
        if hasattr(self,"data"):
            for i in self.data.keys():
                if isinstance(self.data[i], np.ndarray): 
                    try:
                        self.data[i].resize(0)
                    except ValueError:
                        pass
        
            del self.data

class Snapshot(Loader):
    """
    This class loads Gadget snapshots. Currently file format 2 and 3 (hdf5) are supported.
    """
    def __init__(self,filename, format=None, fields=None, parttype=None, combineFiles=False, toDouble=False, onlyHeader=False, verbose=False, **param):     
        """
        *filename* : The name of the snapshot file
        *format* : (optional) file format of the snapshot, otherwise this is guessed from the file name
        *fields* : (optional) list of fields to load, if None, all fields available in the snapshot are loaded
        *parttype* : (optional) array with particle type numbers to load, if None, all particles are loaded
        *combineFiles* : (optinal) if False only on part of the snapshot is loaded at a time, use nextFile() to go the next file.
        *toDouble* : (optinal) converts all values of type float to double precision
        *onlyHeader* : (optinal) load only the snapshot header
        *verbose* : (optional) enable debug output
        
        *num_part* : (ic generation) generate an empty snapshot instead; num_part must be an array with 6 integers, giving the number of particles for each particle species
        *masses* : (ic generation, optinal) array with masses of each particle species, (0 if specified in the mass array)
        """
        super(Snapshot,self).__init__(filename, format=format, fields=fields, parttype=parttype, combineFiles=combineFiles, toDouble=toDouble, onlyHeader=onlyHeader, verbose=verbose, **param)
    
        self.__backend__.load()
        
        self.header = Header(self)
        
        if not self.__onlyHeader__: 
            self.part0 = PartGroup(self,0)
            self.part1 = PartGroup(self,1)
            self.part2 = PartGroup(self,2)
            self.part3 = PartGroup(self,3)
            self.part4 = PartGroup(self,4)
            self.part5 = PartGroup(self,5)
            self.groups = [self.part0, self.part1, self.part2, self.part3, self.part4, self.part5]


        self.__precision__ = None
        
        #set these two, in case we want to add fields to an existing snapshot
        if toDouble:
           self.__precision__ = np.float64
        elif hasattr(self,"flag_doubleprecision"):
            if self.flag_doubleprecision:
                self.__precision__ = np.float64
            else:
                self.__precision__ = np.float32
        elif hasattr(self,"pos"):
            self.__precision__ = self.pos.dtype
            
        if hasattr(self,"id"):
            if self.data['id'].dtype==np.uint64:
                self.__longids__ = True
            else:
                self.__longids__ = False
        else:
             self.__longids__ = False


class ICs(Loader):
    def __init__(self,filename, num_part, format=None, fields=None, masses=None, precision=None, longids=False, verbose=False, **param): 
        """
        Creates an empty snapshot used for ic generation
        
        *filename* : The name of the snapshot file
        *num_part* : (ic generation) generate an empty snapshot instead; num_part must be an array with 6 integers, giving the number of particles for each particle species
        *masses* : (ic generation, optinal) array with masses of each particle species, (0 if specified in the mass array)
        *format* : (optional) file format of the snapshot, otherwise this is guessed from the file name
        *fields* : (optional) list of fields to add beside the basic fields
       
        *verbose* : (optional) enable debug output
        """
        num_part = np.array(num_part)
        if masses != None:
            masses = np.array(masses)
        
        super(ICs,self).__init__(filename, format=format, fields=fields,  verbose=verbose, **param)
        
        if format==2:
            raise Exception( "Creating format2 snapshots is not supported yet")
        
        self.__parttype__ = np.where(num_part>0)[0]
        
        if masses == None:
            masses = np.zeros(6)
            
        if precision == None:
            precision = np.float32
            
        self.__precision__ = precision
        self.__longids__ = longids
            
        self.nparticles = np.longlong(num_part)
        self.nparticlesall = np.longlong(num_part)
        self.npart_loaded = self.nparticles
        self.npart = self.nparticles.sum()
        self.npartall = self.nparticlesall.sum()
        
        
        self.num_files = 1
        self.masses = masses
        
        self.time = 0.
        self.redshift = 0.
        self.boxsize = 0.
        self.omega0 = 0.
        self.omegalambda = 0
        self.hubbleparam = 0
        self.flag_sfr = 0
        self.flag_feedback = 0
        self.flag_cooling = 0
        self.flag_stellarage = 0
        self.flag_metals = 0
        if precision == np.float64:
            self.flag_doubleprecision=1
        else:
            self.flag_doubleprecision=0
        
        self.header = Header(self)
        
        #initiate mass field
        self.__learnPresent__('mass',shape=np.where(masses==0,1,0))
    
        self.part0 = PartGroup(self,0)
        self.part1 = PartGroup(self,1)
        self.part2 = PartGroup(self,2)
        self.part3 = PartGroup(self,3)
        self.part4 = PartGroup(self,4)
        self.part5 = PartGroup(self,5)
        self.groups = [ self.part0, self.part1, self.part2, self.part3, self.part4, self.part5]
        
        for field in flds.default:
            self.addField(field)
        if self.__fields__ != None:
            for field in self.__fields__:
                self.addField(field)

class Subfind(Loader):
    def __init__(self,filename, format=None, fields=None, parttype=None, combineFiles=False, toDouble=False, onlyHeader=False, verbose=False, **param):
        if parttype == None:
            parttype = [0,1]
	
        parttype_filter = []
        for i in parttype:
            if i!=0 and i!=1:
		if verbose:
	            print "ignoring part type %d for subfind output"%i
            else:
                parttype_filter.append(i)

        param['combineParticles'] = False
           
        super(Subfind,self).__init__(filename, format=format, fields=fields, parttype=parttype_filter, combineFiles=combineFiles, toDouble=toDouble, onlyHeader=onlyHeader, verbose=verbose, **param)
        self.__backend__.load()

        self.header = Header(self)
        if not onlyHeader:
            self.group = PartGroup(self,0)
            self.subhalo = PartGroup(self,1)
            self.groups = [self.group, self.subhalo]   
    
        self.__writeable__ = False

    def __getitem__(self, item):
        raise KeyError()
    

class Header(object):
    def __init__(self,parent):
        self.__parent__ = parent
        self.__attrs__ = []
        for entry in flds.headerfields:
            if hasattr(parent,entry):
                self.__attrs__.append(entry)
                
    def __getattr__(self,name):
        if name in self.__attrs__:
            return getattr(self.__parent__,name)
        else:
            raise AttributeError
            
    def __setattr__(self,name, value):
        #we can't handle these
        if name in ["__parent__","__attrs__"]:
            super(Header,self).__setattr__(name,value)
        elif name in self.__attrs__:
            if type(value)==list:
                value = np.array(value)
            setattr(self.__parent__,name,value)
        else:
            raise AttributeError
        
    def __dir__(self):
        return self.__dict__.keys() + self.__attrs__
    
    def __str__(self):
        if isinstance(self.__parent__, Snapshot):
            tmp = "snapshot "+self.__parent__.filename+":\n"
        else:
            tmp = "subfind output "+self.__parent__.filename+":\n"
            
        for entry in flds.headerfields:
            if hasattr(self,entry):
                val = getattr(self,entry)
                if type(val) ==  np.ndarray or type(val) == list:
                    tmp += '  '+entry+': '+', '.join([str(x) for x in val])+'\n'
                else:
                    tmp += '  '+entry+': '+str(val)+'\n'


        return tmp

    def __repr__(self):
        if isinstance(self.__parent__, Snapshot):
            return "snapshot "+self.__parent__.filename
        else:
            return "subfind output "+self.__parent__.filename

class PartGroup(object):
    def __init__(self,parent,num):
        self.__parent__ = parent
        self.__num__ = num

    def __str__(self):
        if isinstance(self.__parent__, Snapshot):
            tmp = "snapshot "+self.__parent__.filename+"\nparticle group %d contains %d particles:\n"%(self.__num__,self.__parent__.npart_loaded[self.__num__])
        else:
            if self.__num__ == 0:
                tmp = "subfind output "+self.__parent__.filename+"\ncontains %d groups:\n"%(self.__parent__.npart_loaded[self.__num__])
            else:
                tmp = "subfind output "+self.__parent__.filename+"\ncontains %d subhalos:\n"%(self.__parent__.npart_loaded[self.__num__])
            
        for i in self.data.keys():
            if flds.shortnames.has_key(i):
                tmp += "  "+i +'/'+flds.shortnames.get(i,i)+"\n"
            elif not i in flds.shortnames.values():
                tmp += "  "+i+"\n"
        return tmp
        
    def __repr__(self):
        if isinstance(self.__parent__, Snapshot):
            return "snapshot "+self.__parent__.filename+", particle group %d contains %d particles"%(self.__num__,self.__parent__.npart_loaded[self.__num__])
        else:
            if self.__num__ == 0:
                return "subfind output "+self.__parent__.filename+", contains %d groups"%(self.__parent__.npart_loaded[self.__num__])
            else:
                return "subfind output "+self.__parent__.filename+", contains %d subhalos"%(self.__parent__.npart_loaded[self.__num__])

    def __getitem__(self, item):
        item = self.__parent__.__normalizeName__(item)
        return self.data[item]
    
    def __getattr__(self,attr):       
        parent = self.__parent__
        num = self.__num__
        attr = self.__parent__.__normalizeName__(attr)
        
        if attr == "data":
            data = {}
            if parent.npart_loaded[num]>0:
                if hasattr(parent,"data"):
                    for key in parent.data.iterkeys():
                        pres = parent.__isPresent__(key)
                        if pres[num]>0:
                            f = parent.data[key]
                            n1 = np.where(pres>0, parent.npart_loaded,np.zeros(6,dtype=np.longlong))
                            tmp = np.sum(n1[0:num])
                            data[key] = f[tmp:tmp+parent.npart_loaded[num]]
            return data  
        elif attr in parent.data:
            pres = parent.__isPresent__(attr)
            if pres[num]>0:
                f = parent.data[attr]
                n1 = np.where(pres>0, parent.npart_loaded,np.zeros(6,dtype=np.longlong))
                tmp = np.sum(n1[0:num])
                return f[tmp:tmp+parent.npart_loaded[num]]
            else:
                raise AttributeError
        else:
            raise AttributeError
        
    def __dir__(self):
        parent = self.__parent__
        num = self.__num__
        
        dir = self.__dict__.keys()
        dir.append('data')
        if parent.npart_loaded[num]>0:
            if hasattr(parent,"data"):
                for key in parent.data.iterkeys():
                    pres = parent.__isPresent__(key)
                    if pres[num]>0:
                        dir.append(key)
                        if flds.shortnames.has_key(key):
                            dir.append(flds.shortnames[key])
        return dir   
        
