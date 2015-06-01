import numpy as np
import re
import functools
import os

import gadget.fields as flds

import gadget.format1
import gadget.format2
import gadget.format3

backends_modules = {3:gadget.format3, 2:gadget.format2, 1:gadget.format1}
backends = {3:gadget.format3.Format3, 2:gadget.format2.Format2, 1:gadget.format1.Format1}

class Loader(object):
       
    def __init__(self,filename, snapshot=None, filenum=None, format=None, fields=None, parttype=None, combineFiles=False, toDouble=False, onlyHeader=False, verbose=False, **param): 
        self.data = {}
                    
        #detect backend
        if format is None:
            if not isinstance(self, gadget.loader.ICs):
                for f in backends_modules.keys():
                    if backends_modules[f].handlesfile(filename, snapshot=snapshot, filenum=filenum, snap=self, **param):
                        format = f
            else:
                for f in backends_modules.keys():
                    if backends_modules[f].writefile(filename):
                        format = f
                    
                
        self.filename = filename
        
        self.filenum = filenum
        self.snapshot = snapshot

        self.__format__ = format
        
        self.__present__ = flds.present.copy()
        self.__headerfields__ = []
        
        self.__fields__ = fields
        self.__normalizeFields__()
        
        self.__writeable__ = True
        self.__verbose__ = verbose
        self.__onlyHeader__ = onlyHeader
        self.__combineFiles__ = combineFiles
        self.__toDouble__ = toDouble
        
        if parttype is None:
            parttype = np.array([0,1,2,3,4,5])
        elif type(parttype) == list:
            parttype = np.array(parttype)
        elif type(parttype) != np.ndarray:
            parttype = np.array([parttype])
        self.__parttype__ = parttype

        self.__backend__ = backends[format](self, **param) 
            
        
    def __normalizeFields__(self):
        if self.__fields__ is None:
            return

        for i in np.arange(len(self.__fields__)):
            self.__fields__[i] = self.__normalizeName__(self.__fields__[i])
                    
    def __getattr__(self,attr):
        if attr in flds.legacy_header_names:
            return getattr(self, flds.legacy_header_names[attr]) 
              
        return self.__getitem__(attr)

    def __setattr__(self,attr,val):
        if attr in flds.legacy_header_names:
            setattr(self, flds.legacy_header_names[attr], val)
            return
            
        attr = self.__normalizeName__(attr)
        
        if attr =='data':
            object.__setattr__(self,attr,val)
        
        if attr in self.data:
            raise AttributeError("you can not exchange the array %s with your own object , write into the array using %s[...] instead"%(attr,attr))
        
        object.__setattr__(self,attr,val)
        
    def __dir__(self):
        dir = self.__dict__.keys()
        
        if hasattr(self, "data"):
            for i in self.data.keys():
                dir.append(i)
                if flds.rev_hdf5toformat2.has_key(i):
                    dir.append(flds.rev_hdf5toformat2[i])
        
        return dir 

    def __getitem__(self, item):
        item = self.__normalizeName__(item)
        
        if item in self.data:
            return self.data[item]
        else:
	    s=item[::-1]
            s=s.split("_",1)

            if(s[0]!=item[::-1]): #string contains an underscore
                g=[s[1][::-1],s[0]]
                                
	        it = self.__normalizeName__(g[0])
                if it in self.data:
	       	    if g[1] == 'x':
		        i = 0
		    elif g[1] == 'y':
		        i = 1
		    elif g[1] == 'z':
		        i = 2
		    else:
		        i = int(g[1])
		    d = self.data[it]
	    	    if d.ndim == 2 and d.shape[1] > i:
		        return d[:,i]
        
        raise AttributeError("unknown field '%s'"%item)
    
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
        if gr !=None:
            pres = np.zeros(6,dtype=np.int64)
            pres[gr] = shape
            if name == 'mass':
                pres[gr] = (0 if self.MassTable[gr]>0  else 1)
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
        
        name = self.__normalizeName__(name)

        if pres != None:
            pres = self.__learnPresent__(name,shape=pres)
        else:
            pres = self.__isPresent__(name)
            
        num = np.where(pres>0,self.npart_loaded,0)
        
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
        
        if filename is None:
            filename = self.filename
            
        if format is None:
            format = self.__format__

        if format == self.__format__:           
            self.__backend__.write(filename=filename)
        else:
            tmp_backend = backends[format](self)
            tmp_backend.write(filename=filename)
    
    def nextFile(self, num=None):
        if self.filenum is None:
            return False
        if num is None:
            if self.filenum < self.NumFilesPerSnapshot-1:
                num = self.filenum+1
            else:
                if self.__verbose__:
                    print "last chunk reached"
                return False
            
        else:
            if num >= self.NumFilesPerSnapshot or num < 0:
                if self.__verbose__:
                    print "invalide file number %d"%num
                return False
            
        if self.filenum == num:
            return True
        
        self.close()


        #open next file
        self.__backend__.load(num, self.snapshot)
        
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

        return True

    def iterFiles(self):
        if self.filenum is None:
            yield self
            return

        for i in np.arange(self.NumFilesPerSnapshot):
            self.nextFile(i)
            yield self
            
    def nextSnapshot(self, snapshot=None):
        if snapshot is None:
            snapshot = self.snapshot + 1
            
        if backends_modules[self.__format__].getFilename(self.filename, snapshot, self.filenum)[0] is None:
            return False
             
        self.close()

        #open next file
        self.__backend__.load(self.filenum, snapshot)
        
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

    def close(self):
        self.__backend__.close()
        
        if hasattr(self,"data"):
            for i in self.data.keys():
                if isinstance(self.data[i], np.ndarray): 
                    try:
                        self.data[i].resize(0)
                    except ValueError:
                        pass
        
            del self.data
            self.data = {}
            
            
        for i in self.__headerfields__:
            delattr(self,i)
        self.__headerfields__ = []
        if hasattr(self,"parameters"):
            del self.parameters
        if hasattr(self,"config"):
            del self.config

class Snapshot(Loader):
    """
    This class loads Gadget snapshots. Currently file format 2 and 3 (hdf5) are supported.
    """
    def __init__(self,filename, snapshot=None, filenum=None, format=None, fields=None, parttype=None, combineFiles=False, toDouble=False, onlyHeader=False, verbose=False, filter=None, sortID=False, **param):     
        """
        *filename* : The name of the snapshot file
        *format* : (optional) file format of the snapshot, otherwise this is guessed from the file name
        *fields* : (optional) list of fields to load, if None, all fields available in the snapshot are loaded
        *parttype* : (optional) array with particle type numbers to load, if None, all particles are loaded
        *combineFiles* : (optinal) if False only on part of the snapshot is loaded at a time, use nextFile() to go the next file.
        *toDouble* : (optinal) converts all values of type float to double precision
        *onlyHeader* : (optinal) load only the snapshot header
        *verbose* : (optional) enable debug output
        *filter* : Only load a filtered subset of the snapshot, specified by the filter object.
        *sortID* : sort all loaded data in each group by particle id

        *num_part* : (ic generation) generate an empty snapshot instead; num_part must be an array with 6 integers, giving the number of particles for each particle species
        *masses* : (ic generation, optinal) array with masses of each particle species, (0 if specified in the mass array)
        """
        super(Snapshot,self).__init__(filename, snapshot=snapshot, filenum=filenum, format=format, fields=fields, parttype=parttype, combineFiles=combineFiles, toDouble=toDouble, onlyHeader=onlyHeader, verbose=verbose, **param)
    
        self.__filter__ = filter
    
        self.__backend__.load(filenum, snapshot)
        
        self.header = Header(self)
        
        if not self.__onlyHeader__: 
            self.part0 = PartGroup(self,0)
            self.part1 = PartGroup(self,1)
            self.part2 = PartGroup(self,2)
            self.part3 = PartGroup(self,3)
            self.part4 = PartGroup(self,4)
            self.part5 = PartGroup(self,5)
            self.groups = [self.part0, self.part1, self.part2, self.part3, self.part4, self.part5]
        else:
            self.groups = []
            
        self.__sortID__ = sortID
        if sortID:
            for gr in self.groups:
                if gr.data.has_key("id"):
                    ind = np.argsort(gr.data['id'])
                    for d in gr.data.values():
                        d[...] = d[ind,...]

        self.__precision__ = None
        #set these two, in case we want to add fields to an existing snapshot
        if toDouble:
           self.__precision__ = np.float64
        elif hasattr(self,"Flag_DoublePrecision"):
            if self.Flag_DoublePrecision:
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
             
    def newFilter(self,filter):
        self.close()
        
        self.__filter__ = filter

        #open next file
        self.__backend__.load(self.filenum, self.snapshot)
        
        if not self.__onlyHeader__:        
            self.part0 = PartGroup(self,0)
            self.part1 = PartGroup(self,1)
            self.part2 = PartGroup(self,2)
            self.part3 = PartGroup(self,3)
            self.part4 = PartGroup(self,4)
            self.part5 = PartGroup(self,5)
            self.groups = [self.part0, self.part1, self.part2, self.part3, self.part4, self.part5]
        else:
            self.groups = []

        return True



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
        
        super(ICs,self).__init__(filename, format=format, verbose=verbose, **param)
        
        if format==2:
            raise Exception( "Creating format2 snapshots is not supported yet")
        
        self.__path__ = os.path.abspath(filename)
        
        self.__parttype__ = np.where(num_part>0)[0]
        
        if masses is None:
            masses = np.zeros(6)
            
        if precision is None:
            precision = np.float32
            
        self.__precision__ = precision
        self.__longids__ = longids
            
        self.NumPart_ThisFile = np.longlong(num_part)
        self.NumPart_Total = np.longlong(num_part)
        self.NumPart_Total_HighWord = np.zeros(6)
        
        self.nparticlesall = np.longlong(self.NumPart_Total)
        self.npart_loaded = self.NumPart_ThisFile
        self.npart = self.NumPart_ThisFile.sum()
        self.npartall = self.nparticlesall.sum()
        
        
        self.NumFilesPerSnapshot = 1
        self.MassTable = masses
        
        self.Time = 0.
        self.Redshift = 0.
        self.BoxSize = 0.
        self.Omega0 = 0.
        self.OmegaLambda = 0
        self.HubbleParam = 0
        self.Flag_Sfr = 0
        self.Flag_Feedback = 0
        self.Flag_Cooling = 0
        self.Flag_StellarAge = 0
        self.Flag_Metals = 0
        if precision == np.float64:
            self.Flag_DoublePrecision=1
        else:
            self.Flag_DoublePrecision=0
            
        self.__headerfields__ = ['Time', 'Redshift', 'BoxSize', 'Omega0', 'OmegaLambda', 'HubbleParam', 'Flag_Sfr',
                                  'Flag_Feedback', 'Flag_Cooling', 'Flag_StellarAge', 'Flag_Metals', 'Flag_DoublePrecision', 
                                  'NumPart_ThisFile', 'NumPart_Total', 'NumPart_Total_HighWord', 'NumFilesPerSnapshot', 'MassTable']
        
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
        
        self.__fields__ = []
        
        for field in flds.default:
            self.__fields__.append(field)
            
        if fields != None:
            for field in fields:
                f = self.__normalizeName__(field)
                if not f in self.__fields__:
                    self.__fields__.append(f)
                
        for field in self.__fields__:
                self.addField(field)

class Subfind(Loader):
    def __init__(self,filename, snapshot=None, filenum=None,  format=None, fields=None, parttype=None, combineFiles=False, toDouble=False, onlyHeader=False, verbose=False, **param):
        if parttype is None:
            parttype = np.array([0,1])
        elif type(parttype) == list:
            parttype = np.array(parttype)
        elif type(parttype) != np.ndarray:
            parttype = np.array([parttype])

        parttype_filter = []
        for i in parttype:
            if i!=0 and i!=1:
		if verbose:
	            print "ignoring part type %d for subfind output"%i
            else:
                parttype_filter.append(i)
           
        super(Subfind,self).__init__(filename, snapshot=snapshot, filenum=filenum, format=format, fields=fields, parttype=parttype_filter, combineFiles=combineFiles, toDouble=toDouble, onlyHeader=onlyHeader, verbose=verbose, **param)
        self.__backend__.load(filenum, snapshot)

        self.header = Header(self)
        
        if not onlyHeader:
            self.group = PartGroup(self,0)
            self.subhalo = PartGroup(self,1)
            self.groups = [self.group, self.subhalo]   
    
        self.__writeable__ = False
    

class Header(object):
    def __init__(self,parent):
        self.__parent__ = parent
                
    def __getattr__(self,name):
        if name in flds.legacy_header_names:
            name = flds.legacy_header_names[name] 
        
        if name in self.__parent__.__headerfields__:
            return getattr(self.__parent__,name)
        else:
            raise AttributeError
            
    def __setattr__(self,name, value):
        #we can't handle these
        if name in ["__parent__","__attrs__"]:
            super(Header,self).__setattr__(name,value)
            return        
        if name in flds.legacy_header_names:
            name = flds.legacy_header_names[name]     
        if name in self.__parent__.__headerfields__:
            if type(value)==list:
                value = np.array(value)
            setattr(self.__parent__,name,value)
        else:
            raise AttributeError
        
    def __dir__(self):
        return self.__dict__.keys() + self.__parent__.__headerfields__
    
    def __str__(self):
        filename = self.__parent__.__path__
        if isinstance(self.__parent__, Snapshot):
            tmp = "snapshot "+filename+"\n"
        elif isinstance(self.__parent__, ICs):
            tmp = "ICs "+filename+"\n"
        else:
            tmp = "subfind output "+filename+"\n"
            
        tmp += "header:\n"
            
        for entry in self.__parent__.__headerfields__:
            val = getattr(self,entry)
            if type(val) ==  np.ndarray or type(val) == list:
                tmp += '  '+entry+': '+', '.join([str(x) for x in val])+'\n'
            else:
                tmp += '  '+entry+': '+str(val)+'\n'
        return tmp

    def __repr__(self):
        filename = self.__parent__.__path__
        if isinstance(self.__parent__, Snapshot):
            return "snapshot "+filename
        elif isinstance(self.__parent__, ICs):
            return "ICs "+filename
        else:
            return "subfind output "+filename
        
class Parameter(object):
    def __init__(self,parent,name):
        self.__parent__ = parent
        self.__name__ = name
        self.__attrs__ = []
            
    def __str__(self):
        filename = self.__parent__.__path__
        if isinstance(self.__parent__, Snapshot):
            tmp = "snapshot "+filename+"\n"
        elif isinstance(self.__parent__, ICs):
            tmp = "ICs "+filename+"\n"
        else:
            tmp = "subfind output "+filename+"\n"
            
        tmp += self.__name__ + ":\n"
            
        for entry in self.__attrs__:
            val = getattr(self,entry)
            if type(val) ==  np.ndarray or type(val) == list:
                tmp += '  '+entry+': '+', '.join([str(x) for x in val])+'\n'
            else:
                tmp += '  '+entry+': '+str(val)+'\n'
        return tmp

    def __repr__(self):
        filename = self.__parent__.__path__
        if isinstance(self.__parent__, Snapshot):
            return "snapshot "+filename
        elif isinstance(self.__parent__, ICs):
            return "ICs "+filename
        else:
            return "subfind output "+filename

class PartGroup(object):
    def __init__(self,parent,num):
        self.__parent__ = parent
        self.__num__ = num

    def __str__(self):
        filename = self.__parent__.__path__
        if isinstance(self.__parent__, Snapshot):
            tmp = "snapshot "+filename+"\nparticle group %d (%d particles):\n"%(self.__num__,self.__parent__.npart_loaded[self.__num__])
        elif isinstance(self.__parent__, ICs):
            tmp = "ICs "+filename+"\nparticle group %d (%d particles):\n"%(self.__num__,self.__parent__.npart_loaded[self.__num__])
        else:
            if self.__num__ == 0:
                tmp = "subfind output "+filename+"\ngroups (%d groups):\n"%(self.__parent__.npart_loaded[self.__num__])
            else:
                tmp = "subfind output "+filename+"\nsubhalos (%d subhalos):\n"%(self.__parent__.npart_loaded[self.__num__])
            
        for i in self.data.keys():
            tmp += "  " + i
            if flds.rev_hdf5toformat2.has_key(i):
                tmp += '/'+flds.rev_hdf5toformat2[i]
            tmp += "\n"
        return tmp
        
    def __repr__(self):
        filename = self.__parent__.__path__
        if isinstance(self.__parent__, Snapshot):
            return "snapshot "+filename+", particle group %d contains %d particles"%(self.__num__,self.__parent__.npart_loaded[self.__num__])
        elif isinstance(self.__parent__, ICs):
            return "ICs "+filename+", particle group %d contains %d particles"%(self.__num__,self.__parent__.npart_loaded[self.__num__])
        else:
            if self.__num__ == 0:
                return "subfind output "+filename+", contains %d groups"%(self.__parent__.npart_loaded[self.__num__])
            else:
                return "subfind output "+filename+", contains %d subhalos"%(self.__parent__.npart_loaded[self.__num__])




    def __getitem__(self, item):
        item = self.__parent__.__normalizeName__(item)
        parent = self.__parent__
        num = self.__num__
        
        f = None
        
        if item in parent.data:
            f = parent.data[item]
            it = item
        else:
            s=item[::-1]
            s=s.split("_",1)

            if(s[0]!=item[::-1]): #string contains an underscore
                g=[s[1][::-1],s[0]]

                it = parent.__normalizeName__(g[0])
                if it in parent.data:
                    if g[1] == 'x':
                        i = 0
                    elif g[1] == 'y':
                        i = 1
                    elif g[1] == 'z':
                        i = 2
                    else:
                        i = int(g[1])
                            
                        d = parent.data[it]
                        if d.ndim == 2 and d.shape[1] > i:
                            f = d[:,i]
        if not f is None:
            pres = parent.__isPresent__(it)
            if pres[num]>0:       
                n1 = np.where(pres>0, parent.npart_loaded,np.zeros(6,dtype=np.longlong))
                tmp = np.sum(n1[0:num])
                return f[tmp:tmp+parent.npart_loaded[num]]
        
        raise AttributeError("unknown field '%s'"%item)
    
    def __setattr__(self,attr,val):
        if attr =='__parent__':
            object.__setattr__(self,attr,val)
        elif attr =='data':
            raise AttributeError("you are not allowed to set attribute 'data'")
        
        attr = self.__parent__.__normalizeName__(attr)
        
        if attr in self.__parent__.data:
            raise AttributeError("you can not exchange the array '%s' with your own object, write into the array using %s[...] instead"%(attr,attr))
        
        object.__setattr__(self,attr,val)
    
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
        else:
            return self.__getitem__(attr)
        
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
                        if flds.rev_hdf5toformat2.has_key(key):
                            dir.append(flds.rev_hdf5toformat2[key])
        return dir   
        
