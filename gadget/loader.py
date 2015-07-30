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
                    
        if format is None:
            raise Exception("could not detect backend for file '%s', snapshot %s, filenum %s. Specify file format or check file name."%(filename, snapshot, filenum))
        
        self.filename = filename
        
        self.filenum = filenum
        self.snapshot = snapshot

        self._format = format
        
        self._present = flds.present.copy()
        self._headerfields = []
        
        self._fields = fields
        self._normalizeFields()
        
        self._writeable = True
        self._verbose = verbose
        self._onlyHeader = onlyHeader
        self._combineFiles = combineFiles
        self._toDouble = toDouble
        
        if parttype is None:
            parttype = np.array([0,1,2,3,4,5])
        elif type(parttype) == list:
            parttype = np.array(parttype)
        elif type(parttype) != np.ndarray:
            parttype = np.array([parttype])
        self._parttype = parttype

        self._backend = backends[format](self, **param) 
            
        
    def _normalizeFields(self):
        if self._fields is None:
            return

        for i in np.arange(len(self._fields)):
            self._fields[i] = self._normalizeName(self._fields[i])
                    
    def __getattr__(self,attr):
        if attr in flds.legacy_header_names:
            return getattr(self, flds.legacy_header_names[attr]) 
              
        return self.__getitem__(attr)

    def __setattr__(self,attr,val):
        if attr in flds.legacy_header_names:
            setattr(self, flds.legacy_header_names[attr], val)
            return
            
        attr = self._normalizeName(attr)
        
        if attr =='data':
            object.__setattr__(self,attr,val)
        
        if attr in self.data:
            raise AttributeError("you can not exchange the array %s with your own object , write into the array using %s[...] instead"%(attr,attr))
        
        object.__setattr__(self,attr,val)
        
    def __dir__(self):
        dir = list(self.__dict__.keys())
        
        if hasattr(self, "data"):
            for i in self.data.keys():
                dir.append(i)
                if i in flds.rev_hdf5toformat2:
                    dir.append(flds.rev_hdf5toformat2[i])
        
        return dir 

    def __getitem__(self, item):
        item = self._normalizeName(item)
        
        if item in self.data:
            return self.data[item]
        else:
            s=item[::-1]
            s=s.split("_",1)

            if(s[0]!=item[::-1]): #string contains an underscore
                g=[s[1][::-1],s[0]]
                                
                it = self._normalizeName(g[0])
                if it in self.data:
                    if g[1] == 'x':
                        i = 0
                    elif g[1] == 'y':
                        i = 1
                    elif g[1] == 'z':
                        i = 2
                    else:
                        try:
                            i = int(g[1])
                        except:
                            raise AttributeError("unknown field '%s'"%item)
                    d = self.data[it]
                    if d.ndim == 2 and d.shape[1] > i:
                        return d[:,i]
        
        raise AttributeError("unknown field '%s'"%item)
    
    def __contains__(self, item):
        item = self._normalizeName(item)
        
        if item in self.data:
            return True
        else:
            s=item[::-1]
            s=s.split("_",1)

            if(s[0]!=item[::-1]): #string contains an underscore
                g=[s[1][::-1],s[0]]
                                
                it = self._normalizeName(g[0])
                if it in self.data:
                    if g[1] == 'x':
                        i = 0
                    elif g[1] == 'y':
                        i = 1
                    elif g[1] == 'z':
                        i = 2
                    else:
                        try:
                            i = int(g[1])
                        except:
                            return False
                    d = self.data[it]
                    if d.ndim == 2 and d.shape[1] > i:
                        return True
        
        return False
    
    def _normalizeName(self, name):
        name = flds.hdf5toformat2.get(name,name)
        name = flds.rev_shortnames.get(name,name)
        return name
        
        
    def _isPresent(self, name):
        try:
            pres = self._present[name]
        except KeyError:
            raise Exception("Unkonwn array shape for field %s"%name)
    
        #filter for loaded particle types
        tmp = np.zeros(6,dtype=np.longlong)
        tmp[self._parttype] = pres[self._parttype]
        return tmp

    def _learnPresent(self, name, gr=None, shape=1):           
        if gr !=None:
            pres = np.zeros(6,dtype=np.int64)
            pres[gr] = shape
            if name == 'mass':
                pres[gr] = (0 if self.MassTable[gr]>0  else 1)
            old = self._present.get(name,np.zeros(6,dtype=np.longlong))
            pres = np.maximum(old,pres)
            self._present[name] = pres
        else:
            shape = np.array(shape)
            if shape.shape!=(6,):
               raise Exception("invalide present array")
            pres = shape
            self._present[name] = pres
    
        #filter for loaded particle types
        tmp = np.zeros(6,dtype=np.longlong)
        tmp[self._parttype] = pres[self._parttype]
        return tmp
    
    def __str__(self):
        tmp = str(self.header)
        
        if not self._onlyHeader:
            for i in np.arange(6):
                if self.npart_loaded[i] > 0:
                    tmp += re.sub("[^\n]*\n","\n",str(self.groups[i]),count=1)

        return tmp
    
    def __repr__(self):
        return repr(self.header)

    def addField(self, name, pres=None, dtype=None):
        if not self._writeable:
            raise Exception("This snapshot can not be modified")
        
        name = self._normalizeName(name)

        if pres != None:
            pres = self._learnPresent(name,shape=pres)
        else:
            pres = self._isPresent(name)
            
        num = np.where(pres>0,self.npart_loaded,0)
        
        if dtype==None:
            if name=='id':
                if self._longids:
                    dtype = np.uint64
                else:
                    dtype = np.uint32
            elif self._precision != None:
                dtype = self._precision
            else:
                dtype = np.float32
             
        if np.max(pres)>1:        
            f = np.zeros([num.sum(),np.max(pres)], dtype=dtype)
        else:
            f = np.zeros([num.sum(),], dtype=dtype)
            
        self.data[name] = f
        
    def write(self, filename=None, format=None):
        if not self._writeable:
            raise Exception("This snapshot can not be written")
        
        if filename is None:
            filename = self.filename
            
        if format is None:
            format = self._format

        if format == self._format:           
            self._backend.write(filename=filename)
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
                if self._verbose:
                    print("last chunk reached")
                return False
            
        else:
            if num >= self.NumFilesPerSnapshot or num < 0:
                if self._verbose:
                    print("invalide file number %d"%num)
                return False
            
        if self.filenum == num:
            return True
        
        self.close()


        #open next file
        self._backend.load(num, self.snapshot)
        
        if self._onlyHeader: 
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
            
        if not backends_modules[self._format].handlesfile(self.filename, snapshot=snapshot, filenum=self.filenum, snap=self):
            return False
             
        self.close()

        #open next file
        self._backend.load(self.filenum, snapshot)
        
        if self._onlyHeader: 
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

    def close(self):
        self._backend.close()
        
        if hasattr(self,"data"):
            for i in self.data.keys():
                if isinstance(self.data[i], np.ndarray): 
                    try:
                        self.data[i].resize(0)
                    except ValueError:
                        pass
        
            del self.data
            self.data = {}
            
            
        for i in self._headerfields:
            delattr(self,i)
        self._headerfields = []
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
    
        self._filter = filter
    
        self._backend.load(filenum, snapshot)
        
        self.header = Header(self)
        
        if not self._onlyHeader: 
            self.part0 = PartGroup(self,0)
            self.part1 = PartGroup(self,1)
            self.part2 = PartGroup(self,2)
            self.part3 = PartGroup(self,3)
            self.part4 = PartGroup(self,4)
            self.part5 = PartGroup(self,5)
            self.groups = [self.part0, self.part1, self.part2, self.part3, self.part4, self.part5]
        else:
            self.groups = []
            
        self._sortID = sortID
        if sortID:
            for gr in self.groups:
                if "id" in gr.data:
                    ind = np.argsort(gr.data['id'])
                    for d in gr.data.values():
                        d[...] = d[ind,...]

        self._precision = None
        #set these two, in case we want to add fields to an existing snapshot
        if toDouble:
           self._precision = np.float64
        elif hasattr(self,"Flag_DoublePrecision"):
            if self.Flag_DoublePrecision:
                self._precision = np.float64
            else:
                self._precision = np.float32
        elif hasattr(self,"pos"):
            self._precision = self.pos.dtype
            
        if hasattr(self,"id"):
            if self.data['id'].dtype==np.uint64:
                self._longids = True
            else:
                self._longids = False
        else:
             self._longids = False
             
    def newFilter(self,filter):
        self.close()
        
        self._filter = filter

        #open next file
        self._backend.load(self.filenum, self.snapshot)
        
        if not self._onlyHeader:        
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
        
        #if format!=2:
        #    raise Exception( "Only Format 3 ICs are supported at the moment")
        
        self._path = os.path.abspath(filename)
        
        self._parttype = np.where(num_part>0)[0]
        
        if masses is None:
            masses = np.zeros(6)
            
        if precision is None:
            precision = np.float32
            
        self._precision = precision
        self._longids = longids
            
        self.NumPart_ThisFile = np.longlong(num_part).copy()
        self.NumPart_Total = np.longlong(num_part).copy()
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
            
        self._headerfields = ['Time', 'Redshift', 'BoxSize', 'Omega0', 'OmegaLambda', 'HubbleParam', 'Flag_Sfr',
                                  'Flag_Feedback', 'Flag_Cooling', 'Flag_StellarAge', 'Flag_Metals', 'Flag_DoublePrecision', 
                                  'NumPart_ThisFile', 'NumPart_Total', 'NumPart_Total_HighWord', 'NumFilesPerSnapshot', 'MassTable']
        
        self.header = Header(self)
        
        #initiate mass field
        self._learnPresent('mass',shape=np.where(masses==0,1,0))
    
        self.part0 = PartGroup(self,0)
        self.part1 = PartGroup(self,1)
        self.part2 = PartGroup(self,2)
        self.part3 = PartGroup(self,3)
        self.part4 = PartGroup(self,4)
        self.part5 = PartGroup(self,5)
        self.groups = [ self.part0, self.part1, self.part2, self.part3, self.part4, self.part5]
        
        self._fields = []
        
        for field in flds.default:
            self._fields.append(field)
            
        if fields != None:
            for field in fields:
                f = self._normalizeName(field)
                if not f in self._fields:
                    self._fields.append(f)
                
        for field in self._fields:
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
                    print("ignoring part type %d for subfind output"%i)
            else:
                parttype_filter.append(i)
           
        super(Subfind,self).__init__(filename, snapshot=snapshot, filenum=filenum, format=format, fields=fields, parttype=parttype_filter, combineFiles=combineFiles, toDouble=toDouble, onlyHeader=onlyHeader, verbose=verbose, **param)
        self._backend.load(filenum, snapshot)

        self.header = Header(self)
        
        if not onlyHeader:
            self.group = PartGroup(self,0)
            self.subhalo = PartGroup(self,1)
            self.groups = [self.group, self.subhalo]   
    
        self._writeable = False
    

class Header(object):
    def __init__(self,parent):
        self._parent = parent
                
    def __getattr__(self,name):       
        return getattr(self._parent,name)
            
    def __setattr__(self,name, value):
        #we can't handle these
        if name in ["_parent","_attrs"]:
            super(Header,self).__setattr__(name,value)
            return        
        if name in flds.legacy_header_names:
            name = flds.legacy_header_names[name]     
        if name in self._parent._headerfields or name=='NumFilesPerSnapshot': #(NumFilesPerSnapshot fix for Subfind Header) 
            if type(value)==list:
                value = np.array(value)
            setattr(self._parent,name,value)
        else:
            raise AttributeError
        
    def __dir__(self):
        return list(self.__dict__.keys()) + self._parent._headerfields
    
    def __str__(self):
        filename = self._parent._path
        if isinstance(self._parent, Snapshot):
            tmp = "snapshot "+filename+"\n"
        elif isinstance(self._parent, ICs):
            tmp = "ICs "+filename+"\n"
        else:
            tmp = "subfind output "+filename+"\n"
            
        tmp += "header:\n"
            
        for entry in self._parent._headerfields:
            val = getattr(self,entry)
            if type(val) ==  np.ndarray or type(val) == list:
                tmp += '  '+entry+': '+', '.join([str(x) for x in val])+'\n'
            else:
                tmp += '  '+entry+': '+str(val)+'\n'
        return tmp

    def __repr__(self):
        filename = self._parent._path
        if isinstance(self._parent, Snapshot):
            return "snapshot "+filename
        elif isinstance(self._parent, ICs):
            return "ICs "+filename
        else:
            return "subfind output "+filename
        
class Parameter(object):
    def __init__(self,parent,name):
        self._parent = parent
        self._name = name
        self._attrs = []
            
    def __str__(self):
        filename = self._parent._path
        if isinstance(self._parent, Snapshot):
            tmp = "snapshot "+filename+"\n"
        elif isinstance(self._parent, ICs):
            tmp = "ICs "+filename+"\n"
        else:
            tmp = "subfind output "+filename+"\n"
            
        tmp += self._name + ":\n"
            
        for entry in self._attrs:
            val = getattr(self,entry)
            if type(val) == np.ndarray or type(val) == list:
                tmp += '  '+entry+': '+', '.join([str(x) for x in val])+'\n'
            elif (type(val) != np.bytes_ and type(val) != np.string_) or len(val) > 0:
                if type(val) is np.bytes_:
                    val = val.decode('UTF-8')
                tmp += '  '+entry+': '+str(val)+'\n'
            else:
                tmp += '  '+entry+'\n'
        return tmp

    def __repr__(self):
        filename = self._parent._path
        if isinstance(self._parent, Snapshot):
            return "snapshot "+filename
        elif isinstance(self._parent, ICs):
            return "ICs "+filename
        else:
            return "subfind output "+filename

class PartGroup(object):
    def __init__(self,parent,num):
        self._parent = parent
        self._num = num

    def __str__(self):
        filename = self._parent._path
        if isinstance(self._parent, Snapshot):
            tmp = "snapshot "+filename+"\nparticle group %d (%d particles):\n"%(self._num,self._parent.npart_loaded[self._num])
        elif isinstance(self._parent, ICs):
            tmp = "ICs "+filename+"\nparticle group %d (%d particles):\n"%(self._num,self._parent.npart_loaded[self._num])
        else:
            if self._num == 0:
                tmp = "subfind output "+filename+"\ngroups (%d groups):\n"%(self._parent.npart_loaded[self._num])
            else:
                tmp = "subfind output "+filename+"\nsubhalos (%d subhalos):\n"%(self._parent.npart_loaded[self._num])
            
        for i in self.data.keys():
            tmp += "  " + i
            if i in flds.rev_hdf5toformat2:
                tmp += '/'+flds.rev_hdf5toformat2[i]
            tmp += "\n"
        return tmp
        
    def __repr__(self):
        filename = self._parent._path
        if isinstance(self._parent, Snapshot):
            return "snapshot "+filename+", particle group %d contains %d particles"%(self._num,self._parent.npart_loaded[self._num])
        elif isinstance(self._parent, ICs):
            return "ICs "+filename+", particle group %d contains %d particles"%(self._num,self._parent.npart_loaded[self._num])
        else:
            if self._num == 0:
                return "subfind output "+filename+", contains %d groups"%(self._parent.npart_loaded[self._num])
            else:
                return "subfind output "+filename+", contains %d subhalos"%(self._parent.npart_loaded[self._num])




    def __getitem__(self, item):
        item = self._parent._normalizeName(item)
        parent = self._parent
        num = self._num
        
        f = None
        
        if item in parent.data:
            f = parent.data[item]
            it = item
        else:
            s=item[::-1]
            s=s.split("_",1)
            
            if(s[0]!=item[::-1]): #string contains an underscore
                g=[s[1][::-1],s[0]]
                                
                it = parent._normalizeName(g[0])
                if it in parent.data:
                    if g[1] == 'x':
                        i = 0
                    elif g[1] == 'y':
                        i = 1
                    elif g[1] == 'z':
                        i = 2
                    else:
                        try:
                            i = int(g[1])
                        except:
                            raise AttributeError("unknown field '%s'"%item)
                    d = parent.data[it]
                    if d.ndim == 2 and d.shape[1] > i:
                        f = d[:,i]

        if not f is None:
            pres = parent._isPresent(it)
            if pres[num]>0:       
                n1 = np.where(pres>0, parent.npart_loaded,np.zeros(6,dtype=np.longlong))
                tmp = np.sum(n1[0:num])
                return f[tmp:tmp+parent.npart_loaded[num]]
        
        raise AttributeError("unknown field '%s'"%item)
    
    def __contains__(self, item):
        item = self._parent._normalizeName(item)
        parent = self._parent
        num = self._num
        
        f = None
        
        if item in parent.data:
            f = parent.data[item]
            it = item
        else:
            s=item[::-1]
            s=s.split("_",1)
            
            if(s[0]!=item[::-1]): #string contains an underscore
                g=[s[1][::-1],s[0]]
                                
                it = parent._normalizeName(g[0])
                if it in parent.data:
                    if g[1] == 'x':
                        i = 0
                    elif g[1] == 'y':
                        i = 1
                    elif g[1] == 'z':
                        i = 2
                    else:
                        try:
                            i = int(g[1])
                        except:
                            return False
                    d = parent.data[it]
                    if d.ndim == 2 and d.shape[1] > i:
                        f = d[:,i]

        if not f is None:
            pres = parent._isPresent(it)
            if pres[num]>0:       
                return True
        
        return False
    
    def __setattr__(self,attr,val):
        if attr =='_parent':
            object.__setattr__(self,attr,val)
        elif attr =='data':
            raise AttributeError("you are not allowed to set attribute 'data'")
        
        attr = self._parent._normalizeName(attr)
        
        if attr in self._parent.data:
            raise AttributeError("you can not exchange the array '%s' with your own object, write into the array using %s[...] instead"%(attr,attr))
        
        object.__setattr__(self,attr,val)
    
    def __getattr__(self,attr):       
        parent = self._parent
        num = self._num
        attr = self._parent._normalizeName(attr)
        
        if attr == "data":
            data = {}
            if parent.npart_loaded[num]>0:
                if hasattr(parent,"data"):
                    for key in parent.data.keys():
                        pres = parent._isPresent(key)
                        if pres[num]>0:
                            f = parent.data[key]
                            n1 = np.where(pres>0, parent.npart_loaded,np.zeros(6,dtype=np.longlong))
                            tmp = np.sum(n1[0:num])
                            data[key] = f[tmp:tmp+parent.npart_loaded[num]]
            return data  
        else:
            return self.__getitem__(attr)
        
    def __dir__(self):
        parent = self._parent
        num = self._num
        
        dir = list(self.__dict__.keys())
        dir.append('data')
        if parent.npart_loaded[num]>0:
            if hasattr(parent,"data"):
                for key in parent.data.keys():
                    pres = parent._isPresent(key)
                    if pres[num]>0:
                        dir.append(key)
                        if key in flds.rev_hdf5toformat2:
                            dir.append(flds.rev_hdf5toformat2[key])
        return dir   
        
