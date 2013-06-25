import numpy as np
import re

import gadget.fields as fields

class Loader(object):
       
    def __init__(self,filename, format=None, fields=None, parttype=None, combineFiles=True, toDouble=False, onlyHeader=False, verbose=False, load=True, **param):     
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
        
        if parttype == None:
            parttype = [0,1,2,3,4,5]
        self.__parttype__ = parttype

        param['format'] = format
        param['parttype'] = parttype
        param['fields'] = fields
        param['toDouble'] = toDouble
        param['onlyHeader'] = onlyHeader
        param['verbose'] = verbose
        param['combineFiles'] = combineFiles


        if format==3:
            import format3
            self.__backend__=format3.Format3(self,filename, **param)

        if format==2:
            if load==False:
                raise Exception( "Creating format2 snapshots is not supported yet")
                
            import format2
            self.__backend__=format2.Format2(self,filename, **param)
                

        if load:
            self.__backend__.load()
            self.__convenience__()
        
    def __normalizeFields__(self):
        if self.__fields__ == None:
            return

        tmp = self.__fields__
        self.__fields__ = []

        rev = dict((v,k) for k, v in fields.shortnames.iteritems())

        for item in tmp:
            if self.__format__==3:
                item = fields.hdf5toformat2.get(item,item)

            self.__fields__.append(rev.get(item,item))
    
    def __convenience__():
        pass
    
    
    def __repr__(self):
        return self.__str__()
    
                
    def __getitem__(self, item):
        return self.data[item]
    
    def nextFile(self, num=None):
        if self.__format__ == 2:
            print "not supported yet"

        self.__backend__.nextFile(num)

    def close(self):
        self.__backend__.close()

class Snapshot(Loader):
    """
    This class loads Gadget snapshots. Currently file format 2 and 3 (hdf5) are supported.
    """
    def __init__(self,filename, format=None, fields=None, parttype=None, num_part=None, masses=None, combineFiles=True, toDouble=False, onlyHeader=False, verbose=False, **param):     
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
                        
        format 3 (hdf5) only:
        *combineParticles* : (optinal) if True arrays containing data of all species are provided on the snapshot object as well (disables mmap)
        """
        
        
        if num_part != None:
            super(Snapshot,self).__init__(filename, format=format, fields=fields, parttype=parttype, combineFiles=combineFiles, toDouble=toDouble, onlyHeader=onlyHeader, verbose=verbose, load=False, **param)
            self.__initEmpty__(np.array(num_part), masses, toDouble)
        else:
            super(Snapshot,self).__init__(filename, format=format, fields=fields, parttype=parttype, combineFiles=combineFiles, toDouble=toDouble, onlyHeader=onlyHeader, verbose=verbose, load=True, **param)
        
    def __initEmpty__(self, num_part, masses=None, toDouble=False):
        self.__parttype__ = np.where(num_part>0)[0]
        
        if masses == None:
            masses = np.zeros(6)
            
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
        if toDouble:
            self.flag_doubleprecision=1
        else:
            self.flag_doubleprecision=0
        
        self.header = Header(self)
        
        #initiate default present fields
        self.__present__ = fields.present
        self.__present__['mass'] = np.where(masses==0,1,0)
        
        self.data = {}
        
        for field in fields.default:
            self.addField(field)
        if self.__fields__ != None:
            for field in self.__fields__:
                self.addField(field)
        
        self.part0 = PartGroup(self,0)
        self.part1 = PartGroup(self,1)
        self.part2 = PartGroup(self,2)
        self.part3 = PartGroup(self,3)
        self.part4 = PartGroup(self,4)
        self.part5 = PartGroup(self,5)
        self.groups = [ self.part0, self.part1, self.part2, self.part3, self.part4, self.part5]
        
                
        self.__convenience__()

    def __convenience__(self):
        items = self.data.keys()
        for i in items:
            setattr(self,i,self.data[i])
            if fields.shortnames.has_key(i):
                setattr(self,fields.shortnames[i],self.data[i])
                
        for gr in (self.part0,self.part1,self.part2,self.part3,self.part4, self.part5):
            gr.__convenience__()


    def __str__(self):
        tmp = self.header.__str__()
        for i in np.arange(0,6):
            if self.nparticlesall[i] > 0:
                tmp += re.sub("[^\n]*\n","\n",self.groups[i].__str__(),count=1)

        return tmp

    def addField(self, name, shape=None, dtype=None):
        name = fields.hdf5toformat2.get(name,name)

        if shape != None:
            pres = fields.isPresent(name,self,learn=True,shape=shape)
        else:
            pres = fields.isPresent(name,self)
            
        tmp = np.where(pres>0,self.nparticles,0)
        num = np.zeros(6,dtype=np.int64)
        num[self.__parttype__] = tmp[self.__parttype__]
        
        if dtype==None:
            if name=='id':
                dtype = np.uint32
            elif self.flag_doubleprecision:
                dtype = np.float64
            else:
                dtype = np.float32
                
        self.data[name] = np.zeros([num.sum(),np.max(pres)], dtype=dtype)
        
        
    def write(self, filename=None, format=None):
        if format==None or format == self.__format__:
            self.__backend__.write(filename=filename)
        else:
            pass
        


class Subfind(Loader):
    def __init__(self,filename, format=None, fields=None, parttype=None, combineFiles=True, toDouble=False, onlyHeader=False, verbose=False, **param):
        if parttype == None:
            parttype = [0,1]

        for i in parttype:
            if i!=0 and i!=1:
                print "ignoring part type %d for subfind output"%i
        
        if parttype == None:
            parttype = [0,1]       

        param['combineParticles'] = False  
           
        super(Subfind,self).__init__(filename, format=format, fields=fields, parttype=parttype, combineFiles=combineFilesm, toDouble=toDouble, onlyHeader=onlyHeader, verbose=verbose, load=True, **param)


    def __convenience__(self):
        items = self.data.keys()
        for i in items:
            setattr(self,i,self.data[i])
            if fields.shortnames.has_key(i):
                setattr(self,fields.shortnames[i],self.data[i])
                
        for gr in (self.group, self.subhalo):
            gr.__convenience__()
            

    def __str__(self):
        tmp = self.header.__str__()
        for i in (self.group,self.subhalo):
            tmp += re.sub("[^\n]*\n","\n",i.__str__(),count=1)

        return tmp

    def __getitem__(self, item):
        pass


class Header(object):
    def __init__(self,parent):
        self.__parent__ = parent
        for entry in fields.headerfields:
            if hasattr(parent,entry):
                setattr(self,entry,getattr(parent,entry))

    def __str__(self):
        if isinstance(self.__parent__, Snapshot):
            tmp = "snapshot "+self.__parent__.filename+":\n"
        else:
            tmp = "subfind output "+self.__parent__.filename+":\n"
            
        for entry in fields.headerfields:
            if hasattr(self,entry):
                val = getattr(self,entry)
                if type(val) ==  np.ndarray or type(val) == list:
                    tmp += '  '+entry+': '+', '.join([str(x) for x in val])+'\n'
                else:
                    tmp += '  '+entry+': '+str(val)+'\n'


        return tmp

    def __repr__(self):
        return self.__str__()

class PartGroup(object):
    def __init__(self,parent,num):
        self.__parent__ = parent
        self.__num__ = num
        self.data = {}
        
        if parent.npart_loaded[num]>0:
            if hasattr(parent,"data"):
                for key in parent.data.iterkeys():
                    pres = fields.isPresent(key, parent)
                    if pres[num]>0:
                        f = parent.data[key]
                        n1 = np.where(pres>0, parent.npart_loaded,np.zeros(6,dtype=np.longlong))
                        tmp = np.sum(n1[0:num])
                        self.data[key] = f[tmp:tmp+parent.npart_loaded[num]]
                        
    def __convenience__(self):
        items = self.data.keys()
        for i in items:
            setattr(self,i,self.data[i])
            if fields.shortnames.has_key(i):
                setattr(self,fields.shortnames[i],self.data[i])


    def __str__(self):
        if isinstance(self.__parent__, Snapshot):
            tmp = "snapshot "+self.__parent__.filename+"\nparticle group %d contains %d particles:\n"%(self.__num__,self.__parent__.npart_loaded[self.__num__])
        else:
            if self.__num__ == 0:
                tmp = "subfind output "+self.__parent__.filename+"\ncontains %d groups:\n"%(self.__parent__.npart_loaded[self.__num__])
            else:
                tmp = "subfind output "+self.__parent__.filename+"\ncontains %d subhalos:\n"%(self.__parent__.npart_loaded[self.__num__])
            
        for i in self.data.keys():
            if fields.shortnames.has_key(i):
                tmp += "  "+i +'/'+fields.shortnames.get(i,i)+"\n"
            elif not i in fields.shortnames.values():
                tmp += "  "+i+"\n"
        return tmp
        
    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        return self.data[item]
        
        
