import numpy as np
import re

import gadget.fields as fields

class Loader:
    def nextChunk(self):
        if self.__format__ == 2:
            print "not supported"

        self.__backend__.next_chunk()

    def close(self):
        self.__backend__.close()
        
    def __init__(self,filename,base=None,num=None, format=None, fields=None, parttype=None, **param):     

        if base!=None:
            filename = base+'/'+filename

        if num!=None:
            filename = filename%num

        #detect backend
        if format==None:
            if filename.endswith('.hdf5') or filename.endswith('.h5'):
                format = 3
            else:
                format = 2


        self.filename = filename
        self.snapshotnum = num

        self.__format__ = format
        self.__fields__ = fields

        if parttype == None:
            parttype = [0,1,2,3,4,5]
        self.__parttype__ = parttype

        self.__normalizeFields__()

        if format==3:
            import format3
            self.__backend__=format3.Format3(self,filename, **param)

        if format==2:
            import format2
            self.__backend__=format2.Format2(self,filename, **param)

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

class Snapshot(Loader):
    """
    This class loads Gadget snapshots. Currently file format 2 and 3 (hdf5) are supported.
    """
    def __init__(self,filename,base=None,num=None, format=None, fields=None, parttype=None, **param):     
        """
        *filename* : The name of the snapshot file
        *base* : (optional) if given, this is added in front of the filename
        *num* : (optional) number of the snapshot to load (requires '%d' pattern in filename)
        *format* : (optional) file format of the snapshot, otherwise this is guessed from the file name
        *fields* : (optional) list of fields to load, if None, all fields available in the snapshot are loaded
        *parttype* : (optional) array with particle type numbers to load, if None, all particles are loaded
        *toDouble* : (optinal) converts all values of type float to double precision
        
        format 3 (hdf5) only:
        *combineParticles* : (optinal) if True arrays containng all particle species are provided as well (disables mmap)
        *combineFiles* : (optinal) if False only on part of the snapshot is loaded at a time, use nextChunk() to get the next part.
        """
        Loader.__init__(self,filename,base,num, format, fields, parttype, **param)


    def __convenience__(self):
        for gr in (self, self.part0,self.part1,self.part2,self.part3,self.part4, self.part5):
            items = gr.data.keys()
            for i in items:
                setattr(gr,i,gr.data[i])
                if fields.shortnames.has_key(i):
                    setattr(gr,fields.shortnames[i],gr.data[i])



    def __str__(self):
        tmp = self.header.__str__()
        for i in np.arange(0,6):
            if self.nparticlesall[i] > 0:
                tmp += re.sub("[^\n]*\n","\n",self.groups[i].__str__(),count=1)

        return tmp



class Subfind(Loader):
    def __init__(self,filename,base=None,num=None, format=None, fields=None, parttype=None, **param):
        if parttype == None:
            parttype = [0,1]

        for i in parttype:
            if i!=0 and i!=1:
                print "ignoring part type %d for subfind output"%i
        
        if parttype == None:
            parttype = [0,1]       

        param['combineParticles'] = False  
           
        Loader.__init__(self,filename,base,num, format, fields, parttype, **param)


    def __convenience__(self):
        for gr in (self.group, self.subhalo):
            items = gr.data.keys()
            for i in items:
                setattr(gr,i,gr.data[i])
                if fields.shortnames.has_key(i):
                    setattr(gr,fields.shortnames[i],gr.data[i])

    def __str__(self):
        tmp = self.header.__str__()
        for i in (self.group,self.subhalo):
            tmp += re.sub("[^\n]*\n","\n",i.__str__(),count=1)

        return tmp

    def __getitem__(self, item):
        pass


class Header:
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

class PartGroup:
    def __init__(self,parent,num):
        self.__parent__ = parent
        self.__num__ = num
        self.data = {}
        
        if parent.npart_loaded[num]>0:
            if hasattr(parent,"data"):
                for key in parent.data.iterkeys():
                    pres = fields.isPresent(key,num, parent)
                    if pres[num]>0:
                        f = parent.data[key]
                        n1 = np.where(pres>0, parent.npart_loaded,np.zeros(6,dtype=np.longlong))
                        tmp = np.sum(n1[0:num])
                        self.data[key] = f[tmp:tmp+parent.npart_loaded[num]]

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
        
        
