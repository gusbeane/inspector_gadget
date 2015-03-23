import h5py
import numpy as np
import os.path as path
import re
import os

import gadget.loader as loader
import gadget.fields as fields

class Format3:


    dict = fields.hdf5toformat2
    rev_dict = fields.rev_hdf5toformat2


    def __init__(self, sn, **param):
        self.sn=sn
        
        
        
    def load(self, num=None):
        self.sn.npart_loaded = np.zeros(6,dtype=np.longlong)
        
        if not path.exists( self.sn.filename ):
            if path.exists( self.sn.filename + ".hdf5" ):
                self.sn.filename += ".hdf5"
            elif  path.exists( self.sn.filename + "0.hdf5" ):
                self.sn.filename += "0.hdf5"
            elif path.exists( self.sn.filename + ".h5" ):
                self.sn.filename += ".h5"
            elif  path.exists( self.sn.filename + "0.h5" ):
                self.sn.filename += "0.h5"
                
        if num==None:
            res = re.findall("\.[0-9]*\.hdf5",self.sn.filename)
            res2 = re.findall("\.[0-9]*\.h5",self.sn.filename)
            if len(res) > 0:
                num = int(res[-1][1:-5])
            elif len(res2) > 0:
                num = int(res[-1][1:-3])
            else:
                num = 0    
                
        filename = self.sn.filename
        if num!=None:
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%num, filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%num, filename) 
        
        try:
            self.file = h5py.File(filename,"r")
        except Exception:
            raise Exception("could not open file %s"%filename)
        
        self.load_header()
        self.load_parameter("Parameters", "parameters")
        self.load_parameter("Config", "config")
        self.file.close()
        del self.file
        
        if not self.sn.__onlyHeader__:
            if isinstance(self.sn, loader.Snapshot):
                self.load_data(filename,num)
            else:
                self.load_data_subfind(filename,num)

        if self.sn.__combineFiles__==False and self.sn.NumFilesPerSnapshot>1:
            self.sn.currFile = num
            self.sn.filename = filename
        else:
            self.sn.currFile = None
            
        self.sn.__path__ = os.path.abspath(self.sn.filename)

    def load_header(self):
        file = self.file
        
        self.sn.__headerfields__ = []
        
        for i in file['/Header'].attrs:
            name = i.replace(" " ,"")
            
            if name == "NumFiles": #fix for subfind cataloges
                name = "NumFilesPerSnapshot"
                
            setattr(self.sn, name, file['/Header'].attrs[i])
            self.sn.__headerfields__.append(name)
        
        if isinstance(self.sn, loader.Snapshot):
            self.sn.nparticlesall = np.longlong(self.sn.NumPart_Total)
            self.sn.nparticlesall += np.longlong(self.sn.NumPart_Total_HighWord)<<32

            self.sn.npart = np.array( self.sn.NumPart_Total ).sum()
            self.sn.npartall = np.array( self.sn.nparticlesall ).sum()
      
        else:
            self.sn.NumPart_ThisFile = np.array([self.sn.Ngroups_ThisFile, self.sn.Nsubgroups_ThisFile,0,0,0,0])
            self.sn.nparticlesall = np.array([self.sn.Ngroups_Total, self.sn.Nsubgroups_Total,0,0,0,0])
            

    def load_parameter(self,group, name):
        file = self.file
        
        if group in file:
            param = loader.Parameter(self.sn,name)
            for i in file[group].attrs:        
                    setattr(param, i, file[group].attrs[i])
                    param.__attrs__.append(i)
    
            setattr(self.sn, name, param)
            
            
    def load_data(self, filename, num):
        self.sn.npart_loaded = np.zeros(6,dtype=np.longlong)
        self.sn.data = {}
        
        loaded = np.zeros(6,dtype=np.longlong)
        
        if hasattr(self.sn,"__filter__") and self.sn.__filter__ != None:
            filter = self.sn.__filter__
            if type(filter) == list:
                filter = np.array(filter)
            elif type(filter) != np.ndarray:
                filter = np.array([filter])
            indices = [[],[],[],[],[],[]]

            for f in filter:
                f.reset()

        else:
            filter = None

        if self.sn.__combineFiles__:
            filesA = 0
            filesB = self.sn.NumFilesPerSnapshot
        else:
            filesA = num
            filesB = num+1
            
        #learn about present fields
        for i in np.arange(filesA,filesB):
            if self.sn.__verbose__:
                print("Learning about file %d"%i)

            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            try:
                self.file = h5py.File(filename,'r')
            except Exception:
                raise Exception("could not open file %s"%filename)
           
            self.load_header()

            for gr in self.sn.__parttype__:
                if "PartType%d"%gr in self.file.keys():
                    for item in self.file["PartType%d"%gr].keys():
                        if not self.dict.has_key(item):
                            if self.sn.__verbose__:
                                print "warning: hdf5 key '%s' could not translated"%item
                                self.dict[item] = item

                        name  = self.dict.get(item,item)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            d = self.file["PartType%d/%s"%(gr,item)]
                            shape = np.array(d.shape)
                            elem = 1
                            if shape.size == 2:
                                elem=shape[1]
                                
                            pres = self.sn.__learnPresent__(name,gr=gr,shape=elem)
                if filter != None:
                    if "PartType%d"%gr in self.file.keys():
                        ind = np.arange(self.sn.NumPart_ThisFile[gr])
                        for f in filter:
                            if gr in f.parttype:
                                data = {}
                                data['NumPart_ThisFile'] = np.longlong(self.file['/Header'].attrs['NumPart_ThisFile'])[gr]
                                data['group'] = gr
                                
                                for fld in f.requieredFields:
                                    fld2 = self.rev_dict.get(fld,fld)
                                    data[fld] = self.file["PartType%d"%gr][fld2][...][ind]
                            
                                ind = ind[f.getIndices(data)]
                            
                        indices[gr].append(np.copy(ind))
                        self.sn.npart_loaded[gr] += len(ind)
                        
                    else:
                        indices[gr].append(np.array([]))
                else:
                    self.sn.npart_loaded[gr] += self.sn.NumPart_ThisFile[gr]
            self.file.close()
                                
        #now load the requested data                        
        for i in np.arange(filesA,filesB):
            if self.sn.__verbose__:
                print("Reading file %d"%i)

            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            try:
                self.file = h5py.File(filename,'r')
            except Exception:
                raise Exception("could not open file %s"%filename)
                
            self.load_header()

            for gr in self.sn.__parttype__:
                if "PartType%d"%gr in self.file.keys():
                    elements = 0
                    for item in self.file["PartType%d"%gr].keys():
                        name  = self.dict.get(item,item)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            pres = self.sn.__isPresent__(name)
                                
                            n1 = np.where(pres > 0, loaded, np.zeros(6,dtype=np.longlong))
                            n2 = np.where(pres > 0, self.sn.npart_loaded, np.zeros(6,dtype=np.longlong))
                            
                            d = self.file["PartType%d/%s"%(gr,item)]
                            shape = np.array(d.shape)
                                
                            if not self.sn.data.has_key(name):
                                num = np.where(pres > 0, self.sn.npart_loaded, np.zeros(6,dtype=np.longlong)).sum()
                
                                #get propertiers of dataset
                                datatype = d.dtype
                                s = np.array(d.shape)
                                s[0] = num
                                if self.sn.__toDouble__ and datatype == np.dtype('float32'):
                                    datatype = np.dtype('float64')
                                        
                                self.sn.data[name] = np.empty(s, dtype=datatype)
                                    
                            if filter is None:
                                self.sn.data[name][n2[0:gr].sum()+n1[gr]:n2[0:gr].sum()+n1[gr]+shape[0]] = d
                                elements = d.shape[0]
                            elif len(indices[gr][i]) > 0:
                                self.sn.data[name][n2[0:gr].sum()+n1[gr]:n2[0:gr].sum()+n1[gr]+len(indices[gr][i])] = d[...][indices[gr][i],...]
                                elements = len(indices[gr][i])
                    loaded[gr] += elements
            
            #self.sn.npart_loaded[self.sn.__parttype__] += self.sn.NumPart_ThisFile[self.sn.__parttype__]
       
            self.file.close()
            del self.file
        
    def load_data_subfind(self, filename, num):
        self.sn.data = {}
        self.sn.npart_loaded = np.zeros(6,dtype=np.longlong)
        
        groupnames = ['Group','Subhalo']

        if self.sn.__combineFiles__:
            filesA = 0
            filesB = self.sn.NumFilesPerSnapshot
        else:
            filesA = num
            filesB = num+1
            
        #learn about present fields
        for i in np.arange(filesA,filesB):
            if self.sn.__verbose__:
                print("Learning about file %d"%i)
                
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            try:
                self.file = h5py.File(filename,'r')
            except Exception:
                raise Exception("could not open file %s"%filename)
           
            self.load_header()

            for gr in self.sn.__parttype__:
                if groupnames[gr] in self.file.keys():
                    for item in self.file[groupnames[gr]].keys():
                        if not self.dict.has_key(item):
                            if self.sn.__verbose__:
                                print "warning: hdf5 key '%s' could not translated"%item
                                self.dict[item] = item
                                
                        name  = self.dict.get(item,item)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            d = self.file["%s/%s"%(groupnames[gr],item)]
                            shape = np.array(d.shape)
                            elem = 1
                            if shape.size == 2:
                                elem=shape[1]
                                
                            pres = self.sn.__learnPresent__(name,gr=gr,shape=elem)
            self.file.close()
                                
        #now load the requested data                    
        for i in np.arange(filesA,filesB):
            if self.sn.__verbose__:
                print("Reading file %d"%i)
                
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            try:
                self.file = h5py.File(filename,'r')
            except Exception:
                raise Exception("could not open file %s"%filename)
                
            self.load_header()  

            for gr in self.sn.__parttype__:
                if groupnames[gr] in self.file.keys():
                    for item in self.file[groupnames[gr]].keys():
                        name  = self.dict.get(item,item)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            pres = self.sn.__isPresent__(name)
                                
                            n1 = np.where(pres > 0, self.sn.npart_loaded, np.zeros(6,dtype=np.longlong))
                                
                            if self.sn.__combineFiles__:
                                n2 = np.where(pres > 0, self.sn.nparticlesall, np.zeros(6,dtype=np.longlong))
                            else:
                                n2 = np.where(pres > 0, self.sn.NumPart_ThisFile, np.zeros(6,dtype=np.longlong))
                            
                            d = self.file["%s/%s"%(groupnames[gr],item)]
                            shape = np.array(d.shape)
                                
                            if not self.sn.data.has_key(name):
                                if self.sn.__combineFiles__:
                                    num = np.where(pres > 0, self.sn.nparticlesall, np.zeros(6,dtype=np.longlong)).sum()
                                else:
                                    num = np.where(pres > 0, self.sn.NumPart_ThisFile, np.zeros(6,dtype=np.longlong)).sum()
                
                                #get propertiers of dataset
                                datatype = d.dtype
                                s = np.array(d.shape)
                                s[0] = num
                                if self.sn.__toDouble__ and datatype == np.dtype('float32'):
                                    datatype = np.dtype('float64')
                                        
                                self.sn.data[name] = np.empty(s, dtype=datatype)
                                    
                            self.sn.data[name][n2[0:gr].sum()+n1[gr]:n2[0:gr].sum()+n1[gr]+shape[0]] = d
            
            self.sn.npart_loaded[self.sn.__parttype__] += self.sn.NumPart_ThisFile[self.sn.__parttype__]
       
            self.file.close()
            del self.file



    def close(self):
        if hasattr(self,"file"):
            self.file.close()
            del self.file

    def write(self,filename):
        if not filename.endswith(".hdf5") and not filename.endswith(".h5"):
            filename += ".hdf5"
            
        #TODO check filename for multipart files    
        try:
            file = h5py.File(filename,"w")
        except Exception:
            raise Exception("could not open file %s for writing"%filename)
        
        
        self.write_header(file)
        
        self.write_particles(file)
        
        file.close()

        
    def write_header(self,file):
        header = file.create_group("/Header")
           
        for i in self.sn.__headerfields__:    
            if hasattr(self.sn, i):
                header.attrs[i] = getattr(self.sn, i) 
        
        
    def write_particles(self,file):
        for i in np.arange(0,6):
            group = file.create_group("PartType%d"%i)
            
            for item in self.sn.data:
                pres = self.sn.__isPresent__(item)
                if pres[i] > 0 and self.sn.npart_loaded[i]>0:
                    #reverse translate name
                    if item in self.dict.values():
                        name = (key for key,value in self.dict.items() if value==item).next()
                    else:
                        name = item
                        
                    f = self.sn.data[item]
                    
                    if pres[i]>1:
                        dataset = group.create_dataset(name, shape=(self.sn.npart_loaded[i],pres[i]), dtype=f.dtype)
                    else:
                        dataset = group.create_dataset(name, shape=(self.sn.npart_loaded[i],), dtype=f.dtype)
                        
                    n1 = np.where(pres>0, self.sn.npart_loaded,np.zeros(6,dtype=np.longlong))
                    tmp = np.sum(n1[0:i])
                    dataset[...] = f[tmp:tmp+self.sn.npart_loaded[i]]
