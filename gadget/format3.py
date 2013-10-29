import h5py
import numpy as np
import os.path as path
import re

import gadget.loader as loader
import gadget.fields as fields


class Format3:


    dict = fields.hdf5toformat2


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
        self.file.close()
        del self.file
        
        if not self.sn.__onlyHeader__:
            if isinstance(self.sn, loader.Snapshot):
                self.load_data(filename,num)
            else:
                self.load_data_subfind(filename,num)

        if self.sn.__combineFiles__==False and self.sn.num_files>1:
            self.sn.currFile = num
            self.sn.filename = filename
        else:
            self.sn.currFile = None
            


    def load_header(self):
        file = self.file
        
        if isinstance(self.sn, loader.Snapshot):
            self.sn.nparticles =  np.longlong(file['/Header'].attrs['NumPart_ThisFile'])
            self.sn.nparticlesall = np.longlong(file['/Header'].attrs['NumPart_Total'])
            self.sn.nparticlesall += np.longlong(file['/Header'].attrs['NumPart_Total_HighWord'])<<32
            self.sn.masses = file['/Header'].attrs['MassTable']
            self.sn.num_files = file['/Header'].attrs['NumFilesPerSnapshot']
            
            self.sn.flag_sfr = file['/Header'].attrs['Flag_Sfr']
            self.sn.flag_cooling = file['/Header'].attrs['Flag_Cooling']
            self.sn.flag_feedback = file['/Header'].attrs['Flag_Feedback']
            self.sn.flag_stellarage = file['/Header'].attrs['Flag_StellarAge']
            self.sn.flag_metals = file['/Header'].attrs['Flag_Metals']

            self.sn.npart = np.array( self.sn.nparticles ).sum()
            self.sn.npartall = np.array( self.sn.nparticlesall ).sum()

        else:
            self.sn.ngroups =  file['/Header'].attrs['Ngroups_ThisFile']
            self.sn.ngroupsall =  file['/Header'].attrs['Ngroups_Total']
            self.sn.nids =  file['/Header'].attrs['Nids_ThisFile']
            self.sn.nidsall =  file['/Header'].attrs['Nids_Total']
            self.sn.nsubgroups =  file['/Header'].attrs['Nsubgroups_ThisFile']
            self.sn.nsubgroupsall =  file['/Header'].attrs['Nsubgroups_Total']
            self.sn.num_files = file['/Header'].attrs['NumFiles']
            
            self.sn.nparticles = np.array([self.sn.ngroups, self.sn.nsubgroups,0,0,0,0])
            self.sn.nparticlesall = np.array([self.sn.ngroupsall, self.sn.nsubgroupsall,0,0,0,0])
        


        self.sn.time = file['/Header'].attrs['Time']
        self.sn.redshift = file['/Header'].attrs['Redshift']
        self.sn.boxsize = file['/Header'].attrs['BoxSize']
        self.sn.omega0 = file['/Header'].attrs['Omega0']
        self.sn.omegalambda = file['/Header'].attrs['OmegaLambda']
        self.sn.hubbleparam = file['/Header'].attrs['HubbleParam']
        
        
        if "Flag_DoublePrecision" in file['/Header'].attrs.keys():
            self.sn.flag_doubleprecision = file['/Header'].attrs['Flag_DoublePrecision']


    def load_data(self, filename, num):
        self.sn.npart_loaded = np.zeros(6,dtype=np.longlong)
        self.sn.data = {}

        if self.sn.__combineFiles__:
            filesA = 0
            filesB = self.sn.num_files
        else:
            filesA = num
            filesB = num+1
            
        #learn about present fields
        for i in np.arange(filesA,filesB):
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            try:
                self.file = h5py.File(filename,'r')
            except Exception:
                raise Exception("could not open file %s"%filename)
           
            self.load_header()
            self.sn.header = loader.Header(self.sn)
                

            for gr in self.sn.__parttype__:
                if "PartType%d"%gr in self.file.keys():
                    for item in self.file["PartType%d"%gr].keys():
                        if not self.dict.has_key(item):
                            if self.sn.__verbose__:
                                print "warning: hdf5 key '%s' could not translated"%key
                                
                        name  = self.dict.get(item,item)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            d = self.file["PartType%d/%s"%(gr,item)]
                            shape = np.array(d.shape)
                            elem = 1
                            if shape.size == 2:
                                elem=shape[1]
                                
                            pres = self.sn.__learnPresent__(name,gr=gr,shape=elem)
            self.file.close()
                                
        #now load the requested data                        
        for i in np.arange(filesA,filesB):
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            try:
                self.file = h5py.File(filename,'r')
            except Exception:
                raise Exception("could not open file %s"%filename)
                
            self.load_header()
            self.sn.header = loader.Header(self.sn)
                

            for gr in self.sn.__parttype__:
                if "PartType%d"%gr in self.file.keys():
                    for item in self.file["PartType%d"%gr].keys():
                        name  = self.dict.get(item,item)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            pres = self.sn.__isPresent__(name)
                                
                            n1 = np.where(pres > 0, self.sn.npart_loaded, np.zeros(6,dtype=np.longlong))
                                
                            if self.sn.__combineFiles__:
                                n2 = np.where(pres > 0, self.sn.nparticlesall, np.zeros(6,dtype=np.longlong))
                            else:
                                n2 = np.where(pres > 0, self.sn.nparticles, np.zeros(6,dtype=np.longlong))
                            
                            d = self.file["PartType%d/%s"%(gr,item)]
                            shape = np.array(d.shape)
                                
                            if not self.sn.data.has_key(name):
                                if self.sn.__combineFiles__:
                                    num = np.where(pres > 0, self.sn.nparticlesall, np.zeros(6,dtype=np.longlong)).sum()
                                else:
                                    num = np.where(pres > 0, self.sn.nparticles, np.zeros(6,dtype=np.longlong)).sum()
                
                                #get propertiers of dataset
                                datatype = d.dtype
                                s = np.array(d.shape)
                                s[0] = num
                                if self.sn.__toDouble__ and datatype == np.dtype('float32'):
                                    datatype = np.dtype('float64')
                                        
                                self.sn.data[name] = np.empty(s, dtype=datatype)
                                    
                            self.sn.data[name][n2[0:gr].sum()+n1[gr]:n2[0:gr].sum()+n1[gr]+shape[0]] = d
            
            self.sn.npart_loaded[self.sn.__parttype__] += self.sn.nparticles[self.sn.__parttype__]
       
            self.file.close()
            del self.file
        
    def load_data_subfind(self, filename, num):
        self.sn.data = {}
        self.sn.npart_loaded = np.zeros(6,dtype=np.longlong)
        
        groupnames = ['Group','Subhalo']

        if self.sn.__combineFiles__:
            filesA = 0
            filesB = self.sn.num_files
        else:
            filesA = num
            filesB = num+1
            
        #learn about present fields
        for i in np.arange(filesA,filesB):
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            try:
                self.file = h5py.File(filename,'r')
            except Exception:
                raise Exception("could not open file %s"%filename)
           
            self.load_header()
            self.sn.header = loader.Header(self.sn)
                

            for gr in self.sn.__parttype__:
                if groupnames[gr] in self.file.keys():
                    for item in self.file[groupnames[gr]].keys():
                        if not self.dict.has_key(item):
                            if self.sn.__verbose__:
                                print "warning: hdf5 key '%s' could not translated"%key
                                
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
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            try:
                self.file = h5py.File(filename,'r')
            except Exception:
                raise Exception("could not open file %s"%filename)
                
            self.load_header()
            self.sn.header = loader.Header(self.sn)
                

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
                                n2 = np.where(pres > 0, self.sn.nparticles, np.zeros(6,dtype=np.longlong))
                            
                            d = self.file["%s/%s"%(groupnames[gr],item)]
                            shape = np.array(d.shape)
                                
                            if not self.sn.data.has_key(name):
                                if self.sn.__combineFiles__:
                                    num = np.where(pres > 0, self.sn.nparticlesall, np.zeros(6,dtype=np.longlong)).sum()
                                else:
                                    num = np.where(pres > 0, self.sn.nparticles, np.zeros(6,dtype=np.longlong)).sum()
                
                                #get propertiers of dataset
                                datatype = d.dtype
                                s = np.array(d.shape)
                                s[0] = num
                                if self.sn.__toDouble__ and datatype == np.dtype('float32'):
                                    datatype = np.dtype('float64')
                                        
                                self.sn.data[name] = np.empty(s, dtype=datatype)
                                    
                            self.sn.data[name][n2[0:gr].sum()+n1[gr]:n2[0:gr].sum()+n1[gr]+shape[0]] = d
            
            self.sn.npart_loaded[self.sn.__parttype__] += self.sn.nparticles[self.sn.__parttype__]
       
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
        header.attrs['NumPart_ThisFile'] = self.sn.nparticles
        header.attrs['NumPart_Total'] = self.sn.nparticlesall
        header.attrs['NumPart_Total_HighWord'] = [0,0,0,0,0,0]
        header.attrs['MassTable'] = self.sn.masses
        
        header.attrs['Time'] = self.sn.time
        header.attrs['NumFilesPerSnapshot'] = self.sn.num_files
        header.attrs['Redshift'] = self.sn.redshift
        header.attrs['BoxSize'] = self.sn.boxsize

        header.attrs['Omega0'] =  self.sn.omega0
        header.attrs['OmegaLambda'] = self.sn.omegalambda
        header.attrs['HubbleParam'] =  self.sn.hubbleparam
        header.attrs['Flag_Sfr'] =  self.sn.flag_sfr
        header.attrs['Flag_Cooling'] = self.sn.flag_cooling
        header.attrs['Flag_StellarAge'] = self.sn.flag_stellarage
        header.attrs['Flag_Metals'] =  self.sn.flag_metals
        header.attrs['Flag_Feedback'] =  self.sn.flag_feedback
        if hasattr(self.sn, "flag_doubleprecision"):
            header.attrs['Flag_DoublePrecision'] = self.sn.flag_doubleprecision
        
        
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
