import h5py
import numpy as np
import os.path as path
import re

import gadget.loader as loader
import gadget.fields as fields


class Format3:


    dict = fields.hdf5toformat2


    def __init__(self,sn,filename, verbose=False, onlyHeader=False, nommap=False, combineParticles=True, combineFiles=True, toDouble = False, **param):
        self.sn=sn
        
        res = re.findall("\.[0-9]*\.hdf5",filename)
        res2 = re.findall("\.[0-9]*\.h5",filename)
        if len(res) > 0:
            self.sn.currFile = int(res[-1][1:-5])
        elif len(res2) > 0:
            self.sn.currFile = int(res[-1][1:-3])
        else:
            self.sn.currFile=0

        self.combineParticles = combineParticles
        self.combineFiles = combineFiles
        self.toDouble = toDouble
        
        self.onlyHeader = onlyHeader
        self.verbose = verbose
        self.nommap = nommap
        

    def load(self):
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
        
        self.file = h5py.File(self.sn.filename,"r")
        self.load_header()
        self.file.close()
        del self.file
        self.sn.header = loader.Header(self.sn)
        
        if self.onlyHeader:
            if isinstance(self.sn, loader.Snapshot):
                self.sn.part0 = loader.PartGroup(self.sn,0)
                self.sn.part1 = loader.PartGroup(self.sn,1)
                self.sn.part2 = loader.PartGroup(self.sn,2)
                self.sn.part3 = loader.PartGroup(self.sn,3)
                self.sn.part4 = loader.PartGroup(self.sn,4)
                self.sn.part5 = loader.PartGroup(self.sn,5)
                self.sn.groups = [ self.sn.part0, self.sn.part1, self.sn.part2, self.sn.part3, self.sn.part4, self.sn.part5]
            else:
                self.sn.group = loader.PartGroup(self.sn,0)
                self.sn.subhalo = loader.PartGroup(self.sn,1)
            return

        if self.combineParticles or (self.combineFiles and self.sn.num_files>1) or self.toDouble or self.nommap:
            if isinstance(self.sn, loader.Snapshot):
                self.load_data()
                self.sn.part0 = loader.PartGroup(self.sn,0)
                self.sn.part1 = loader.PartGroup(self.sn,1)
                self.sn.part2 = loader.PartGroup(self.sn,2)
                self.sn.part3 = loader.PartGroup(self.sn,3)
                self.sn.part4 = loader.PartGroup(self.sn,4)
                self.sn.part5 = loader.PartGroup(self.sn,5)
                self.sn.groups = [ self.sn.part0, self.sn.part1, self.sn.part2, self.sn.part3, self.sn.part4, self.sn.part5]
            else:
                self.load_data_subfind()
                self.sn.group = loader.PartGroup(self.sn,0)
                self.sn.subhalo = loader.PartGroup(self.sn,1)
        else:
            if isinstance(self.sn, loader.Snapshot):
                self.sn.part0 = loader.PartGroup(self.sn,0)
                self.sn.part1 = loader.PartGroup(self.sn,1)
                self.sn.part2 = loader.PartGroup(self.sn,2)
                self.sn.part3 = loader.PartGroup(self.sn,3)
                self.sn.part4 = loader.PartGroup(self.sn,4)
                self.sn.part5 = loader.PartGroup(self.sn,5)
                self.sn.groups = [ self.sn.part0, self.sn.part1, self.sn.part2, self.sn.part3, self.sn.part4, self.sn.part5]
                self.load_data_map(((self.sn.part0, 'PartType0'),(self.sn.part1, 'PartType1'),(self.sn.part2, 'PartType2'),(self.sn.part3, 'PartType3'),(self.sn.part4, 'PartType4'),(self.sn.part5, 'PartType5')))
                
            else:
                self.sn.group = loader.PartGroup(self.sn,0)
                self.sn.subhalo = loader.PartGroup(self.sn,1)
                self.load_data_map(((self.sn.group, 'Group'),(self.sn.subhalo, 'Subhalo')))
            
            self.sn.__convenience__()
            self.sn.__writable__ = False
            


    def load_header(self):
        file = self.file
        
        if isinstance(self.sn, loader.Snapshot):
            self.sn.nparticles =  np.longlong(file['/Header'].attrs['NumPart_ThisFile'])
            self.sn.nparticlesall = np.longlong(file['/Header'].attrs['NumPart_Total'])
            self.sn.nparticlesall += np.longlong(file['/Header'].attrs['NumPart_Total_HighWord'])<<32
            self.sn.masses = file['/Header'].attrs['MassTable']
            self.sn.num_files = file['/Header'].attrs['NumFilesPerSnapshot']
            
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
        
        self.sn.flag_sfr = file['/Header'].attrs['Flag_Sfr']
        self.sn.flag_cooling = file['/Header'].attrs['Flag_Cooling']
        self.sn.flag_feedback = file['/Header'].attrs['Flag_Feedback']
        self.sn.flag_stellarage = file['/Header'].attrs['Flag_StellarAge']
        self.sn.flag_metals = file['/Header'].attrs['Flag_Metals']
        if "Flag_DoublePrecision" in file['/Header'].attrs.keys():
	           self.sn.flag_doubleprecision = file['/Header'].attrs['Flag_DoublePrecision']

    def load_data_map(self, groups):  
        self.file = h5py.File(self.sn.filename,"r")

        for (gr, hgr) in groups:
            if gr.__num__ in self.sn.__parttype__:
                self.sn.npart_loaded[gr.__num__] = self.sn.header.nparticles[gr.__num__]
                if hgr in self.file.keys():
                    for key in self.file[hgr].keys():
                        if not self.dict.has_key(key):
                            if self.verbose:
                                print "warning: hdf5 key '%s' could not translated"%key

                        name = self.dict.get(key,key)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            gr.data[str(name)]=self.file[hgr+'/'+key]
                            
        #TODO update npart_loaded

    def load_data(self):
        self.sn.npart_loaded = np.zeros(6,dtype=np.longlong)
        self.data = {}

        if self.combineFiles:
            filesA = 0
            filesB = self.sn.num_files
        else:
            filesA = self.sn.currFile
            filesB = self.sn.currFile+1
            
        #learn about present fields
        for i in np.arange(filesA,filesB):
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            self.file = h5py.File(filename,'r')
            self.sn.currFile = i
           
            self.load_header()
            self.sn.header = loader.Header(self.sn)
                

            for gr in self.sn.__parttype__:
                if "PartType%d"%gr in self.file.keys():
                    for item in self.file["PartType%d"%gr].keys():
                        if not self.dict.has_key(item):
                            if self.verbose:
                                print "warning: hdf5 key '%s' could not translated"%key
                                
                        name  = self.dict.get(item,item)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            d = self.file["PartType%d/%s"%(gr,item)]
                            shape = np.array(d.shape)
                            elem = 1
                            if shape.size == 2:
                                elem=shape[1]
                                
                            pres = fields.isPresent(name,self.sn,learn=True,gr=gr,shape=elem)
            self.file.close()
                                
        #now load the requested data                        
        for i in np.arange(filesA,filesB):
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            self.file = h5py.File(filename,'r')
            self.sn.currFile = i
                
            self.load_header()
            self.sn.header = loader.Header(self.sn)
                

            for gr in self.sn.__parttype__:
                if "PartType%d"%gr in self.file.keys():
                    for item in self.file["PartType%d"%gr].keys():
                        name  = self.dict.get(item,item)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            pres = fields.isPresent(name,self.sn)
                                
                            n1 = np.where(pres > 0, self.sn.npart_loaded, np.zeros(6,dtype=np.longlong))
                                
                            if self.combineFiles:
                                n2 = np.where(pres > 0, self.sn.nparticlesall, np.zeros(6,dtype=np.longlong))
                            else:
                                n2 = np.where(pres > 0, self.sn.nparticles, np.zeros(6,dtype=np.longlong))
                            
                            d = self.file["PartType%d/%s"%(gr,item)]
                            shape = np.array(d.shape)
                                
                            if not self.sn.data.has_key(name):
                                if self.combineFiles:
                                    num = np.where(pres > 0, self.sn.nparticlesall, np.zeros(6,dtype=np.longlong)).sum()
                                else:
                                    num = np.where(pres > 0, self.sn.nparticles, np.zeros(6,dtype=np.longlong)).sum()
                
                                #get propertiers of dataset
                                datatype = d.dtype
                                s = np.array(d.shape)
                                s[0] = num
                                if self.toDouble and datatype == np.dtype('float32'):
                                    datatype = np.dtype('float64')
                                        
                                self.sn.data[name] = np.empty(s, dtype=datatype)
                                    
                            self.sn.data[name][n2[0:gr].sum()+n1[gr]:n2[0:gr].sum()+n1[gr]+shape[0]] = d
            
            self.sn.npart_loaded[self.sn.__parttype__] += self.sn.nparticles[self.sn.__parttype__]
       
            self.file.close()
            del self.file
        
    def load_data_subfind(self):
        self.sn.data = {}
        self.sn.npart_loaded = np.zeros(6,dtype=np.longlong)
        
        groupnames = ['Group','Subhalo']

        if self.combineFiles:
            filesA = 0
            filesB = self.sn.num_files
        else:
            filesA = self.sn.currFile
            filesB = self.sn.currFile+1
            
        #learn about present fields
        for i in np.arange(filesA,filesB):
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            self.file = h5py.File(filename,'r')
            self.sn.currFile = i
           
            self.load_header()
            self.sn.header = loader.Header(self.sn)
                

            for gr in self.sn.__parttype__:
                if groupnames[gr] in self.file.keys():
                    for item in self.file[groupnames[gr]].keys():
                        if not self.dict.has_key(item):
                            if self.verbose:
                                print "warning: hdf5 key '%s' could not translated"%key
                                
                        name  = self.dict.get(item,item)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            d = self.file["%s/%s"%(groupnames[gr],item)]
                            shape = np.array(d.shape)
                            elem = 1
                            if shape.size == 2:
                                elem=shape[1]
                                
                            pres = fields.isPresent(name,self.sn,learn=True,gr=gr,shape=elem)
            self.file.close()
                                
        #now load the requested data                    
        for i in np.arange(filesA,filesB):
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            self.file = h5py.File(filename,'r')
            self.sn.currFile = i
                
            self.load_header()
            self.sn.header = loader.Header(self.sn)
                

            for gr in self.sn.__parttype__:
                if groupnames[gr] in self.file.keys():
                    for item in self.file[groupnames[gr]].keys():
                        name  = self.dict.get(item,item)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            pres = fields.isPresent(name,self.sn)
                                
                            n1 = np.where(pres > 0, self.sn.npart_loaded, np.zeros(6,dtype=np.longlong))
                                
                            if self.combineFiles:
                                n2 = np.where(pres > 0, self.sn.nparticlesall, np.zeros(6,dtype=np.longlong))
                            else:
                                n2 = np.where(pres > 0, self.sn.nparticles, np.zeros(6,dtype=np.longlong))
                            
                            d = self.file["%s/%s"%(groupnames[gr],item)]
                            shape = np.array(d.shape)
                                
                            if not self.sn.data.has_key(name):
                                if self.combineFiles:
                                    num = np.where(pres > 0, self.sn.nparticlesall, np.zeros(6,dtype=np.longlong)).sum()
                                else:
                                    num = np.where(pres > 0, self.sn.nparticles, np.zeros(6,dtype=np.longlong)).sum()
                
                                #get propertiers of dataset
                                datatype = d.dtype
                                s = np.array(d.shape)
                                s[0] = num
                                if self.toDouble and datatype == np.dtype('float32'):
                                    datatype = np.dtype('float64')
                                        
                                self.sn.data[name] = np.empty(s, dtype=datatype)
                                    
                            self.sn.data[name][n2[0:gr].sum()+n1[gr]:n2[0:gr].sum()+n1[gr]+shape[0]] = d
            
            self.sn.npart_loaded[self.sn.__parttype__] += self.sn.nparticles[self.sn.__parttype__]
       
            self.file.close()
            del self.file

    def nextFile(self, num=None):
        if (num == None and self.sn.currFile < (self.sn.num_files-1) ) or (num != None and num < self.sn.num_files and num >= 0):
            if num == None:
                self.sn.currFile = self.sn.currFile+1
            else:
                self.sn.currFile = num
                
            if isinstance(self.sn, loader.Snapshot):
                del self.sn.data
                del self.sn.part0
                del self.sn.part1
                del self.sn.part2
                del self.sn.part3
                del self.sn.part4
                del self.sn.part5
            else:
                del self.sn.group
                del self.sn.subhalo
                
            del self.sn.groups
            self.close()

            #open next file
            self.sn.npart_loaded = np.zeros(6,dtype=np.longlong)
            self.sn.data = {}
            
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%self.sn.currFile, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%self.sn.currFile, filename)
            self.sn.filename = filename

            self.load()
            
            return True

        else:
            if self.verbose:
                if num == None:
                    print "last chunk reached"
                else:
                    print "invalid file number: %d"%num
                
            return False

    def close(self):
        if hasattr(self,"file"):
            self.file.close()
            del self.file

    def write(self,filename):
        if not filename.endswith(".hdf5") and not filename.endswith(".h5"):
            filename += ".hdf5"
            
        #TODO check filename for multipart files    
            
        file = h5py.File(filename,"w")
        
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
        if hasattr(self, "flag_doubleprecision"):
            header.attrs['Flag_DoublePrecision'] = self.sn.flag_doubleprecision
        
        
    def write_particles(self,file):
        for i in np.arange(0,6):
            group = file.create_group("PartType%d"%i)
            
            for item in self.sn.data:
                pres = fields.isPresent(item,self.sn)
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
