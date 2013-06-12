import h5py
import numpy as np
import os.path as path
import re

import gadget.loader as loader
import gadget.fields as fields


class Format3:


    dict = fields.hdf5toformat2


    def __init__(self,sn,filename,combineParticles=False, combineFiles=True, toDouble = False):
        self.sn=sn

        if not path.exists( filename ):
            if path.exists( filename + ".hdf5" ):
                filename += ".hdf5"
            elif  path.exists( filename + "0.hdf5" ):
                filename += "0.hdf5"
            elif path.exists( filename + ".h5" ):
                filename += ".h5"
            elif  path.exists( filename + "0.h5" ):
                filename += "0.h5"

        self.sn.filename = filename
        self.currFile=0

        self.combineParticles = combineParticles
        self.combineFiles = combineFiles
        self.toDouble = toDouble

        self.sn.numpart_loaded = np.zeros(6,dtype=np.longlong)

    def load(self):
        self.file = h5py.File(self.sn.filename,'r')

        self.load_header()
        self.sn.header = loader.Header(self.sn)

        if self.combineParticles or (self.combineFiles and self.sn.nfiles>1) or self.toDouble:
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
                self.sn.group = loader.PartGroup(self.sn,0)
                self.sn.subhalo = loader.PartGroup(self.sn,1)
                self.load_data_subfind()
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
            


    def load_header(self):
        file = self.file
        if isinstance(self.sn, loader.Snapshot):
            self.sn.nparticles =  np.longlong(file['/Header'].attrs['NumPart_ThisFile'])
            self.sn.nparticlesall = np.longlong(file['/Header'].attrs['NumPart_Total'])
            self.sn.nparticlesall += np.longlong(file['/Header'].attrs['NumPart_Total_HighWord'])<<32
            self.sn.masses = file['/Header'].attrs['MassTable']
            self.sn.nfiles = file['/Header'].attrs['NumFilesPerSnapshot']
            
            self.sn.npart = np.array( self.sn.nparticles ).sum()
            self.sn.npartall = np.array( self.sn.nparticlesall ).sum()

        else:
            self.sn.ngroups =  file['/Header'].attrs['Ngroups_ThisFile']
            self.sn.ngroupsall =  file['/Header'].attrs['Ngroups_Total']
            self.sn.nids =  file['/Header'].attrs['Nids_ThisFile']
            self.sn.nidsall =  file['/Header'].attrs['Nids_Total']
            self.sn.nsubgroups =  file['/Header'].attrs['Nsubgroups_ThisFile']
            self.sn.nsubgroupsall =  file['/Header'].attrs['Nsubgroups_Total']
            self.sn.nfiles = file['/Header'].attrs['NumFiles']
            
            self.sn.nparticles = [self.sn.ngroups, self.sn.nsubgroups,0,0,0,0]
            self.sn.nparticlesall = [self.sn.ngroupsall, self.sn.nsubgroupsall,0,0,0,0]
        


        self.sn.time = file['/Header'].attrs['Time']
        self.sn.redshift = file['/Header'].attrs['Redshift']
        self.sn.boxsize = file['/Header'].attrs['BoxSize']
        self.sn.omega0 = file['/Header'].attrs['Omega0']
        self.sn.omegalambda = file['/Header'].attrs['OmegaLambda']
        self.sn.hubbleparam = file['/Header'].attrs['HubbleParam']
        #TODO read flags
        

    def load_data_map(self, groups):      
        self.sn.data = {}

        for (gr, hgr) in groups:
            if gr.__num__ in self.sn.__parttype__:
                self.sn.numpart_loaded[gr.__num__] = self.sn.header.nparticles[gr.__num__]
                if hgr in self.file.keys():
                    for key in self.file[hgr].keys():
                        if not self.dict.has_key(key):
                            print "warning: hdf5 key '%s' could not translated"%key

                        name = self.dict.get(key,key)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            gr.data[str(name)]=self.file[hgr+'/'+key]


    def load_data(self):
        self.sn.data = {}
        self.sn.numpart_loaded = np.zeros(6,dtype=np.longlong)

        if self.combineFiles:
            filesA = 0
            filesB = self.sn.nfiles
        else:
            filesA = self.currFile
            filesB = self.currFile+1
            
        #learn about present fields
        for i in np.arange(filesA,filesB):
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            self.file = h5py.File(filename,'r')
            self.currFile = i
           
            self.load_header()
            self.sn.header = loader.Header(self.sn)
                

            for gr in self.sn.__parttype__:
                if "PartType%d"%gr in self.file.keys():
                    for item in self.file["PartType%d"%gr].keys():
                        name  = self.dict.get(item,item)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            d = self.file["PartType%d/%s"%(gr,item)]
                            shape = np.array(d.shape)
                            elem = 1
                            if shape.size == 2:
                                elem=shape[1]
                                
                            pres = fields.isPresent(name,gr,self.sn,learn=True,shape=elem)
            self.close()
                                
        #now load the requested data                        
        for i in np.arange(filesA,filesB):
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            self.file = h5py.File(filename,'r')
            self.currFile = i
                
            self.load_header()
            self.sn.header = loader.Header(self.sn)
                

            for gr in self.sn.__parttype__:
                if "PartType%d"%gr in self.file.keys():
                    for item in self.file["PartType%d"%gr].keys():
                        name  = self.dict.get(item,item)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            pres = fields.isPresent(name,gr,self.sn)
                                
                            n1 = np.where(pres > 0, self.sn.numpart_loaded, np.zeros(6,dtype=np.longlong))
                                
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
            
            self.sn.numpart_loaded[self.sn.__parttype__] += self.sn.nparticles[self.sn.__parttype__]
       
            self.close()
        
    def load_data_subfind(self):
        self.sn.data = {}
        self.sn.numpart_loaded = np.zeros(6,dtype=np.longlong)

        if self.combineFiles:
            filesA = 0
            filesB = self.sn.nfiles
        else:
            filesA = self.currFile
            filesB = self.currFile+1
            
        #learn about present fields
        for i in np.arange(filesA,filesB):
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            self.file = h5py.File(filename,'r')
            self.currFile = i
           
            self.load_header()
            self.sn.header = loader.Header(self.sn)
                

            for gr in self.sn.__parttype__:
                if gr in self.file.keys():
                    for item in self.file[gr].keys():
                        name  = self.dict.get(item,item)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            d = self.file["%s/%s"%(gr,item)]
                            shape = np.array(d.shape)
                            elem = 1
                            if shape.size == 2:
                                elem=shape[1]
                                
                            pres = fields.isPresent(name,gr,self.sn,learn=True,shape=elem)
            self.close()
                                
        #now load the requested data                    
        for i in np.arange(filesA,filesB):
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            self.file = h5py.File(filename,'r')
            self.currFile = i
                
            self.load_header()
            self.sn.header = loader.Header(self.sn)
                

            for gr in self.sn.__parttype__:
                if gr in self.file.keys():
                    for item in self.file[gr].keys():
                        name  = self.dict.get(item,item)
                        if self.sn.__fields__==None or name in self.sn.__fields__:
                            pres = fields.isPresent(name,gr,self.sn)
                                
                            n1 = np.where(pres > 0, self.sn.numpart_loaded, np.zeros(6,dtype=np.longlong))
                                
                            if self.combineFiles:
                                n2 = np.where(pres > 0, self.sn.nparticlesall, np.zeros(6,dtype=np.longlong))
                            else:
                                n2 = np.where(pres > 0, self.sn.nparticles, np.zeros(6,dtype=np.longlong))
                            
                            d = self.file["%s/%s"%(gr,item)]
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
            
            self.sn.numpart_loaded[self.sn.__parttype__] += self.sn.nparticles[self.sn.__parttype__]
       
            self.close()


    def next_chunk(self):
        if self.currFile < (self.sn.nfiles-1):
            self.currFile = self.currFile+1
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
            self.sn.numpart_loaded = np.zeros(6,dtype=np.longlong)
            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%self.currFile, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%self.currFile, filename)
            self.sn.filename = filename

            self.load()

        else:
            print "last chunk reached"

    def close(self):
        if hasattr(self,"file"):
            self.file.close()
            del self.file

