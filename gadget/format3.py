import h5py
import numpy as np
import os.path as path
import re
import os

import gadget
import gadget.fields as fields
import gadget.units as units

def handlesfile(filename, snapshot=None, filenum=None, snapprefix=None, dirprefix=None, snap=None, **param):
    if getFilename(filename, snapshot, filenum, snapprefix, dirprefix, snap)[0] is not None:
        return True
    else:
        return False
    
def writefile(filename):
    if filename.endswith(".hdf5") or filename.endswith(".h5"):
        return True
    else:
        return False
    
    
def getFilename(filename, snapshot=None, filenum=None, snapprefix = None, dirprefix=None, snap = None):
    if snapprefix is None:
        if isinstance(snap, gadget.loader.Subfind):
            snapprefix = ["fof_subhalo_tab"]
        else:
            snapprefix = ["snap","snapshot"]
            
    else:
            snapprefix = [snapprefix]
            
    if dirprefix is None:
        if isinstance(snap, gadget.loader.Subfind):
            dirprefix = "groups"
        else:
            dirprefix = "snapdir"
    
    if snapshot is not None:
        filename = re.sub(r"%s_[0-9]+"%dirprefix,r"%s_%03d"%(dirprefix,snapshot), filename)
        for sp in snapprefix:
            filename = re.sub(r"%s_[0-9]+\."%sp,r"%s_%03d."%(sp,snapshot), filename)
    
    if filenum is not None:
        filename = re.sub(r"\.[0-9]+\.(hdf5|h5)$",r".%d.\1"%filenum, filename)
        
    if filenum is None:
        filenum = 0
        
    if filename == "":
        filename = "."
    
    if path.isfile( filename ) and (filename.endswith(".hdf5") or filename.endswith(".h5")):
        filename = filename  
    else: 
        filename_candidates = []    
        if snapshot is not None:
            for sp in snapprefix:
                filename_candidates.append(filename + "/" + sp +"_%03d"%snapshot + ".hdf5")
                filename_candidates.append(filename + "/" + sp +"_%03d"%snapshot + ".h5")
        
                filename_candidates.append(filename + "/" + sp +"_%03d"%snapshot + ".%d"%filenum + ".hdf5")
                filename_candidates.append(filename + "/" + sp +"_%03d"%snapshot + ".%d"%filenum + ".h5")
            
                filename_candidates.append(filename + "/" + dirprefix + "_%03d/"%snapshot + sp +"_%03d"%snapshot + ".%d"%filenum + ".hdf5")
                filename_candidates.append(filename + "/" + dirprefix + "_%03d/"%snapshot + sp +"_%03d"%snapshot + ".%d"%filenum + ".h5")
        
        found = False
        for fname in filename_candidates:
            if path.isfile(fname):
                filename = fname
                found = True
                break
            
        if not found:
            return (None, None, None)
        
        
    res = re.findall(r"_([0-9]+)\.(([0-9]*)\.)?(hdf5|h5)$", filename)
    
    if len(res) > 0:
        if res[0][2] != "":
            filenum = int(res[0][2])
        else:
            filenum = None
    
        if res[0][0] != "":
            snapshot = int(res[0][0])
        else:
            snapshot = None
    else:
        filenum = None
        snapshot = None
           
    return (filename, filenum, snapshot)

class Format3:
    def __init__(self, sn, snapprefix=None, dirprefix=None):          
        self.sn=sn
        self.snapprefix = snapprefix
        self.dirprefix = dirprefix
        
        
    def load(self, num_load, snapshot_load):
        if num_load is None:
            num_load = self.sn.filenum
            
        if snapshot_load is None:
            snapshot_load = self.sn.snapshot
        
        filename, num, snapshot = getFilename(self.sn.filename, snapshot_load, num_load, self.snapprefix, self.dirprefix, self.sn)
        
        try:
            self.file = h5py.File(filename,"r")
        except Exception:
            raise Exception("could not open file '%s'"%str((self.sn.filename, snapshot_load, num_load, self.snapprefix, self.dirprefix)))
        
        self.load_parameter("Parameters", "parameters")
        self.load_parameter("Config", "config")
        self.load_header()
        self.file.close()
        del self.file
        
        if not self.sn._onlyHeader:
            self.load_data(filename,num)
                
        if self.sn._combineFiles:
            self.sn.filenum = None
        else:
            self.sn.filenum = num
            
        self.sn.snapshot = snapshot
        self.sn.filename = filename
            
        self.sn._path = path.abspath(self.sn.filename)

    def load_header(self):
        file = self.file
        
        self.sn._headerfields = []
        
        for i in file['/Header'].attrs:
            name = i.replace(" " ,"")
                
            setattr(self.sn, name, file['/Header'].attrs[i])
            self.sn._headerfields.append(name)
        
        if isinstance(self.sn, gadget.loader.Snapshot):
            self.sn.nparticlesall = np.longlong(self.sn.NumPart_Total)
            self.sn.nparticlesall += np.longlong(self.sn.NumPart_Total_HighWord)<<32

            self.sn.npart = np.array(self.sn.NumPart_ThisFile).sum()
            self.sn.npartall = np.array(self.sn.nparticlesall).sum()
            
            self.sn.ntypes = len(self.sn.NumPart_ThisFile)
            
            if hasattr(self.sn, "parameters"):
                if self.sn.parameters.ComovingIntegrationOn == 0:
                    mu = gadget.units.Unit(0.,0.,{'M':1.},self.sn.parameters.UnitMass_in_g)
                else:
                    mu = gadget.units.Unit(0.,-1.,{'M':1.},self.sn.parameters.UnitMass_in_g)
                
                if self.sn.parameters.ComovingIntegrationOn == 0:
                    lu = gadget.units.Unit(0.,0.,{'L':1.},self.sn.parameters.UnitMass_in_g)
                else:
                    lu = gadget.units.Unit(1.,-1.,{'L':1.},self.sn.parameters.UnitLength_in_cm)
                    
                if self.sn.parameters.ComovingIntegrationOn == 0:
                    tu = gadget.units.Unit(0.,0.,{'L':1.,'V':-1.},self.sn.parameters.UnitLength_in_cm/self.sn.parameters.UnitVelocity_in_cm_per_s)
                    self.sn.Time = gadget.units.Quantity(self.sn.Time,tu)
                     
                self.sn.MassTable = gadget.units.Quantity(self.sn.MassTable, mu)
                self.sn.BoxSize = gadget.units.Quantity(self.sn.BoxSize, lu)
                
      
        else:
            self.sn.NumPart_ThisFile = np.array([self.sn.Ngroups_ThisFile, self.sn.Nsubgroups_ThisFile])
            self.sn.nparticlesall = np.array([self.sn.Ngroups_Total, self.sn.Nsubgroups_Total])
            
            self.sn.ntypes = 2
        
        if self.sn._parttype is None:
            self.sn._parttype = np.arange(self.sn.ntypes)
            

    def load_parameter(self,group, name):
        file = self.file
        
        if group in file:
            param = gadget.loader.Parameter(self.sn,name)
            for i in file[group].attrs:        
                    setattr(param, i, file[group].attrs[i])
                    param._attrs.append(i)
    
            setattr(self.sn, name, param)
            
            
    def load_data(self, filename, num):
        self.sn.npart_loaded = np.zeros(self.sn.ntypes,dtype=np.longlong)
        self.sn.data = {}
        
        loaded = np.zeros(self.sn.ntypes,dtype=np.longlong)
        
        if isinstance(self.sn, gadget.loader.Subfind):
            groupnames = ['Group','Subhalo']
        else:
            groupnames = ["PartType%d"%i for i in np.arange(self.sn.ntypes)]
        
        if hasattr(self.sn,"_filter") and self.sn._filter != None:
            filter = self.sn._filter
            if type(filter) == list:
                filter = np.array(filter)
            elif type(filter) != np.ndarray:
                filter = np.array([filter])
            indices = [[],[],[],[],[],[]]

            for f in filter:
                f.reset()

        else:
            filter = None

        if self.sn._combineFiles:
            filesA = 0
            filesB = self.sn.NumFilesPerSnapshot
        elif num is not None:
            filesA = num
            filesB = num+1
        else:
            filesA = 0
            filesB = 1
            
        #learn about present fields
        for i in np.arange(filesA,filesB):
            if self.sn._verbose:
                print("Learning about file %d"%i)

            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            try:
                self.file = h5py.File(filename,'r')
            except Exception:
                raise Exception("could not open file %s"%filename)
           
            self.load_header()

            for gr in self.sn._parttype:
                if groupnames[gr] in self.file.keys():
                    for item in self.file[groupnames[gr]].keys():
                        name  = self.sn._normalizeName(item)
                        if self.sn._fields==None or name in self.sn._fields:
                            d = self.file["%s/%s"%(groupnames[gr],item)]
                            shape = np.array(d.shape)
                            elem = 1
                            if shape.size == 2:
                                elem=np.int32(shape[1])
                                
                            pres = self.sn._learnPresent(name,gr=gr,shape=elem)
                if filter != None:
                    if groupnames[gr] in self.file.keys():
                        ind = np.arange(self.sn.NumPart_ThisFile[gr])
                        for f in filter:
                            if gr in f.parttype:
                                data = {}
                                data['NumPart_ThisFile'] = np.longlong(self.file['/Header'].attrs['NumPart_ThisFile'])[gr]
                                data['group'] = gr
                                
                                for fld in f.requieredFields:
                                    fld2 = fields.rev_hdf5toformat2.get(fld,fld)
                                    data[fld] = self.file[groupnames[gr]][fld2][...][ind]
                                ind_tmp = f.getIndices(data)
                                ind = ind[ind_tmp]
                            
                        indices[gr].append(np.copy(ind))
                        self.sn.npart_loaded[gr] += len(ind)
                        
                    else:
                        indices[gr].append(np.array([]))
                else:
                    self.sn.npart_loaded[gr] += self.sn.NumPart_ThisFile[gr]
            self.file.close()
                                
        #now load the requested data                        
        for i in np.arange(filesA,filesB):
            if self.sn._verbose:
                print("Reading file %d"%i)

            filename = re.sub("\.[0-9]*\.hdf5",".%d.hdf5"%i, self.sn.filename)
            filename = re.sub("\.[0-9]*\.h5",".%d.h5"%i, filename)
            self.sn.filename = filename
            try:
                self.file = h5py.File(filename,'r')
            except Exception:
                raise Exception("could not open file %s"%filename)
                
            self.load_header()

            for gr in self.sn._parttype:
                if groupnames[gr] in self.file.keys():
                    elements = 0
                    for item in self.file[groupnames[gr]].keys():
                        name  = self.sn._normalizeName(item)
                        if self.sn._fields==None or name in self.sn._fields:
                            pres = self.sn._isPresent(name)
                                
                            n1 = np.where(pres > 0, loaded, np.zeros(self.sn.ntypes,dtype=np.longlong))
                            n2 = np.where(pres > 0, self.sn.npart_loaded, np.zeros(self.sn.ntypes,dtype=np.longlong))
                            
                            d = self.file["%s/%s"%(groupnames[gr],item)]
                            shape = np.array(d.shape)
                                
                            if not name in self.sn.data:
                                dtype = d.dtype
                                if self.sn._toDouble and dtype == np.float32:
                                    dtype = np.float64
                                    
                                unit = None
                                if 'a_scaling' in d.attrs:
                                    unit = units.Unit(d.attrs['a_scaling'], d.attrs['h_scaling'], {'L':d.attrs['length_scaling'], 'M':d.attrs['mass_scaling'], 'V':d.attrs['velocity_scaling']},d.attrs['to_cgs'])
                                    
                                self.sn.addField(name, dtype=dtype, unit = unit)
                                
                                
                            if filter is None:
                                self.sn.data[name][n2[0:gr].sum()+n1[gr]:n2[0:gr].sum()+n1[gr]+shape[0]] = d
                                elements = d.shape[0]
                            elif len(indices[gr][i]) > 0:
                                self.sn.data[name][n2[0:gr].sum()+n1[gr]:n2[0:gr].sum()+n1[gr]+len(indices[gr][i])] = d[...][indices[gr][i],...]
                                elements = len(indices[gr][i])
                    loaded[gr] += elements
       
            self.file.close()
            del self.file


    def close(self):
        if hasattr(self,"file"):
            self.file.close()
            del self.file

    def write(self,filename):
        if not filename.endswith(".hdf5") and not filename.endswith(".h5"):
            filename += ".hdf5"
    
        try:
            file = h5py.File(filename,"w")
        except Exception:
            raise Exception("could not open file %s for writing"%filename)        
        
        self.write_header(file)
        
        if hasattr(self.sn, "parameters"):
            self.write_parameter(self.sn.parameters, "Parameters", file)
            
        if hasattr(self.sn, "config"):
            self.write_parameter(self.sn.config, "Config", file)
        
        self.write_particles(file)
        
        file.close()

        
    def write_header(self,file):
        header = file.create_group("/Header")
           
        for i in self.sn._headerfields:    
            if hasattr(self.sn, i):
                value = getattr(self.sn, i)
                if i == "Flag_DoublePrecision":
                    if self.sn._precision == np.float64:
                        value = 1
                    else:
                        value = 0
                        
                if i == "NumPart_ThisFile":
                    value = self.sn.npart_loaded
                
                header.attrs[i] = value  
    
    def write_parameter(self,param, name, file):
        group = file.create_group(name)
        
        for i in param._attrs:
            if hasattr(param, i):
                group.attrs[i] = getattr(param, i) 
        
    def write_particles(self,file):
        if isinstance(self.sn, gadget.loader.Subfind):
            groupnames = ['Group','Subhalo']
        else:
            groupnames = ["PartType%d"%i for i in np.arange(self.sn.ntypes)]
            
        for i in np.arange(self.sn.ntypes):
            if self.sn.npart_loaded[i] == 0:
                continue
            
            group = file.create_group(groupnames[i])
            
            for item in self.sn.data:
                pres = self.sn._isPresent(item)
                if pres[i] > 0 and self.sn.npart_loaded[i]>0:
                    #reverse translate name
                    if item in fields.rev_hdf5toformat2:
                        name = fields.rev_hdf5toformat2[item]
                    else:
                        name = item
                        
                    f = self.sn.data[item]
                    
                    if pres[i]>1:
                        dataset = group.create_dataset(name, shape=(self.sn.npart_loaded[i],pres[i]), dtype=f.dtype)
                    else:
                        dataset = group.create_dataset(name, shape=(self.sn.npart_loaded[i],), dtype=f.dtype)
                        
                    n1 = np.where(pres>0, self.sn.npart_loaded,np.zeros(self.sn.ntypes,dtype=np.longlong))
                    tmp = np.sum(n1[0:i])
                    dataset[...] = f[tmp:tmp+self.sn.npart_loaded[i]]
