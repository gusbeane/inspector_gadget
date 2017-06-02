import numpy as np
import re
import functools
import os

import gadget
import gadget.units as units

import h5py


class header:
    def __init__(self,file):
        ## load all attributes
        for attr in file["/Header"].attrs:
          name = attr.replace(" " ,"")
          setattr(self, name, file['/Header'].attrs[attr])

        ## load arrays in header
        d = file["/Header/TreeNHalos"]
        self.TreeNHalos = np.array(d)

        d = file["/Header/TotNsubhalos"]
        self.TotNsubhalos = np.array(d)

        d = file["/Header/Redshifts"]
        self.Redshifts = np.array(d)

class halo:
    def __init__(self, tree, i_halo, fields=False, verbose=False):
        if not fields:    ## if not specified, get all fields that are arrays
            fields = tree.__dict__
        self.fields = fields
        for item in fields:
            data = getattr(tree,item)
            if isinstance(data,np.ndarray):
                if verbose:
                    print("copying ", item)
                setattr(self,item,data[i_halo])


class LHaloTree:
    def __init__(self, filename, i_tree, fields=False, verbose=False):
        file = h5py.File(filename,"r")

        self.header = header(file)
        ## load i'th tree
        Name = "/Tree%d" % np.int16(i_tree)
        ## get groups, either all, or only specified ones
        if not fields:
            fields = file[Name].keys()
        self.fields = fields

        for item in self.fields:
            if verbose:
                print("reading in ", item)
            setattr(self, item, np.array(file["%s/%s"%(Name,item)]) )
        file.close()

    def getMainBranch(self, i_in_tree, fields=False, verbose=False):
        i_halo = i_in_tree
        ## trace back to last snapshot
        while self.Descendant[i_halo] != -1:
            i_halo = self.Descendant[i_halo]
        i_next = self.FirstProgenitor[i_halo]
        ## debug check
        if(self.SnapNum[i_halo] < self.header.LastSnapshotNr):
            print("error: something is wrong here! SnapNum[0] != header.LastSnapshotNr")
            return -1
        ## walk the tree and get properties
        MainBranch = []
        MainBranch.append( halo(self,i_halo,fields=fields, verbose=verbose) )
        while i_next > 0:  ## actually = is not necessary, as this is the first one
            i_halo = i_next
            i_next = self.FirstProgenitor[i_halo]
            MainBranch.append( halo(self,i_halo,fields=fields, verbose=verbose) )

        return MainBranch

    def getHaloProperty(self, function, i_sub, RunOutputDir=None,parttype=None,fields=None,haloFields=None,verbose=False):
        if not RunOutputDir:
            RunOutputDir = self.header.RunOutputDir
        SnapNum = self.SnapNum[i_sub]

        if verbose:
            print("getHaloProperty: opening group catalogue", RunOutputDir+"/groups_%03d/fof_subhalo_tab_%03d.0.hdf5"%(SnapNum, SnapNum))
        sub = gadget.Subfind(RunOutputDir+"/groups_%03d/fof_subhalo_tab_%03d.0.hdf5"%(SnapNum, SnapNum),combineFiles=True,fields=haloFields )

        i_halo = np.int(sub.SubhaloGrNr[i_sub])
        Filter = gadget.filter.Halo(sub,halo=i_halo)

        if verbose:
            print("geHaloProperty: getting particle data: ", RunOutputDir+"/snapdir_%03d/snap_%03d.0.hdf5"%(SnapNum,SnapNum))
        snap = gadget.Simulation(RunOutputDir+"/snapdir_%03d/snap_%03d.0.hdf5"%(SnapNum,SnapNum),combineFiles=True,filter=Filter,parttype=parttype,fields=fields)

        if verbose:
            print("getHaloProperty: calling function!")
        Property = function(snap=snap,sub=sub,i_halo=i_halo,i_sub=i_sub)

        snap.close()
        sub.close()

        return Property

    def getMainBranchProperty(self,function, RunOutputDir=None,parttype=None,fields=None,haloFields=None,verbose=False):
        i_in_tree = 0
        i_next_in_tree = self.FirstProgenitor[i_in_tree]
        i_sub = self.SubhaloNumber[i_in_tree]

        Redshift = []
        Property = []

        Redshift.append( self.header.Redshifts[self.SnapNum[i_sub]] )
        Property.append( self.getHaloProperty(function, i_sub, RunOutputDir=RunOutputDir, parttype=parttype, fields=fields, haloFields=haloFields, verbose=verbose) )

        while i_next_in_tree > 0:
            i_in_tree = i_next_in_tree
            i_next_in_tree = self.FirstProgenitor[i_in_tree]
            i_sub = self.SubhaloNumber[i_in_tree]

            Redshift.append( self.header.Redshifts[self.SnapNum[i_sub]] )
            Property.append( self.getHaloProperty(function, i_sub, RunOutputDir=RunOutputDir, parttype=parttype, fields=fields, haloFields=haloFields, verbose=verbose) )

        return np.array(Redshift), Property



