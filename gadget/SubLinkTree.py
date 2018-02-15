import numpy as np
import re
import functools
import os

import gadget
import gadget.units as units

import h5py

import gadget.LHaloTree


class SubLinkTree:
    def __init__(self, filename, fields=False, verbose=False):
        file = h5py.File(filename, 'r')
        
        if not fields:
            fields = file.keys()
        self.fields = fields
        
        for item in self.fields:
            if verbose:
                print("reading in ", item)
            setattr( self, item, np.array( file[ "%s" % item ] ) )
            
        if hasattr(self, 'SubhaloID'):
            ## set up dictionary to translate subhaloIDs in indices
            self.ID_to_index = {
                                self.SubhaloID[i]: i
                                for i in np.arange(len(self.SubhaloID))
                                }
        
        file.close()
        
    def getMainBranch(self, i_in_tree, fields = False, verbose = False ):
        if not hasattr(self, 'ID_to_index'):
            print("ERROR: getMainBranch neads ID_to_index dictionary, which can only be created if you load the field 'SubhaloID' ")
        
        i_in_tree =  self.ID_to_index[ self.RootDescendantID[i_in_tree] ]
        id_prog = self.FirstProgenitorID[i_in_tree]
        
        MainBranch = []
        MainBranch.append( LHaloTree.halo(self, i_in_tree, fields=fields, verbose=verbose) )
        
        while id_prog > 0:
            i_prog = self.ID_to_index[ id_prog ]
            
            MainBranch.append( LHaloTree.halo( self, i_prog, fields = fields, verbose = verbose ) )
            id_prog = self.FirstProgenitorID[ i_prog ]
            
        return MainBranch