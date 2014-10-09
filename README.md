# Installation #

You need numpy, scipy and h5py installed. Then checkout the git repository and run

```
#!bash

python setup.py install
```


# Inspector Gadget Examples #

First load the module (both are the same):

```
#!python
import arepo
# or
import gadget
```

## Loading a Snapshot ##

```
#!python

# load a snapshot (only part type 0 and 4, and only the given fields)
# skip these attributes if you want to load everything; combineFiles=True, would load all snapshot files
sn = arepo.Snapshot('./snapdir_050/snap_050.0.hdf5', parttype=[0,4],  fields =['mass','GFM_StellarFormationTime'])

# get an overview of whats in the snapshot
print sn

# get a header attribute
sn.header.time
sn.time

# both point to the same array
sn.part0.mass
sn.part0['mass']

# this array contains the masses of all particles (with sn.header.masses[i] == 0)
sn.mass
sn['mass']

#move on to the next file 
sn.next()
```

## Loading a Subfind Catalog ##



```
#!python

# this will load the whole catalog (the interface is very similar to the snapshot interface above)
 sub = arepo.Subfind('./groups_135/fof_subhalo_tab_135.0.hdf5', combineFiles=True)

#access the halos (similar ways as in the snapshot example above are possible):
sub.group.GroupCM

#and the subhalos:
sub.subhalo.SubhaloCM 
```



## Load only one (sub) halo ##


```
#!python

# select the Halo; sub is a subhalo catalog (see above)
#    you need at least the following fields: fields=['GroupNsubs','SubhaloLenType','GroupLenType']
#    and combineFiles = True
# and load only real star particles
# the Halo filter must be first, then the Stars filter
filter = [arepo.filter.Halo(sub,halo=0), arepo.filter.Stars() ]

# and load the data
sn = arepo.Snapshot('./snapdir_135/snap_135.0.hdf5', parttype=[4], filter=filter, combineFiles=True, verbose=True)

```


## Some simple plots ##
The code used in the following examples is still a bit messy and should be considered experimental. Use them at your own risk and read the code if you are in doubt.
These functions should be cleaned up and extended by other small functions to get a quick look at your data. Patches are welcome.


```
#!python

# The Simulation class extands the capabilities of the Snapshot class with some basic analysis/visualization functions
sn = arepo.Simulation("snap_003.hdf5")

# A slice through the Voronoi mesh
sn.plot_Aslice("rho")

# A slice through an AMR mesh
sn.plot_AMRslice("rho")

```