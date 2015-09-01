Overview
============================================

Documentation of the python API:

.. toctree::
   :maxdepth: 2
   
   gadget



Installation
============

You need the following python modules installed: numpy, scipy and h5py. 
To install inspector gadget, checkout the git repository from https://bitbucket.org/abauer/inspector_gadget and run::

  python setup.py install

Now the module should be available systemwide.

Usage
=====

Inspector gadget installs two command line tools.
``arepoinfo`` gives shows you the header and contents of a snapshot and ``arepoconvert`` can convert snapshots to format 3 snapshots. Start them with ``--help`` to get a list of options

To use inspector gadget, start ipython and load the module (both modules are the same)::

  import arepo
  # or
  import gadget

Loading a Snapshot
******************
A snapshot is stored in a Snapshot object:

  sn.gadget.Snapshot("output", 10)
  
will load the 10-th snapshot in the folder output. Alternatively the full path of the snapshot file can be specified::

  sn.gadget.Snapshot("output/snap_010.hdf5")

Formats 1,2 and 3 have reading support. Writing is however only supported for format 3. See documentation of the snapshot class for details

You can specify which particle types and which fields should be loaded through the ``parttype`` and ``fields`` options. To load only type 0 and 4 particles and only the masses and velocities, you would use::

  sn = gadet.Snapshot('./snapdir_050/snap_050.0.hdf5', parttype=[0,4],  fields =['mass','Velocities'])

If the snapshot is a multipart file, by default only one file is loaded at a time. Use the option ``combineFiles`` to load all files at once. Otherwise you can  specific the file number using the ``filenumber`` option or by providing the correct path. The method ``nextFile`` iterates over the individual files of a snapshot.

Working with a Snapshot
***********************

To obtain a quick overview what was loaded use::

  print(sn)
  
The header attributes can be accessed via::

  sn.header.time
  #or
  sn.time

The data of a single particle type can be retrieved through::

  sn.part0.Masses
  #or
  sn.part0['Masses']

If you want to access a field for all particles, you may use::

  sn.Masses
  sn['Masses']
  
Note: ``Masses`` only contains the masses of particle types which have a zero in the Mass Table  (``sn.header.MassTable``). In ipython, tab completion is provided. 

You can not exchange these arrays, but have to write into them, i.e.::

  sn.Masses = np.ones(2*128**3) #wrong
  sn.Masses[:] = 1. #ok


Loading a Subfind Catalog
*************************

For subfind catalogs a similar interface is provided. A catalog can be loaded using the Subfind class::

   sub = gadget.Subfind('./groups_135/fof_subhalo_tab_135.0.hdf5')

data can be accessed in a similar way::

  sub.group.GroupCM
  #or
  sub.GroupCM
  
  sub.subhalo.SubhaloCM 
  #or
  sub.SubhaloCM


Using filters
*************

While loading a snapshot, the ``filter`` option provides a powerful way of loading only a subset of the data. For example only a spherical or rectangular region can be loaded. The following example shows how to load only one friends-of-friends halo::

  # select the Halo; sub is a subhalo catalog (see above)
  #    you need at least the following fields: fields=['GroupNsubs','SubhaloLenType','GroupLenType']
  #    and combineFiles = True
  # and load only real star particles
  # the Halo filter must be first, then the Stars filter
  filter = [gadget.filter.Halo(sub,halo=0), gadget.filter.Stars() ]

  # and load the data
  sn = gadget.Snapshot('./snapdir_135/snap_135.0.hdf5', parttype=[4], filter=filter, combineFiles=True, verbose=True)


Simple analysis and plotting routines
*************************************
The ``Snapshot`` class provides only loading functionality. All methods of that class load the data as it is (except for optional conversion to double precision). Additional data manipulation and analysis functionality is provided through subclasses. The ``Simulation`` subclass contains a few basic analysis and plotting methods:: 

  sn = gadget.Simulation("snap_003.hdf5")

  # A slice through the Voronoi mesh
  sn.plot_Aslice("rho")

  # A slice through an AMR mesh
  sn.plot_AMRslice("rho")
  
  #obtaining a radial profile
  sn.get_radprof("u")
  
  #plotting the profile right away
  sn.plot_radprof("u")

Initial Conditions
******************

Initial conditions can be created using the ``ICs`` class. An empty snapshot is created, which can be filled and written to a file. All default fields are present. Additional fields can be added using the ``addField`` method::

  #create ICS with 128**3 gas cells and 128**3 DM particles
  ics = gadget.ICs("ics.hdf5", [128**3,128**3])
  
  #write into an array
  ics.ids[:] = np.arange(2*128**3) + 1
  
  ics.write()

