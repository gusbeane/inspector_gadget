import numpy as np

shortnames = {
    "pos":"position",
    "u":"internalenergy",
    "rho":"density",
    "vol":"volume",
    "hsml":"smoothinglength",
    "vel":"velocity",
    "pres":"pressure",
    "acce":"acceleration",
    "pot":"potential",
    "grar":"densitygradient",
    "grav":"velocitygradient",
    "grap":"pressuregradient",
    "temp":"temperature"
}

rev_shortnames = dict((v,k) for k, v in shortnames.iteritems())

hdf5toformat2 = {
    'Coordinates':'pos',
    'Velocities':'vel',
    'InternalEnergy':'u',
    'ParticleIDs' :'id',
    'Masses' : 'mass',
    'InternalEnergy' : 'u',
    'Density' :'rho',
    'Volume' : 'vol',
    'Pressure' : 'pres',
    'SmoothingLength' : 'hsml',
    'StarFormationRate' :'sfr',
    'StellarFormationTime' : 'age',
    'Metallicity' : 'z',
    'Potential' :'pot',
    'Acceleration' : 'acce',
    'TimeStep' : 'tstp',
    'MagneticField' : 'bfld',
    'DivBCleening' : 'psi',
    'SmoothedMagneticField' : 'bfsm',
    'RateOfChangeOfMagneticField' : 'dbdt',
    'DivergenceOfMagneticField' : 'divb',
    'PressureGradient' : 'grap',
    'DensityGradient' : 'grar',
    'VelocityGradient' : 'grav',
    'Center-of-Mass' : 'cmce',
    'Surface Area' : 'area',
    'Number of faces of cell' : 'nfac',
    'VertexVelocity' : 'veve',
    'Divergence of VertexVelocity' : 'divv',
    'Temperature' : 'temp',
    'Vorticity' : 'vort'
    }

rev_hdf5toformat2 = dict((v,k) for k, v in hdf5toformat2.iteritems())

default = [
    'id',
    'pos',
    'vel',
    'u',
    'mass'
    ]

present = {
    'id' : np.array([1,1,1,1,1,1]),
    'pos' : np.array([3,3,3,3,3,3]),
    'vel' : np.array([3,3,3,3,3,3]),
    'u'  : np.array([1,0,0,0,0,0]),           
    }

legacy_header_names = {
    'nparticles': 'NumPart_ThisFile',
    'masses': 'MassTable',
    'num_files': 'NumFilesPerSnapshot',
    'flag_sfr': 'Flag_Sfr',
    'flag_cooling': 'Flag_Cooling',
    'flag_feedback': 'Flag_Feedback',
    'flag_stellarage': 'Flag_StellarAge',
    'flag_metals': 'Flag_Metals',
    'ngroups': 'Ngroups_ThisFile',
    'ngroupsall': 'Ngroups_Total',
    'nids': 'Nids_ThisFile',
    'nidsall': 'Nids_Total',
    'nsubgroups': 'Nsubgroups_ThisFile',
    'nsubgroupsall': 'Nsubgroups_Total',
    'NumFiles': 'NumFilesPerSnapshot',
    'time': 'Time',
    'redshift': 'Redshift',
    'boxsize': 'BoxSize',
    'omega0': 'Omega0',
    'omegalambda': 'OmegaLambda',
    'hubbleparam': 'HubbleParam',
    'flag_doubleprecision': 'Flag_DoublePrecision'
    }

