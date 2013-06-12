import numpy as np

shortnames = {
    "pos":"position",
    "u":"internalenergy",
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

hdf5toformat2 = {'Coordinates':'pos',
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
        'Vorticity' : 'vort'}

def isPresent(name, gr , snapshot, learn=False, shape=1):
    if learn:
        pres = np.zeros(6,dtype=np.longlong)
        pres[gr] = shape
          
        if name == 'mass':
            pres[gr] = (0 if snapshot.masses[gr]>0  else 1)
        if not hasattr(snapshot,"__present__"):
            snapshot.__present__ = {}
            
        old = snapshot.__present__.get(name,np.zeros(6,dtype=np.longlong))
        pres = np.maximum(old,pres)
        snapshot.__present__[name] = pres
        
    else:
        pres = snapshot.__present__[name]
        
    return pres

headerfields = [
    'nparticles', 
    'nparticlesall', 
    'nparticlesallhighword',
    'nfiles',
    'masses',
    'time',
    'redshift',
    'boxsize',
    'omega0',
    'omegalambda',
    'hubbleparam',
    'flag_sfr',
    'flag_feedback',
    'flag_cooling',
    'flag_stellarage',
    'flag_metals',
    'flag_entropy_instead_u',
    'flag_doubleprecision',
    'flag_lpt_ics',
    
    'ngroups',
    'ngroupsall',
    'nids',
    'nidsall',
    'nsubhalos',
    'nsubhalosall'
]
