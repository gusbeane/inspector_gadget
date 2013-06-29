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

headerfields = [
    'nparticles', 
    'nparticlesall', 
    'nparticlesallhighword',
    'num_files',
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

def normalizeName(name):
    name = hdf5toformat2.get(name,name)
    name = rev_shortnames.get(name,name)
    return name
        
        
def isPresent(name, snapshot, learn=False, gr=None, shape=1):
    if learn:
        if not hasattr(snapshot,"__present__"):
            snapshot.__present__ = present
            
        if gr !=None:
            pres = np.zeros(6,dtype=np.int64)
            pres[gr] = shape
            if name == 'mass':
                pres[gr] = (0 if snapshot.masses[gr]>0  else 1)
            old = snapshot.__present__.get(name,np.zeros(6,dtype=np.longlong))
            pres = np.maximum(old,pres)
            snapshot.__present__[name] = pres
        else:
            shape = np.array(shape)
            if shape.shape!=(6,):
               raise Exception("invalide present array")
            pres = shape
            snapshot.__present__[name] = pres
    else:
        try:
            pres = snapshot.__present__[name]
        except KeyError:
            raise Exception("Unkonwn array shape for field %s"%name)
    
    #filter for loaded particle types
    tmp = np.zeros(6,dtype=np.longlong)
    tmp[snapshot.__parttype__] = pres[snapshot.__parttype__]
    return tmp


