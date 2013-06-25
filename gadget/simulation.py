import numpy as np
import os.path as path
import pylab
import matplotlib
import struct
import time
import scipy.linalg


from gadget.loader import Snapshot
import gadget.calcGrid as calcGrid


class Simulation(Snapshot):
    
    def __init__(self,filename, format=None, fields=None, parttype=None, **param):
        
        param['combineParticles'] = True
        super(Simulation,self).__init__(filename, format=format, fields=fields, parttype=parttype,**param)
        

        center = np.ones(3) * 0.5 * self.boxsize
        self.set_center(center)
        
        self.twodim = False
        if np.abs( self.data['pos'][:,2] ).max() == 0.:
            self.center[2] = 0.
            self.twodim = True
        
        
        
        
    def set_center( self, center ):
        if type( center ) == list:
            self.center = pylab.array( center )
        elif type( center ) == np.ndarray:
            self.center = center
        else:
            raise Exception( "center has to be of type list or numpy.ndarray" )
        return
    
    """
    
    def cosmology_init( self ):
        self.cosmo = CosmologicalFactors( my_h = self.hubbleparam, my_OmegaMatter = self.omega0, my_OmegaLambda = self.omegalambda )
        self.cosmo.SetLookbackTimeTable()
        self.transform_to_physical_units()
        return
    
    def cosmology_get_lookback_time_from_a( self, a ):
        return self.cosmo.LookbackTime_a_in_Gyr( a )

    def transform_to_physical_units( self ):
        self.pos *= self.time / self.hubbleparam
        self.vel *= sqrt( self.time )
        self.data['rho'] /= self.time**3 * self.hubbleparam**2
        self.data['vol'] *= self.time**3 / self.hubbleparam**3
        if self.data.has_key( 'pres' ):
            self.data['pres'] /= self.time**3 * self.hubbleparam**2
        self.data['mass'] /= self.hubbleparam
        self.masses /= self.hubbleparam
        self.center *= self.time / self.hubbleparam
        if self.data.has_key( 'bfld' ):
            self.data['bfld'] /= self.time**2
        return

    def calc_mean_a( self, speciesfile="../species.txt" ):
        sp = loaders.load_species( speciesfile )

        if sp['count'] != self.nspecies:
            print "Number of species in speciesfile (%d) and snapshot (%d) don't match." % (sp['count'],self.nspecies)

        self.data['mean_a'] = np.zeros( self.nparticlesall[0] )
        for i in range( self.nspecies ):
            self.data['mean_a'] += self.data['xnuc'][:,i] * sp['na'][i]
        return

    def calc_sf_indizes( self, sf, verbose=False ):
        halo_indizes = np.zeros( self.npartall, dtype='int32' )
        
        if verbose:
            print "Total particles per type:", self.nparticlesall

        idx = 0
        for ptype in range( 6 ):
            if verbose:
                print "Particle type %d starts at index %d." % (ptype, idx)
            for halo in range( sf.nsubgroups ):
                halo_indizes[ idx ] = halo
                idx += 1

        self.data['halo'] = halo_indizes
        return


        
    def r( self, center=False ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        radius = pylab.sqrt( (self.data[ "pos" ][:,0]-center[0])**2+(self.data[ "pos" ][:,1]-center[1])**2+(self.data[ "pos" ][:,2]-center[2])**2 )
        return radius

    def rid( self ):
        rids = pylab.zeros( self.npart+1, dtype='int32' )
        rids[0] = -1
        rids[1:] = self.data['id'].argsort()
        return rids

    def get_principal_axis( self, idx, L=None ):
        tensor = pylab.zeros( (3,3) )

        mass = self.data['mass'][idx]
        px = self.pos[idx,0]
        py = self.pos[idx,1]
        pz = self.pos[idx,2]
        
        tensor[0,0] = (mass * (py*py + pz*pz)).sum()
        tensor[1,1] = (mass * (px*px + pz*pz)).sum()
        tensor[2,2] = (mass * (px*px + py*py)).sum()

        tensor[0,1] = - (mass * px * py).sum()
        tensor[1,0] = tensor[0,1]
        tensor[0,2] = - (mass * px * pz).sum()
        tensor[2,0] = tensor[0,2]
        tensor[1,2] = - (mass * py * pz).sum()
        tensor[2,1] = tensor[1,2]

        eigval, eigvec = scipy.linalg.eig( tensor )

        if L == None:
            maxval = eigval.argsort()[-1]
            return eigvec[:,maxval]
        else:
            A1 = (L * eigvec[:,0]).sum()
            A2 = (L * eigvec[:,1]).sum()
            A3 = (L * eigvec[:,2]).sum()

            A = np.abs( np.array( [A1, A2, A3] ) )
            i, = np.where( A == A.max() )
            xdir = eigvec[:,i[0]]

            if (xdir * L).sum() < 0:
                xdir *= -1.0

            j, = np.where( A != A.max() )
            i2 = eigval[j].argsort()
            ydir = eigvec[:,j[i2[1]]]
            
            if ydir[0] < 0:
                ydir *= -1.0

            zdir = np.cross( xdir, ydir )

            return xdir, ydir, zdir
    """
    def centerat( self, center ):
        self.data['pos'] -= pylab.array( center )[None,:]
        self.center = pylab.zeros( 3 )
        return
    """    
    def rotateto( self, dir, dir2=None, dir3=None, verbose=False ):
        if dir2 == None or dir3 == None:
            # get normals
            dir2 = pylab.zeros( 3 )
            if dir[0] != 0 and dir[1] != 0:
                dir2[0] = -dir[1]
                dir2[1] = dir[0]
            else:
                dir2[0] = 1
            dir2 /= sqrt( (dir2**2).sum() )
                dir3 = np.cross( dir, dir2 )

        matrix = pylab.array( [dir,dir2,dir3] )

            for value in self.data.keys():
            if self.data[value].ndim == 2 and pylab.shape( self.data[value] )[1] == 3:
                self.rotate_value( value, matrix )
                if verbose:
                    print "Rotated %s." % value
        self.convenience()
        return
    
    def rotate_value( self, value, matrix ):
        new_value = pylab.zeros( pylab.shape(self.data[value]) )
        for i in range( 3 ):
            new_value[:,i] = (self.data[value] * matrix[i,:][None,:]).sum(axis=1)
        self.data[value] = new_value
        return


    def select_halo( self, sf, haloid=0, remove_bulk_vel=True, galradfac=0.1, rotate_disk=True, use_principal_axis=True, verbose=True ):
        self.centerat( sf.data['fpos'][haloid,:] )
        
        galrad = galradfac * sf.data['frc2'][haloid]

        if remove_bulk_vel:
            iall, = np.where( self.r() < galrad )
            mass = self.data['mass'].astype('float64')
            vel = (self.vel[iall,:] * mass[iall][:,None]).sum(axis=0) / mass[iall].sum()
            self.vel -= vel[None,:]

        if rotate_disk:
            istars, = np.where( (self.r() < galrad) & (self.type == 4) & (self.data['halo'] == haloid) )
            print "Found %d stars." % np.size(istars)
            mass = self.data['mass'].astype('float64')
            L = np.cross( self.pos[istars,:].astype('float64'), (self.vel[istars,:].astype('float64') * mass[istars][:,None]) )
            Ltot = L.sum( axis=0 )
            Ldir = Ltot / sqrt( (Ltot**2).sum() )
            
            if use_principal_axis:
                xdir, ydir, zdir = self.get_principal_axis( istars, L=Ldir )
                self.rotateto( xdir, dir2=ydir, dir3=zdir, verbose=verbose )
                return np.array( [xdir, ydir, zdir] )
            else:
                dir = Ldir
                self.rotateto( dir, verbose=verbose )
                return dir
            return
        
        return False
    """
    def get_slice( self, value, box=[0,0], nx=200, ny=200, center=False, axes=[0,1], group=None):
        if group == None:
            group = self
            
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        
        dim0 = axes[0]
        dim1 = axes[1]
        
        if (box[0] == 0 and box[1] == 0):
            box[0] = max( abs( group.data[ "pos" ][:,dim0] ) ) * 2
            box[1] = max( abs( group.data[ "pos" ][:,dim1] ) ) * 2

        if (value == "mass"):
            return calcGrid.calcDensSlice( group.data["pos"].astype('float64'), group.data["hsml"].astype('float64'), group.data[value].astype('float64'), nx, ny, box[0], box[1], center[0], center[1], center[2], dim0, dim1 )
        else:
            return calcGrid.calcSlice( group.data["pos"].astype('float64'), group.data["hsml"].astype('float64'), group.data["mass"].astype('float64'), group.data["rho"].astype('float64'), group.data[value].astype('float64'), nx, ny, box[0], box[1], center[0], center[1], center[2], dim0, dim1 )

    def get_raddens( self, nshells=200, dr=0, center=False, group=None):
        if group == None:
            group = self
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        return calcGrid.calcRadialProfile( group.data['pos'].astype('float64'), group.data["mass"].astype('float64'), 1, nshells, dr, center[0], center[1], center[2] )

    def get_radprof( self, value, nshells=200, dr=0, center=False, mode=2, group=None ):
        if group == None:
            group = self
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        return calcGrid.calcRadialProfile( group.data['pos'].astype('float64'), group.data[value].astype('float64'), mode, nshells, dr, center[0], center[1], center[2] )
    
    def get_radmassprof( self, nshells, dr=0, center=False, solarmass=False, group=None):
        if group == None:
            group = self
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center        

        p = calcGrid.calcRadialProfile( group.data['pos'].astype('float64'), group.data["mass"].astype('float64'), 0, nshells, dr, center[0], center[1], center[2] )
        for i in range( 1, nshells ):
            p[0,i] += p[0,i-1]
        if solarmass:
            p[0,:] /= 1.989e33
        return p

    def plot_raddens( self, log=False, nshells=200, dr=0, center=False, color='k', group=None):
        if group == None:
            group = self
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center    
        
        p = self.get_raddens( nshells, dr, center, group )
        if log:
            pylab.semilogy( p[1,:], p[0,:], color )
        else:
            pylab.plot( p[1,:], p[0,:], color )

    def plot_radprof( self, value, log=False, nshells=200, dr=0, center=False, color='k', mode=2, group=None ):
        if group == None:
            group = self
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center    
        
        p = self.get_radprof( value, nshells, dr, center, mode=mode,group=group )
        if log:
            pylab.semilogy( p[1,:], p[0,:], color )
        else:
            pylab.plot( p[1,:], p[0,:], color )

    def plot_radmassprof( self, log=False, nshells=200, dr=0, center=False, color='k', solarmass=False, group=None ):
        if group == None:
            group = self
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        
        p = self.get_radmassprof( nshells, dr, center, solarmass,group )
        if log:
            pylab.semilogy( p[1,:], p[0,:], color )
        else:
            pylab.plot( p[1,:], p[0,:], color )
    """  
    def plot_radvecprof( self, value, log=False, nshells=200, dr=0, center=False, color='k' ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        
        vec = (self.data[value]*self.data['pos']).sum(axis=1)/self.r()
        p = calcGrid.calcRadialProfile( self.pos.astype('float64'), vec.astype('float64'), 2, nshells, dr, center[0], center[1], center[2] )
        if log:
            pylab.semilogy( p[1,:], p[0,:], color )
        else:
            pylab.plot( p[1,:], p[0,:], color )
    """
    def plot_pos( self, axes=[0,1], group=None ):
        if group == None:
            group = self
        pylab.plot( group.data['pos'][:,axes[0]], group.data['pos'][:,axes[1]], ',' )
        pylab.axis( "scaled" )

    """
    def print_abundances( self ):
        print "Total abundances in solar masses:"
        for i in range( self.nspecies ):
            print "Species %d: %g" % (i,(self.data['mass'][:self.nparticlesall[0]].astype('float64') * self.data['xnuc'][:,i]).sum()/msol)
        return
    """

    def plot_slice( self, value, logplot=True, colorbar=False, box=[0,0], nx=200, ny=200, center=False, axes=[0,1], minimum=1e-8, newfig=True, nolabels=False, cmap=False, vrange=False, rasterized=True, cblabel=False, logfm=True ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        
        dim0 = axes[0]
        dim1 = axes[1]
        
        if (box[0] == 0 and box[1] == 0):
            box[0] = max( abs( self.data[ "pos" ][:,dim0] ) ) * 2
            box[1] = max( abs( self.data[ "pos" ][:,dim1] ) ) * 2

        slice = self.get_slice( value, box, nx, ny, center, axes )
        x = (pylab.array( range( nx+1 ) ) - nx/2.) / nx * box[0]
        y = (pylab.array( range( ny+1 ) ) - ny/2.) / ny * box[1]

        if newfig:
            fig = pylab.figure( figsize = ( 13, int(12*box[1]/box[0] + 0.5) ) )
            
        if cmap:
            pylab.set_cmap( cmap )

        if logplot:
            slice = pylab.maximum( slice, minimum )

        if not vrange:
            vrange = [ slice.min(), slice.max() ]

        if logplot:
            if logfm:
                pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.LogNorm(vmin=vrange[0], vmax=vrange[1]), rasterized=rasterized )
            else:
                pc = pylab.pcolormesh( x, y, pylab.transpose( pylab.log10(slice) ), shading='flat', rasterized=rasterized )
        else:
            pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', rasterized=rasterized )
        
        if colorbar:
            if logplot:
                cb = pylab.colorbar( format=matplotlib.ticker.LogFormatterMathtext() )
            else:
                fmt = matplotlib.ticker.ScalarFormatter( useMathText=True )
                fmt.set_powerlimits( (-2, 2) )
                fmt.set_useOffset( False )
                cb = pylab.colorbar( format=fmt )
            if cblabel:
                    cb.set_label( cblabel )
        
        pylab.axis( "image" )

        if not nolabels:
            xticklabels = []
            for tick in pc.axes.get_xticks():
                if (tick == 0):
                    xticklabels += [ r'$0.0$' ]
                else:
                    xticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_xticklabels( xticklabels, size=16, y=-0.1, va='baseline' )

            yticklabels = []
            for tick in pc.axes.get_yticks():
                if (tick == 0):
                    yticklabels += [ r'$0.0$' ]
                else:
                    yticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_yticklabels( yticklabels, size=16, ha='right' )
        return pc

    def plot_cylav( self, value, logplot=True, box=[0,0], nx=512, ny=512, center=False, minimum=1e-8 ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        
        if (box[0] == 0 and box[1] == 0):
            box[0] = max( abs( self.data[ "pos" ][:,0] ) ) * 2
            box[1] = max( abs( self.data[ "pos" ][:,1:] ) ) * 2

        grid = calcGrid.calcGrid( self.pos.astype('float64'), self.data["hsml"].astype('float64'), self.data["mass"].astype('float64'), self.data["rho"].astype('float64'), self.data[value].astype('float64').astype('float64'), nx, ny, ny, box[0], box[1], box[1], 0, 0, 0 )
        cylav = calcGrid.calcCylinderAverage( grid )
        x = (pylab.array( range( nx+1 ) ) - nx/2.) / nx * box[0]
        y = (pylab.array( range( ny+1 ) ) - ny/2.) / ny * box[1]

        fig = pylab.figure( figsize = ( 13, int(12*box[1]/box[0] + 0.5) ) )
        pylab.spectral()
        
        if logplot:
            pc = pylab.pcolor( x, y, pylab.transpose( pylab.log10( pylab.maximum( cylav, minimum ) ) ), shading='flat' )
        else:
            pc = pylab.pcolor( x, y, pylab.transpose( slice ), shading='flat' )

        pylab.axis( "image" )
        xticklabels = []
        for tick in pc.axes.get_xticks():
            if (tick == 0):
                xticklabels += [ r'$0.0$' ]
            else:
                xticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
        pc.axes.set_xticklabels( xticklabels, size=16, y=-0.1, va='baseline' )

        yticklabels = []
        for tick in pc.axes.get_yticks():
            if (tick == 0):
                yticklabels += [ r'$0.0$' ]
            else:
                yticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
        pc.axes.set_yticklabels( yticklabels, size=16, ha='right' )
        return pc

    def getbound( self, center=False, vel=[0,0,0] ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center
        
        start = time.time()
        radius = pylab.zeros( self.npartall )
        for i in range( 3 ):
            radius += (self.data[ "pos" ][:,i] - center[i])**2
        radius = pylab.sqrt( radius )
        rs = radius.argsort()
        
        mass = 0.
        bcount = 0.
        bmass = 0.
        bcenter = [0., 0., 0.]
        bparticles = []
        for part in range( self.npart ):
            if (part == 0) or (( ( self.vel[rs[part],:] - vel )**2. ).sum() < 2.*G*mass/radius[rs[part]]):
                bcount += 1.
                bmass += self.data['mass'][rs[part]]
                bcenter += self.pos[rs[part],:]
                bparticles += [self.id[rs[part]]]
            mass += self.data['mass'][rs[part]]
        
        bobject = {}
        bobject['mass'] = bmass
        bobject['center'] = bcenter / bcount
        bobject['count'] = bcount
        bobject['particles'] = bparticles
        
        print "Calculation took %gs." % (time.time()-start)
        return bobject

    def mapOnCartGrid( self, value, center=False, box=False, res=512, saveas=False ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        if type( box ) == list:
            box = pylab.array( box )
        elif type( box ) != np.ndarray:
            box = np.array( [self.boxsize,self.boxsize,self.boxsize] )

        pos = self.pos[:self.nparticlesall[0],:].astype( 'float64' )
        px = np.abs( pos[:,0] - center[0] )
        py = np.abs( pos[:,1] - center[1] )
        pz = np.abs( pos[:,2] - center[2] )

        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < 0.5*box[2]) )
        print "Selected %d of %d particles." % (pp.size,self.npart)

        posdata = pos[pp]
        valdata = self.data[value][pp].astype('float64')

        data = calcGrid.calcASlice( posdata, valdata, nx=res, ny=res, nz=res, boxx=box[0], boxy=box[1], boxz=box[2], 
                        centerx=center[0], centery=center[1], centerz=center[2], grid3D=True )
        
        grid = data[ "grid" ]
        if saveas:
            grid.tofile( saveas )

        return grid

    def get_Aslice( self, value, grad=False, res=1024, center=False, axes=[0,1], box=False ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        if type( box ) == list:
            box = pylab.array( box )
        elif type( box ) != np.ndarray:
            box = np.array( [self.boxsize,self.boxsize] )
        
        axis0 = axes[0]
        axis1 = axes[1]

        c = pylab.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        pos = self.pos.astype( 'float64' )
        px = np.abs( pos[:,axis0] - c[0] )
        py = np.abs( pos[:,axis1] - c[1] )
        pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

        zdist = 2. * self.data['vol'].astype('float64')**(1./3.)
        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < zdist) )
        print "Selected %d of %d particles." % (pp.size,self.npart)

        posdata = pos[pp,:]
        valdata = self.data[value][pp].astype('float64')
        if not grad:
            data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1 )
        else:
            data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, self.data[grad][pp].astype('float64') )
        data[ "neighbours" ] = pp[ data["neighbours"] ]
        
        return data

    def plot_Aslice( self, value, grad=False, logplot=False, colorbar=False, contour=False, res=1024, center=False, axes=[0,1], minimum=1e-8, newfig=True, newlabels=False, cmap=False, vrange=False, cblabel=False, rasterized=False, box=False, proj=False ):
        if type( center ) == list:
            center = pylab.array( center )
        elif type( center ) != np.ndarray:
            center = self.center

        if type( box ) == list:
            box = pylab.array( box )
        elif type( box ) != np.ndarray:
            box = np.array( [self.boxsize,self.boxsize] )
        
        axis0 = axes[0]
        axis1 = axes[1]

        c = pylab.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        pos = self.pos.astype( 'float64' )
        px = np.abs( pos[:,axis0] - c[0] )
        py = np.abs( pos[:,axis1] - c[1] )
        pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

        zdist = 2. * self.data['vol'].astype('float64')**(1./3.)
        if proj:
            zdist[:] = 0.8 * box.max()

        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < zdist) )
        print "Selected %d of %d particles." % (pp.size,self.npart)

        posdata = pos[pp,:]
        valdata = self.data[value][pp].astype('float64')
        if not grad:
            data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, proj=proj )
        else:
            graddata = self.data[grad][pp,:].astype('float64')
            data = calcGrid.calcASlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, graddata, proj=proj )
        slice = data[ "grid"]
        if (not proj):
            neighbours = data[ "neighbours" ]
            contours = data[ "contours" ]
        x = pylab.arange( res+1, dtype="float64" ) / res * box[0] - .5 * box[0] + c[0]
        y = pylab.arange( res+1, dtype="float64" ) / res * box[1] - .5 * box[1] + c[1]

        if newfig:
            fig = pylab.figure()

        if cmap:
            pylab.set_cmap( cmap )

        if logplot:
            slice = pylab.maximum( slice, minimum )

        if not vrange:
            vrange = [ slice.min(), slice.max() ]
        
        if logplot:
            pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', norm=matplotlib.colors.LogNorm(vmin=vrange[0], vmax=vrange[1]), rasterized=rasterized )
        else:
            pc = pylab.pcolormesh( x, y, pylab.transpose( slice ), shading='flat', rasterized=rasterized, vmin=vrange[0], vmax=vrange[1] )
        
        if colorbar:
            if logplot:
                cb = pylab.colorbar( format=matplotlib.ticker.LogFormatterMathtext() )
            else:
                fmt = matplotlib.ticker.ScalarFormatter( useMathText=True )
                fmt.set_powerlimits( (-2, 2) )
                fmt.set_useOffset( False )
                cb = pylab.colorbar( format=fmt )
            if cblabel:
                cb.set_label( cblabel )
        
        if contour and not proj:
            x = ( pylab.arange( res, dtype="float64" ) + 0.5 ) / res * box[0] - .5 * box[0] + center[0]
            y = ( pylab.arange( res, dtype="float64" ) + 0.5 ) / res * box[1] - .5 * box[1] + center[1]
            pylab.contour( x, y, pylab.transpose( contours ), levels=[0.99], colors="w" )

        pylab.axis( "image" )

        if newlabels:
            xticklabels = []
            for tick in pc.axes.get_xticks():
                if (tick == 0):
                    xticklabels += [ r'$0.0$' ]
                else:
                    xticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_xticklabels( xticklabels, size=24, y=-0.1, va='baseline' )

            yticklabels = []
            for tick in pc.axes.get_yticks():
                if (tick == 0):
                    yticklabels += [ r'$0.0$' ]
                else:
                    yticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(ceil(log10(abs(tick)))), ceil(log10(abs(tick)))) ]
            pc.axes.set_yticklabels( yticklabels, size=24, ha='right' )
        return
