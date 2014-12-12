import numpy as np
import matplotlib.pyplot as p
import matplotlib
import time

from gadget.loader import Snapshot
import gadget.calcGrid as calcGrid


class Simulation(Snapshot):
    
    def __init__(self,filename, format=None, fields=None, parttype=None, **param):
        param['combineParticles'] = True
        super(Simulation,self).__init__(filename, format=format, fields=fields, parttype=parttype,**param)
        

        self.numdims = 3
        self.twodim = False
        self.onedim = False
                
        if hasattr(self,'pos'):
            if np.abs( self['pos'][:,2] ).max() == 0.:
                self.twodim = True
                self.numdims = 2
            if self.twodim and np.abs( self['pos'][:,1] ).max() == 0.:
                self.onedim = True
                self.numdims = 1
            
        self.set_center(None)
        
    def __validate_vector__(self, vector, default, len=None):
        if len == None:
            len = self.numdims

        if vector == None:
            if type(default) == np.ndarray or type(default) == list:
                default = default[:len]
            vector = default
        
        v = np.zeros(3)
        
        if len == 0:
            return v
        
        v[:len] = vector
        
        return v
        
    def set_center(self, center):
        self.center = self.__validate_vector__(center, self.boxsize/2)
        return
    
    def r(self, center=None, periodic=True, group=None):
        if group == None:
            group = self
            
        center = self.__validate_vector__(center, self.center)

        dx = group["pos"][:,0]-center[0]
        dy = group["pos"][:,1]-center[1]
        dz = group["pos"][:,2]-center[2]
        
        if periodic:
            dx = np.where(dx > self.boxsize/2, dx-self.boxsize/2,dx)
            dx = np.where(dx < -self.boxsize/2, dx+self.boxsize/2,dx)
            dy = np.where(dy > self.boxsize/2, dy-self.boxsize/2,dy)
            dy = np.where(dy < -self.boxsize/2, dy+self.boxsize/2,dy)
            dz = np.where(dz > self.boxsize/2, dz-self.boxsize/2,dz)
            dz = np.where(dz < -self.boxsize/2, dz+self.boxsize/2,dz)
            
        radius = np.sqrt(dx**2+dy**2+dz**2)

        return radius
    
    def centerat(self, center, group=None):
        if group == None:
            group = self
            
        center = self.__validate_vector__(center, self.center)
        
        group['pos'] -= center[None,:]
        self.center = np.zeros( 3 )
        
        return
    
    def __get_radhist__(self, value, center=None, bins=100, range=None, log=False, periodic=True, group=None):
        if group == None:
            group = self

        center = self.__validate_vector__(center, self.center)
        
        radius = self.r(center=center, periodic=periodic, group=group)
        
        if type(range) == list:
            range = np.array( range )    
        if range == None:
            range = np.array([np.min(radius),np.max(radius)])
            
        if log:
            logbins = np.linspace(np.log(range[0]), np.log(range[1]), bins+1)
            xbins = np.exp(logbins)
            logpos = 0.5 * (logbins[:-1] + logbins[1:])
            xpos = np.exp(logpos)
        else:
            xbins = np.linspace(range[0], range[1], bins+1)
            xpos = 0.5 * (xbins[:-1] + xbins[1:])
            
        if value == None:
            val = None
        else:
            val = group[value].astype('float64')
            
        profile, xbins = np.histogram(radius,bins=xbins,weights=val)
        
        return (profile, xpos, xbins, range)

    def get_raddens(self, value='mass', center=None, bins=100, range=None, log=False, periodic=True, group=None):
        (profile, xpos, xbins, range) = self.__get_radhist__(value=value, center=center, bins=bins ,range=range, log=log, periodic=periodic, group=group)
        profile /= 4./3*np.pi * (xbins[1:]**3-xbins[:1]**3)
        
        return (profile,xpos)
    
    def get_radprof(self, value, weights=None, center=None, bins=100, range=None, log=False, periodic=True, group=None):
        (profile, xpos, xbins, range) = self.__get_radhist__(value=value, center=center, bins=bins ,range=range, log=log, periodic=periodic, group=group)
        (norm, xpos, xbins, range) = self.__get_radhist__(value=weights, center=center, bins=bins ,range=range, log=log, periodic=periodic, group=group)
        
        profile /= norm
        
        return (profile,xpos)
    
    def plot_raddens(self, value='mass', center=None, bins=100, range=None, log=False, periodic=True, group=None, **params):
        (profile, xpos) = self.get_raddens(value=value, center=center, bins=bins, range=range, log=log, periodic=periodic, group=group)
        if log:
            p.loglog(xpos, profile, **params)
        else:
            p.plot(xpos, profile, **params)

    def plot_radprof(self, value, weights=None, center=None, bins=100, range=None, log=False, periodic=True, group=None, **params):
        (profile, xpos) = self.get_radprof(value=value, weights=weights, center=center, bins=bins, range=range, log=log, periodic=periodic, group=group)
        if log:
            p.loglog(xpos,profile, **params)
        else:
            p.plot(xpos, profile, **params)


    def plot_pos(self, center=None, axis=[0,1], box=None, periodic=True, group=None, newfig=True, axes=None, **params):
        if group == None:
            group = self
               
        center = self.__validate_vector__(center,self.center)
        box = self.__validate_vector__(box,self.boxsize)
                
        if newfig and axes==None:
            fig = p.figure()
            axes = p.gca()
        elif axes==None:
            axes = p.gca()
            
        axis0 = axis[0]
        axis1 = axis[1]

        c = np.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        x = group["pos"][:,axis0]
        y = group["pos"][:,axis1]
        z = group["pos"][:,3 - axis0 - axis1]
        
        if periodic:
            x = np.where(x - c[0] > self.boxsize/2, x-self.boxsize/2,x)
            x = np.where(x - c[0] < -self.boxsize/2, x+self.boxsize/2,x)
            y = np.where(y - c[1] > self.boxsize/2, y-self.boxsize/2,y)
            y = np.where(y - c[1] < -self.boxsize/2, y+self.boxsize/2,y)
            z = np.where(z - c[2] > self.boxsize/2, z-self.boxsize/2,z)
            z = np.where(z - c[2] < -self.boxsize/2, z+self.boxsize/2,z)

        pp, = np.where( (np.abs(x-c[0]) <= 0.5*box[0]) & (np.abs(y-c[1]) <= 0.5*box[1]) & (np.abs(z-c[2]) <= 0.5*box[2]) )
        
        axes.scatter(x[pp], y[pp], **params)
        axes.axis( "scaled" )
        
    def get_Aslice( self, value, gradient=None, res=1024, center=None, axis=[0,1], box=None, group=None):
        if group == None:
            group = self
               
        center = self.__validate_vector__(center, self.center)
        box = self.__validate_vector__(box, self.boxsize,len=2)
            
        axis0 = axis[0]
        axis1 = axis[1]

        c = np.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        pos = group.pos.astype( 'float64' )
        px = np.abs( pos[:,axis0] - c[0] )
        py = np.abs( pos[:,axis1] - c[1] )
        pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

        zdist = 2. * group['vol'].astype('float64')**(1./3.)

        pp, = np.where( (px <= 0.5*box[0]) & (py <= 0.5*box[1]) & (pz <= zdist) )
        print "Selected %d of %d particles." % (pp.size,self.npart)

        posdata = pos[pp,:]
        valdata = group[value][pp].astype('float64')
        
        if gradient==None:
            data = calcGrid.calcASlice(posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, boxz=box[2])
        else:
            data = calcGrid.calcASlice(posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, group[gradient][pp].astype('float64'), boxz=box[2])
        data[ "neighbours" ] = pp[ data["neighbours"] ]
        
        data['name'] = value
        data['x'] = np.arange( res+1, dtype="float64" ) / res * box[0] - .5 * box[0] + c[0]
        data['y'] = np.arange( res+1, dtype="float64" ) / res * box[1] - .5 * box[1] + c[1]
        data['x2'] = (np.arange( res, dtype="float64" ) + 0.5) / res * box[0] - .5 * box[0] + center[0]
        data['y2'] = (np.arange( res, dtype="float64" ) + 0.5) / res * box[1] - .5 * box[1] + center[1]
        
        return data
    
    def plot_Aslice(self, value, gradient=None, log=False, res=1024, center=None, axis=[0,1], box=None, group=None, vmin=None, vmax=None, dvalue=None, dgradient=None, colorbar=True, cblabel=None, contour=False, newlabels=False, newfig=True, axes=None, **params):
        result = self.get_Aslice(value=value, gradient=gradient, res=res, center=center, axis=axis, box=box, group=group)
        
        dresult = None
        if dvalue != None:
            dresult = self.get_Aslice(value=dvalue, gradient=dgradient, res=res, center=center, axis=axis, box=box, group=group)
            
        self.__plot_Slice__(result,log=log, vmin=vmin, vmax=vmax, dresult=dresult, colorbar=colorbar, cblabel=cblabel, contour=contour, newlabels=newlabels, newfig=newfig, axes=axes, **params)

        
    def get_AMRslice(self, value, gradient=None, res=1024, center=None, axis=[0,1], box=None, group=None):
        if group == None:
            group = self.part0
               
        center = self.__validate_vector__(center, self.center)
        box = self.__validate_vector__(box, self.boxsize,len=2)
        
        axis0 = axis[0]
        axis1 = axis[1]

        c = np.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]
        

        domainlen = self.boxsize        
        domainc = np.zeros(3)
        domainc[0] = self.boxsize/2
        domainc[1] = self.boxsize/2.
        
        if self.numdims >2:
            domainc[2] = self.boxsize/2.

        posdata = group['pos'].astype('float64')
        valdata = group[value].astype('float64')
        
        if not gradient:
            data = calcGrid.calcAMRSlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, axis0, axis1, boxz=box[2])
        else:
            graddata = group[gradient].astype('float64')
            data = calcGrid.calcAMRSlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, axis0, axis1, graddata, boxz=box[2])
        
        data['name'] = value
        data['x'] = np.arange( res+1, dtype="float64" ) / res * box[0] - .5 * box[0] + c[0]
        data['y'] = np.arange( res+1, dtype="float64" ) / res * box[1] - .5 * box[1] + c[1]
        data['x2'] = (np.arange( res, dtype="float64" ) + 0.5) / res * box[0] - .5 * box[0] + center[0]
        data['y2'] = (np.arange( res, dtype="float64" ) + 0.5) / res * box[1] - .5 * box[1] + center[1]
        
        return data
    

    def plot_AMRslice(self, value, gradient=None, log=False, res=1024, center=None, axis=[0,1], box=None, group=None, vmin=None, vmax=None, dvalue=None, dgradient=None, colorbar=True, cblabel=None, contour=False, newlabels=False, newfig=True, axes=None, **params):
        result = self.get_AMRslice(value, gradient=gradient, res=res, center=center, axis=axis, box=box, group=group)
        
        dresult = None
        if dvalue != None:
            dresult = self.get_AMRslice(dvalue, gradient=dgradient, res=res, center=center, axis=axis, box=box, group=group)
            
        self.__plot_Slice__(result,log=log, vmin=vmin, vmax=vmax, dresult=dresult, colorbar=colorbar, cblabel=cblabel, contour=contour, newlabels=newlabels, newfig=newfig, axes=axes, **params)
        
        
    def get_SPHproj( self, value, hsml="hsml", weights=None, normalized=True, res=1024, center=None, axis=[0,1], box=None, group=None):
        if group == None:
            group = self
               
        center = self.__validate_vector__(center, self.center)
        box = self.__validate_vector__(box, self.boxsize)
            
        axis0 = axis[0]
        axis1 = axis[1]

        c = np.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        pos = group.pos.astype( 'float64' )
        px = np.abs( pos[:,axis0] - c[0] )
        py = np.abs( pos[:,axis1] - c[1] )
        pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

        pp, = np.where( (px <= 0.5*box[0]) & (py <= 0.5*box[1]) & (pz <= 0.5*box[2]) )
        print "Selected %d of %d particles." % (pp.size,self.npart)

        posdata = pos[pp,:]
        valdata = group[value][pp].astype('float64')
        hsmldata = group[hsml][pp].astype("float64")

        
        if weights==None:
            grid = calcGrid.calcGrid(posdata, hsmldata, valdata, res, res, res, box[0], box[1], box[2], c[0], c[1], c[2], proj=True, norm=normalized )
        else:
            weightdata = group[weights].astype("float64")
            grid = calcGrid.calcGrid(posdata, hsmldata, valdata, res, res, res, box[0], box[1], box[2], c[0], c[1], c[2], proj=True, norm=normalized, weights=weightdata )
            
        data = {}
        data['grid'] = grid
        
        data['name'] = value
        data['x'] = np.arange( res+1, dtype="float64" ) / res * box[0] - .5 * box[0] + c[0]
        data['y'] = np.arange( res+1, dtype="float64" ) / res * box[1] - .5 * box[1] + c[1]
        data['x2'] = (np.arange( res, dtype="float64" ) + 0.5) / res * box[0] - .5 * box[0] + center[0]
        data['y2'] = (np.arange( res, dtype="float64" ) + 0.5) / res * box[1] - .5 * box[1] + center[1]
        
        return data
    
    def plot_SPHproj(self, value, hsml="hsml", weights=None, normalized=True, log=False, res=1024, center=None, axis=[0,1], box=None, group=None, vmin=None, vmax=None, dvalue=None, dweights=None, colorbar=True, cblabel=None, contour=False, newlabels=False, newfig=True, axes=None, **params):
        result = self.get_SPHproj(value, hsml=hsml, weights=weights, normalized=normalized, res=res, center=center, axis=axis, box=box, group=group)
        
        dresult = None
        if dvalue != None:
            dresult = self.get_SPHproj(dvalue, hsml=hsml, weights=dweights, normalized=normalized, res=res, center=center, axis=axis, box=box, group=group)
            
        self.__plot_Slice__(result,log=log, vmin=vmin, vmax=vmax, dresult=dresult, colorbar=colorbar, cblabel=cblabel, contour=contour, newlabels=newlabels, newfig=newfig, axes=axes, **params)
        
        
    def __plot_Slice__(self, result, log=False, vmin=None, vmax=None, dresult=None, colorbar=True, cblabel=None, contour=False, newlabels=False, newfig=True, axes=None, **params):          
        slice = result['grid']
        x = result['x']
        y = result['y']
            
        
        if newfig and axes==None:
            fig = p.figure()
            axes = p.gca()
        elif axes==None:
            axes = p.gca()

        if vmin == None:
            vmin = np.min(slice)
            
        if vmax == None:
            vmax = np.max(slice)
            
        if dresult == None:
            dresult = result
        
        if log:
            pc = axes.imshow(slice.T, origin='lower', interpolation='nearest', extent=[x.min(), x.max(), y.min(), y.max()], norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), **params)
        else:
            pc = axes.imshow(slice.T, origin='lower', interpolation='nearest', extent=[x.min(), x.max(), y.min(), y.max()], vmin=vmin, vmax=vmax, **params)
        
        if colorbar:
            if log:
                cb = p.colorbar(pc, ax=axes, format=matplotlib.ticker.LogFormatterMathtext())
            else:
                fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((-2, 2))
                fmt.set_useOffset(False)
                cb = p.colorbar(pc, ax=axes, format=fmt)
            if cblabel != None:
                cb.set_label(cblabel)
        
        if contour:
            x2 = result['x2']
            y2 = result['y2']
            contours = result['contours']
            axes.contour(x2, y2, contours.T, levels=[0.99], colors="black")
        
        axes.axis( "image" )
        
            
        dval = dresult["grid"].T
        dxmin = dresult["x"].min()
        dymin = dresult["y"].min()
            
        ddx = dresult["x"][1] - dresult["x"][0]
        ddy = dresult["y"][1] - dresult["y"][0]
            
        dxmax = dval.shape[0]
        dymax = dval.shape[1]
            
        def format_coord(x, y):
            col = int((x-dxmin)/ddx+0.5)
            row = int((y-dymin)/ddy+0.5)
            if col>=0 and col<dxmax and row>=0 and row<dymax:
                z = dval[row,col]
                if dresult["name"] == "id":
                    return 'x=%1.4f, y=%1.4f, %s=%d'%(x, y, dresult["name"], int(z))
                else:
                    return 'x=%1.4f, y=%1.4f, %s=%.4e'%(x, y, dresult["name"], z)
            else:
                return 'x=%1.4f, y=%1.4f'%(x, y)

        axes.format_coord = format_coord
        
        if newlabels:
            xticklabels = []
            for tick in pc.axes.get_xticks():
                if (tick == 0):
                    xticklabels += [ r'$0.0$' ]
                else:
                    xticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(np.ceil(np.log10(np.abs(tick)))), np.ceil(np.log10(np.abs(tick)))) ]
            pc.axes.set_xticklabels( xticklabels, size=24, y=-0.1, va='baseline' )

            yticklabels = []
            for tick in pc.axes.get_yticks():
                if (tick == 0):
                    yticklabels += [ r'$0.0$' ]
                else:
                    yticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(np.ceil(np.log10(np.abs(tick)))), np.ceil(np.log10(np.abs(tick)))) ]
            pc.axes.set_yticklabels( yticklabels, size=24, ha='right' )
        
        return

        

    def get_Agrid( self, value, gradient=None, res=1024, center=None, box=None, group=None):
        if self.numdims != 3:
            raise Exception( "not supported" )
        
        if group == None:
            group = self
            
        center = self.__validate_vector__(center, self.center)
        box = self.__validate_vector__(box, self.boxsize)

        c = center

        pos = group.pos.astype( 'float64' )
        px = np.abs( pos[:,0] - c[0] )
        py = np.abs( pos[:,1] - c[1] )
        pz = np.abs( pos[:,2] - c[2] )

        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < 0.5*box[2]) )
        print "Selected %d of %d particles." % (pp.size,self.npart)

        posdata = pos[pp,:]
        valdata = group[value][pp].astype('float64')
        
        if  gradient == None:
            data = calcGrid.calcASlice(posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], 0, 1, boxz=box[2], grid3D=True)
        else:
            data = calcGrid.calcASlice(posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], 0, 1, group[gradient][pp].astype('float64'), boxz=box[2], grid3D=True)
        
        data[ "neighbours" ] = pp[ data["neighbours"] ]
        
        data["name"] = value
        
        return data

    def get_AMRgrid( self, value, gradient=None, res=1024, center=None, box=None, group=None):
        if self.numdims != 3:
            raise Exception( "not supported" )
        
        if group == None:
            group = self.part0
            
        center = self.__validate_vector__(center, self.center)
        box = self.__validate_vector__(box, self.boxsize)

        c = center
        
        domainlen = self.boxsize
        
        domainc = np.zeros(3)
        domainc[0] = self.boxsize/2
        domainc[1] = self.boxsize/2.
        
        if self.numdims >2:
            domainc[2] = self.boxsize/2.

        posdata = group.pos.astype( 'float64' )
        valdata = group[value].astype('float64')
        
        if gradient==None:
            data = calcGrid.calcAMRSlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, 0, 1,boxz=box[2], grid3D=True)
        else:
            data = calcGrid.calcAMRSlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, 0, 1, group[gradient][pp].astype('float64'), boxz=box[2], grid3D=True)

        data["name"] = value
        
        return data
    
    def get_SPHgrid( self, value, hsml="hsml", weights=None, normalized=True, res=1024, center=None, box=None, group=None):
        if self.numdims != 3:
            raise Exception( "not supported" )
        
        if group == None:
            group = self
            
        center = self.__validate_vector__(center, self.center)
        box = self.__validate_vector__(box, self.boxsize)

        c = center
        
        domainlen = self.boxsize
        
        domainc = np.zeros(3)
        domainc[0] = self.boxsize/2
        domainc[1] = self.boxsize/2.
        
        if self.numdims >2:
            domainc[2] = self.boxsize/2.

        posdata = group.pos.astype( 'float64' )
        valdata = group[value].astype('float64')
        hsmldata = group[hsml].astype("float64")

        
        if weights==None:
            grid = calcGrid.calcGrid(posdata, hsmldata, valdata, res, res, res, box[0], box[1], box[2], c[0], c[1], c[2], proj=False, norm=normalized )
        else:
            weightdata = group[weights].astype("float64")
            grid = calcGrid.calcGrid(posdata, hsmldata, valdata, res, res, res, box[0], box[1], box[2], c[0], c[1], c[2], proj=False, norm=normalized, weights=weightdata )
        
        return grid
    
    
    
