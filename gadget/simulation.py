import numpy as np
import time

try:
    import matplotlib.pyplot as p
    from matplotlib import collections  as mc
    import matplotlib
except:
    print("Could not load matplotlib, plotting function will not work")



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
        self.set_box(None)
        
        self.__domain__ = np.zeros(3)
        self.__domain__[:] = self.box
        
    def __validate_vector__(self, vector, default, len=None, req=None):
        if len is None:
            len = self.numdims
            
        if req is None:
            req = len

        if vector is None:
            if type(default) == np.ndarray or type(default) == list:
                default = default[:len]
            vector = default
        
        v = np.zeros(3)
        
        if len == 0:
            return v
        
        if type(vector) == np.ndarray or type(vector) == list:
            v[:req] = vector[:req]
        else:
            v[:req] = vector
        
        return v
    
    def __validate_value__(self, value, length, group=None):
        if value is None:
            return None
        elif type(value) == str:
            if group is None:
                group = self
            return group[value]
        elif type(value) == np.ndarray:
            if value.shape[0] != length:
                raise Exception("wrong array length: %s\n"%str(value.shape))
            return value
        else:
            return np.ones(length) * value
        
        
    def set_center(self, center):
        c =  self.__validate_vector__(center, self.BoxSize/2, len=3, req=self.numdims)
        
        if hasattr(self,"config"):
            if hasattr(self.config,"LONG_X"):
                c[0] *= self.config.LONG_X
            if hasattr(self.config,"LONG_Y"):
                c[1] *= self.config.LONG_Y
            if hasattr(self.config,"LONG_Z"):
                c[2] *= self.config.LONG_Z
                
        self.center = c
        return
    
    def set_box(self, box):
        c = self.__validate_vector__(box, self.BoxSize, len=3, req=self.numdims)
        
        if hasattr(self,"config"):
            if hasattr(self.config,"LONG_X"):
                c[0] *= self.config.LONG_X
            if hasattr(self.config,"LONG_Y"):
                c[1] *= self.config.LONG_Y
            if hasattr(self.config,"LONG_Z"):
                c[2] *= self.config.LONG_Z
                
        self.box = c
        return
    
    def r(self, center=None, periodic=True, group=None):
        if group is None:
            group = self
            
        center = self.__validate_vector__(center, self.center, len=3)

        dx = group.pos[:,0]-center[0]
        dy = group.pos[:,1]-center[1]
        dz = group.pos[:,2]-center[2]
        
        if periodic:
            dx = np.where(dx > self.__domain__[0]/2, dx-self.__domain__[0]/2,dx)
            dx = np.where(dx < -self.__domain__[0]/2, dx+self.__domain__[0]/2,dx)
            dy = np.where(dy > self.__domain__[1]/2, dy-self.__domain__[1]/2,dy)
            dy = np.where(dy < -self.__domain__[1]/2, dy+self.__domain__[1]/2,dy)
            dz = np.where(dz > self.__domain__[2]/2, dz-self.__domain__[2]/2,dz)
            dz = np.where(dz < -self.__domain__[2]/2, dz+self.__domain__[2]/2,dz)
            
        radius = np.sqrt(dx**2+dy**2+dz**2)

        return radius
    
    def centerat(self, center, group=None):
        if group is None:
            group = self
            
        center = self.__validate_vector__(center, self.center)
        
        group.pos -= center[None,:]
        self.set_center(np.zeros(3))
        
        return
    
    def __get_radhist__(self, value, center=None, bins=100, range=None, log=False, periodic=True, group=None):
        if group is None:
            group = self

        center = self.__validate_vector__(center, self.center)
        
        radius = self.r(center=center, periodic=periodic, group=group)
        
        if type(range) == list:
            range = np.array( range )    
        if range is None:
            range = np.array([np.min(radius),np.max(radius)])
            
        if log:
            logbins = np.linspace(np.log(range[0]), np.log(range[1]), bins+1)
            xbins = np.exp(logbins)
            logpos = 0.5 * (logbins[:-1] + logbins[1:])
            xpos = np.exp(logpos)
        else:
            xbins = np.linspace(range[0], range[1], bins+1)
            xpos = 0.5 * (xbins[:-1] + xbins[1:])
            
        if value is None:
            val = None
        else:
            val = self.__validate_value__(value, radius.shape[0], group).astype('float64')
            
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
        if group is None:
            group = self.part0
               
        center = self.__validate_vector__(center,self.center)
        box = self.__validate_vector__(box,self.box)
                
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

        x = group.pos[:,axis0]
        y = group.pos[:,axis1]
        z = group.pos[:,3 - axis0 - axis1]
        
        if periodic:
            x = np.where(x - c[0] > self.__domain__[0]/2, x-self.__domain__[0]/2,x)
            x = np.where(x - c[0] < -self.__domain__[0]/2, x+self.__domain__[0]/2,x)
            y = np.where(y - c[1] > self.__domain__[1]/2, y-self.__domain__[1]/2,y)
            y = np.where(y - c[1] < -self.__domain__[1]/2, y+self.__domain__[1]/2,y)
            z = np.where(z - c[2] > self.__domain__[2]/2, z-self.__domain__[2]/2,z)
            z = np.where(z - c[2] < -self.__domain__[2]/2, z+self.__domain__[2]/2,z)

        pp, = np.where( (np.abs(x-c[0]) <= 0.5*box[0]) & (np.abs(y-c[1]) <= 0.5*box[1]) & (np.abs(z-c[2]) <= 0.5*box[2]) )
        
        axes.scatter(x[pp], y[pp], **params)
        axes.axis( "scaled" )
        
    def get_Aslice( self, value, gradient=None, res=1024, center=None, axis=[0,1], box=None, group=None):
        if group is None:
            group = self.part0
                        
        axis0 = axis[0]
        axis1 = axis[1]
        
        b = np.zeros(2)
        b[0] = self.box[axis0]
        b[1] = self.box[axis1]
               
        center = self.__validate_vector__(center, self.center)
        box = self.__validate_vector__(box, b, len=2)

        c = np.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        pos = group.pos.astype( 'float64' )
        px = np.abs( pos[:,axis0] - c[0] )
        py = np.abs( pos[:,axis1] - c[1] )
        pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

        zdist = 2. * group.vol.astype('float64')**(1./3.)

        pp, = np.where( (px <= 0.5*box[0]) & (py <= 0.5*box[1]) & (pz <= zdist) )
        print "Selected %d of %d particles." % (pp.size,self.npart)

        posdata = pos[pp,:]
        valdata = self.__validate_value__(value, posdata.shape[0], group)[pp].astype('float64')
        
        if gradient is None:
            data = calcGrid.calcASlice(posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1)
        else:
            graddata = self.__validate_value__(gradient, posdata.shape[0], group)[pp].astype('float64')
            data = calcGrid.calcASlice(posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], axis0, axis1, grad=graddata)
        data[ "neighbours" ] = pp[ data["neighbours"] ]
        
        if type(value) == str:
            data['name'] = value
        else:
            data['name'] = ""
        data['x'] = np.arange( res+1, dtype="float64" ) / res * box[0] - .5 * box[0] + c[0]
        data['y'] = np.arange( res+1, dtype="float64" ) / res * box[1] - .5 * box[1] + c[1]
        data['x2'] = (np.arange( res, dtype="float64" ) + 0.5) / res * box[0] - .5 * box[0] + c[0]
        data['y2'] = (np.arange( res, dtype="float64" ) + 0.5) / res * box[1] - .5 * box[1] + c[1]
        
        return data
    
    def plot_Aslice(self, value, gradient=None, log=False, res=1024, center=None, axis=[0,1], box=None, group=None, vmin=None, vmax=None, dvalue=None, dgradient=None, colorbar=True, cblabel=None, contour=False, newlabels=False, newfig=True, axes=None, **params):
        result = self.get_Aslice(value=value, gradient=gradient, res=res, center=center, axis=axis, box=box, group=group)
        
        dresult = None
        if dvalue != None:
            dresult = self.get_Aslice(value=dvalue, gradient=dgradient, res=res, center=center, axis=axis, box=box, group=group)
            
        self.__plot_Slice__(result,log=log, vmin=vmin, vmax=vmax, dresult=dresult, colorbar=colorbar, cblabel=cblabel, contour=contour, newlabels=newlabels, newfig=newfig, axes=axes, **params)

        
    def get_AMRslice(self, value, gradient=None, res=1024, center=None, axis=[0,1], box=None, group=None):
        if group is None:
            group = self.part0
                
        axis0 = axis[0]
        axis1 = axis[1]
        
        b = np.zeros(2)
        b[0] = self.box[axis0]
        b[1] = self.box[axis1]
               
        center = self.__validate_vector__(center, self.center)
        box = self.__validate_vector__(box, b, len=2)

        c = np.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        domainlen = np.max(self.__domain__)        
        domainc = np.zeros(3)
        domainc[0] = np.max(self.__domain__)/2
        domainc[1] = np.max(self.__domain__)/2.
        
        if self.numdims >2:
            domainc[2] = np.max(self.__domain__)/2.

        posdata = group.pos.astype('float64')
        valdata = self.__validate_value__(value, posdata.shape[0], group).astype('float64')
        
        if gradient is None:
            data = calcGrid.calcAMRSlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, axis0, axis1)
        else:
            graddata = self.__validate_value__(gradient, posdata.shape[0], group).astype('float64')
            data = calcGrid.calcAMRSlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, axis0, axis1, grad=graddata)
        
        if type(value) == str:
            data['name'] = value
        else:
            data['name'] = ""
        data['x'] = np.arange( res+1, dtype="float64" ) / res * box[0] - .5 * box[0] + c[0]
        data['y'] = np.arange( res+1, dtype="float64" ) / res * box[1] - .5 * box[1] + c[1]
        data['x2'] = (np.arange( res, dtype="float64" ) + 0.5) / res * box[0] - .5 * box[0] + c[0]
        data['y2'] = (np.arange( res, dtype="float64" ) + 0.5) / res * box[1] - .5 * box[1] + c[1]
        
        return data
    

    def plot_AMRslice(self, value, gradient=None, log=False, res=1024, center=None, axis=[0,1], box=None, group=None, vmin=None, vmax=None, dvalue=None, dgradient=None, colorbar=True, cblabel=None, contour=False, newlabels=False, newfig=True, axes=None, **params):
        result = self.get_AMRslice(value, gradient=gradient, res=res, center=center, axis=axis, box=box, group=group)
        
        dresult = None
        if dvalue != None:
            dresult = self.get_AMRslice(dvalue, gradient=dgradient, res=res, center=center, axis=axis, box=box, group=group)
            
        self.__plot_Slice__(result,log=log, vmin=vmin, vmax=vmax, dresult=dresult, colorbar=colorbar, cblabel=cblabel, contour=contour, newlabels=newlabels, newfig=newfig, axes=axes, **params)


    def __add_square(self, lines, i, cx,cy,edgelength):
        e05=edgelength*0.5;
        self.__add_line(lines, i, cx-e05,cy-e05, cx+e05,cy-e05)
        self.__add_line(lines, i+1, cx+e05,cy-e05, cx+e05,cy+e05)
        self.__add_line(lines, i+2, cx+e05,cy+e05, cx-e05,cy+e05)
        self.__add_line(lines, i+3, cx-e05,cy+e05, cx-e05,cy-e05)

    def __add_line(self, lines, i, x1,y1,x2,y2):
        lines[i,0,0]=x1
        lines[i,0,1]=y1
        lines[i,1,0]=x2
        lines[i,1,1]=y2

   
    def plot_AMRmesh(self, res=1024, center=None, axis=[0,1], box=None, group=None, newfig=False, axes=None, **params):
        if(group==None):
            group=self.part0

        if(newfig and axes==None):
            fig = p.figure()
            axes = p.gca()
        elif axes==None:
            axes = p.gca()

        ids=np.unique(self.get_AMRslice("id",box=box,center=center,axis=axis,res=res,group=self.part0)["grid"])
        ids=ids.astype(self.id.dtype)

        lines=np.zeros((4*np.shape(ids)[0],2,2))

        j=0

        for i in ids:
            index=np.where(self.id==i)[0][0]
            self.__add_square(lines, j, group.pos[index,axis[0]], group.pos[index,axis[1]], group.volume[index]**(1./(self.numdims)))
            j=j+4

        if(not 'color' in params):
            params['color']='black'    

        lc = mc.LineCollection(lines, **params)
        ax=p.subplot(1,1,1)
        ax.add_collection(lc)


    def get_AMRline(self, value, gradient=None, res=1024, center=None, axis=0, box=None, group=None):
        if group is None:
            group = self.part0
               
        center = self.__validate_vector__(center, self.center)
                
        axis0 = axis
        if axis0 == 0:
            axis1 = 1
        else:
            axis1 = 0
            
        b = self.box[axis0]
        box0 = self.__validate_vector__(box, b, len=1)[0]

        resx=res
        resy=1

        c = np.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]
        
        domainlen = np.max(self.__domain__)        
        domainc = np.zeros(3)
        domainc[0] = np.max(self.__domain__)/2
        domainc[1] = np.max(self.__domain__)/2.
        if self.numdims > 2:
            domainc[2] = np.max(self.__domain__)/2.

        posdata = group.pos.astype('float64')
        valdata = self.__validate_value__(value, posdata.shape[0], group).astype('float64')
        
        if gradient is None:
            data = calcGrid.calcAMRSlice( posdata, valdata, resx, resy, box0, 0., c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, axis0, axis1)
        else:
            graddata = self.__validate_value__(gradient, posdata.shape[0], group).astype('float64')
            data = calcGrid.calcAMRSlice( posdata, valdata, resx, resy, box0, 0., c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, axis0, axis1, grad=graddata)
        
        if type(value) == str:
            data['name'] = value
        else:
            data['name'] = ""
        data['grid'] = data['grid'][:,0]
        data['x'] = np.arange( resx+1, dtype="float64" ) / resx * box0 - .5 * box0 + c[0]
        data['x2'] = (np.arange( resx, dtype="float64" ) + 0.5) / resx * box0 - .5 * box0 + c[0]
        
        return data

    def plot_AMRline(self, value, gradient=None, log=False, res=1024, center=None, axis=0, box=None, group=None, newlabels=False, newfig=True, axes=None, **params):
        result = self.get_AMRline(value, gradient=gradient, res=res, center=center, axis=axis, box=box, group=group)
            
        self.__plot_Line__(result,log=log, newlabels=newlabels, newfig=newfig, axes=axes, **params)        
        
    def get_SPHproj( self, value, hsml="hsml", weights=None, normalized=True, res=1024, center=None, axis=[0,1], box=None, group=None):
        if group is None:
            group = self.part0
            
        axis0 = axis[0]
        axis1 = axis[1]
        
        b = np.zeros(3)
        b[0] = self.box[axis0]
        b[1] = self.box[axis1]
        b[2] = self.box[3 - axis0 - axis1]
               
        center = self.__validate_vector__(center, self.center)
        box = self.__validate_vector__(box, b)

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
        valdata = self.__validate_value__(value, posdata.shape[0], group)[pp].astype('float64')
        hsmldata = self.__validate_value__(hsml, posdata.shape[0], group)[pp].astype("float64")

        
        if weights is None:
            grid = calcGrid.calcGrid(posdata, hsmldata, valdata, res, res, res, box[0], box[1], box[2], c[0], c[1], c[2], proj=True, norm=normalized )
        else:
            weightdata = self.__validate_value__(weights, posdata.shape[0], group).astype("float64")
            grid = calcGrid.calcGrid(posdata, hsmldata, valdata, res, res, res, box[0], box[1], box[2], c[0], c[1], c[2], proj=True, norm=normalized, weights=weightdata )
            
        data = {}
        data['grid'] = grid
        
        if type(value) == str:
            data['name'] = value
        else:
            data['name'] = ""
        data['x'] = np.arange( res+1, dtype="float64" ) / res * box[0] - .5 * box[0] + c[0]
        data['y'] = np.arange( res+1, dtype="float64" ) / res * box[1] - .5 * box[1] + c[1]
        data['x2'] = (np.arange( res, dtype="float64" ) + 0.5) / res * box[0] - .5 * box[0] + c[0]
        data['y2'] = (np.arange( res, dtype="float64" ) + 0.5) / res * box[1] - .5 * box[1] + c[1]
        
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

        if vmin is None:
            vmin = np.min(slice)
            
        if vmax is None:
            vmax = np.max(slice)
            
        if dresult is None:
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
            col = int((x-dxmin)/ddx)
            row = int((y-dymin)/ddy)
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


    def __plot_Line__(self, result, log=False, newlabels=False, newfig=True, axes=None, **params):          
        slice = result['grid']
        x = result['x2']
                   
        
        if newfig and axes==None:
            fig = p.figure()
            axes = p.gca()
        elif axes==None:
            axes = p.gca()

     
        if log:
            pc = axes.semilogy(x, slice, **params)
        else:
            pc = axes.plot(x, slice, **params)

        
        if newlabels:
            xticklabels = []
            for tick in pc.axes.get_xticks():
                if (tick == 0):
                    xticklabels += [ r'$0.0$' ]
                else:
                    xticklabels += [ r'$%.2f \cdot 10^{%d}$' % (tick/10**(np.ceil(np.log10(np.abs(tick)))), np.ceil(np.log10(np.abs(tick)))) ]
            pc.axes.set_xticklabels( xticklabels, size=24, y=-0.1, va='baseline' )
        
        return        

    def get_Agrid( self, value, gradient=None, res=1024, center=None, box=None, group=None):
        if self.numdims != 3:
            raise Exception( "not supported" )
        
        if group is None:
            group = self.part0
            
        center = self.__validate_vector__(center, self.center)
        box = self.__validate_vector__(box, self.box)

        c = center

        pos = group.pos.astype( 'float64' )
        px = np.abs( pos[:,0] - c[0] )
        py = np.abs( pos[:,1] - c[1] )
        pz = np.abs( pos[:,2] - c[2] )

        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < 0.5*box[2]) )
        print "Selected %d of %d particles." % (pp.size,self.npart)

        posdata = pos[pp,:]
        valdata = self.__validate_value__(value, posdata.shape[0], group)[pp].astype('float64')
        
        if  gradient is None:
            data = calcGrid.calcASlice(posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], 0, 1, boxz=box[2], grid3D=True)
        else:
            data = calcGrid.calcASlice(posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], 0, 1, grad=self.__validate_value__(gradient, posdata.shape[0], group)[pp].astype('float64'), boxz=box[2], grid3D=True)
        
        data[ "neighbours" ] = pp[ data["neighbours"] ]
        
        if type(value) == str:
            data['name'] = value
        else:
            data['name'] = ""
        
        return data

    def get_AMRgrid( self, value, gradient=None, res=1024, center=None, box=None, group=None):
        if self.numdims != 3:
            raise Exception( "not supported" )
        
        if group is None:
            group = self.part0
            
        center = self.__validate_vector__(center, self.center)
        box = self.__validate_vector__(box, self.box)

        c = center
        
        domainlen = np.max(self.__domain__)        
        domainc = np.zeros(3)
        domainc[0] = np.max(self.__domain__)/2
        domainc[1] = np.max(self.__domain__)/2.
        if self.numdims > 2:
            domainc[2] = np.max(self.__domain__)/2.

        posdata = group.pos.astype( 'float64' )
        valdata = self.__validate_value__(value, posdata.shape[0], group).astype('float64')
        
        if gradient is None:
            data = calcGrid.calcAMRSlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, 0, 1,boxz=box[2], grid3D=True)
        else:
            data = calcGrid.calcAMRSlice( posdata, valdata, res, res, box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, 0, 1, grad=self.__validate_value__(gradient, posdata.shape[0], group)[pp].astype('float64'), boxz=box[2], grid3D=True)

        if type(value) == str:
            data['name'] = value
        else:
            data['name'] = ""
        
        return data
    
    def get_SPHgrid( self, value, hsml="hsml", weights=None, normalized=True, res=1024, center=None, box=None, group=None):
        if self.numdims != 3:
            raise Exception( "not supported" )
        
        if group is None:
            group = self.part0
            
        center = self.__validate_vector__(center, self.center)
        box = self.__validate_vector__(box, self.box)

        c = center

        posdata = group.pos.astype( 'float64' )
        valdata = self.__validate_value__(value, posdata.shape[0], group).astype('float64')
        hsmldata = self.__validate_value__(hsml, posdata.shape[0], group).astype("float64")

        
        if weights is None:
            grid = calcGrid.calcGrid(posdata, hsmldata, valdata, res, res, res, box[0], box[1], box[2], c[0], c[1], c[2], proj=False, norm=normalized )
        else:
            weightdata = self.__validate_value__(weights, posdata.shape[0], group).astype("float64")
            grid = calcGrid.calcGrid(posdata, hsmldata, valdata, res, res, res, box[0], box[1], box[2], c[0], c[1], c[2], proj=False, norm=normalized, weights=weightdata )
        
        return grid
    
    
