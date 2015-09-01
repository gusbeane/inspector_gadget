import numpy as np

try:
    import matplotlib.pyplot as p
    from matplotlib import collections  as mc
    import matplotlib
except:
    print("Could not load matplotlib, plotting functions will not work")

from gadget.loader import Snapshot

try:
    import gadget.calcGrid as calcGrid
except:
    print("Could not load calcGrid, plotting function will not work")


class Simulation(Snapshot):
    
    def __init__(self,filename, snapshot=None, filenum=None, format=None, fields=None, parttype=None, **param):
        param['combineFiles'] = True
        super(Simulation,self).__init__(filename, snapshot=snapshot, filenum=filenum, format=format, fields=fields, parttype=parttype,**param)
        

        self.numdims = np.int32(3)
        self.twodim = False
        self.onedim = False
                
        if hasattr(self,'pos'):
            if np.abs( self['pos'][:,2] ).max() == 0.:
                self.twodim = True
                self.numdims = np.int32(2)
            if self.twodim and np.abs( self['pos'][:,1] ).max() == 0.:
                self.onedim = True
                self.numdims = np.int32(1)
            
        self.set_center(None)
        self.set_box(None)
        
        self._shift = np.zeros(3)
        self._trafomatrix = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
        
        self._domain = np.zeros(3)
        self._domain[:] = self.box
        
    def _validate_vector(self, vector, default, len=None, req=None):
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
    
    def _validate_value(self, value, length, group=None, components=None):
        if value is None:
            return None
        elif type(value) == str:
            if group is None:
                group = self
            ret = group[value]
        elif type(value) == np.ndarray:
            if value.shape[0] != length:
                raise Exception("wrong array length: %s\n"%str(value.shape))
            ret = value
        else:
            ret = np.ones(length) * value
         
        if components is not None:
            if components > 1:
                if ret.shape[1] < components:
                    raise Exception("wrong array format: %s\n"%str(value.shape))
                
        return ret
        
    def set_center(self, center):
        c =  self._validate_vector(center, self.BoxSize/2, len=3, req=self.numdims)
        
        if center is None:
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
        c = self._validate_vector(box, self.BoxSize, len=3, req=self.numdims)
        
        if box is None:
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
            
        center = self._validate_vector(center, self.center, len=3)

        dx = group.pos[:,0]-center[0]
        dy = group.pos[:,1]-center[1]
        dz = group.pos[:,2]-center[2]
        
        if periodic:
            dx = np.where(dx > self._domain[0]/2, dx-self._domain[0]/2,dx)
            dx = np.where(dx < -self._domain[0]/2, dx+self._domain[0]/2,dx)
            dy = np.where(dy > self._domain[1]/2, dy-self._domain[1]/2,dy)
            dy = np.where(dy < -self._domain[1]/2, dy+self._domain[1]/2,dy)
            dz = np.where(dz > self._domain[2]/2, dz-self._domain[2]/2,dz)
            dz = np.where(dz < -self._domain[2]/2, dz+self._domain[2]/2,dz)
            
        radius = np.sqrt(dx**2+dy**2+dz**2)

        return radius
    
    def centerat(self, center, group=None):
        if group is None:
            group = self
            
        center = self._validate_vector(center, self.center)
        
        group.pos[...] -= (center[None,:] - self._shift[None,:])
        
        self._shift = center
        
        self.set_center(np.zeros(3))
        
        return

    def coordtransform(self, zaxis, center=None, xaxis=None, fields=["pos", "vel"]):


        if(xaxis != None):
            axx=np.array(xaxis)

        elif(zaxis[0]==0 and zaxis[1]==0 and zaxis[2]==1):
            axx=np.array([1,0,0])

        elif(zaxis[0]!=0 and zaxis[2]!=0):
            axx=np.array([zaxis[2],0,-zaxis[0]])

        axz=np.array(zaxis)
        axy=np.cross(axz,axx)

        axx=axx/np.sqrt(np.sum(axx*axx))
        axy=axy/np.sqrt(np.sum(axy*axy))
        axz=axz/np.sqrt(np.sum(axz*axz))

        if(np.abs(np.dot(axx,axz))>1e-12):
            raise Exception("The coordinate axes have to be orthogonal!\n")

        M=np.matrix([[axx[0], axy[0], axz[0]], [axx[1], axy[1], axz[1]], [axx[2], axy[2], axz[2]]])

        if(center!=None):
            cn=center
            self.centerat(cn)

        for field in fields:
            field=self[field]
            field[...] = np.dot(field, np.dot(self._trafomatrix,M))           

        self._trafomatrix = np.linalg.inv(M)
    
    def _get_radhist(self, value, center=None, bins=100, range=None, log=False, periodic=True, group=None):
        if group is None:
            group = self

        center = self._validate_vector(center, self.center)
        
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
            val = self._validate_value(value, radius.shape[0], group).astype('float64')
            
        profile, xbins = np.histogram(radius,bins=xbins,weights=val)
        
        return (profile, xpos, xbins, range)

    def get_raddens(self, value='mass', center=None, bins=100, range=None, log=False, periodic=True, group=None):
        (profile, xpos, xbins, range) = self._get_radhist(value=value, center=center, bins=bins ,range=range, log=log, periodic=periodic, group=group)
        profile /= 4./3*np.pi * (xbins[1:]**3-xbins[:1]**3)
        
        return (profile,xpos)
    
    def get_radprof(self, value, weights=None, center=None, bins=100, range=None, log=False, periodic=True, group=None):
        (profile, xpos, xbins, range) = self._get_radhist(value=value, center=center, bins=bins ,range=range, log=log, periodic=periodic, group=group)
        (norm, xpos, xbins, range) = self._get_radhist(value=weights, center=center, bins=bins ,range=range, log=log, periodic=periodic, group=group)
        
        profile /= norm
        
        return (profile,xpos)
    
    def plot_raddens(self, value='mass', center=None, bins=100, range=None, log=False, periodic=True, group=None, **params):
        (profile, xpos) = self.get_raddens(value=value, center=center, bins=bins, range=range, log=log, periodic=periodic, group=group)
        if log:
            myplot = p.loglog(xpos, profile, **params)
        else:
            myplot = p.plot(xpos, profile, **params)

        return myplot

    def plot_radprof(self, value, weights=None, center=None, bins=100, range=None, log=False, periodic=True, group=None, **params):
        (profile, xpos) = self.get_radprof(value=value, weights=weights, center=center, bins=bins, range=range, log=log, periodic=periodic, group=group)
        if log:
            myplot = p.loglog(xpos,profile, **params)
        else:
            myplot = p.plot(xpos, profile, **params)

        return myplot


    def plot_pos(self, center=None, axis=[0,1], box=None, periodic=True, group=None, newfig=True, axes=None, **params):
        if group is None:
            group = self.part0
               
        center = self._validate_vector(center,self.center)
        box = self._validate_vector(box,self.box)
                
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
            x = np.where(x - c[0] > self._domain[0]/2, x-self._domain[0]/2,x)
            x = np.where(x - c[0] < -self._domain[0]/2, x+self._domain[0]/2,x)
            y = np.where(y - c[1] > self._domain[1]/2, y-self._domain[1]/2,y)
            y = np.where(y - c[1] < -self._domain[1]/2, y+self._domain[1]/2,y)
            z = np.where(z - c[2] > self._domain[2]/2, z-self._domain[2]/2,z)
            z = np.where(z - c[2] < -self._domain[2]/2, z+self._domain[2]/2,z)

        pp, = np.where( (np.abs(x-c[0]) <= 0.5*box[0]) & (np.abs(y-c[1]) <= 0.5*box[1]) & (np.abs(z-c[2]) <= 0.5*box[2]) )
        
        myplot = axes.scatter(x[pp], y[pp], **params)
        axes.axis( "scaled" )

        return myplot
        
    def get_Aslice( self, value, gradient=None, res=None, center=None, axis=[0,1], box=None, group=None):
        if group is None:
            group = self.part0
                        
        axis0 = axis[0]
        axis1 = axis[1]
        
        b = np.zeros(2)
        b[0] = self.box[axis0]
        b[1] = self.box[axis1]
               
        center = self._validate_vector(center, self.center)
        box = self._validate_vector(box, b, len=2)
        
        r = np.zeros(2)
        r[0] = int(1024 * box[0]/box.max())
        r[1] = int(1024 * box[1]/box.max())
        res = self._validate_vector(res, r, len=2).astype(np.int)

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
        print("Selected %d of %d particles." % (pp.size,self.npart))

        posdata = pos[pp,:]
        valdata = self._validate_value(value, posdata.shape[0], group)[pp].astype('float64')
        
        if gradient is None:
            data = calcGrid.calcASlice(posdata, valdata, res[0], res[1], box[0], box[1], c[0], c[1], c[2], axis0, axis1)
        else:
            graddata = self._validate_value(gradient, posdata.shape[0], group)[pp].astype('float64')
            data = calcGrid.calcASlice(posdata, valdata, res[0], res[1], box[0], box[1], c[0], c[1], c[2], axis0, axis1, grad=graddata)
        data[ "neighbours" ] = pp[ data["neighbours"] ]
        
        if type(value) == str:
            data['name'] = value
        else:
            data['name'] = ""
        data['x'] = np.arange( res[0]+1, dtype="float64" ) / res[0] * box[0] - .5 * box[0] + c[0]
        data['y'] = np.arange( res[1]+1, dtype="float64" ) / res[1] * box[1] - .5 * box[1] + c[1]
        data['x2'] = (np.arange( res[0], dtype="float64" ) + 0.5) / res[0] * box[0] - .5 * box[0] + c[0]
        data['y2'] = (np.arange( res[1], dtype="float64" ) + 0.5) / res[1] * box[1] - .5 * box[1] + c[1]
        
        return data
    
    def plot_Aslice(self, value, gradient=None, log=False, res=None, center=None, axis=[0,1], box=None, group=None, vmin=None, vmax=None, dvalue=None, dgradient=None, colorbar=True, cblabel=None, contour=False, newlabels=False, newfig=True, axes=None, **params):
        result = self.get_Aslice(value=value, gradient=gradient, res=res, center=center, axis=axis, box=box, group=group)
        
        dresult = None
        if dvalue != None:
            dresult = self.get_Aslice(value=dvalue, gradient=dgradient, res=res, center=center, axis=axis, box=box, group=group)
            
        myplot = self._plot_Slice(result,log=log, vmin=vmin, vmax=vmax, dresult=dresult, colorbar=colorbar, cblabel=cblabel, contour=contour, newlabels=newlabels, newfig=newfig, axes=axes, **params)

        return myplot

        
    def get_AMRslice(self, value, gradient=None, res=None, center=None, axis=[0,1], box=None, group=None):
        if group is None:
            group = self.part0
                
        axis0 = axis[0]
        axis1 = axis[1]
        
        b = np.zeros(2)
        b[0] = self.box[axis0]
        b[1] = self.box[axis1]
               
        center = self._validate_vector(center, self.center)
        box = self._validate_vector(box, b, len=2)
        
        r = np.zeros(2)
        r[0] = int(1024 * box[0]/box.max())
        r[1] = int(1024 * box[1]/box.max())
        res = self._validate_vector(res, r, len=2).astype(np.int)

        c = np.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        domainlen = np.max(self._domain)        
        domainc = np.zeros(3)
        domainc[0] = np.max(self._domain)/2
        domainc[1] = np.max(self._domain)/2.
        
        if self.numdims >2:
            domainc[2] = np.max(self._domain)/2.

        posdata = group.pos.astype('float64')
        valdata = self._validate_value(value, posdata.shape[0], group).astype('float64')
        
        if gradient is None:
            data = calcGrid.calcAMRSlice( posdata, valdata, res[0], res[1], box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, axis0, axis1)
        else:
            graddata = self._validate_value(gradient, posdata.shape[0], group).astype('float64')
            data = calcGrid.calcAMRSlice( posdata, valdata, res[0], res[1], box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, axis0, axis1, grad=graddata)
        
        if type(value) == str:
            data['name'] = value
        else:
            data['name'] = ""
        data['x'] = np.arange( res[0]+1, dtype="float64" ) / res[0] * box[0] - .5 * box[0] + c[0]
        data['y'] = np.arange( res[1]+1, dtype="float64" ) / res[1] * box[1] - .5 * box[1] + c[1]
        data['x2'] = (np.arange( res[0], dtype="float64" ) + 0.5) / res[0] * box[0] - .5 * box[0] + c[0]
        data['y2'] = (np.arange( res[1], dtype="float64" ) + 0.5) / res[1] * box[1] - .5 * box[1] + c[1]
        
        return data
    

    def plot_AMRslice(self, value, gradient=None, log=False, res=None, center=None, axis=[0,1], box=None, group=None, vmin=None, vmax=None, dvalue=None, dgradient=None, colorbar=True, cblabel=None, contour=False, newlabels=False, newfig=True, axes=None, **params):
        result = self.get_AMRslice(value, gradient=gradient, res=res, center=center, axis=axis, box=box, group=group)
        
        dresult = None
        if dvalue != None:
            dresult = self.get_AMRslice(dvalue, gradient=dgradient, res=res, center=center, axis=axis, box=box, group=group)
            
        myplot = self._plot_Slice(result,log=log, vmin=vmin, vmax=vmax, dresult=dresult, colorbar=colorbar, cblabel=cblabel, contour=contour, newlabels=newlabels, newfig=newfig, axes=axes, **params)

        return myplot


    def _add_square(self, lines, i, cx,cy,edgelength):
        e05=edgelength*0.5;
        self._add_line(lines, i, cx-e05,cy-e05, cx+e05,cy-e05)
        self._add_line(lines, i+1, cx+e05,cy-e05, cx+e05,cy+e05)
        self._add_line(lines, i+2, cx+e05,cy+e05, cx-e05,cy+e05)
        self._add_line(lines, i+3, cx-e05,cy+e05, cx-e05,cy-e05)

    def _add_line(self, lines, i, x1,y1,x2,y2):
        lines[i,0,0]=x1
        lines[i,0,1]=y1
        lines[i,1,0]=x2
        lines[i,1,1]=y2

   
    def plot_AMRmesh(self, res=None, center=None, axis=[0,1], box=None, group=None, newfig=False, axes=None, **params):
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
            self._add_square(lines, j, group.pos[index,axis[0]], group.pos[index,axis[1]], group.volume[index]**(1./(self.numdims)))
            j=j+4

        if(not 'color' in params):
            params['color']='black'    

        lc = mc.LineCollection(lines, **params)

        if(axes==None):
            ax=p.subplot(1,1,1)
        else:
            ax=axes

        ax.add_collection(lc)


    def get_AMRline(self, value, gradient=None, res=1024, center=None, axis=0, box=None, group=None):
        if group is None:
            group = self.part0
               
        resx=res
        resy=1
        
        axis0 = axis
        if axis0 == 0:
            axis1 = 1
        else:
            axis1 = 0
            
        b = self.box[axis0]
        box0 = self._validate_vector(box, b, len=1)[0]

        center = self._validate_vector(center, self.center)
        c = np.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]
        
        domainlen = np.max(self._domain)        
        domainc = np.zeros(3)
        domainc[0] = np.max(self._domain)/2
        domainc[1] = np.max(self._domain)/2.
        if self.numdims > 2:
            domainc[2] = np.max(self._domain)/2.

        posdata = group.pos.astype('float64')
        valdata = self._validate_value(value, posdata.shape[0], group).astype('float64')
        
        if gradient is None:
            data = calcGrid.calcAMRSlice( posdata, valdata, resx, resy, box0, 0., c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, axis0, axis1)
        else:
            graddata = self._validate_value(gradient, posdata.shape[0], group).astype('float64')
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
            
        myplot = self._plot_Line(result,log=log, newlabels=newlabels, newfig=newfig, axes=axes, **params)        
        
        return myplot

    def get_SPHproj( self, value, hsml="hsml", weights=None, normalized=True, res=None, center=None, axis=[0,1], box=None, group=None):
        if group is None:
            group = self.part0
            
        axis0 = axis[0]
        axis1 = axis[1]
        
        b = np.zeros(3)
        b[0] = self.box[axis0]
        b[1] = self.box[axis1]
        b[2] = self.box[3 - axis0 - axis1]
               
        center = self._validate_vector(center, self.center)
        box = self._validate_vector(box, b)
        
        r = np.zeros(3)
        r[0] = int(1024 * box[0]/box.max())
        r[1] = int(1024 * box[1]/box.max())
        r[2] = int(1024 * box[2]/box.max())
        res = self._validate_vector(res, r, len=3).astype(np.int)

        c = np.zeros( 3 )
        c[0] = center[axis0]
        c[1] = center[axis1]
        c[2] = center[3 - axis0 - axis1]

        pos = group.pos.astype( 'float64' )
        px = np.abs( pos[:,axis0] - c[0] )
        py = np.abs( pos[:,axis1] - c[1] )
        pz = np.abs( pos[:,3 - axis0 - axis1] - c[2] )

        pp, = np.where( (px <= 0.5*box[0]) & (py <= 0.5*box[1]) & (pz <= 0.5*box[2]) )
        print("Selected %d of %d particles." % (pp.size,self.npart))

        posdata = pos[pp,:]
        valdata = self._validate_value(value, posdata.shape[0], group)[pp].astype('float64')
        hsmldata = self._validate_value(hsml, posdata.shape[0], group)[pp].astype("float64")

        
        if weights is None:
            grid = calcGrid.calcGrid(posdata, hsmldata, valdata, res[0], res[1], res[2], box[0], box[1], box[2], c[0], c[1], c[2], proj=True, norm=normalized )
        else:
            weightdata = self._validate_value(weights, posdata.shape[0], group).astype("float64")
            grid = calcGrid.calcGrid(posdata, hsmldata, valdata, res[0], res[1], res[2], box[0], box[1], box[2], c[0], c[1], c[2], proj=True, norm=normalized, weights=weightdata )
            
        data = {}
        data['grid'] = grid
        
        if type(value) == str:
            data['name'] = value
        else:
            data['name'] = ""
        data['x'] = np.arange( res[0]+1, dtype="float64" ) / res[0] * box[0] - .5 * box[0] + c[0]
        data['y'] = np.arange( res[1]+1, dtype="float64" ) / res[1] * box[1] - .5 * box[1] + c[1]
        data['x2'] = (np.arange( res[0], dtype="float64" ) + 0.5) / res[0] * box[0] - .5 * box[0] + c[0]
        data['y2'] = (np.arange( res[1], dtype="float64" ) + 0.5) / res[1] * box[1] - .5 * box[1] + c[1]
        
        return data
    
    def plot_SPHproj(self, value, hsml="hsml", weights=None, normalized=True, log=False, res=None, center=None, axis=[0,1], box=None, group=None, vmin=None, vmax=None, dvalue=None, dweights=None, colorbar=True, cblabel=None, contour=False, newlabels=False, newfig=True, axes=None, **params):
        result = self.get_SPHproj(value, hsml=hsml, weights=weights, normalized=normalized, res=res, center=center, axis=axis, box=box, group=group)
        
        dresult = None
        if dvalue != None:
            dresult = self.get_SPHproj(dvalue, hsml=hsml, weights=dweights, normalized=normalized, res=res, center=center, axis=axis, box=box, group=group)
            
        myplot = self._plot_Slice(result,log=log, vmin=vmin, vmax=vmax, dresult=dresult, colorbar=colorbar, cblabel=cblabel, contour=contour, newlabels=newlabels, newfig=newfig, axes=axes, **params)

        return myplot
        
        
    def _plot_Slice(self, result, log=False, vmin=None, vmax=None, dresult=None, colorbar=True, cblabel=None, contour=False, newlabels=False, newfig=True, axes=None, **params):          
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
        
        return pc


    def _plot_Line(self, result, log=False, newlabels=False, newfig=True, axes=None, **params):          
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
        
        return pc      

    def get_Agrid( self, value, gradient=None, res=None, center=None, box=None, group=None):
        if self.numdims != 3:
            raise Exception( "not supported" )
        
        if group is None:
            group = self.part0
            
        center = self._validate_vector(center, self.center)
        box = self._validate_vector(box, self.box)
        
        r = np.zeros(3)
        r[0] = int(1024 * box[0]/box.max())
        r[1] = int(1024 * box[1]/box.max())
        r[2] = int(1024 * box[2]/box.max())
        res = self._validate_vector(res, r, len=3).astype(np.int)

        c = center

        pos = group.pos.astype( 'float64' )
        px = np.abs( pos[:,0] - c[0] )
        py = np.abs( pos[:,1] - c[1] )
        pz = np.abs( pos[:,2] - c[2] )

        pp, = np.where( (px < 0.5*box[0]) & (py < 0.5*box[1]) & (pz < 0.5*box[2]) )
        print("Selected %d of %d particles." % (pp.size,self.npart))

        posdata = pos[pp,:]
        valdata = self._validate_value(value, posdata.shape[0], group)[pp].astype('float64')
        
        if  gradient is None:
            data = calcGrid.calcASlice(posdata, valdata, res[0], res[1], box[0], box[1], c[0], c[1], c[2], 0, 1, boxz=box[2], nz=res[2], grid3D=True)
        else:
            data = calcGrid.calcASlice(posdata, valdata, res[0], res[1], box[0], box[1], c[0], c[1], c[2], 0, 1, grad=self._validate_value(gradient, posdata.shape[0], group)[pp].astype('float64'), boxz=box[2], nz=res[2], grid3D=True)
        
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
            
        center = self._validate_vector(center, self.center)
        box = self._validate_vector(box, self.box)
        
        r = np.zeros(3)
        r[0] = int(1024 * box[0]/box.max())
        r[1] = int(1024 * box[1]/box.max())
        r[2] = int(1024 * box[2]/box.max())
        res = self._validate_vector(res, r, len=3).astype(np.int)

        c = center
        
        domainlen = np.max(self._domain)        
        domainc = np.zeros(3)
        domainc[0] = np.max(self._domain)/2
        domainc[1] = np.max(self._domain)/2.
        domainc[2] = np.max(self._domain)/2.

        posdata = group.pos.astype( 'float64' )
        valdata = self._validate_value(value, posdata.shape[0], group).astype('float64')
        
        if gradient is None:
            data = calcGrid.calcAMRSlice( posdata, valdata, res[0], res[1], box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, 0, 1,boxz=box[2], nz=res[2], grid3D=True)
        else:
            data = calcGrid.calcAMRSlice( posdata, valdata, res[0], res[1], box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, 0, 1, grad=self._validate_value(gradient, posdata.shape[0], group)[pp].astype('float64'), boxz=box[2], nz=res[2], grid3D=True)

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
            
        center = self._validate_vector(center, self.center)
        box = self._validate_vector(box, self.box)
        
        r = np.zeros(3)
        r[0] = int(1024 * box[0]/box.max())
        r[1] = int(1024 * box[1]/box.max())
        r[2] = int(1024 * box[2]/box.max())
        res = self._validate_vector(res, r, len=3).astype(np.int)

        c = center

        posdata = group.pos.astype( 'float64' )
        valdata = self._validate_value(value, posdata.shape[0], group).astype('float64')
        hsmldata = self._validate_value(hsml, posdata.shape[0], group).astype("float64")

        
        if weights is None:
            grid = calcGrid.calcGrid(posdata, hsmldata, valdata, res[0], res[1], res[2], box[0], box[1], box[2], c[0], c[1], c[2], proj=False, norm=normalized )
        else:
            weightdata = self._validate_value(weights, posdata.shape[0], group).astype("float64")
            grid = calcGrid.calcGrid(posdata, hsmldata, valdata, res[0], res[1], res[2], box[0], box[1], box[2], c[0], c[1], c[2], proj=False, norm=normalized, weights=weightdata )
        
        return grid
    
    
