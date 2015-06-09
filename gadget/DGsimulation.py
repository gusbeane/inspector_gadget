import numpy as np

try:
    import matplotlib.pyplot as p
    import matplotlib
except:
    print("Could not load matplotlib, plotting function will not work")

from gadget.loader import Snapshot
from gadget.simulation import Simulation

import gadget.calcGrid as calcGrid

class DGSimulation(Simulation):

    def __init__(self, filename, snapshot=None, filenum=None, format=None, fields=None, parttype=None, **param):
        """
        *filename* : The name of the snapshot file
        *format* : (optional) file format of the snapshot, otherwise this is guessed from the file name
        *fields* : (optional) list of fields to load, if None, all fields available in the snapshot are loaded
        *parttype* : (optional) array with particle type numbers to load, if None, all particles are loaded
        *toDouble* : (optional) converts all values of type float to double precision
        *onlyHeader* : (optiional) load only the snapshot header
        *verbose* : (optional) enable debug output
        *filter* : Only load a filtered subset of the snapshot, specified by the filter object.
        """
        super(DGSimulation, self).__init__(filename, snapshot=snapshot, filenum=filenum, format=format, fields=fields, parttype=parttype,**param)

        #store the polynomials in an array
        self._polynomials=[self._P0, self._P1, self._P2, self._P3, self._P4, self._P5]

        #set the number of base functions
        if(self.numdims==2):
            self.Nof_base_functions=(self.Degree_K+1)*(self.Degree_K+2)/2
        else:
            self.Nof_base_functions=(self.Degree_K+1)*(self.Degree_K+2)*(self.Degree_K+3)/6


        #set the base function table
        self.index_to_base_function_table = np.zeros([self.Nof_base_functions,self.numdims],dtype=np.int32)

        if(self.numdims==2):
            for i in np.arange(0,self.Nof_base_functions):
                px,py = self.index_to_base_function2d(i);
                self.index_to_base_function_table[i,0]=px
                self.index_to_base_function_table[i,1]=py
        else:
            for i in np.arange(0,self.Nof_base_functions):
                px,py,pz = self.index_to_base_function3d(i);
                self.index_to_base_function_table[i,0]=px
                self.index_to_base_function_table[i,1]=py
                self.index_to_base_function_table[i,2]=pz

    def _P0(self,x):
        return 1

    def _P1(self,x):
        return np.sqrt(3)*x

    def _P2(self,x):
        return np.sqrt(5.)*0.5*(-1.+x*(x*3.))

    def _P3(self,x):
        return np.sqrt(7.)*0.5*(x*(-3.+x*(x*5.)))

    def _P4(self,x):
        return 3./8.*(3.+x*(x*(-30.+x*(x*35.))))

    def _P5(self,x):
        return np.sqrt(11.)/8.*(x*(15.+x*(x*(-70.+x*(x*63.)))))




    def index_to_base_function2d(self,k):
        degree = 0
        counter = 0

        while(True):
            if(k<=counter):
                break
            else:
                degree=degree+1
                counter = counter + degree+1

        Px=0
        Py=degree

        while(k != counter):
            counter=counter-1
            Px=Px+1
            Py=Py-1

        return (Px,Py)


    def index_to_base_function3d(self,k):
        counter=0

        for deg_k in np.arange(0,self.Degree_K+1):
            for u in np.arange(0, deg_k+1):
                for v in np.arange(0,deg_k-u+1):
                    for w in np.arange(0,deg_k-u-v+1):
                        if(u+v+w==deg_k):
                            if(counter==k):
                                Px=w
                                Py=v
                                Pz=u
                                return (Px,Py,Pz)
                            else:
                                counter=counter+1

        print("Index:", k)
        raise Exception("shouldn't be reached!")



    def base_function_value(self, index, cell_x, cell_y, cell_z, cell_dl, x, y, z):
        xi_1=2./cell_dl*(x-cell_x)
        xi_2=2./cell_dl*(y-cell_y)
        xi_3=2./cell_dl*(z-cell_z)

        if(self.numdims==2):
            px=self.index_to_base_function_table[index,0]
            py=self.index_to_base_function_table[index,1]

            return self._polynomials[px](xi_1)*self._polynomials[py](xi_2)

        else:
            px=self.index_to_base_function_table[index,0]
            py=self.index_to_base_function_table[index,1]
            pz=self.index_to_base_function_table[index,2]

            return self._polynomials[px](xi_1)*self._polynomials[py](xi_2)*self._polynomials[pz](xi_3)


    def get_DGvalue(self, value, x, y, z=0, group=None):
        if group is None:
            group = self.part0

        id=np.unique(self.get_AMRline("id",box=[0,0,0],center=[x,y,z],res=1)["grid"])
        id=id.astype(self.id.dtype)
        id=id[0]

        cell_index=np.where(self.id==id)[0][0]

        result = 0

        cell_x=self.pos[cell_index][0]
        cell_y=self.pos[cell_index][1]
        cell_z=self.pos[cell_index][2]
        cell_dl=np.max(self._domain)/(2.**self.amrlevel[cell_index])
        
        posdata = group.pos
        valdata = self._validate_value(value, posdata.shape[0], group)

        for i in np.arange(0,self.Nof_base_functions):
            result = result + valdata[cell_index][i] * self.base_function_value(i, cell_x, cell_y, cell_z, cell_dl, x, y, z)

        return result


    def get_DGslice(self, value, res=1024, center=None, axis=[0,1], box=None, group=None):
        if group is None:
            group = self.part0

        center = self._validate_vector(center, self.center)
        box = self._validate_vector(box, self.BoxSize,len=2)

        axis0 = axis[0]
        axis1 = axis[1]

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
        amrlevel = group.amrlevel.astype('int32')
        dgdims = self.numdims.astype('int32')
        degree_k = self.Degree_K.astype('int32')
        valdata = self._validate_value(value, posdata.shape[0], group).astype('float64')

        data = calcGrid.calcDGSlice( posdata, valdata, amrlevel, dgdims, degree_k, res, res, box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, axis0, axis1, boxz=box[2])


        if type(value) == str:
            data['name'] = value
        else:
            data['name'] = ""
        data['x'] = np.arange( res+1, dtype="float64" ) / res * box[0] - .5 * box[0] + c[0]
        data['y'] = np.arange( res+1, dtype="float64" ) / res * box[1] - .5 * box[1] + c[1]
        data['x2'] = (np.arange( res, dtype="float64" ) + 0.5) / res * box[0] - .5 * box[0] + center[0]
        data['y2'] = (np.arange( res, dtype="float64" ) + 0.5) / res * box[1] - .5 * box[1] + center[1]

        return data


    def plot_DGslice(self, value, log=False, res=1024, center=None, axis=[0,1], box=None, group=None, vmin=None, vmax=None, dvalue=None, dgradient=None, colorbar=True, cblabel=None, contour=False, newlabels=False, newfig=True, axes=None, **params):
        result = self.get_DGslice(value, res=res, center=center, axis=axis, box=box, group=group)

        dresult = None
        if dvalue != None:
            dresult = self.get_DGslice(dvalue, res=res, center=center, axis=axis, box=box, group=group)

        myplot = self._plot_Slice(result,log=log, vmin=vmin, vmax=vmax, dresult=dresult, colorbar=colorbar, cblabel=cblabel, contour=contour, newlabels=newlabels, newfig=newfig, axes=axes, **params)

        return myplot
    
    def get_DGgrid( self, value, res=1024, center=None, box=None, group=None):
        if self.numdims != 3:
            raise Exception( "not supported" )
        
        if group is None:
            group = self.part0
            
        center = self._validate_vector(center, self.center)
        box = self._validate_vector(box, self.box)

        c = center
        
        domainlen = np.max(self._domain)        
        domainc = np.zeros(3)
        domainc[0] = np.max(self._domain)/2
        domainc[1] = np.max(self._domain)/2.
        domainc[2] = np.max(self._domain)/2.
        
        posdata = group.pos.astype('float64')
        amrlevel = group.amrlevel.astype('int32')
        dgdims = self.numdims.astype('int32')
        degree_k = self.Degree_K.astype('int32')
        valdata = self._validate_value(value, posdata.shape[0], group).astype('float64')
        
        data = calcGrid.calcDGSlice( posdata, valdata, amrlevel, dgdims, degree_k, res, res, box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, 0, 1,boxz=box[2], grid3D=True)
                
        if type(value) == str:
            data['name'] = value
        else:
            data['name'] = ""
        
        return data

    def plot_DGline(self, value="dgw0", res=1024, res_per_cell=100, ylim=None, box=None, center=None, group=None, shift=0, axis=0, newfig=True, axes=None,colorful=False,colors=[None],**params):
        if newfig and axes==None:
            fig = p.figure()
            axes = p.gca()
        elif axes==None:
            axes = p.gca()

        ids=np.unique(self.get_AMRline("id",box=box,center=center,axis=axis,res=res,group=group)["grid"])

        center = self._validate_vector(center, self.center)
        box = self._validate_vector(box, self.BoxSize,len=2)
  
        myplot=None

        if(colors[0]!=None):
            self._color_counter=0

        for i in ids:
            index=np.where(self.id==i)[0][0]
            myplot,=self._plot_1dsolution(cell_index=index, value=value, axis=axis, axes=axes, center=center, res=res_per_cell, group=group, shift=shift,colorful=colorful,colors=colors,**params)

        if(ylim!=None):
            p.ylim(ylim[0],ylim[1])

        p.show()

        return myplot


    def plot_DGline_dir(self, value="dgw0", res=4096, group=None, ylim=None, newfig=True, start=[0.5,0.5,0.5], end=[1,0.5,0.5], shift=0, axes=None, colorful=False,colors=[None],**params):
        if group is None:
            group = self.part0
            
        if newfig and axes==None:
            fig = p.figure()
            axes = p.gca()
        elif axes==None:
            axes = p.gca()

        if not 'color' in params and colors[0]==None and colorful==False:
            params['color']="black"

        if colors[0] != None:
            self._color_counter=0

        #plot
        xvals=np.linspace(start[0],end[0], res)
        yvals=np.linspace(start[1],end[1], res)
        zvals=np.linspace(start[2],end[2], res)

        start=np.array(start)
        end=np.array(end)
        l=start-end

        l=np.sqrt(l[0]**2+l[1]**2+l[2]**2)

        X=np.linspace(0,l,res)
        Y=np.zeros(res)

        id=np.int(self.get_AMRline("id",center=[xvals[0],yvals[0],zvals[0]],box=[0,0,0],res=1)['grid'][0])
        cell_index=np.where(self.id==id)[0][0]

        cmin=0
        cnof=0

        cell_x=self.pos[cell_index][0]
        cell_y=self.pos[cell_index][1]
        cell_z=self.pos[cell_index][2]
        cell_dl=np.max(self._domain)/(2.**self.amrlevel[cell_index])

        cells_crossed=0
        
        posdata = group.pos
        valdata = self._validate_value(value, posdata.shape[0], group)

        for j in np.arange(0,res):

              if(not(xvals[j]<=cell_x+0.5*cell_dl and xvals[j]>=cell_x-0.5*cell_dl and yvals[j]<=cell_y+0.5*cell_dl and yvals[j]>=cell_y-0.5*cell_dl and zvals[j]<=cell_z+0.5*cell_dl and zvals[j]>=cell_z-0.5*cell_dl)):
                  #cell crossed
                  if colors[0] != None:
                      params['color']=colors[self._color_counter%np.shape(colors)[0]] 
                      self._color_counter = self._color_counter+1
  
                  myplot,=axes.plot(X[cmin:cmin+cnof]+shift,Y[cmin:cmin+cnof],**params)
                  cmin=cmin+cnof
                  cnof=0
          
                  id=np.int(self.get_AMRline("id",center=[xvals[j],yvals[j],zvals[j]],box=[0,0,0],res=1,group=group)['grid'][0])


                  print("xval, yval, zval", (xvals[j],yvals[j],zvals[j]))
                  print("cx, cy, cz, dl", (cell_x,cell_y,cell_z, cell_dl))

                  cell_index=np.where(self.id==id)[0][0]

                  cell_x=self.pos[cell_index][0]
                  cell_y=self.pos[cell_index][1]
                  cell_z=self.pos[cell_index][2]
                  cell_dl=np.max(self._domain)/(2.**self.amrlevel[cell_index])
                
                  cells_crossed=cells_crossed+1

              result=0

              for i in np.arange(0,self.Nof_base_functions):
                  result = result + valdata[cell_index][i] * self.base_function_value(i, cell_x, cell_y, cell_z, cell_dl, xvals[j], yvals[j], zvals[j])

              Y[j]=result
              cnof=cnof+1


        if(ylim!=None):
            p.ylim(ylim[0],ylim[1])

        print("cells crossed:", cells_crossed)

        if colors[0] != None:
            params['color']=colors[self._color_counter%np.shape(colors)[0]] 
            self._color_counter = self._color_counter+1

        myplot,=axes.plot(X[cmin:cmin+cnof]+shift,Y[cmin:cmin+cnof],**params)
        p.show()

        return myplot

    def _plot_1dsolution(self, cell_index, value, axis, axes, center, res, group=None, shift=0, colorful=False, colors=[None], **params):
        if group is None:
            group = self.part0
            
        cell_x=self.pos[cell_index][0]
        cell_y=self.pos[cell_index][1]
        cell_z=self.pos[cell_index][2]
        cell_dl=np.max(self._domain)/(2.**self.amrlevel[cell_index])
        
        posdata = group.pos
        valdata = self._validate_value(value, posdata.shape[0], group)

        #line in x-direction
        if(axis==0):
            X=np.linspace(cell_x-0.5*cell_dl,cell_x+0.5*cell_dl,res)
            Y=np.zeros(res)
            yval=center[1]
            zval=center[2]

            j=0

            for xval in X:
                result = 0

                for i in np.arange(0,self.Nof_base_functions):
                    result = result + valdata[cell_index][i] * self.base_function_value(i, cell_x, cell_y, cell_z, cell_dl, xval, yval, zval)

                Y[j]=result
                j=j+1


        #line in y-direction
        elif(axis==1):
            X=np.linspace(cell_y-0.5*cell_dl,cell_y+0.5*cell_dl,res)
            Y=np.zeros(res)
            xval=center[0]
            zval=center[2]

            j=0

            for yval in X:
                result = 0

                for i in np.arange(0,self.Nof_base_functions):
                    result = result + valdata[cell_index][i] * self.base_function_value(i, cell_x, cell_y, cell_z, cell_dl, xval, yval, zval)

                Y[j]=result
                j=j+1

    #line in z-direction
        elif(axis==2):
            X=np.linspace(cell_z-0.5*cell_dl,cell_z+0.5*cell_dl,res)
            Y=np.zeros(res)
            xval=center[0]
            yval=center[1]

            j=0

            for zval in X:
                result = 0

                for i in np.arange(0,self.Nof_base_functions):
                    result = result + valdata[cell_index][i] * self.base_function_value(i, cell_x, cell_y, cell_z, cell_dl, xval, yval, zval)

                Y[j]=result
                j=j+1

        else:
            raise Exception("axis not valid!, axis=1/2/3")


        if not 'color' in params and colors[0]==None and colorful==False:
            params['color']="black"


        if colors[0] != None:
            params['color']=colors[self._color_counter%np.shape(colors)[0]] 
            self._color_counter = self._color_counter+1


        myplot=axes.plot(X+shift,Y,**params)

        return myplot
