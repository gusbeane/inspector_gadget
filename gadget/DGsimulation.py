import numpy as np
import matplotlib.pyplot as p
import matplotlib
import time
import sys

from gadget.loader import Snapshot
from gadget.simulation import Simulation

import gadget.calcGrid as calcGrid 

class DGSimulation(Simulation):

	def __init__(self, *args, **kwargs):
                """
                *filename* : The name of the snapshot file
                *format* : (optional) file format of the snapshot, otherwise this is guessed from the file name
                *fields* : (optional) list of fields to load, if None, all fields available in the snapshot are loaded
                *parttype* : (optional) array with particle type numbers to load, if None, all particles are loaded
                *combineFiles* : (optional) if False only on part of the snapshot is loaded at a time, use nextFile() to go the next file.
                *toDouble* : (optional) converts all values of type float to double precision
                *onlyHeader* : (optiional) load only the snapshot header
                *verbose* : (optional) enable debug output
                *filter* : Only load a filtered subset of the snapshot, specified by the filter object.
                """

                super(DGSimulation, self).__init__(*args, **kwargs)
		self.__polynomials=[self.__P0, self.__P1, self.__P2, self.__P3, self.__P4, self.__P5]


	def __P0(self,x):
		return 1

	def __P1(self,x):
		return np.sqrt(3)*x

	def __P2(self,x):
		return np.sqrt(5.)*0.5*(-1.+x*(x*3.))

	def __P3(self,x):
		return np.sqrt(7.)*0.5*(x*(-3.+x*(x*5.)))

	def __P4(self,x):
		return 3./8.*(3.+x*(x*(-30.+x*(x*35.))))

	def __P5(self,x):
		return np.sqrt(11.)/8.*(x*(15.+x*(x*(-70.+x*(x*63.)))))




	def __index_to_base_function(self,k):
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


	def base_function_value(self, index, cell_x, cell_y, cell_dl, x, y):

		xi_1=2./cell_dl*(x-cell_x)
		xi_2=2./cell_dl*(y-cell_y)

		px,py = self.__index_to_base_function(index);

		return self.__polynomials[px](xi_1)*self.__polynomials[py](xi_2)


    
            
 	def get_DGslice(self, value, res=1024, center=None, axis=[0,1], box=None, group=None):
		if group is None:
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

		posdata = group.pos.astype('float64')
		amrlevel = group.amrlevel.astype('int32')
		valdata = self.__validate_value__(value, posdata.shape[0], group).astype('float64')
		
		data = calcGrid.calcDGSlice( posdata, valdata, amrlevel, res, res, box[0], box[1], c[0], c[1], c[2], domainc[0], domainc[1], domainc[2], domainlen, axis0, axis1, boxz=box[2])

		
		data['name'] = value
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
		    
		self.__plot_Slice__(result,log=log, vmin=vmin, vmax=vmax, dresult=dresult, colorbar=colorbar, cblabel=cblabel, contour=contour, newlabels=newlabels, newfig=newfig, axes=axes, **params)

	def plot_DGline(self, value="dgw0", res=1024, res_per_cell=100, ylim=None, box=None, center=None,axis=[0,1], newfig=True, axes=None,colorful=False,**params):

		if newfig and axes==None:
		    fig = p.figure()
		    axes = p.gca()
		elif axes==None:
		    axes = p.gca()

		ids=np.unique(self.get_AMRline("id",box=box,center=center,axis=axis,res=res)["grid"])

		for i in ids:
			index=np.where(self.id==i)[0][0]
			self.__plot_1dsolution(cell_index=index, value=value, axis=axis, res=res_per_cell, axes=axes,colorful=colorful,**params)

		if(ylim!=None):
			p.ylim(ylim[0],ylim[1])

		p.show()

	def __plot_1dsolution(self, cell_index, value, axis, axes, res, colorful=False,**params):

		nof_base_functions=np.shape(self.dgw0)[1]
		cell_x=self.pos[cell_index][0]
		cell_y=self.pos[cell_index][1]
		cell_dl=self.boxsize/(2.**self.amrlevel[cell_index])

		#line in x-direction
		if(axis==[0,1]):
			X=np.linspace(cell_x-0.5*cell_dl,cell_x+0.5*cell_dl,res)
			Y=np.zeros(res)
			yval=cell_y

			j=0

			for xval in X:
				result = 0

				for i in np.arange(0,nof_base_functions):
					result = result + self.data[value][cell_index][i] * self.base_function_value(i, cell_x, cell_y, cell_dl, xval, yval)

				Y[j]=result
				j=j+1


		#line in y-direction
		elif(axis==[1,0]):
			X=np.linspace(cell_y-0.5*cell_dl,cell_y+0.5*cell_dl,res)
			Y=np.zeros(res)
			xval=cell_x

			j=0

			for yval in X:
				result = 0

				for i in np.arange(0,nof_base_functions):
					result = result + self.data[value][cell_index][i] * self.base_function_value(i, cell_x, cell_y, cell_dl, xval, yval)

				Y[j]=result
				j=j+1

		else:	
			sys.exit("axis not valid!")
			
	
		if not 'color' in params and colorful==False:
			params['color']="black"

		axes.plot(X,Y,**params)
