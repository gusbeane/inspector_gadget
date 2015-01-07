import numpy as np
import matplotlib.pyplot as p
import matplotlib
import time

from gadget.loader import Snapshot
from gadget.simulation import Simulation

import gadget.calcGrid as calcGrid


class DGSimulation(Simulation):
    
            
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

        posdata = group['pos'].astype('float64')
        amrlevel = group['amrlevel'].astype('int32')
        valdata = group[value].astype('float64')
        
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