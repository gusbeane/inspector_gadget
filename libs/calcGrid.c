//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "sph.h"
#include "dg.h"

#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))
#define sqr(x) ((x)*(x))

inline void PyDict_SetStolenItem(PyObject *dict, const char *key, PyObject *object) {
	PyDict_SetItemString(dict, key, object);
	Py_DECREF(object);
}

inline double _getkernel( double h, double r2 ) {
	double coeff1, coeff2, coeff5;
	double hinv, hinv3, u;
	coeff1 = 8.0 / M_PI;
	coeff2 = coeff1 * 6.0;
	coeff5 = coeff1 * 2.0;

	hinv = 1.0 / h;
	hinv3 = hinv*hinv*hinv;
	u = sqrt(r2)*hinv;
	if (u < 0.5) {
		return hinv3 * ( coeff1 + coeff2*(u-1.0)*u*u );
	} else {
	  	return hinv3 * coeff5 * (1.0-u) * (1.0-u) * (1.0-u);
	}
}

PyObject* _calcGrid(PyObject *self, PyObject *args, PyObject *kwargs) {
	PyArrayObject *pos, *hsml,*value, *weights, *pyGrid, *pyWeightGrid;
	int npart, nx, ny, nz, cells;
	int dims[3];
	double bx, by, bz, cx, cy, cz;
	double *data_pos, *data_hsml, *data_weights, *data_value;
	double *grid, *weightGrid;
	int part, proj, norm;
	double px, py, pz, h, h2, weight, v, cpx, cpy, cpz, r2, sum;
	int x, y, z0, z1;
	int xmin, xmax, ymin, ymax, zmin, zmax, zmid;
	double cellsizex, cellsizey, cellsizez;
	clock_t start;
	
	int w;

	start = clock();

	proj = 0;
	norm = 0;
	weights = 0;

	char *kwlist[] = {"pos", "hsml", "value", "nx", "ny", "nz", "boxx", "boxy", "boxz", "centerx", "centery", "centerz", "proj", "norm", "weights", NULL} ;
	if (!PyArg_ParseTupleAndKeywords( args, kwargs, "O!O!O!iiidddddd|iiO!:calcGrid( pos, hsml, value, nx, ny, nz, boxx, boxy, boxz, centerx, centery, centerz, [proj, norm, weights] )", kwlist, &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &value, &nx, &ny, &nz, &bx, &by, &bz, &cx, &cy, &cz, &proj, &norm,&PyArray_Type, &weights))
	  {
		return 0;
	  }

	if (pos->nd != 2 || pos->dimensions[1] != 3 || pos->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "pos has to be of dimensions [n,3] and type double" );
		return 0;
	}

	if (hsml->nd != 1 || hsml->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "hsml has to be of dimension [n] and type double" );
		return 0;
	}

	if (value->nd != 1 || value->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "value has to be of dimension [n] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0]  || npart != value->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos and hsml have to have the same size in the first dimension" );
		return 0;
	}

	if (proj) {
		dims[0] = nx;
		dims[1] = ny;
		pyGrid = (PyArrayObject *)PyArray_FromDims( 2, dims, PyArray_DOUBLE );
		grid = (double*)pyGrid->data;
		cells = nx*ny;
	} else {
		dims[0] = nx;
		dims[1] = ny;
		dims[2] = nz;
		pyGrid = (PyArrayObject *)PyArray_FromDims( 3, dims, PyArray_DOUBLE );
		grid = (double*)pyGrid->data;
		cells = nx*ny*nz;		
	}

	if(weights)
	  {
	    w = 1;

	    if (proj) {
	        pyWeightGrid = (PyArrayObject *)PyArray_FromDims( 2, dims, PyArray_DOUBLE );
	    } else {
	        pyWeightGrid = (PyArrayObject *)PyArray_FromDims( 3, dims, PyArray_DOUBLE );
	    }
	    weightGrid = (double*)pyWeightGrid->data;
	    memset( weightGrid, 0, cells*sizeof(double) );

	    data_weights = (double*) weights->data;

	    if (npart != weights->dimensions[0] )
	      {
	        PyErr_SetString( PyExc_ValueError, "weights and pos have to have the same size in the first dimension" );
	        return 0;
	      }
	  }
	else
	  {
	    w = 0;
	  }

	memset( grid, 0, cells*sizeof(double) );

	cellsizex = bx / nx;
	cellsizey = by / ny;
	cellsizez = bz / nz;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_value = (double*)value->data;

	for (part=0; part<npart; part++) {
		px = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		py = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		pz = *data_pos;
		data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);
		
		h = *data_hsml;
		data_hsml = (double*)((char*)data_hsml + hsml->strides[0]);
		h2 = h*h;

		v = *data_value;
		data_value = (double*)((char*)data_value + value->strides[0]);

		if(w)
		  {
		    weight = *data_weights;
		    data_weights = (double*)((char*)data_weights + weights->strides[0]);
		  }

		xmin = max( floor( (px - h - cx + 0.5*bx) / cellsizex ), 0 );
		xmax = min( ceil( (px + h - cx + 0.5*bx) / cellsizex ), nx-1 );
		ymin = max( floor( (py - h - cy + 0.5*by) / cellsizey ), 0 );
		ymax = min( ceil( (py + h - cy + 0.5*by) / cellsizey ), ny-1 );
		zmin = max( floor( (pz - h - cz + 0.5*bz) / cellsizez ), 0 );
		zmax = min( ceil( (pz + h - cz + 0.5*bz) / cellsizez ), nz-1 );

		zmid = floor( 0.5 * (zmin+zmax) + 0.5 );

		if (xmin < nx && ymin < ny && xmax >= 0 && ymax >= 0 && zmin < nz && zmax >= 0) {
			if (norm) {
				sum = 0.;
				for (x=xmin; x<=xmax; x++) {
					cpx = -0.5*bx + bx*(x+0.5)/nx;
					for (y=ymin; y<=ymax; y++) {
						cpy = -0.5*by + by*(y+0.5)/ny;

						if (proj) {
							r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) );
							if (r2 < h2) {
								sum += h * _getkernel( h, r2 );
							}
						} else {				
							for (z0=zmid; z0>=zmin; z0--) {
								cpz = -0.5*bz + bz*(z0+0.5)/nz;
								r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
								if (r2 > h2) break;
								sum += _getkernel( h, r2 );
							}

							for (z1=zmid+1; z1<=zmax; z1++) {
								cpz = -0.5*bz + bz*(z1+0.5)/nz;
								r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
								if (r2 > h2) break;
								sum += _getkernel( h, r2 );
							}
						}
					}
				}
			} else {
				sum = 1.0;
			}
			
			for (x=xmin; x<=xmax; x++) {
				cpx = -0.5*bx + bx*(x+0.5)/nx;
				for (y=ymin; y<=ymax; y++) {
					cpy = -0.5*by + by*(y+0.5)/ny;
					
					if (proj) {
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) );
						if (r2 < h2) {
						    if(w)
						      {
						        grid[x*ny + y] += h * _getkernel( h, r2 ) * v * weight / sum;
						        weightGrid[x*ny + y] += h * _getkernel( h, r2 ) * weight / sum;
						      }
						    else
						      {
							grid[x*ny + y] += h * _getkernel( h, r2 ) * v / sum;
						      }
						}
					} else {				
					    for (z0=zmid; z0>=zmin; z0--) {
					        cpz = -0.5*bz + bz*(z0+0.5)/nz;
					        r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
					        if (r2 > h2) break;


					        if(w)
					          {
					            grid[(x*ny + y)*nz + z0] += _getkernel( h, r2 ) * v * weight / sum;
					            weightGrid[(x*ny + y)*nz + z0] += _getkernel( h, r2 ) * weight / sum;
					          }
					        else
					          {
					            grid[x*ny + y] += _getkernel( h, r2 ) * v / sum;
					          }
						}

						for (z1=zmid+1; z1<=zmax; z1++) {
							cpz = -0.5*bz + bz*(z1+0.5)/nz;
							r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
							if (r2 > h2) break;
	                                                if(w)
	                                                  {
	                                                    grid[(x*ny + y)*nz + z1] += _getkernel( h, r2 ) * v * weight / sum;
	                                                    weightGrid[(x*ny + y)*nz + z1] += _getkernel( h, r2 ) * weight / sum;
	                                                  }
	                                                else
	                                                  {
	                                                    grid[(x*ny + y)*nz + z1] += _getkernel( h, r2 ) * v / sum;
	                                                  }
						}
					}
				}
			}	
		}
	}

	if(w)
	  {
      int i;
	    for(i = 0; i < cells; i++)
	      {
	        if(weightGrid[i] > 0.)
	          {
	            grid[i] = grid[i] / weightGrid[i];
	          }
	      }

	  }

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyGrid );
}


PyObject* _calcASlice(PyObject *self, PyObject *args, PyObject *kwargs) {
        PyArrayObject *pos, *value, *grad, *pyGrid, *pyNeighbours, *pyContours;
	int npart, nx, ny, nz, axis0, axis1, proj, grid3D, ngbs;
	int x, y, z, i, j, cell;;                                                                               
	double bx, by, bz, cx, cy, cz;
	double *real_pos, *real_value, *real_grad;
	double *grid, coord[3];
	double cellsizex, cellsizey, cellsizez;
	int neighbour, *neighbours, *contours;
	t_sph_tree tree;
	PyObject *dict;
	clock_t start;
        char *kwlist[] = {"pos", "value", "nx", "ny", "boxx", "boxy", "centerx", "centery", "centerz", "axis0", "axis1", "proj", "grad", "nz", "boxz", "grid3D", "ngbs", NULL} ;
	
	start = clock();

	axis0 = 0;
	axis1 = 1;
        proj = 0;
	grad = NULL;
        nz = 0;
        bz = 0;
	grid3D = 0;
	ngbs = 1;
	if (!PyArg_ParseTupleAndKeywords( args, kwargs, "O!O!iiddddd|iiiO!idii:calcASlice( pos, value, nx, ny, boxx, boxy, centerx, centery, centerz, [axis0, axis1, proj, grad, nz, boxz, grid3D, ngbs] )", kwlist, &PyArray_Type, &pos, &PyArray_Type, &value, &nx, &ny, &bx, &by, &cx, &cy, &cz, &axis0, &axis1, &proj, &PyArray_Type, &grad, &nz, &bz, &grid3D, &ngbs )) {
		return 0;
	}
	
	if (proj || grid3D)
	{
                if (nz == 0)
                        nz = max( nx, ny );
	        if (bz == 0)
                        bz = max( bx, by );
	}
	else
	{
                nz = 1;
                bz = 0;
	}
	
	if (proj || grid3D)
		ngbs = 0;

	if (pos->nd != 2 || pos->dimensions[1] != 3 || pos->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "pos has to be of dimensions [n,3] and type double" );
		return 0;
	}

	if (value->nd != 1 || value->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "value has to be of dimension [n] and type double" );
		return 0;
	}

	
	if (grad && (grad->nd != 2 || grad->dimensions[1] != 3 || value->descr->type_num != PyArray_DOUBLE)) {
		PyErr_SetString( PyExc_ValueError, "grad has to be of dimension [n,3] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != value->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos and value have to have the same size in the first dimension" );
		return 0;
	}
	
	if (grid3D) {
		npy_intp dims[3];
		dims[0] = nx;
		dims[1] = ny;
		dims[2] = nz;
		pyGrid = (PyArrayObject *)PyArray_SimpleNew( 3, (npy_intp*)dims, NPY_DOUBLE );
		grid = (double*)pyGrid->data;
		memset( grid, 0, nx * ny * nz * sizeof(double) );
		printf( "Doing 3D Grid of size %d x %d x %d\n", nx, ny, nz );
	} else {
		npy_intp dims[2];
		dims[0] = nx;
		dims[1] = ny;
		pyGrid = (PyArrayObject *)PyArray_SimpleNew( 2, (npy_intp*)dims, NPY_DOUBLE );
		grid = (double*)pyGrid->data;
		memset( grid, 0, nx * ny * sizeof(double) );
		
		if (ngbs) {
		        pyNeighbours = (PyArrayObject *)PyArray_SimpleNew( 2, (npy_intp*)dims, NPY_INT );
		        neighbours = (int*)pyNeighbours->data;
		        pyContours = (PyArrayObject *)PyArray_SimpleNew( 2, (npy_intp*)dims, NPY_INT );
		        contours = (int*)pyContours->data;
		        memset( contours, 0, nx * ny * sizeof(int) );
	        }
	}

	real_pos = (double*)malloc( 3 * npart * sizeof( double ) );
	real_value = (double*)malloc( npart * sizeof( double ) );
	if (grad) real_grad = (double*)malloc( 3 * npart * sizeof( double ) );
	for (i=0; i<npart; i++) {
	  for (j=0; j<3; j++) {
	    real_pos[i*3+j] = *(double*)((char*)pos->data + i*pos->strides[0] + j*pos->strides[1]);
	    if (grad) real_grad[i*3+j] = *(double*)((char*)grad->data + i*grad->strides[0] + j*grad->strides[1]);
	  }
	  real_value[i] = *(double*)((char*)value->data + i*value->strides[0]);
	}

	double center[] = {0.,0.,0.};
	createTree( &tree, npart, real_pos , getDomainLen(npart, real_pos), center);

	cellsizex = bx / nx;
	cellsizey = by / ny;

	if (proj || grid3D)
		cellsizez = bz / nz;
	else
		cellsizez = 0;

	coord[ 3 - axis0 - axis1 ] = cz;
	neighbour = 0;
	cell = 0;

	for (x=0; x<nx; x++) {
	  coord[ axis0 ] = cx - 0.5 * bx + cellsizex * (0.5 + x);
	  for (y=0; y<ny; y++) {
	    coord[ axis1 ] = cy - 0.5 * by + cellsizey * (0.5 + y);
	    for (z=0; z<nz; z++) {
    	      coord[ 3-axis0-axis1 ] = cz - 0.5 * bz + cellsizez * (0.5 + z);

	      getNearestNeighbour( &tree, real_pos, coord, &neighbour );

	      if (!grad)
	        grid[ cell ] += real_value[ neighbour ];
	      else
	        grid[ cell ] += real_value[ neighbour ]
		+ (coord[0] - real_pos[ neighbour*3 + 0 ]) * real_grad[ neighbour*3 + 0 ]
		+ (coord[1] - real_pos[ neighbour*3 + 1 ]) * real_grad[ neighbour*3 + 1 ]
		+ (coord[2] - real_pos[ neighbour*3 + 2 ]) * real_grad[ neighbour*3 + 2 ];
	      if (ngbs) neighbours[ cell ] = neighbour;
	
	      if (grid3D) cell++; /* 3d grid */
            }
	    if (!grid3D) cell++; /* 2d projection or grid */
          }

	  if (ngbs) neighbour = neighbours[ cell - ny ];
	}

	freeTree( &tree );

	free( real_pos );
	free( real_value );
	if (grad) free( real_grad );

        if (ngbs)
	  for (x=1; x<nx-1; x++) {
	    for (y=1; y<ny-1; y++) {
	      neighbour = neighbours[ x*ny + y ];
              if (neighbours[ (x-1)*ny + y-1 ] != neighbour ||
	          neighbours[  x   *ny + y-1 ] != neighbour ||
	          neighbours[ (x+1)*ny + y-1 ] != neighbour ||
	          neighbours[ (x-1)*ny + y   ] != neighbour ||
	          neighbours[ (x+1)*ny + y   ] != neighbour ||
	          neighbours[ (x-1)*ny + y+1 ] != neighbour ||
	          neighbours[  x   *ny + y+1 ] != neighbour ||
	          neighbours[ (x+1)*ny + y+1 ] != neighbour) {
	        contours[ x*ny + y ] = 1;
	      } else {
                contours[ x*ny + y ] = 0;
	      }
            }
          }

	dict = PyDict_New();
	PyDict_SetStolenItem( dict, "grid", (PyObject*)pyGrid );
	
	if (ngbs)
	{
		PyDict_SetStolenItem( dict, "neighbours", (PyObject*)pyNeighbours );
		PyDict_SetStolenItem( dict, "contours", (PyObject*)pyContours );
	}

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return dict;
}

PyObject* _calcAMRSlice(PyObject *self, PyObject *args, PyObject *kwargs) {
        PyArrayObject *pos, *value, *grad, *pyGrid, *pyNeighbours, *pyContours;
        int npart, nx, ny, nz, axis0, axis1, proj, grid3D, ngbs;
        int x, y, z, i, j, cell;;
        double bx, by, bz, cx, cy, cz;
        double domainx, domainy, domainz, domainlen;
        double *real_pos, *real_value, *real_grad;
        double *grid, coord[3];
        double cellsizex, cellsizey, cellsizez;
        int neighbour, *neighbours, *contours;
        t_sph_tree tree;
        PyObject *dict;
        clock_t start;
        char *kwlist[] = {"pos", "value", "nx", "ny", "boxx", "boxy", "centerx", "centery", "centerz", "domainx", "domainy", "domainz", "domainlen", "axis0", "axis1", "proj", "grad", "nz", "boxz", "grid3D", "ngbs", NULL} ;

        start = clock();

        axis0 = 0;
        axis1 = 1;
        proj = 0;
        grad = NULL;
        nz = 0;
        bz = 0;
        grid3D = 0;
        ngbs = 1;
        if (!PyArg_ParseTupleAndKeywords( args, kwargs, "O!O!iiddddddddd|iiiO!idii:calcAMRSlice( pos, value, nx, ny, boxx, boxy, centerx, centery, centerz, domainx, domainy, domainz, domainlen, [axis0, axis1, proj, grad, nz, boxz, grid3D, ngbs] )", kwlist, &PyArray_Type, &pos, &PyArray_Type, &value, &nx, &ny, &bx, &by, &cx, &cy, &cz, &domainx, &domainy, &domainz, &domainlen, &axis0, &axis1, &proj, &PyArray_Type, &grad, &nz, &bz, &grid3D, &ngbs )) {
                return 0;
        }

        if (proj || grid3D)
        {
                if (nz == 0)
                        nz = max( nx, ny );
                if (bz == 0)
                        bz = max( bx, by );
        }
        else
        {
                nz = 1;
                bz = 0;
        }

        if (proj || grid3D)
                ngbs = 0;

        if (pos->nd != 2 || pos->dimensions[1] != 3 || pos->descr->type_num != PyArray_DOUBLE) {
                PyErr_SetString( PyExc_ValueError, "pos has to be of dimensions [n,3] and type double" );
                return 0;
        }

        if (value->nd != 1 || value->descr->type_num != PyArray_DOUBLE) {
                PyErr_SetString( PyExc_ValueError, "value has to be of dimension [n] and type double" );
                return 0;
        }


        if (grad && (grad->nd != 2 || grad->dimensions[1] != 3 || value->descr->type_num != PyArray_DOUBLE)) {
                PyErr_SetString( PyExc_ValueError, "grad has to be of dimension [n,3] and type double" );
                return 0;
        }

        npart = pos->dimensions[0];
        if (npart != value->dimensions[0]) {
                PyErr_SetString( PyExc_ValueError, "pos and value have to have the same size in the first dimension" );
                return 0;
        }

        if (grid3D) {
                npy_intp dims[3];
                dims[0] = nx;
                dims[1] = ny;
                dims[2] = nz;
                pyGrid = (PyArrayObject *)PyArray_SimpleNew( 3, (npy_intp*)dims, NPY_DOUBLE );
                grid = (double*)pyGrid->data;
                memset( grid, 0, nx * ny * nz * sizeof(double) );
                printf( "Doing 3D Grid of size %d x %d x %d\n", nx, ny, nz );
        } else {
                npy_intp dims[2];
                dims[0] = nx;
                dims[1] = ny;
                pyGrid = (PyArrayObject *)PyArray_SimpleNew( 2, (npy_intp*)dims, NPY_DOUBLE );
                grid = (double*)pyGrid->data;
                memset( grid, 0, nx * ny * sizeof(double) );

                if (ngbs) {
                        pyNeighbours = (PyArrayObject *)PyArray_SimpleNew( 2, (npy_intp*)dims, NPY_INT );
                        neighbours = (int*)pyNeighbours->data;
                        pyContours = (PyArrayObject *)PyArray_SimpleNew( 2, (npy_intp*)dims, NPY_INT );
                        contours = (int*)pyContours->data;
                        memset( contours, 0, nx * ny * sizeof(int) );
                }
        }

        real_pos = (double*)malloc( 3 * npart * sizeof( double ) );
        real_value = (double*)malloc( npart * sizeof( double ) );
        if (grad) real_grad = (double*)malloc( 3 * npart * sizeof( double ) );
        for (i=0; i<npart; i++) {
          for (j=0; j<3; j++) {
            real_pos[i*3+j] = *(double*)((char*)pos->data + i*pos->strides[0] + j*pos->strides[1]);
            if (grad) real_grad[i*3+j] = *(double*)((char*)grad->data + i*grad->strides[0] + j*grad->strides[1]);
          }
          real_value[i] = *(double*)((char*)value->data + i*value->strides[0]);
        }

        double center[3];
        center[0] = domainx;
        center[1] = domainy;
        center[2] = domainz;

        createTree( &tree, npart, real_pos , domainlen, center);

        cellsizex = bx / nx;
        cellsizey = by / ny;

        if (proj || grid3D)
                cellsizez = bz / nz;
        else
                cellsizez = 0;

        coord[ 3 - axis0 - axis1 ] = cz;
        neighbour = 0;
        cell = 0;

        for (x=0; x<nx; x++) {
          coord[ axis0 ] = cx - 0.5 * bx + cellsizex * (0.5 + x);
          for (y=0; y<ny; y++) {
            coord[ axis1 ] = cy - 0.5 * by + cellsizey * (0.5 + y);
            for (z=0; z<nz; z++) {
              coord[ 3-axis0-axis1 ] = cz - 0.5 * bz + cellsizez * (0.5 + z);

              neighbour = getNearestNode( &tree, coord );

              if (!grad)
                grid[ cell ] += real_value[ neighbour ];
              else
                grid[ cell ] += real_value[ neighbour ]
                + (coord[0] - real_pos[ neighbour*3 + 0 ]) * real_grad[ neighbour*3 + 0 ]
                + (coord[1] - real_pos[ neighbour*3 + 1 ]) * real_grad[ neighbour*3 + 1 ]
                + (coord[2] - real_pos[ neighbour*3 + 2 ]) * real_grad[ neighbour*3 + 2 ];
              if (ngbs) neighbours[ cell ] = neighbour;

              if (grid3D) cell++; /* 3d grid */
            }
            if (!grid3D) cell++; /* 2d projection or grid */
          }

          if (ngbs) neighbour = neighbours[ cell - ny ];
        }

        freeTree( &tree );

        free( real_pos );
        free( real_value );
        if (grad) free( real_grad );

        if (ngbs)
          for (x=1; x<nx-1; x++) {
            for (y=1; y<ny-1; y++) {
              neighbour = neighbours[ x*ny + y ];
              if (neighbours[ (x-1)*ny + y-1 ] != neighbour ||
                  neighbours[  x   *ny + y-1 ] != neighbour ||
                  neighbours[ (x+1)*ny + y-1 ] != neighbour ||
                  neighbours[ (x-1)*ny + y   ] != neighbour ||
                  neighbours[ (x+1)*ny + y   ] != neighbour ||
                  neighbours[ (x-1)*ny + y+1 ] != neighbour ||
                  neighbours[  x   *ny + y+1 ] != neighbour ||
                  neighbours[ (x+1)*ny + y+1 ] != neighbour) {
                contours[ x*ny + y ] = 1;
              } else {
                contours[ x*ny + y ] = 0;
              }
            }
          }

        dict = PyDict_New();
        PyDict_SetStolenItem( dict, "grid", (PyObject*)pyGrid );

        if (ngbs)
        {
                PyDict_SetStolenItem( dict, "neighbours", (PyObject*)pyNeighbours );
                PyDict_SetStolenItem( dict, "contours", (PyObject*)pyContours );
        }

        printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
        return dict;
}

PyObject* _calcDGSlice(PyObject *self, PyObject *args, PyObject *kwargs) {
        PyArrayObject *pos, *value, *amrlevel, *pyGrid, *pyNeighbours, *pyContours;
        int npart, nx, ny, nz, axis0, axis1, proj, grid3D, ngbs;
        int x, y, z, i, j, cell;
        int dgdims,degree_k;
        double bx, by, bz, cx, cy, cz;
        double domainx, domainy, domainz, domainlen;
        double *real_pos, *real_value;
        double *grid, coord[3];
        int *real_level;
        double cellsizex, cellsizey, cellsizez;
        int neighbour, *neighbours, *contours;
        t_sph_tree tree;
        PyObject *dict;
        clock_t start;
        char *kwlist[] = {"pos", "value", "amrlevel", "dgdims", "degree_k", "nx", "ny", "boxx", "boxy", "centerx", "centery", "centerz", "domainx", "domainy", "domainz", "domainlen", "axis0", "axis1", "proj", "nz", "boxz", "grid3D", "ngbs", NULL} ;

        start = clock();

        axis0 = 0;
        axis1 = 1;
        proj = 0;
        nz = 0;
        bz = 0;
        grid3D = 0;
        ngbs = 1;
        if (!PyArg_ParseTupleAndKeywords( args, kwargs, "O!O!O!iiiiddddddddd|iiiidii:calcDGSlice( pos, value, amrlevel, dgdims, degree_k, nx, ny, boxx, boxy, centerx, centery, centerz, domainx, domainy, domainz, domainlen, [axis0, axis1, proj, nz, boxz, grid3D, ngbs] )", kwlist, &PyArray_Type, &pos, &PyArray_Type, &value, &PyArray_Type, &amrlevel, &dgdims, &degree_k, &nx, &ny, &bx, &by, &cx, &cy, &cz, &domainx, &domainy, &domainz, &domainlen, &axis0, &axis1, &proj, &nz, &bz, &grid3D, &ngbs )) {
                return 0;
        }

        if (proj || grid3D)
        {
                if (nz == 0)
                        nz = max( nx, ny );
                if (bz == 0)
                        bz = max( bx, by );
        }
        else
        {
                nz = 1;
                bz = 0;
        }

        if (proj || grid3D)
                ngbs = 0;

        if (pos->nd != 2 || pos->dimensions[1] != 3 || pos->descr->type_num != PyArray_DOUBLE) {
                PyErr_SetString( PyExc_ValueError, "pos has to be of dimensions [n,3] and type double" );
                return 0;
        }

        if (value->nd != 2 || value->descr->type_num != PyArray_DOUBLE) {
                PyErr_SetString( PyExc_ValueError, "value has to be of dimension [n,nof_base_functions] and type double" );
                return 0;
        }

        if (amrlevel->nd != 1 || !PyArray_ISINTEGER(amrlevel)) {
                PyErr_SetString( PyExc_ValueError, "value has to be of dimension [n] and type int" );
                return 0;
        }

        double amr_length[40];
        amr_length[0] = domainlen;
        for(i = 1; i < 40; i++)
          {
            amr_length[i] = amr_length[i - 1] / 2;

          }

        int nof_base_fcts = value->dimensions[1];

        npart = pos->dimensions[0];
        if (npart != value->dimensions[0]) {
                PyErr_SetString( PyExc_ValueError, "pos and value have to have the same size in the first dimension" );
                return 0;
        }

        if (grid3D) {
                npy_intp dims[3];
                dims[0] = nx;
                dims[1] = ny;
                dims[2] = nz;
                pyGrid = (PyArrayObject *)PyArray_SimpleNew( 3, (npy_intp*)dims, NPY_DOUBLE );
                grid = (double*)pyGrid->data;
                memset( grid, 0, nx * ny * nz * sizeof(double) );
                printf( "Doing 3D Grid of size %d x %d x %d\n", nx, ny, nz );
        } else {
                npy_intp dims[2];
                dims[0] = nx;
                dims[1] = ny;
                pyGrid = (PyArrayObject *)PyArray_SimpleNew( 2, (npy_intp*)dims, NPY_DOUBLE );
                grid = (double*)pyGrid->data;
                memset( grid, 0, nx * ny * sizeof(double) );

                if (ngbs) {
                        pyNeighbours = (PyArrayObject *)PyArray_SimpleNew( 2, (npy_intp*)dims, NPY_INT );
                        neighbours = (int*)pyNeighbours->data;
                        pyContours = (PyArrayObject *)PyArray_SimpleNew( 2, (npy_intp*)dims, NPY_INT );
                        contours = (int*)pyContours->data;
                        memset( contours, 0, nx * ny * sizeof(int) );
                }
        }

        real_pos = (double*)malloc( 3 * npart * sizeof( double ) );
        real_value = (double*)malloc( npart * nof_base_fcts * sizeof( double ) );
        real_level = (int*)malloc( npart * sizeof(int));

        for (i=0; i<npart; i++) {
          for (j=0; j<3; j++) {
            real_pos[i*3+j] = *(double*)((char*)pos->data + i*pos->strides[0] + j*pos->strides[1]);
          }
          for(j = 0; j < nof_base_fcts; j++)
            {
              real_value[i*nof_base_fcts + j] = *(double*)((char*)value->data + i*value->strides[0] + j*value->strides[1]);
            }

          real_level[i] = *(int*)((char*)amrlevel->data + i*amrlevel->strides[0]);
        }

        double center[3];
        center[0] = domainx;
        center[1] = domainy;
        center[2] = domainz;

        createTree( &tree, npart, real_pos , domainlen, center);

        cellsizex = bx / nx;
        cellsizey = by / ny;

        if (proj || grid3D)
                cellsizez = bz / nz;
        else
                cellsizez = 0;

        coord[ 3 - axis0 - axis1 ] = cz;
        neighbour = 0;
        cell = 0;

        for (x=0; x<nx; x++) {
          coord[ axis0 ] = cx - 0.5 * bx + cellsizex * (0.5 + x);
          for (y=0; y<ny; y++) {
            coord[ axis1 ] = cy - 0.5 * by + cellsizey * (0.5 + y);
            for (z=0; z<nz; z++) {
              coord[ 3-axis0-axis1 ] = cz - 0.5 * bz + cellsizez * (0.5 + z);

              neighbour = getNearestNode( &tree, coord );


              if(dgdims==2)
              {
                grid[ cell ] += dg_get_value2d(&real_value[ nof_base_fcts * neighbour ], nof_base_fcts, real_pos[3 * neighbour + 0],  real_pos[3 * neighbour + 1], amr_length[real_level[neighbour]], coord[ axis0 ], coord[ axis1 ]  );
              }
              else if(dgdims==3)
              {
                grid[ cell ] += dg_get_value3d(&real_value[ nof_base_fcts * neighbour ], nof_base_fcts, degree_k, real_pos[3 * neighbour + 0],  real_pos[3 * neighbour + 1],  real_pos[3 * neighbour + 2], amr_length[real_level[neighbour]], coord[ axis0 ], coord[ axis1 ], coord[3-axis0-axis1] );
              }
              else
              {
		PyErr_SetString( PyExc_ValueError, "dgdims has to be 2 or 3!\n");
		return 0;
              }

              if (ngbs) neighbours[ cell ] = neighbour;

              if (grid3D) cell++; /* 3d grid */
            }
            if (!grid3D) cell++; /* 2d projection or grid */
          }

          if (ngbs) neighbour = neighbours[ cell - ny ];
        }

        freeTree( &tree );

        free(real_level);
        free( real_pos );
        free( real_value );

        if (ngbs)
          for (x=1; x<nx-1; x++) {
            for (y=1; y<ny-1; y++) {
              neighbour = neighbours[ x*ny + y ];
              if (neighbours[ (x-1)*ny + y-1 ] != neighbour ||
                  neighbours[  x   *ny + y-1 ] != neighbour ||
                  neighbours[ (x+1)*ny + y-1 ] != neighbour ||
                  neighbours[ (x-1)*ny + y   ] != neighbour ||
                  neighbours[ (x+1)*ny + y   ] != neighbour ||
                  neighbours[ (x-1)*ny + y+1 ] != neighbour ||
                  neighbours[  x   *ny + y+1 ] != neighbour ||
                  neighbours[ (x+1)*ny + y+1 ] != neighbour) {
                contours[ x*ny + y ] = 1;
              } else {
                contours[ x*ny + y ] = 0;
              }
            }
          }

        dict = PyDict_New();
        PyDict_SetStolenItem( dict, "grid", (PyObject*)pyGrid );

        if (ngbs)
        {
                PyDict_SetStolenItem( dict, "neighbours", (PyObject*)pyNeighbours );
                PyDict_SetStolenItem( dict, "contours", (PyObject*)pyContours );
        }

        printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
        return dict;
}



static PyMethodDef calcGridmethods[] = {
	{ "calcGrid", (PyCFunction)_calcGrid, METH_VARARGS | METH_KEYWORDS, "" },
	{ "calcASlice", (PyCFunction)_calcASlice, METH_VARARGS | METH_KEYWORDS, "" },
        { "calcAMRSlice", (PyCFunction)_calcAMRSlice, METH_VARARGS | METH_KEYWORDS, "" },
        { "calcDGSlice", (PyCFunction)_calcDGSlice, METH_VARARGS | METH_KEYWORDS, "" },
	{ NULL, NULL, 0, NULL }
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "calcGrid", /* m_name */
  NULL,      /* m_doc */
  -1,                  /* m_size */
  calcGridmethods,     /* m_methods */
  NULL,                /* m_reload */
  NULL,                /* m_traverse */
  NULL,                /* m_clear */
  NULL,                /* m_free */
};

PyMODINIT_FUNC PyInit_calcGrid(void)
{
        import_array();
        return PyModule_Create( &moduledef );
}
#else
PyMODINIT_FUNC initcalcGrid(void)
{
        Py_InitModule( "calcGrid", calcGridmethods );
        import_array();
}
#endif

