#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "sph.h"

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

PyObject* _calcGrid(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *rho, *value, *pyGrid;
	int npart, nx, ny, nz, cells;
	int dims[3];
	double bx, by, bz, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass, *data_rho, *data_value;
	double *grid;
	int part, proj, norm;
	double px, py, pz, h, h2, m, r, v, cpx, cpy, cpz, r2, sum;
	int x, y, z0, z1;
	int xmin, xmax, ymin, ymax, zmin, zmax, zmid;
	double cellsizex, cellsizey, cellsizez;
	clock_t start;
	
	start = clock();

	proj = 0;
	norm = 0;
	if (!PyArg_ParseTuple( args, "O!O!O!O!O!iiidddddd|ii:calcGrid( pos, hsml, mass, rho, value, nx, ny, nz, boxx, boxy, boxz, centerx, centery, centerz, [proj, norm] )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &PyArray_Type, &rho, &PyArray_Type, &value, &nx, &ny, &nz, &bx, &by, &bz, &cx, &cy, &cz, &proj, &norm )) {
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

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	if (rho->nd != 1 || rho->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "rho has to be of dimension [n] and type double" );
		return 0;
	}

	if (value->nd != 1 || value->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "value has to be of dimension [n] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0]  || npart != rho->dimensions[0] || npart != value->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml, rho and value have to have the same size in the first dimension" );
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
	memset( grid, 0, cells*sizeof(double) );

	cellsizex = bx / nx;
	cellsizey = by / ny;
	cellsizez = bz / nz;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;
	data_rho = (double*)rho->data;
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

		m = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		r = *data_rho;
		data_rho = (double*)((char*)data_rho + rho->strides[0]);

		v = *data_value;
		data_value = (double*)((char*)data_value + value->strides[0]);

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
							grid[x*ny + y] += h * _getkernel( h, r2 ) * m * v / r / sum;
						}
					} else {				
						for (z0=zmid; z0>=zmin; z0--) {
							cpz = -0.5*bz + bz*(z0+0.5)/nz;
							r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
							if (r2 > h2) break;
							grid[(x*ny + y)*nz + z0] += _getkernel( h, r2 ) * m * v / r / sum;
						}

						for (z1=zmid+1; z1<=zmax; z1++) {
							cpz = -0.5*bz + bz*(z1+0.5)/nz;
							r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
							if (r2 > h2) break;
							grid[(x*ny + y)*nz + z1] += _getkernel( h, r2 ) * m * v / r / sum;
						}
					}
				}
			}	
		}
	}

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyGrid );
}

PyObject* _gatherAbundGrid(PyObject *self, PyObject *args, PyObject *kwargs) {
        PyArrayObject *pos, *mass, *rho, *abund, *pyGrid, *pyDensityField;
	t_sph_tree tree;
	int npart, nx, ny, nz, nspecies, nneighbours;
	size_t cells;
	npy_intp dims[4];
	double bx, by, bz, cx, cy, cz;
	double *data_mass, *real_pos;
	double grid_pos[3], grid_hsml, grid_rho;
	double *grid;
	float *fgrid;
	int part, species, *neighbours, nneighbours_real, neighbourcount;
	size_t cell;
	double h2, wk, m, r, a, cpx, cpy, cpz, r2;
	double posz, poszmin, poszmax;
	long i, j, x, y, z, id;
	int gcheck, forceneighbourcount, is2d, converged, single_precision;
	size_t count_notconverged;
        time_t start, now;
	double runtime;
	double *abundsum, masssum, *abundmin, *abundmax, *dens, maxratio, ratio, densitycut;
	int *specieslist, iratio, store, dummy, slowly;
	size_t nextoutput;
	FILE *fcheck;
        char *kwlist[] = {"pos", "mass", "rho", "abund", "nneighbours", "nx", "ny", "nz", "boxx", "boxy", "boxz", "centerx", "centery", "centerz", "forceneighbourcount", "densitycut", "densityfield", "gradientcheck", "single_precision", NULL};
	
	start = time( NULL );
	slowly = 0;

	gcheck = 0;
	pyDensityField = NULL;
	densitycut = 0;
	forceneighbourcount = 0;
        single_precision = 0;
	if (!PyArg_ParseTupleAndKeywords( args, kwargs, "O!O!O!O!iiiidddddd|idO!ii:gatherAbundGrid(pos, mass, rho, abund, nneighbours, nx, ny, nz, boxx, boxy, boxz, centerx, centery, centerz, forceneighbourcount=0, densitycut=0, densityfield=None, gradientcheck=0, single_precision=False)", kwlist, &PyArray_Type, &pos, &PyArray_Type, &mass, &PyArray_Type, &rho, &PyArray_Type, &abund, &nneighbours, &nx, &ny, &nz, &bx, &by, &bz, &cx, &cy, &cz, &forceneighbourcount, &densitycut, &PyArray_Type, &pyDensityField, &gcheck, &single_precision )) {
		return 0;
	}

	printf( "This is gatherAbundGrid.\n" );

	if (pos->nd != 2 || pos->dimensions[1] != 3 || pos->descr->type_num != PyArray_DOUBLE
	    || pos->descr->byteorder != '=') {
		PyErr_SetString( PyExc_ValueError, "pos has to be of dimensions [n,3], type double and native byte order" );
		return 0;
	}
	npart = pos->dimensions[0];

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE || mass->descr->byteorder != '=') {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n], type double and native byte order" );
		return 0;
	}

	if (rho->nd != 1 || rho->descr->type_num != PyArray_DOUBLE || rho->descr->byteorder != '=') {
		PyErr_SetString( PyExc_ValueError, "rho has to be of dimension [n], type double and native byte order" );
		return 0;
	}

	if (abund->nd != 2 || abund->descr->type_num != PyArray_DOUBLE || abund->descr->byteorder != '=') {
		PyErr_SetString( PyExc_ValueError, "value has to be of dimension [n,nspecies], type double and native byte order" );
		return 0;
	}
	nspecies = abund->dimensions[1];
	
	if (npart != mass->dimensions[0]  || npart != rho->dimensions[0] || npart != abund->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, rho and abund have to have the same size in the first dimension" );
		return 0;
	}

	printf( "%d particles with %d species found.\n", npart, nspecies );

	if (densitycut > 0.) {
	  if (!pyDensityField ||
	      pyDensityField->nd != 3 || 
	      pyDensityField->dimensions[0] != nx || 
              pyDensityField->dimensions[1] != ny ||
	      pyDensityField->dimensions[2] != nz ||
	      pyDensityField->descr->type_num != PyArray_DOUBLE) {
	        PyErr_SetString( PyExc_ValueError, "densityfield has to be of dimension [nx,ny,nz] and type double" );
		return 0;
	  }

	  printf( "Using densitycut: %g g/ccm\n", densitycut );
	}
	
	poszmin =  1e200;
	poszmax = -1e200;
	for (id=0; id<npart; id++) {
		posz = *(double*)((char*)pos->data + id*pos->strides[0] + 2*pos->strides[1]);
		if (posz < poszmin) poszmin = posz;
		if (posz > poszmax) poszmax = posz;
	}
	
	if (poszmin == poszmax) {
		is2d = 1;
		printf( "Using 2d mode.\n" );
		if (!forceneighbourcount) {
                  printf( "Using exact number of neighbours!\n" );
		  forceneighbourcount = 1;
		}
	} else {
		is2d = 0;
	}

	if (gcheck) {
		abundsum = (double*)malloc( nspecies * sizeof(double) );
		memset( abundsum, 0, nspecies * sizeof(double) );
		masssum = 0;

		for (id=0; id<npart; id++) {
			m = *(double*)((char*)mass->data + id*mass->strides[0]);
			masssum += m;

			for (species=0; species<nspecies; species++) {
				abundsum[species] += *(double*)((char*)abund->data + id*abund->strides[0] + species*abund->strides[1]) * m;
			}
		}

		for (species=0; species<nspecies; species++) {
			abundsum[species] /= masssum;
		}

		specieslist = (int*)malloc( gcheck * sizeof(int) );
		for (i=0; i<gcheck; i++) {
			j = 0;
			while ((j < i) && (abundsum[specieslist[j]] > abundsum[i])) j++;

			store = i;
			while (j <= i) {
				dummy = specieslist[j];
				specieslist[j] = store;
				store = dummy;
				j++;
			}
		}

		for (i=gcheck; i<nspecies; i++) {
			j = gcheck;

			while ((j > 0) && (abundsum[specieslist[j-1]] < abundsum[i])) j--;
			
			if (j < gcheck) {
				store = i;
				while (j < gcheck) {
					dummy = specieslist[j];
					specieslist[j] = store;
					store = dummy;
					j++;
				}
			}
		}

		abundmin = (double*)malloc( gcheck * sizeof(double) );
		abundmax = (double*)malloc( gcheck * sizeof(double) );

		fcheck = fopen( "gradientcheck.dat", "w" );
		fwrite( &gcheck, sizeof(int), 1, fcheck );
		fwrite( specieslist, sizeof(int), gcheck, fcheck );

		free( abundsum );
	}

	dims[0] = nx;
	dims[1] = ny;
	dims[2] = nz;
	dims[3] = nspecies+1;
	printf( "Allocating %dx%dx%d grid for %d species, total size=%g GB.\n", (int)dims[0], (int)dims[1], (int)dims[2], (int)dims[3],
		1e-9*dims[0]*dims[1]*dims[2]*dims[3]*8 );
	pyGrid = (PyArrayObject *)PyArray_SimpleNew( 4, dims, single_precision ? PyArray_FLOAT : PyArray_DOUBLE );
	if (!pyGrid) {
		PyErr_SetString( PyExc_MemoryError, "Could not allocate memory for grid array." );
		return 0;
	}
		

	cells = nx*ny*nz;
        if (single_precision) {
		fgrid = (float*)pyGrid->data;
		memset( fgrid, 0, cells*(nspecies+1)*sizeof(*fgrid) );
        }
        else {
		grid = (double*)pyGrid->data;
		memset( grid, 0, cells*(nspecies+1)*sizeof(*grid) );
        }

	
	if (forceneighbourcount) {
          dens = malloc( cells * sizeof(double) );
          memset( dens, 0, cells * sizeof(double) );
	}
	
	data_mass = (double*)mass->data;
	
	real_pos = (double*)malloc( 3 * npart * sizeof( double ) );
	for (i=0; i<npart; i++) {
	  for (j=0; j<3; j++) {
	    real_pos[i*3+j] = *(double*)((char*)pos->data + i*pos->strides[0] + j*pos->strides[1]);
	  }
	}

	createTree( &tree, npart, real_pos );

	cell = 0;
	nextoutput = cells / 100;
	count_notconverged = 0;
	for (x=0; x<nx; x++) {
		cpx = -0.5*bx + bx*(x+0.5)/nx + cx;
		for (y=0; y<ny; y++) {
			cpy = -0.5*by + by*(y+0.5)/ny + cy;
			for (z=0; z<nz; z++) {
				cpz = -0.5*bz + bz*(z+0.5)/nz + cz;

				if (densitycut > 0 &&
				    *(double*)((char*)pyDensityField->data  + x*pyDensityField->strides[0] + 
					       y*pyDensityField->strides[1] + z*pyDensityField->strides[2]) <= densitycut) {
				      cell++;
				      continue;
				}
				
				grid_pos[0] = cpx;
				grid_pos[1] = cpy;
				grid_pos[2] = cpz;
				
				if (forceneighbourcount) {
					/* this function allocates memory for neighbours */
				        grid_hsml = getNNeighbours( &tree, grid_pos, real_pos, nneighbours, &nneighbours_real, &neighbours, &converged );
					neighbourcount = nneighbours_real;

					if (!converged) count_notconverged++;
				} else {
					grid_hsml = 0;
					calcHsml( &tree, grid_pos, real_pos, data_mass, nneighbours, &grid_hsml, &grid_rho );
					/* this function allocates memory for neighbours */
					neighbourcount = getNeighbours( &tree, grid_pos, real_pos, grid_hsml, &neighbours );
				}
				h2 = grid_hsml * grid_hsml;

				if (gcheck) {
					for (i=0; i<gcheck; i++) {
						abundmin[i] = 1.0;
						abundmax[i] = 0.0;
					}
				}
				
				for (part=0; part<neighbourcount; part++) {
					id = neighbours[part];
					
					r2 = (grid_pos[0]-real_pos[id*3+0])*(grid_pos[0]-real_pos[id*3+0])
					   + (grid_pos[1]-real_pos[id*3+1])*(grid_pos[1]-real_pos[id*3+1])
					   + (grid_pos[2]-real_pos[id*3+2])*(grid_pos[2]-real_pos[id*3+2]);
					  
					if (r2 < h2) {
						wk = _getkernel( grid_hsml, r2 );
						
						/* if 2d => correct normalisation */
						if (is2d) wk *= grid_hsml;
						
						m = *(double*)((char*)mass->data + id*mass->strides[0]);
						/*r = *(double*)((char*)rho->data + id*rho->strides[0]);*/
						
#define tmp(g) {							\
	       	g[((x*ny + y)*nz + z)*(nspecies+1)+nspecies] += wk * m; /* density */ \
									\
		for (species=0; species<nspecies; species++) {		\
			a = *(double*)((char*)abund->data + id*abund->strides[0] + species*abund->strides[1]); \
									\
			g[((x*ny + y)*nz + z)*(nspecies+1) + species] += wk * m * a; /* abundances */ \
		}							\
						}
                                                if (single_precision) tmp(fgrid)
						else tmp(grid)
#undef tmp
						
						if (forceneighbourcount) {
							r = *(double*)((char*)rho->data + id*rho->strides[0]);
							dens[ (x*ny + y)*nz + z ] += wk * m * r;
						}
						
						if (gcheck) {
							for (i=0; i<gcheck; i++) {
								species = specieslist[i];							 
								a = *(double*)((char*)abund->data + id*abund->strides[0] + species*abund->strides[1]);
								if (a < abundmin[i]) abundmin[i] = a;
								if (a > abundmax[i]) abundmax[i] = a;
							}
						}
					}					
				}
				
				if (gcheck && neighbourcount) {
					maxratio = 0;
					iratio = -1;
					for (i=0; i<gcheck; i++) {
						if (abundmax[i] > 1e-3) {
							ratio = (abundmax[i]-abundmin[i])/(abundmax[i]+abundmin[i]);
							if ( ratio > maxratio ) {
								maxratio = ratio;
								iratio = i;
							}
						}
					}

					if (maxratio > 0.1) {
						printf( "r: %g km, massratio: %g, min: %g, max: %g\n", sqrt(r2) / 1e5, maxratio, abundmin[iratio], abundmax[iratio] );
						fwrite( &maxratio, sizeof(double), 1, fcheck );
						fwrite( &grid_pos, sizeof(double), 3, fcheck );
						fwrite( abundmin, sizeof(double), gcheck, fcheck );
						fwrite( abundmax, sizeof(double), gcheck, fcheck );

						fwrite( &neighbourcount, sizeof(int), 1, fcheck );
						fwrite( neighbours, sizeof(int), neighbourcount, fcheck );
					}
				}	  

				free( neighbours );
				
				cell++;
				if (cell >= nextoutput) {
				  now = time( NULL );
				  runtime = difftime( now, start );
				  
				  if (nextoutput == cells / 100) {
				    if ( runtime > 60. ) {
				      slowly = 1;
				    } else {
				      nextoutput = 0;
				    }
				  }

				  printf( "%zd / %zd cells done (%d%%): %ds elapsed, ~%ds remaining\n", cell, cells, (int)floor(100.0*(double)cell/(double)cells), (int)(runtime), (int)(runtime/cell*(cells-cell)) );

				  if (slowly)
				    nextoutput += cells / 100;
				  else
				    nextoutput += cells /  10;
				}
			}
		}
	}

	if (count_notconverged) {
		printf( "Neighbour search not converged for %zd cells => be careful\n", count_notconverged );
	}

	if (gcheck) {
		fclose( fcheck );
		free( specieslist );
		free( abundmin );
		free( abundmax );
	}
	
	free( real_pos );
	
#define tmp(g)								\
	for (cell=0; cell<cells; cell++) {				\
		r = g[cell*(nspecies+1)+nspecies];			\
									\
		if (r > 0) {						\
			for (species=0; species<nspecies; species++) {	\
				g[cell*(nspecies+1)+species] /= r;	\
			}						\
									\
			if (forceneighbourcount) g[cell*(nspecies+1)+nspecies] = dens[cell] / r; \
		}							\
	}
        if (single_precision) tmp(fgrid)
	else tmp(grid)
#undef tmp

	if (forceneighbourcount) free( dens );

	freeTree( &tree );

	now = time( NULL );
	runtime = difftime( now, start );
	printf( "Calculation took %ds\n", (int)runtime );
	return PyArray_Return( pyGrid );
}

PyObject* _calcSlice(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *rho, *value, *pyGrid;
	int npart, nx, ny, cells;
	int dims[2];
	double bx, by, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass, *data_rho, *data_value;
	double *grid;
	int part;
	double px, py, pz, h, m, r, v, cpx, cpy, r2, h2;
	double p[3];
	int x, y;
	int xmin, xmax, ymin, ymax, axis0, axis1;
	double cellsizex, cellsizey;
	clock_t start;
	
	start = clock();

	axis0 = 0;
	axis1 = 1;
	if (!PyArg_ParseTuple( args, "O!O!O!O!O!iiddddd|ii:calcSlice( pos, hsml, mass, rho, value, nx, ny, boxx, boxy, centerx, centery, centerz, [axis0, axis1] )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &PyArray_Type, &rho, &PyArray_Type, &value, &nx, &ny, &bx, &by, &cx, &cy, &cz, &axis0, &axis1 )) {
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

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	if (rho->nd != 1 || rho->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "rho has to be of dimension [n] and type double" );
		return 0;
	}

	if (value->nd != 1 || value->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "value has to be of dimension [n] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0]  || npart != rho->dimensions[0] || npart != value->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml, mass, rho and value have to have the same size in the first dimension" );
		return 0;
	}
	dims[0] = nx;
	dims[1] = ny;
	pyGrid = (PyArrayObject *)PyArray_FromDims( 2, dims, PyArray_DOUBLE );
	grid = (double*)pyGrid->data;
	cells = nx*ny;
	memset( grid, 0, cells*sizeof(double) );

	cellsizex = bx / nx;
	cellsizey = by / ny;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;
	data_rho = (double*)rho->data;
	data_value = (double*)value->data;

	for (part=0; part<npart; part++) {
		p[0] = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		p[1] = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		p[2] = *data_pos;
		data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);
		
		px = p[ axis0 ];
		py = p[ axis1 ];
		pz = p[ 3 - axis0 - axis1 ];
		
		h = *data_hsml;
		data_hsml = (double*)((char*)data_hsml + hsml->strides[0]);
		h2 = h*h;

		m = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		r = *data_rho;
		data_rho = (double*)((char*)data_rho + rho->strides[0]);

		v = *data_value;
		data_value = (double*)((char*)data_value + value->strides[0]);

		xmin = max( floor( (px - h - cx + 0.5*bx) / cellsizex ), 0 );
		xmax = min( ceil( (px + h - cx + 0.5*bx) / cellsizex ), nx-1 );
		ymin = max( floor( (py - h - cy + 0.5*by) / cellsizey ), 0 );
		ymax = min( ceil( (py + h - cy + 0.5*by) / cellsizey ), ny-1 );

		if (xmin < nx && ymin < ny && xmax >= 0 && ymax >= 0 && abs(pz-cz) < h) {
			for (x=xmin; x<=xmax; x++) {
				cpx = -0.5*bx + bx*(x+0.5)/nx;
				for (y=ymin; y<=ymax; y++) {
					cpy = -0.5*by + by*(y+0.5)/ny;
					r2 = sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cz);
					if (r2 > h2) continue;
					grid[x*ny + y] += _getkernel( h, r2 ) * m * v / r;
				}
			}	
		}
	}

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyGrid );
}

PyObject* _calcGridMassWeight(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *value, *pyGridMass, *pyGridValue;
	int npart, nx, ny, nz, cells;
	int dims[3];
	double bx, by, bz, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass, *data_value;
	double *gridmass, *gridvalue, *massend, *massiter, *valueiter;
	int part;
	double px, py, pz, h, h2, m, v, cpx, cpy, cpz, r2, dmass;
	int x, y, z0, z1;
	int xmin, xmax, ymin, ymax, zmin, zmax, zmid;
	double cellsizex, cellsizey, cellsizez;

	if (!PyArg_ParseTuple( args, "O!O!O!O!O!iiidddddd:calcGridMassWeight( pos, hsml, mass, value, massgrid, nx, ny, nz, boxx, boxy, boxz, centerx, centery, centerz )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &PyArray_Type, &value, &PyArray_Type, &pyGridMass, &nx, &ny, &nz, &bx, &by, &bz, &cx, &cy, &cz )) {
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

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	if (value->nd != 1 || value->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "value has to be of dimension [n] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0] || npart != value->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml and value have to have the same size in the first dimension" );
		return 0;
	}

	if (pyGridMass->nd != 3 || pyGridMass->dimensions[0] != nx || pyGridMass->dimensions[1] != ny || pyGridMass->dimensions[2] != nz) {
		PyErr_SetString( PyExc_ValueError, "massgrid has to have 3 dimensions: [nx,ny,nz]" );
		return 0;
	}

	dims[0] = nx;
	dims[1] = ny;
	dims[2] = nz;
	cells = nx*ny*nz;
	
	gridmass = (double*)pyGridMass->data;
	
	pyGridValue = (PyArrayObject *)PyArray_FromDims( 3, dims, PyArray_DOUBLE );
	gridvalue = (double*)pyGridValue->data;
	memset( gridvalue, 0, cells*sizeof(double) );

	cellsizex = bx / nx;
	cellsizey = by / ny;
	cellsizez = bz / nz;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;
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

		m = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		v = *data_value;
		data_value = (double*)((char*)data_value + value->strides[0]);

		xmin = max( floor( (px - h - cx + 0.5*bx) / cellsizex ), 0 );
		xmax = min( ceil( (px + h - cx + 0.5*bx) / cellsizex ), nx-1 );
		ymin = max( floor( (py - h - cy + 0.5*by) / cellsizey ), 0 );
		ymax = min( ceil( (py + h - cy + 0.5*by) / cellsizey ), ny-1 );
		zmin = max( floor( (pz - h - cz + 0.5*bz) / cellsizez ), 0 );
		zmax = min( ceil( (pz + h - cz + 0.5*bz) / cellsizez ), nz-1 );

		zmid = floor( 0.5 * (zmin+zmax) + 0.5 );

		if (xmin < nx && ymin < ny && xmax >= 0 && ymax >= 0 && zmin < nz && zmax >= 0) {
			for (x=xmin; x<=xmax; x++) {
				cpx = -0.5*bx + bx*(x+0.5)/nx;
				for (y=ymin; y<=ymax; y++) {
					cpy = -0.5*by + by*(y+0.5)/ny;
					for (z0=zmid; z0>=zmin; z0--) {
						cpz = -0.5*bz + bz*(z0+0.5)/nz;
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
						if (r2 > h2) break;
						
						dmass = _getkernel( h, r2 ) * m;
						gridvalue[(x*ny + y)*nz + z0] += dmass * v;
					}

					for (z1=zmid+1; z1<=zmax; z1++) {
						cpz = -0.5*bz + bz*(z1+0.5)/nz;
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
						if (r2 > h2) break;
						
						dmass = _getkernel( h, r2 ) * m;
						gridvalue[(x*ny + y)*nz + z1] += dmass * v;
					}
				}
			}	
		}
	}
	
	massend = &gridmass[ cells ];
	for (massiter = gridmass, valueiter = gridvalue; massiter != massend; massiter++, valueiter++) {
		if (*massiter > 0)
			*valueiter /= *massiter;
	}
	
	return PyArray_Return( pyGridValue );
}

PyObject* _calcDensGrid(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *pyGrid;
	int npart, nx, ny, nz, cells;
	int dims[3];
	double bx, by, bz, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass;
	double *grid;
	int part;
	double px, py, pz, h, h2, v, cpx, cpy, cpz, r2;
	int x, y, z0, z1;
	int xmin, xmax, ymin, ymax, zmin, zmax, zmid;
	double cellsizex, cellsizey, cellsizez;
	clock_t start;
	
	start = clock();
	
	if (!PyArg_ParseTuple( args, "O!O!O!iiidddddd:calcDensGrid( pos, hsml, mass, nx, ny, nz, boxx, boxy, boxz, centerx, centery, centerz )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &nx, &ny, &nz, &bx, &by, &bz, &cx, &cy, &cz )) {
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

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml and mass have to have the same size in the first dimension" );
		return 0;
	}

	dims[0] = nx;
	dims[1] = ny;
	dims[2] = nz;
	pyGrid = (PyArrayObject *)PyArray_FromDims( 3, dims, PyArray_DOUBLE );
	grid = (double*)pyGrid->data;
	cells = nx*ny*nz;
	memset( grid, 0, cells*sizeof(double) );

	cellsizex = bx / nx;
	cellsizey = by / ny;
	cellsizez = bz / nz;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;
	
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

		v = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		xmin = max( floor( (px - h - cx + 0.5*bx) / cellsizex ), 0 );
		xmax = min( ceil( (px + h - cx + 0.5*bx) / cellsizex ), nx-1 );
		ymin = max( floor( (py - h - cy + 0.5*by) / cellsizey ), 0 );
		ymax = min( ceil( (py + h - cy + 0.5*by) / cellsizey ), ny-1 );
		zmin = max( floor( (pz - h - cz + 0.5*bz) / cellsizez ), 0 );
		zmax = min( ceil( (pz + h - cz + 0.5*bz) / cellsizez ), nz-1 );

		zmid = floor( 0.5 * (zmin+zmax) + 0.5 );

		if (xmin < nx && ymin < ny && xmax >= 0 && ymax >= 0 && zmin < nz && zmax >= 0) {
			for (x=xmin; x<=xmax; x++) {
				cpx = -0.5*bx + bx*(x+0.5)/nx;
				for (y=ymin; y<=ymax; y++) {
					cpy = -0.5*by + by*(y+0.5)/ny;
					for (z0=zmid; z0>=zmin; z0--) {
						cpz = -0.5*bz + bz*(z0+0.5)/nz;
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
						if (r2 > h2) break;
						grid[(x*ny + y)*nz + z0] += _getkernel( h, r2 ) * v;
					}

					for (z1=zmid+1; z1<=zmax; z1++) {
						cpz = -0.5*bz + bz*(z1+0.5)/nz;
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
						if (r2 > h2) break;
						grid[(x*ny + y)*nz + z1] += _getkernel( h, r2 ) * v;
					}
				}
			}	
		}
	}

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyGrid );
}

PyObject* _calcDensSlice(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *pyGrid;
	int npart, nx, ny, cells;
	int dims[2];
	double bx, by, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass;
	double *grid;
	int part;
	double px, py, pz, h, v, cpx, cpy, r2, h2;
	double p[3];
	int x, y;
	int xmin, xmax, ymin, ymax, axis0, axis1;
	double cellsizex, cellsizey;
	clock_t start;
	
	start = clock();

	axis0 = 0;
	axis1 = 1;
	if (!PyArg_ParseTuple( args, "O!O!O!iiddddd|ii:calcDensSlice( pos, hsml, mass, nx, ny, boxx, boxy, centerx, centery, centerz, [axis0, axis1] )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &nx, &ny, &bx, &by, &cx, &cy, &cz, &axis0, &axis1 )) {
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

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml and mass have to have the same size in the first dimension" );
		return 0;
	}

	dims[0] = nx;
	dims[1] = ny;
	pyGrid = (PyArrayObject *)PyArray_FromDims( 2, dims, PyArray_DOUBLE );
	grid = (double*)pyGrid->data;
	cells = nx*ny;
	memset( grid, 0, cells*sizeof(double) );

	cellsizex = bx / nx;
	cellsizey = by / ny;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;

	for (part=0; part<npart; part++) {
		p[0] = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		p[1] = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		p[2] = *data_pos;
		data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);
		
		px = p[ axis0 ];
		py = p[ axis1 ];
		pz = p[ 3 - axis0 - axis1 ];
		
		h = *data_hsml;
		data_hsml = (double*)((char*)data_hsml + hsml->strides[0]);
		h2 = h*h;

		v = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		xmin = max( floor( (px - h - cx + 0.5*bx) / cellsizex ), 0 );
		xmax = min( ceil( (px + h - cx + 0.5*bx) / cellsizex ), nx-1 );
		ymin = max( floor( (py - h - cy + 0.5*by) / cellsizey ), 0 );
		ymax = min( ceil( (py + h - cy + 0.5*by) / cellsizey ), ny-1 );

		if (xmin < nx && ymin < ny && xmax >= 0 && ymax >= 0 && abs(pz-cz) < h) {
			for (x=xmin; x<=xmax; x++) {
				cpx = -0.5*bx + bx*(x+0.5)/nx;
				for (y=ymin; y<=ymax; y++) {
					cpy = -0.5*by + by*(y+0.5)/ny;
					r2 = sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cz);
					if (r2 > h2) continue;
					grid[x*ny + y] += _getkernel( h, r2 ) * v;
				}
			}	
		}
	}

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyGrid );
}

PyObject* _calcCylinderAverage(PyObject *self, PyObject *args) {
	PyArrayObject *pyGrid, *pyNewgrid;
	int dims[2], cells;
	double *newgrid, *count;
	int x, y, z, nx, ny, nz, nr, r;

	if (!PyArg_ParseTuple( args, "O!:calcCylinderAverage( grid )", &PyArray_Type, &pyGrid )) {
		return 0;
	}

	if (pyGrid->nd != 3 || pyGrid->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "grid has to be of dimensions [nx,ny,nz] and type double" );
		return 0;
	}

	nx = pyGrid->dimensions[0];
	ny = pyGrid->dimensions[1];
	nz = pyGrid->dimensions[2];
	nr = min( ny, nz );
	
	dims[0] = nx;
	dims[1] = nr;
	cells = nx*nr;
	pyNewgrid = (PyArrayObject *)PyArray_FromDims( 2, dims, PyArray_DOUBLE );
	newgrid = (double*)pyNewgrid->data;
	memset( newgrid, 0, cells*sizeof(double) );

	count = (double*)malloc( cells*sizeof(double) );
	memset( count, 0, cells*sizeof(double) );

	for (x=0; x<nx; x++) for (y=0; y<ny; y++) for (z=0; z<nz; z++) {
		r = floor( sqrt( sqr(y-ny/2.0+0.5) + sqr(z-nz/2.0+0.5) ) );
		if (r >= nr/2) continue;

		newgrid[x*nr+r+nr/2] += *(double*)( pyGrid->data + pyGrid->strides[0]*x + pyGrid->strides[1]*y + pyGrid->strides[2]*z );
		count[x*nr+r+nr/2] += 1;
	}
	
	for (x=0; x<nx; x++) for (r=0; r<nr/2; r++) {
		if (count[x*nr+r+nr/2] > 0) {
			newgrid[x*nr+r+nr/2] /= count[x*nr+r+nr/2];
			newgrid[x*nr-r+nr/2-1] = newgrid[x*nr+r+nr/2];
		}
	}

	free( count );
	return PyArray_Return( pyNewgrid );
}

PyObject* _calcRadialProfile(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *data, *pyProfile;
	int npart, nshells, mode;
	int dims[2];
	int *count;
	double cx, cy, cz, dr;
	double *data_pos, *data_data;
	double *profile;
	int part, shell;
	double px, py, pz, d, rr, v;
	clock_t start;
	
	start = clock();

	mode = 1;
	nshells = 200;
	dr = 0;
	cx = cy = cz = 0;
	if (!PyArg_ParseTuple( args, "O!O!|iidddd:calcRadialProfile( pos, data, mode, nshells, dr, centerx, centery, centerz )", &PyArray_Type, &pos, &PyArray_Type, &data, &mode, &nshells, &dr, &cx, &cy, &cz )) {
		return 0;
	}

	if (pos->nd != 2 || pos->dimensions[1] != 3 || pos->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "pos has to be of dimensions [n,3] and type double" );
		return 0;
	}

	if (data->nd != 1 || data->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "data has to be of dimension [n] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != data->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos and data have to have the same size in the first dimension" );
		return 0;
	}
	dims[0] = 2;
	dims[1] = nshells;
	pyProfile = (PyArrayObject *)PyArray_FromDims( 2, dims, PyArray_DOUBLE );
	profile = (double*)pyProfile->data;
	memset( profile, 0, 2*nshells*sizeof(double) );

	count = (int*)malloc( nshells*sizeof(int) );
	memset( count, 0, nshells*sizeof(int) );

	if (!dr) {
		data_pos = (double*)pos->data;
		for (part=0; part<npart; part++) {
			px = *data_pos;
			data_pos = (double*)((char*)data_pos + pos->strides[1]);
			py = *data_pos;
			data_pos = (double*)((char*)data_pos + pos->strides[1]);
			pz = *data_pos;
			data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);

			rr = sqrt( sqr(px-cx) + sqr(py-cy) + sqr(pz-cz) );
			if (rr > dr)
				dr = rr;
		}
		dr /= nshells;
		printf( "dr set to %g\n", dr );
	}

	data_pos = (double*)pos->data;
	data_data = (double*)data->data;

	for (part=0; part<npart; part++) {
		px = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		py = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		pz = *data_pos;
		data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);

		d = *data_data;
		data_data = (double*)((char*)data_data + data->strides[0]);

		rr = sqrt( sqr(px-cx) + sqr(py-cy) + sqr(pz-cz) );
		shell = floor( rr / dr );

		if (shell < nshells) {
			profile[ shell ] += d;
			count[ shell ] += 1;
		}
	}

	for (shell=0; shell<nshells; shell++) {
		profile[ nshells + shell ] = dr * (shell + 0.5);
	}

	switch (mode) {
		// sum
		case 0:
			break;
		// density
		case 1:
			for (shell=0; shell<nshells; shell++) {
				v = 4.0 / 3.0 * M_PI * dr*dr*dr * ( ((double)shell+1.)*((double)shell+1.)*((double)shell+1.) - (double)shell*(double)shell*(double)shell );
				profile[shell] /= v;
			}
			break;
		// average
		case 2:
			for (shell=0; shell<nshells; shell++) if (count[shell] > 0) profile[shell] /= count[shell];
			break;
	}

	free( count );

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyProfile );
}

PyObject* _calcAbundGrid(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *abund, *pyGrid;
	int npart, nx, ny, nz, nspecies;
	size_t cell, cells;
	int dims[4];
	double bx, by, bz, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass, *data_abund;
	double *grid;
	int part, species;
	double *xnuc;
	double px, py, pz, h, h2, m, cpx, cpy, cpz, r2, kk, rho;
	size_t x, y, z0, z1;
	int xmin, xmax, ymin, ymax, zmin, zmax, zmid;
	int slowly, nextoutput;
	double cellsizex, cellsizey, cellsizez, runtime;
	clock_t start;
	
	start = clock();

	if (!PyArg_ParseTuple( args, "O!O!O!O!iiidddddd:calcAbundGrid( pos, hsml, mass, abund, nx, ny, nz, boxx, boxy, boxz, centerx, centery, centerz )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &PyArray_Type, &abund, &nx, &ny, &nz, &bx, &by, &bz, &cx, &cy, &cz )) {
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

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	if (abund->nd != 2 || abund->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "abund has to be of dimension [n,nspecies] and type double" );
		return 0;
	}
	
	nspecies = abund->dimensions[1];

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0]  || npart != abund->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml and abund have to have the same size in the first dimension" );
		return 0;
	}
	
	xnuc = (double*)malloc( nspecies * sizeof( double ) );

	dims[0] = nx;
	dims[1] = ny;
	dims[2] = nz;
	dims[3] = nspecies+1;
	pyGrid = (PyArrayObject *)PyArray_FromDims( 4, dims, PyArray_DOUBLE );
	grid = (double*)pyGrid->data;
	cells = nx*ny*nz;
	memset( grid, 0, cells*(nspecies+1)*sizeof(double) );

	cellsizex = bx / nx;
	cellsizey = by / ny;
	cellsizez = bz / nz;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;
	data_abund = (double*)abund->data;

	slowly = 0;
	nextoutput = npart / 100;

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

		m = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		for (species = 0; species < nspecies; species++ ) {
			xnuc[species] = *data_abund;
			data_abund = (double*)((char*)data_abund + abund->strides[1]);
		}
		data_abund = (double*)((char*)data_abund - nspecies*abund->strides[1] + abund->strides[0]);

		xmin = max( floor( (px - h - cx + 0.5*bx) / cellsizex ), 0 );
		xmax = min( ceil( (px + h - cx + 0.5*bx) / cellsizex ), nx-1 );
		ymin = max( floor( (py - h - cy + 0.5*by) / cellsizey ), 0 );
		ymax = min( ceil( (py + h - cy + 0.5*by) / cellsizey ), ny-1 );
		zmin = max( floor( (pz - h - cz + 0.5*bz) / cellsizez ), 0 );
		zmax = min( ceil( (pz + h - cz + 0.5*bz) / cellsizez ), nz-1 );

		zmid = floor( 0.5 * (zmin+zmax) + 0.5 );

		if (xmin < nx && ymin < ny && xmax >= 0 && ymax >= 0 && zmin < nz && zmax >= 0) {
			for (x=xmin; x<=xmax; x++) {
				cpx = -0.5*bx + bx*(x+0.5)/nx;
				for (y=ymin; y<=ymax; y++) {
					cpy = -0.5*by + by*(y+0.5)/ny;
					for (z0=zmid; z0>=zmin; z0--) {
						cpz = -0.5*bz + bz*(z0+0.5)/nz;
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
						if (r2 > h2) break;
						
						kk = _getkernel( h, r2 ) * m;
						for (species = 0; species < nspecies; species++ )
							grid[((x*ny + y)*nz + z0)*(nspecies+1) + species] += kk * xnuc[species];
						grid[((x*ny + y)*nz + z0)*(nspecies+1) + nspecies] += kk;
					}

					for (z1=zmid+1; z1<=zmax; z1++) {
						cpz = -0.5*bz + bz*(z1+0.5)/nz;
						r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
						if (r2 > h2) break;
						
						kk = _getkernel( h, r2 ) * m;
						for (species = 0; species < nspecies; species++ )
							grid[((x*ny + y)*nz + z1)*(nspecies+1) + species] += kk * xnuc[species];
						grid[((x*ny + y)*nz + z1)*(nspecies+1) + nspecies] += kk;
					}
				}
			}	
		}

		if (part >= nextoutput) {
			runtime = ((double)clock()-(double)start)/CLOCKS_PER_SEC;
				  
			if (nextoutput == npart / 100) {
			  if ( runtime > 60. ) {
			    slowly = 1;
			  } else {
			    nextoutput = 0;
			  }
			}

			printf( "%d / %d particles done (%d%%): %ds elapsed, ~%ds remaining\n", part, npart, (int)floor(100.0*(double)part/(double)npart), (int)(runtime), (int)(runtime/part*(npart-part)) );

			if (slowly)
			  nextoutput += npart / 100;
			else
			  nextoutput += npart /  10;
		}
	}
	
	free( xnuc );

	for (cell=0; cell<cells; cell++) {		
		rho = grid[cell*(nspecies+1)+nspecies];
		if (rho > 0) {
			for (species=0; species<nspecies; species++) {
			  grid[cell*(nspecies+1)+species] /= rho;
			}						
		}							
	}

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyGrid );
}

PyObject* _calcAbundSphere(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *abund, *pyGrid;
	int npart, nradius, ntheta, nphi, cells, nspecies;
	int dims[4];
	double radius, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass, *data_abund;
	double *grid;
	int part, species;
	double *xnuc;
	double px, py, pz, h, h2, m, r, dr, cpx, cpy, cpz, r2, kk;
	double vr, vtheta, vphi;
	int ir, itheta, iphi;
	int minradius, maxradius;
	clock_t start;
	
	start = clock();

	if (!PyArg_ParseTuple( args, "O!O!O!O!iiidddd:calcAbundSphere( pos, hsml, mass, abund, nradius, ntheta, nphi, radius, centerx, centery, centerz )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &PyArray_Type, &abund, &nradius, &ntheta, &nphi, &radius, &cx, &cy, &cz )) {
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

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	if (abund->nd != 2 || abund->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "abund has to be of dimension [n,nspecies] and type double" );
		return 0;
	}
	
	nspecies = abund->dimensions[1];

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0]  || npart != abund->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml and abund have to have the same size in the first dimension" );
		return 0;
	}
	
	xnuc = (double*)malloc( nspecies * sizeof( double ) );

	dims[0] = nradius;
	dims[1] = ntheta;
	dims[2] = nphi;
	dims[3] = nspecies+1;
	pyGrid = (PyArrayObject *)PyArray_FromDims( 4, dims, PyArray_DOUBLE );
	grid = (double*)pyGrid->data;
	cells = nradius*ntheta*nphi*(nspecies+1);
	memset( grid, 0, cells*sizeof(double) );

	dr = radius / nradius;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;
	data_abund = (double*)abund->data;
		
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

		m = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		for (species = 0; species < nspecies; species++ ) {
			xnuc[species] = *data_abund;
			data_abund = (double*)((char*)data_abund + abund->strides[1]);
		}
		data_abund = (double*)((char*)data_abund - nspecies*abund->strides[1] + abund->strides[0]);

		r = sqrt( px*px + py*py + pz*pz );
		minradius = max( 0, floor( (r-h-0.5) / dr ) );
		maxradius = min( nradius-1, floor( (r+h+0.5) / dr ) );

		for (ir=minradius; ir<=maxradius; ir++)
		for (itheta=0; itheta<ntheta; itheta++)
		for (iphi=0; iphi<nphi; iphi++) {
	        	vr = radius * (ir+0.5) / nradius;
			vtheta = M_PI * (itheta+0.5) / ntheta;
			vphi = 2. * M_PI * (iphi+0.5) / nphi;
			cpx = vr * sin( vtheta ) * cos( vphi );
			cpy = vr * sin( vtheta ) * sin( vphi );
			cpz = vr * cos( vtheta );

			r2 = ( sqr(px-cpx-cx) + sqr(py-cpy-cy) + sqr(pz-cpz-cz) );
			if (r2 > h2) continue;
						
			kk = _getkernel( h, r2 ) * m;
			for (species = 0; species < nspecies; species++ ) {
				grid[((ir*ntheta + itheta)*nphi + iphi)*(nspecies+1) + species] += kk * xnuc[species];
			}
			grid[((ir*ntheta + itheta)*nphi + iphi)*(nspecies+1) + nspecies] += kk;
		}
	}
	
	free( xnuc );

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

	createTree( &tree, npart, real_pos );

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

PyObject* _calcDensProjection(PyObject *self, PyObject *args) {
	PyArrayObject *pos, *hsml, *mass, *pyGrid;
	int npart, nx, ny, cells;
	int dims[2];
	double bx, by, bz, cx, cy, cz;
	double *data_pos, *data_hsml, *data_mass;
	double *grid;
	int part;
	double px, py, pz, h, v, cpx, cpy, r2, h2;
	double p[3], weight;
	int x, y;
	int xmin, xmax, ymin, ymax, axis0, axis1;
	double cellsizex, cellsizey;
	clock_t start;
	
	start = clock();

	axis0 = 0;
	axis1 = 1;
	if (!PyArg_ParseTuple( args, "O!O!O!iidddddd|ii:calcDensProjection( pos, hsml, mass, nx, ny, boxx, boxy, boxz, centerx, centery, centerz, [axis0, axis1] )", &PyArray_Type, &pos, &PyArray_Type, &hsml, &PyArray_Type, &mass, &nx, &ny, &bx, &by, &bz, &cx, &cy, &cz, &axis0, &axis1 )) {
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

	if (mass->nd != 1 || mass->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString( PyExc_ValueError, "mass has to be of dimension [n] and type double" );
		return 0;
	}

	npart = pos->dimensions[0];
	if (npart != hsml->dimensions[0] || npart != mass->dimensions[0]) {
		PyErr_SetString( PyExc_ValueError, "pos, hsml and mass have to have the same size in the first dimension" );
		return 0;
	}

	dims[0] = nx;
	dims[1] = ny;
	pyGrid = (PyArrayObject *)PyArray_FromDims( 2, dims, PyArray_DOUBLE );
	grid = (double*)pyGrid->data;
	cells = nx*ny;
	memset( grid, 0, cells*sizeof(double) );

	cellsizex = bx / nx;
	cellsizey = by / ny;

	data_pos = (double*)pos->data;
	data_hsml = (double*)hsml->data;
	data_mass = (double*)mass->data;

	for (part=0; part<npart; part++) {
		p[0] = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		p[1] = *data_pos;
		data_pos = (double*)((char*)data_pos + pos->strides[1]);
		p[2] = *data_pos;
		data_pos = (double*)((char*)data_pos - 2*pos->strides[1] + pos->strides[0]);
		
		px = p[ axis0 ];
		py = p[ axis1 ];
		pz = p[ 3 - axis0 - axis1 ];
		
		h = *data_hsml;
		data_hsml = (double*)((char*)data_hsml + hsml->strides[0]);
		h2 = h*h;

		v = *data_mass;
		data_mass = (double*)((char*)data_mass + mass->strides[0]);

		xmin = max( floor( (px - h - cx + 0.5*bx) / cellsizex ), 0 );
		xmax = min( ceil( (px + h - cx + 0.5*bx) / cellsizex ), nx-1 );
		ymin = max( floor( (py - h - cy + 0.5*by) / cellsizey ), 0 );
		ymax = min( ceil( (py + h - cy + 0.5*by) / cellsizey ), ny-1 );
		
		if (xmin < nx && ymin < ny && xmax >= 0 && ymax >= 0 && pz > cz - 0.5*bz && pz < cz + 0.5*bz) {
                        weight = 0;
			for (x=xmin; x<=xmax; x++) {
				cpx = -0.5*bx + bx*(x+0.5)/nx;
				for (y=ymin; y<=ymax; y++) {
					cpy = -0.5*by + by*(y+0.5)/ny;
					r2 = sqr(px-cpx-cx) + sqr(py-cpy-cy);
					if (r2 > h2) continue;
					weight += h * _getkernel( h, r2 );
				}
			}
			
			if (weight > 0)
			  {
    			for (x=xmin; x<=xmax; x++) {
    				cpx = -0.5*bx + bx*(x+0.5)/nx;
    				for (y=ymin; y<=ymax; y++) {
    					cpy = -0.5*by + by*(y+0.5)/ny;
    					r2 = sqr(px-cpx-cx) + sqr(py-cpy-cy);
    					if (r2 > h2) continue;
    					grid[x*ny + y] += h * _getkernel( h, r2 ) * v / weight;
    				}
    			}
    	  }
		}
	}

	printf( "Calculation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	return PyArray_Return( pyGrid );
}

static PyMethodDef calcGridmethods[] = {
	{ "calcGrid", _calcGrid, METH_VARARGS, "" },
	{ "calcSlice", _calcSlice, METH_VARARGS, "" },
	{ "calcDensGrid", _calcDensGrid, METH_VARARGS, "" },
	{ "calcDensSlice", _calcDensSlice, METH_VARARGS, "" },
	{ "calcGridMassWeight", _calcGridMassWeight, METH_VARARGS, "" },
	{ "calcCylinderAverage", _calcCylinderAverage, METH_VARARGS, "" },
	{ "calcRadialProfile", _calcRadialProfile, METH_VARARGS, "" },
	{ "calcAbundGrid", _calcAbundGrid, METH_VARARGS, "" },
	{ "calcAbundSphere", _calcAbundSphere, METH_VARARGS, "" },
	{ "gatherAbundGrid", (PyCFunction)_gatherAbundGrid, METH_VARARGS | METH_KEYWORDS, "" },
	{ "calcASlice", (PyCFunction)_calcASlice, METH_VARARGS | METH_KEYWORDS, "" },
	{ "calcDensProjection", _calcDensProjection, METH_VARARGS, "" },
	{ NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initcalcGrid(void)
{
	Py_InitModule( "calcGrid", calcGridmethods );
	import_array();
}
