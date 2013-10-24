#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include "sph.h"
#include "mersenne.h"

static inline void _sphgetkernel( double h, double r, double *wk, double *dwk ) {
	double coeff1, coeff2, coeff3, coeff4, coeff5, coeff6;
	double hinv, hinv3, hinv4, u, oneminusu;
	coeff1 = 8.0 / M_PI;
	coeff2 = coeff1 * 6.0;
	coeff3 = coeff1 * 18.0;
	coeff4 = coeff1 * 12.0;
	coeff5 = coeff1 * 2.0;
	coeff6 = -coeff1 * 6.0;

	hinv = 1.0 / h;
	hinv3 = hinv*hinv*hinv;
	hinv4 = hinv3*hinv;
	u = r*hinv;
	if (u < 0.5) {
		*wk = hinv3 * ( coeff1 + coeff2*(u-1.0)*u*u );
		*dwk = hinv4 * u * ( coeff3 * u - coeff4 );
	} else {
		oneminusu = 1.0 - u;
		*wk = hinv3 * coeff5 * oneminusu * oneminusu * oneminusu;
		*dwk = hinv4 * coeff6 * oneminusu * oneminusu;
	}
}

int createTree( t_sph_tree *tree, int npart, double *pos , double domainLen, double* center) {
	int maxnodes, result;
	//double domainLen;
	time_t start;
	
	start = clock();
	
	maxnodes = (int)(1.1 * npart);
	printf( "Creating tree for %d particles with %d nodes.\n", npart, maxnodes );
	
	initTree( tree, npart, maxnodes );
	
	result = makeTree( tree, pos, domainLen, center );
	while ( result != 0 ) {
		switch (result) {
		case 1:
			/* failed => increase maxnodes */
			maxnodes = (int)(1.1 * maxnodes);
			printf( "Repeating tree construction with %d nodes\n", maxnodes );
			break;
		}
		
		/* reinit tree */
		freeTree( tree );
		initTree( tree, npart, maxnodes );
		
		result = makeTree( tree, pos, domainLen, center );
	}
	
	organizeTree( tree );
	
	getParticles( tree, tree->topnode );
	
	printf( "Tree creation took %gs\n", ((double)clock()-(double)start)/CLOCKS_PER_SEC );
	
	return 0;
}

void initTree( t_sph_tree *tree, int npart, int maxnodes ) {
	tree->npart = npart;
	tree->maxnodes = maxnodes;
	tree->nnodes = 0;
	tree->nodes = (t_sph_treenode*)malloc( maxnodes * sizeof(t_sph_treenode) );
}

void freeTree( t_sph_tree *tree ) {
	free( tree->nodes );
}

double getDomainLen( int npart, double *pos ) {
	double len, *positer, *posend;
	
	len = 0;
	posend = &pos[npart*3];
	for (positer = pos; positer != posend; positer++) {
		if (fabs(*positer) > len) len = fabs(*positer);
	}
	
	printf( "Domainlen: %g\n", len*2.1 );
	
	return len*2.1;
}

int makeTree( t_sph_tree *tree, double *pos, double domainLen, double* center) {
	int nodecount, subnode;
	int i, j, p;
	t_sph_treenode *node, *parent;
	
	nodecount = tree->npart;
	tree->topnode = nodecount;
	
	node = &tree->nodes[nodecount];
	
	node->len = domainLen;
	for (j=0; j<3; j++) node->center[j] = center[j];
	for (j=0; j<8; j++) node->sons[j] = -1;
	
	nodecount++;
	node++;
	
	for (i=0; i<tree->npart; i++) {
		node = &tree->nodes[ tree->topnode ];
		
		while (1) {
			subnode = 0;
			if (pos[i*3+0] > node->center[0]) subnode += 1;
			if (pos[i*3+1] > node->center[1]) subnode += 2;
			if (pos[i*3+2] > node->center[2]) subnode += 4;
			
			if (node->sons[subnode] == -1) {
				/* we found an empty node */
				node->sons[subnode] = i;
				break;
			}
			
			if (node->sons[subnode] < tree->npart) {
				/* there is already a particle in this node, so we move it down one level and try again */
				
				/* check whether there is an empty node available */
				if (nodecount == tree->maxnodes) {
					return 1;
				}
				
				/* store the particle */
				p = node->sons[subnode];
				
				/* get new node */
				node->sons[subnode] = nodecount;
				parent = node;
				node = &tree->nodes[nodecount];
				nodecount++;
				
				/* initialize new node */
				node->len = 0.5 * parent->len;
				if (subnode & 1)
					node->center[0] = parent->center[0] + 0.5 * node->len;
				else
					node->center[0] = parent->center[0] - 0.5 * node->len;
				if (subnode & 2)
					node->center[1] = parent->center[1] + 0.5 * node->len;
				else
					node->center[1] = parent->center[1] - 0.5 * node->len;
				if (subnode & 4)
					node->center[2] = parent->center[2] + 0.5 * node->len;
				else
					node->center[2] = parent->center[2] - 0.5 * node->len;
				
				for (j=0; j<8; j++) node->sons[j] = -1;
				
				/* add the stored particle again */
				subnode = 0;
				if (pos[p*3+0] > node->center[0]) subnode += 1;
				if (pos[p*3+1] > node->center[1]) subnode += 2;
				if (pos[p*3+2] > node->center[2]) subnode += 4;

				/* add offset to deal with particles at the same position */
				if (node->len < 1.0e-8 * domainLen) {
					subnode = (subnode + 1) % 8;
				}
				
				node->sons[subnode] = p;
			}
			
			if (node->sons[subnode] >= tree->npart) {
				/* we have to look deeper */
				node = &tree->nodes[ node->sons[subnode] ];
			}
		}
	}
	
	tree->usednodes = nodecount;
	
	return 0;
}

int organizeTree( t_sph_tree *tree ) {
	t_sph_treenode *node, *father;
	int i, j, k;
	
	for (i=tree->npart; i<tree->usednodes; i++) {
		node = &tree->nodes[i];
		
		/* get firstborn */
		for (j=0; j<8; j++) {
			if (node->sons[j] > -1) {
				node->firstborn = node->sons[j];
				break;
			}
		}
		
		/* do all sons */
		for (j=0; j<8; j++) {
			if (node->sons[j] > -1) {
				tree->nodes[ node->sons[j] ].father = i;
				tree->nodes[ node->sons[j] ].son = j;
				
				/* get sibling */
				for (k=j+1; k<8; k++)
					if (node->sons[k] > -1) break;
				
				if (k < 8) {
					/* we have another sibling */
					tree->nodes[ node->sons[j] ].sibling = node->sons[k];
				} else {
					/* we do not have another sibling => do that later */
					tree->nodes[ node->sons[j] ].sibling = -1;
				}
			}
		}
	}
	
	node = &tree->nodes[ tree->topnode ];
	node->father = -1;
	node->sibling = -1;
	node->son = -1;
	
	/* do not do topnode here */
	for (i=0; i<tree->usednodes; i++) {
		node = &tree->nodes[i];
		
		if (node->sibling == -1 && node->father != -1) {
			father = &tree->nodes[ node->father ];
			
			while (father->father != -1) {
				/* the father itself can not help us (we tried before) => so directly go one level higher in the first step */
				node = father;
				father = &tree->nodes[ node->father ];
				
				/* find next sibling */
				for (k=node->son+1; k<8; k++)
					if (father->sons[k] > -1) break;
				
				/* if we found a sibling, its our next target. if not, we have to repeat it one level higher */
				if (k < 8) {
					tree->nodes[i].sibling = father->sons[k];
					break;
				}
			}
			
			/* if we did not find a sibling now, the node is an end point => its sibling stays -1 */
		}
	}
	
	return 0;
}

int getParticles( t_sph_tree *tree, int node ) {
	t_sph_treenode *pnode;
	int j;
	
	/* check if this is a particle node */
	if (node < tree->npart) {
	  tree->nodes[ node ].npart = 1;
	  return 1;
	}
	 
	pnode = &tree->nodes[ node ];
	pnode->npart = 0;
	
	for (j=0; j<8; j++) {
		if (pnode->sons[j] > -1) {
			pnode->npart += getParticles( tree, pnode->sons[j] );
		}
	}
	
	return pnode->npart;
}

int getNearestNode( t_sph_tree *tree, double *coord ) {
	int node, subnode;
	t_sph_treenode *pnode;
	
	node = tree->topnode;
	pnode = &tree->nodes[node];

	while (node >= tree->npart) {
	  	subnode = 0;
		if (coord[0] > pnode->center[0]) subnode += 1;
		if (coord[1] > pnode->center[1]) subnode += 2;
		if (coord[2] > pnode->center[2]) subnode += 4;

		if (pnode->sons[subnode] > -1) {
		  node = pnode->sons[subnode];
		  pnode = &tree->nodes[ node ];
		} else {
		  return node;
		}
	}

	return node;
}

int getNearestNeighbour( t_sph_tree *tree, double *pos, double *coord, int *neighbour ) {
	int *worklist;
	int workcount, nearestNeighbour;
	double distance, distance_sqr, distance_sqr_new;
	int son, node;

	if (neighbour && *neighbour >= 0 && *neighbour < tree->npart) {
		nearestNeighbour = *neighbour;
		distance_sqr = (pos[nearestNeighbour*3  ] - coord[0]) * (pos[nearestNeighbour*3  ] - coord[0]) +
			       (pos[nearestNeighbour*3+1] - coord[1]) * (pos[nearestNeighbour*3+1] - coord[1]) +
			       (pos[nearestNeighbour*3+2] - coord[2]) * (pos[nearestNeighbour*3+2] - coord[2]);
		distance = sqrt( distance_sqr );
	} else {
		nearestNeighbour = -1;

		node = getNearestNode( tree, coord );
		if (node >= tree->npart) {
		  t_sph_treenode *pnode = &tree->nodes[node];
		  distance = 2. * pnode->len;
		  distance_sqr = distance * distance;
		} else {
		  nearestNeighbour = node;
		  distance_sqr = (pos[nearestNeighbour*3  ] - coord[0]) * (pos[nearestNeighbour*3  ] - coord[0]) +
		    (pos[nearestNeighbour*3+1] - coord[1]) * (pos[nearestNeighbour*3+1] - coord[1]) +
		    (pos[nearestNeighbour*3+2] - coord[2]) * (pos[nearestNeighbour*3+2] - coord[2]);
		  distance = sqrt( distance_sqr );
		}
	}

	worklist = malloc( tree->usednodes * sizeof(int) );
	worklist[0] = tree->topnode;
	workcount = 1;

	while( workcount > 0 ) {
		node = worklist[ workcount-1 ];
		workcount--;

		if (node < tree->npart) {
			/* node is a particle: check if its closer then our currently nearest neighbour */
			distance_sqr_new = (pos[node*3  ] - coord[0]) * (pos[node*3  ] - coord[0]) +
					   (pos[node*3+1] - coord[1]) * (pos[node*3+1] - coord[1]) +
					   (pos[node*3+2] - coord[2]) * (pos[node*3+2] - coord[2]);
			if (distance_sqr_new < distance_sqr) {
				nearestNeighbour = node;
				distance_sqr = distance_sqr_new;
				distance = sqrt( distance_sqr );
			}
		} else {
			/* check whether any particle of this node could be closer to the target than
			   our current nearest neighbour. if yes, add all subnodes to worklist */
			t_sph_treenode *pnode = &tree->nodes[node];
			if ( sqrt( (pnode->center[0] - coord[0]) * (pnode->center[0] - coord[0]) +
			           (pnode->center[1] - coord[1]) * (pnode->center[1] - coord[1]) +
			           (pnode->center[2] - coord[2]) * (pnode->center[2] - coord[2]) )
			     - 0.5 * pnode->len * sqrt( 3. ) < distance) {
				for (son=0; son<8; son++) {
					if (pnode->sons[son] > -1) {
						worklist[ workcount ] = pnode->sons[son];
						workcount++;
					}
				}
			}
		}
	}

	if (neighbour) *neighbour = nearestNeighbour;

	free( worklist );
	return nearestNeighbour;
}

double calcDensity( t_sph_tree *tree, double *coord, double hsml, double *pos, double *mass, double *density, double *dhsmldensity ) {
	/* Input: coord => coordinates of the point where the density should be evaluated
	 *        hsml => smoothing length of the particle at coord
	 * 		  pos, mass => positions and masses of the particles
	 * Output: density => the density at coord
	 */
	int node, nneighbours;
	double r, r2, hinv, hsml2, hsml3, dist, wk, dwk;
	double dhsml, weighted_neighbours;
	double *pcoord, *ppos;
	t_sph_treenode *pnode, *nodes;
	
	node = tree->topnode;
	hinv = 1. / hsml;
	hsml2 = hsml * hsml;
	hsml3 = hsml2 * hsml;
	
	nodes = tree->nodes;

	*density = 0;
	dhsml = 0;
	nneighbours = 0;
	weighted_neighbours = 0;
	while (node != -1) {
		if (node < tree->npart) {
			/* we found a particle node, that contains only one particle */			
			/* check if the particle is within a sphere of radius hsml around coord */
			/* for (i=0, r2=0; i<3; i++) r2 += (coord[i]-pos[node*3+i])*(coord[i]-pos[node*3+i]); */
			pcoord = coord;
			ppos = &pos[node*3];
			r2 = (*pcoord-*ppos)*(*pcoord-*ppos);
			pcoord++;
			ppos++;
			r2 += (*pcoord-*ppos)*(*pcoord-*ppos);
			pcoord++;
			ppos++;
			r2 += (*pcoord-*ppos)*(*pcoord-*ppos);
			if (r2 < hsml2) {
				r = sqrt( r2 );
				_sphgetkernel( hsml, r, &wk, &dwk );
				*density += wk * mass[node];
				dhsml -= mass[node] * hinv * (3. * wk + r * dwk);
				
				nneighbours++;
				weighted_neighbours += 4./3. * M_PI * wk * hsml3;
			}
			
			/* go on with next particle */
			node = (nodes[ node ]).sibling;
		} else {
			/* we found a real node, so lets check if we can exclude it or have to dig deeper */
			/* check if any particle within this node is able to overlap with  a sphere of radius hsml around coord */
			pnode = &nodes[node];
			dist = hsml + 0.5 * pnode->len;
			
			if ( (fabs( coord[0] - pnode->center[0] ) < dist) &&
			     (fabs( coord[1] - pnode->center[1] ) < dist) &&
			     (fabs( coord[2] - pnode->center[2] ) < dist) ) {
				node = (nodes[ node ]).firstborn;
			} else {
				node = (nodes[ node ]).sibling;
			}
		}
	}
	
	if (dhsmldensity) {
	  dhsml *= hsml / ( 3. * (*density) );
	  if (dhsml > -0.9) {
	    *dhsmldensity = 1. / (1. + dhsml);
	  } else {
	    *dhsmldensity = 1.;
	  }
	}

	return weighted_neighbours;
}

double calcHsml( t_sph_tree *tree, double *coord, double *pos, double *mass, int nneighbours, double *hsml, double *density ) {
	/* Input:  coord => coordinates of the point where the density should be evaluated
	 * 		   pos, mass => positions and masses of the particles
	 *         nneighbours => the desired (weighted!!!) number of neighbours (not equal to actual numbers of neighbours within sphere of radius hsml)
	 * Output: density => the density at coord
	 *         hsml => the smoothing length of the particle at coord
	 *         
	 *         significant parts of this routine have been copied from the GADGET3 code
	 */
	int node, iter;
	t_sph_treenode *pnode, *nodes;
	double weighted_neighbours, dhsmldensity;
	double left, right, fac;
	
	nodes = tree->nodes;

	if (*hsml == 0) {
		/* get initial guess */
		node = getNearestNode( tree, coord );
		pnode = &nodes[node];
		
		while (pnode->npart < 10*nneighbours && pnode->father != -1) {
			node = pnode->father;
			pnode = &nodes[node];
		}
		
		*hsml = pow( 3.0 / (4 * M_PI) * nneighbours / pnode->npart, 1.0 / 3 ) * pnode->len;
	}
	
	left = right = 0;
	iter = 0;
	
	while (1) {
		weighted_neighbours = calcDensity( tree, coord, *hsml, pos, mass, density, &dhsmldensity );
		
		if (weighted_neighbours < nneighbours-NEIGHBOURTOLERANCE ||
			weighted_neighbours > nneighbours+NEIGHBOURTOLERANCE) {
			/* we have to improve our guess */
			
			if (left > 0 && right > 0 && right-left < 1.0e-3*left)
				/* its close enough */
				break;
			
			if (weighted_neighbours < nneighbours-NEIGHBOURTOLERANCE) {
				left = max( *hsml, left );
			} else {
				if (right != 0) {
					if (*hsml < right)
						right = *hsml;
				} else {
					right = *hsml;
				}
			}
			
			if (right > 0 && left > 0) {
				*hsml = pow( 0.5 * (left*left*left + right*right*right), 1./3. );
			} else {
				if (right == 0 && left == 0) {
					printf( "Something very bad just happened...\n" );
					return -1;
				}
			
				if (right == 0 && left > 0) {
					if (fabs(weighted_neighbours - nneighbours) < 0.5 * nneighbours) {
						fac = 1.0 - (weighted_neighbours - nneighbours) / (3. * weighted_neighbours) * dhsmldensity;
						if (fac < 1.26) {
							*hsml *= fac;
						} else {
							*hsml *= 1.26;
						}
					} else {
						*hsml *= 1.26;
					}
				}
			
				if (right > 0 && left == 0) {
					if (fabs(weighted_neighbours - nneighbours) < 0.5 * nneighbours) {
						fac = 1.0 - (weighted_neighbours - nneighbours) / (3. * weighted_neighbours) * dhsmldensity;
						if (fac > 1./1.26) {
							*hsml *= fac;
						} else {
							*hsml /= 1.26;
						}
					} else {
						*hsml /= 1.26;
					}
				}
		    }
		} else {
			/* we are done */
			break;
		}
		
		iter++;
		if (iter > MAXITER) {
			printf( "Neighbour iteration did not converge.\n" );
			break;
		}
	}
	
	return weighted_neighbours;
}

double getNNeighbours( t_sph_tree *tree, double *coord, double *pos, int nneighbours, int *nneighbours_real, int **neighbours, int *converged ) {
	/* Input: coord => coordinates of the point where the density should be evaluated
	 *        nneighbours => number of neighbours required
	 * 		  pos => positions of the particles
	 * Output: neighbours => a list of particles within a sphere of radius hsml around coord
	 */

	double minradius, maxradius, radius;
	int node, neighbourcount, iter;
	t_sph_treenode *pnode;

	node = getNearestNode( tree, coord );

	while (tree->nodes[ node ].npart < nneighbours) {
		node = tree->nodes[ node ].father;
	}

	pnode = &tree->nodes[ node ];

	maxradius = pnode->len;
	minradius = pnode->len / 2.;

	while ( (neighbourcount = getNeighbours( tree, coord, pos, maxradius, 0 )) < nneighbours ) {
		minradius = maxradius;
		maxradius *= 2.;
	}

	while ( (neighbourcount = getNeighbours( tree, coord, pos, minradius, 0 )) > nneighbours ) {
		maxradius = minradius;
		minradius /= 2.;
	}

 	neighbourcount = -1;
	iter = 0;
	while ( neighbourcount != nneighbours && iter < 50) {
		radius = 0.5 * (minradius + maxradius);
		neighbourcount = getNeighbours( tree, coord, pos, radius, 0 );
		
		if (neighbourcount > nneighbours) {
			maxradius = radius;
		}

		if (neighbourcount < nneighbours) {
			minradius = radius;
		}

		if ((maxradius-minradius)/radius < 1e-3) {
			break;
		}

		iter++;
	}

	if (iter < 50) {
		*converged = 1;
	} else {
		*converged = 0;
	}

	radius = 0.5 * (minradius + maxradius);
	neighbourcount = getNeighbours( tree, coord, pos, radius, neighbours );
	if (nneighbours_real) *nneighbours_real = neighbourcount;
	return radius;
}

int getNeighbours( t_sph_tree *tree, double *coord, double *pos, double hsml, int **neighbours ) {
	/* Input: coord => coordinates of the point where the density should be evaluated
	 *        hsml => smoothing length of the particle at coord
	 * 		  pos => positions of the particles
	 * Output: neighbours => a list of particles within a sphere of radius hsml around coord
	 */
	int node, nneighbours, maxneighbours, *pneighbours, *dummy;
	double r2, hsml2, dist;
	double *pcoord, *ppos;
	t_sph_treenode *pnode, *nodes;
	
	node = tree->topnode;
	hsml2 = hsml * hsml;

	nodes = tree->nodes;

	nneighbours = 0;
	maxneighbours = 50;
	pneighbours = (int*)malloc( maxneighbours*sizeof(int) );
	
	while (node != -1) {
		if (node < tree->npart) {
			/* we found a particle node, that contains only one particle */			
			/* check if the particle is within a sphere of radius hsml around coord */
			/* for (i=0, r2=0; i<3; i++) r2 += (coord[i]-pos[node*3+i])*(coord[i]-pos[node*3+i]); */
			pcoord = coord;
			ppos = &pos[node*3];
			r2 = (*pcoord-*ppos)*(*pcoord-*ppos);
			pcoord++;
			ppos++;
			r2 += (*pcoord-*ppos)*(*pcoord-*ppos);
			pcoord++;
			ppos++;
			r2 += (*pcoord-*ppos)*(*pcoord-*ppos);
			if (r2 < hsml2) {
				if (nneighbours == maxneighbours) {
					dummy = pneighbours;
					pneighbours = (int*)malloc( 2*maxneighbours*sizeof(int) );
					memcpy( pneighbours, dummy, maxneighbours*sizeof(int) );
					free( dummy );
					maxneighbours *= 2;
				}
				pneighbours[nneighbours] = node;
				nneighbours++;
			}
			
			/* go on with next particle */
			node = (nodes[ node ]).sibling;
		} else {
			/* we found a real node, so lets check if we can exclude it or have to dig deeper */
			/* check if any particle within this node is able to overlap with  a sphere of radius hsml around coord */
			pnode = &nodes[node];
			dist = hsml + 0.5 * pnode->len;
			
			if ( (fabs( coord[0] - pnode->center[0] ) < dist) &&
			     (fabs( coord[1] - pnode->center[1] ) < dist) &&
			     (fabs( coord[2] - pnode->center[2] ) < dist) ) {
				node = (nodes[ node ]).firstborn;
			} else {
				node = (nodes[ node ]).sibling;
			}
		}
	}

	if (neighbours) {
		*neighbours = pneighbours;
	} else {
		free( pneighbours );
	}
	return nneighbours;
}

int relaxData( int npart, double *pos, double *mass, double *hsml, int nneighbours, int ncells, double dr, double *rho, int nsteps ) {
	t_sph_tree tree;
	double r2, density;
	double err_min, err_max, err_avg, error;
	double coord[3], *errors;
	int i, j, idx, iter;
	
	errors = (double*)malloc( npart * sizeof( double ) );
	
	double center[] = {0.,0.,0.};
	createTree( &tree, npart, pos, getDomainLen(npart, pos) , center);
	
	err_min = 1e30;
	err_max = 0;
	err_avg = 0;
	for (i=0; i<npart; i++) {
		for (j=0, r2=0; j<3; j++) r2 += pos[i*3+j] * pos[i*3+j];
		idx = sqrt( r2 ) / dr;
		if (idx > ncells)
				idx = ncells-1;
		
		calcHsml( &tree, &pos[i*3], pos, mass, nneighbours, &hsml[i], &density );
		error = fabs( rho[idx] - density ) / rho[idx];
		errors[i] = error;
		
		if (error < err_min) err_min = error;
		if (error > err_max) err_max = error;
		err_avg += error;
	}
	
	printf( "Min Error: %g\n", err_min );
	printf( "Max Error: %g\n", err_max );
	printf( "Avg Error: %g\n", err_avg / (double)npart );
	
	iter = 0;
	while (iter < nsteps) {
		if (iter > 0) {
			freeTree( &tree );
			createTree( &tree, npart, pos, getDomainLen(npart, pos), center  );
		}
		
		err_min = 1e30;
		err_max = 0;
		err_avg = 0;
	
		for (i=0; i<npart; i++) {
			for (j=0, r2=0; j<3; j++) {
				coord[j] = (-1. + 2.*randMT()) * 0.5 * hsml[i] + pos[i*3+j];
				r2 += coord[j] * coord[j];
			}
			
			calcHsml( &tree, coord, pos, mass, nneighbours, &hsml[i], &density );
			
			idx = sqrt( r2 ) / dr;
			if (idx > ncells)
				idx = ncells-1;
			error = fabs( rho[idx] - density ) / rho[idx];
			
			if ((error < errors[i]) || (exp(-(error-errors[i])*1e6) > randMT()) ) {
				for (j=0; j<3; j++)
					pos[i*3+j] = coord[j];
			} else {
				for (j=0, r2=0; j<3; j++)
					r2 += pos[i*3+j] * pos[i*3+j];
				
				idx = sqrt( r2 ) / dr;
				if (idx > ncells)
					idx = ncells-1;
				error = fabs( rho[idx] - density ) / rho[idx];
			}
			
			errors[i] = error;
			
			if (error < err_min) err_min = error;
			if (error > err_max) err_max = error;
			err_avg += error;
		}
		
		iter++;
		printf( "Step %d, Min Error: %g\n", iter, err_min );
		printf( "Step %d, Max Error: %g\n", iter, err_max );
		printf( "Step %d, Avg Error: %g\n", iter, err_avg / (double)npart );
	}
	
	freeTree( &tree );
	free( errors );
	
	return 0;
}
