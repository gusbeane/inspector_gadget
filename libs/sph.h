#ifndef SPH_H
#define SPH_H

#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))

#define NEIGHBOURTOLERANCE 1
#define MAXITER 150

typedef struct {
	double len;
	double center[3];
	int father;
	int sibling;
	int firstborn;
	int son;
	int sons[8];
	int npart;
} t_sph_treenode;

typedef struct {
	int npart;
	int maxnodes;
	int nnodes;
	int topnode;
	int usednodes;
	t_sph_treenode *nodes; /* stores the nodes */
} t_sph_tree;

int createTree( t_sph_tree *tree, int npart, double *pos );
void initTree( t_sph_tree *tree, int npart, int maxnodes );
void freeTree( t_sph_tree *tree );
double getDomainLen( int npart, double *pos );
int makeTree( t_sph_tree *tree, double *pos, double domainLen );
int organizeTree( t_sph_tree *tree );
int getParticles( t_sph_tree *tree, int node );
int getNearestNode( t_sph_tree *tree, double *coord );
int getNearestNeighbour( t_sph_tree *tree, double *pos, double *coord, int *neighbour );
double calcDensity( t_sph_tree *tree, double *coord, double hsml, double *pos, double *mass, double *density, double *dhsmldensity );
double calcHsml( t_sph_tree *tree, double *coord, double *pos, double *mass, int nneighbours, double *hsml, double *density );
double getNNeighbours( t_sph_tree *tree, double *coord, double *pos, int nneighbours, int *nneighbours_real, int **neighbours, int *converged);
int getNeighbours( t_sph_tree *tree, double *coord, double *pos, double hsml, int **neighbours );
int relaxData( int npart, double *pos, double *mass, double *hsml, int nneighbours, int ncells, double dr, double *rho, int nsteps );


#endif /* SPH_H */
