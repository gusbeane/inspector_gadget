#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double P_0(double x)
{
  return 1;
}

double P_1(double x)
{
  return sqrt(3)*x;
}

double P_2(double x)
{
  return sqrt(5.)*0.5*(-1.+x*(x*3.));
}

double P_3(double x)
{
  return sqrt(7.)*0.5*(x*(-3.+x*(x*5.)));
}

double P_4(double x)
{
  return 3./8.*(3.+x*(x*(-30.+x*(x*35.))));
}

double P_5(double x)
{
  return sqrt(11.)/8.*(x*(15+x*(x*(-70+x*(x*63)))));
}

double P_k(int k, double xi)
{
    switch(k)
  {
  case 0:
    return P_0(xi);
    break;
  case 1:
    return P_1(xi);
    break;
  case 2:
    return P_2(xi);
    break;
  case 3:
    return P_3(xi);
    break;
  case 4:
    return P_4(xi);
    break;
  case 5:
    return P_5(xi);
    break;
  default:
    printf("Order of base function not supported!\n");
    return 0;
  }
}

void index_to_base_function2d(int k, int* Px, int* Py)
{
  int degree = 0;
  int counter = 0;

  while(1)
  {
    if(k<=counter)
    {
      break;
    }
    else
    {
      degree++;
      counter += degree+1;
    }
  }

  *Px=0;
  *Py=degree;

  while(k != counter)
  {
    counter--;
    *Px=*Px+1;
    *Py=*Py-1;
  }

  return;
}

void index_to_base_function3d(int k, int *Px, int *Py, int *Pz, int degree_k)
{
  int DEGREE_K = degree_k;

  int u,v,w,deg_k;

  int counter = 0;

  for(deg_k=0;deg_k<=DEGREE_K;deg_k++)
    {
      for(u=0;u<=deg_k;u++)
        {
          for(v=0;v<=deg_k-u;v++)
            {
              for(w=0;w<=deg_k-u-v;w++)
                {
                  if(u+v+w==deg_k)
                    {
                      if(counter==k)
                        {
                          *Px=w;
                          *Py=v;
                          *Pz=u;
                         
                          return;
                        }
                      else
                        {
                          counter++;
                        }
                    }
                }
            }
        }
    }

  *Px=0;
  *Py=0;
  *Pz=0;

  printf("Error in index_to_base_function3d: Couldn't calculate index from base function!\n");
  return;
}

double dg_get_value2d(double* weights, int nof_base_fcts, double cell_x, double cell_y, double cell_dl, double x, double y)
{
  double xi_1,xi_2;

  //transform to cell coordinates
  xi_1=2./cell_dl*(x-cell_x);
  xi_2=2./cell_dl*(y-cell_y);

  int px;
  int py;

  int index;
  double value = 0.;

  for(index = 0; index < nof_base_fcts; index++)
    {
      index_to_base_function2d(index, &px, &py);

      value += weights[index] * P_k(px, xi_1)*P_k(py, xi_2);
    }

  return value;
}

double dg_get_value3d(double* weights, int nof_base_fcts, int degree_k, double cell_x, double cell_y, double cell_z, double cell_dl, double x, double y, double z)
{
  double xi_1,xi_2, xi_3;

  //transform to cell coordinates
  xi_1=2./cell_dl*(x-cell_x);
  xi_2=2./cell_dl*(y-cell_y);
  xi_3=2./cell_dl*(z-cell_z);

  int px;
  int py;
  int pz;

  int index;
  double value = 0.;
  for(index = 0; index < nof_base_fcts; index++)
    {
      index_to_base_function3d(index, &px, &py, &pz, degree_k);
      
      value += weights[index] * P_k(px, xi_1)*P_k(py, xi_2)*P_k(pz, xi_3);
    }

  return value;
}
