#ifndef PACO_L12_H
#define PACO_L12_H
/**
 * L1,2 cost function for groups of size 2.
 * Useful for computing isotropic TV norms:
 * 
 * f(x) = \sum_{i=1}^{m} f_i( x_{2i-1}, x_{2i} ) = \sum_{i=1}^{m} \sqrt{ x_{2i-1}^2 + x_{2i}^2 }
 * 
 * proximal soft vector thresholding on groups of size 2:
 * 
 * (z_{2i-1},z_{2i}) = prox_{af_i}(x_{2i-1},x_{2i}) = (1- a/max{a,b_i} ) * ( x_{2i-1}^2 , x_{2i} )
 * where b_i = \sqrt{ x_{2i-1}^2 + x_{2i}^2 } 
 */
#include "paco_types.h"
#include "paco_function.h"

struct paco_problem;

paco_function_st paco_l12();

void paco_l12_create ( const struct paco_problem* problem );

void paco_l12_fit ( const struct paco_problem* problem );

double paco_l12_eval ( const gsl_matrix *B );

void paco_l12_prox ( gsl_matrix *SB, const gsl_matrix *B, const double tau );

void paco_l12_prox_vec ( gsl_vector *SB, const gsl_vector *B, const double tau );

void paco_l12_destroy();


#endif
