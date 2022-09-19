#ifndef PACO_SPARSE_L1
#define PACO_SPARSE_L1

#include "paco_types.h"
#include "paco_function.h"

struct paco_problem;

paco_function_st paco_sparse_l1();

void paco_sparse_l1_create ( const struct paco_problem* problem );

void paco_sparse_l1_fit ( const struct paco_problem* problem );

double paco_sparse_l1_eval ( const gsl_matrix *B );

void paco_sparse_l1_prox ( gsl_matrix *SB, const gsl_matrix *B, const double tau );

void paco_sparse_l1_prox_vec ( gsl_vector *SB, const gsl_vector *B, const double tau );

void paco_sparse_l1_destroy();


#endif