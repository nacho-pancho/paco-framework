#ifndef PACO_TV_H
#define PACO_TV_H

#include "paco_types.h"
#include "paco_function.h"

struct paco_problem;

paco_function_st paco_tv();

void paco_tv_create ( const struct paco_problem* problem );

void paco_tv_fit ( const struct paco_problem* problem );

double paco_tv_eval ( const gsl_matrix *B );

void paco_tv_prox ( gsl_matrix *SB, const gsl_matrix *B, const double tau );

void paco_tv_prox_vec ( gsl_vector *SB, const gsl_vector *B, const double tau );

void paco_tv_destroy();


#endif
