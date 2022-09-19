#ifndef PACO_MOE_H
#define PACO_MOE_H

#include "paco_types.h"

paco_function_st paco_moe();

void paco_moe_create ( const struct paco_problem* problem );

void paco_moe_fit ( const struct paco_problem* problem );

double paco_moe_eval ( const gsl_matrix *B );

void paco_moe_prox ( gsl_matrix *SB, const gsl_matrix *B, const double tau );

void paco_moe_prox_vec ( gsl_vector *SB, const gsl_vector *B, const double tau );

void paco_moe_destroy();



#endif