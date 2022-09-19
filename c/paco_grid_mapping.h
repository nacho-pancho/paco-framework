#ifndef PACO_GRID_MAPPING_H
#define PACO_GRID_MAPPING_H
#include "paco_mapping.h"

paco_mapping_st paco_grid_mapping();

struct paco_problem;

void paco_grid_mapping_create ( struct paco_problem* problem );

void paco_grid_mapping_destroy();

void paco_grid_mapping_stitch ( gsl_matrix *I, const gsl_matrix *X );

void paco_grid_mapping_extract ( gsl_matrix *X, const gsl_matrix *I );

#endif

