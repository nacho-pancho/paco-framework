#ifndef PACO_UTIL_H
#define PACO_UTIL_H

#include "paco_types.h"

gsl_matrix *subsample_patches ( const gsl_matrix *X, const index_t nsub, gsl_matrix **pXsub );

gsl_matrix *paco_dict_mosaic ( const gsl_matrix *D, gsl_matrix **pI,
                               int margin, int mag, double bg );

sample_t gsl_vector_mean ( const gsl_vector* x );

double l2dist ( const gsl_vector* A, const gsl_vector* B );

double frobenius_squared_dist ( const gsl_matrix* A, const gsl_matrix* B );


#endif