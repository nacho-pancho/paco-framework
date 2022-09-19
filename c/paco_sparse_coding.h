#ifndef PACO_SPARSE_CODING_H
#define PACO_SPARSE_CODING_H
#include <gsl/gsl_matrix.h>

/**
 * Solve a = arg min ||a||_1 subject to Da = x
 */
void paco_basis_pursuit ( const gsl_matrix *D, const gsl_matrix *patches, gsl_matrix *coeffs, const double eps );

void paco_basis_pursuit_double ( const gsl_matrix *D, const gsl_matrix *patches, gsl_matrix *coeffs, const double eps );

void paco_dict_learn ( const gsl_matrix *patches, gsl_matrix *dict, gsl_matrix *coeffs, const double eps );

void paco_dict_learn_double ( const gsl_matrix *patches, gsl_matrix *dict, gsl_matrix *coeffs, const double eps );

#endif
