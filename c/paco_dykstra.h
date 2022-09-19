#ifndef PACO_DYKSTRA_H
#define PACO_DYKSTRA_H
#include <gsl/gsl_matrix.h>

/**
 * \brief general interface for projection operators to be used by Dykstra's algorithm
 */
typedef void ( *paco_proj_f ) ( gsl_matrix *PA, const gsl_matrix *A, void *params );

/**
 * \brief Dykstra's algorithm for Z = proj_C[X] where C is the intersection of two convex sets A and B
 * The projections onto A and B are provided by two functions passed as arguments as projA, projB
 */
void paco_dykstra ( gsl_matrix *Z, const gsl_matrix *X, paco_proj_f projA, void *paramsA, paco_proj_f projB, void *paramsB );


/**
 * \brief Generic Dykstra's algorithm for Z = proj_C[X] where C is the intersection of J>2 convex sets A[i], i = 2,...,J
 * The projections onto A and B are provided by J functions passed as arguments in the array proj_fun
 */
void paco_dykstra_gen ( gsl_matrix *Z, const gsl_matrix *X, paco_proj_f *proj_fun, void **proj_param, const index_t J );

#endif