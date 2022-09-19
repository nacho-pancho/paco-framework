#ifndef PACO_FUNCTION_H
#define PACO_FUNCTION_H

#include <gsl/gsl_matrix.h>
#include "paco_config.h"

struct paco_problem;
/**
 * \brief Prototype for fiting parameters of a given function
 */
typedef  void ( *paco_function_create_f ) ( const struct paco_problem* problem );

/**
 * \brief Prototype for fiting parameters of a given function
 */
typedef  void ( *paco_function_fit_f ) ( const struct paco_problem* problem );

/**
 * \brief Prototype for evaluating a function of the PACO problem.
 * Implementation must be provided at linkage time by some C module
 */
typedef  double ( *paco_function_eval_f ) ( const gsl_matrix *X );

/**
 * \brief Proximal operator prototype for a given function
 */
typedef void ( *paco_function_prox_f ) ( gsl_matrix *SA, const gsl_matrix *A, const double tau );


/**
 * \brief Prototype for destroying a function.
 */
typedef  void ( *paco_function_destroy_f ) (  );

typedef struct paco_function {
    char name[64];
    paco_function_create_f create;
    paco_function_fit_f fit;
    paco_function_eval_f eval;
    paco_function_prox_f prox;
    paco_function_destroy_f destroy;
} paco_function_st;

int paco_function_validate ( const paco_function_st* f );

double paco_dummy_eval ( const gsl_matrix* X );

void paco_dummy_fit ( const struct paco_problem* problem );

void paco_dummy_destroy();

#endif