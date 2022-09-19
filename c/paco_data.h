#ifndef PACO_DATA_H
#define PACO_DATA_H
#include <gsl/gsl_matrix.h>

typedef struct paco_data {
    const gsl_matrix* input;
    const gsl_matrix* mask;
    const gsl_matrix* reference;
    gsl_matrix* initial;
} paco_data_st;



typedef int ( *paco_data_validate_f ) ( const paco_data_st* data );

int paco_data_validate_default ( const paco_data_st* data );

int paco_data_validate_input_and_mask ( const paco_data_st* data );

#endif