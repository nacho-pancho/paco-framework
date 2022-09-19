#include "paco_data.h"
#include "paco_log.h"

int paco_data_validate_default ( const paco_data_st* data ) {
    const gsl_matrix* mask = data->mask;
    const gsl_matrix* input = data->input;
    const gsl_matrix* ref = data->reference;
    const gsl_matrix* init = data->initial;

    if ( !input ) {
        paco_error ( "Input not defined.\n" );
        return 0;
    }

    if ( mask ) {
        if ( mask->size1 != input->size1 ) {
            paco_error ( "Error: number of rows in input and mask do not match.\n" );
            return 0;
        }

        if ( mask->size2 != input->size2 ) {
            paco_error ( "Error: number of columns in input and mask do not match.\n" );
            return 0;
        }
    }

    if ( init ) {
        if ( init->size1 != input->size1 ) {
            paco_error ( "Error: number of rows in initialization and input do not match.\n" );
            return 0;
        }

        if ( init->size2 != input->size2 ) {
            paco_error ( "Error: number of columns in initialization and input do not match.\n" );
            return 0;
        }
    }
    if ( ref ) {
        if ( ref->size1 != input->size1 ) {
            paco_error ( "Error: number of rows in reference and input do not match.\n" );
            return 0;
        }

        if ( ref->size2 != input->size2 ) {
            paco_error ( "Error: number of columns in reference and input do not match.\n" );
            return 0;
        }
    }
    return 1;
}

int paco_data_validate_input_and_mask ( const paco_data_st* data ) {
    if ( !data->mask ) {
        paco_error ( "Mask not defined!\n" );
        return 0;
    }
    return paco_data_validate_default ( data );
}
