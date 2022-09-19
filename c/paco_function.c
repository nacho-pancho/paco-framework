#include <gsl/gsl_matrix.h>
#include "paco_function.h"
#include "paco_log.h"
#include <string.h>

double paco_dummy_eval ( const gsl_matrix* X ) {
    paco_debug ( "(dummy eval)\n" );
    return 0;
}

void paco_dummy_fit ( const struct paco_problem* problem ) {
    paco_debug ( "(dummy fit)\n" );
}

int paco_function_validate ( const paco_function_st* f ) {
    if ( !f ) {
        paco_error ( "Null function pointer.\n" );
    } else if ( !f->fit ) {
        paco_error ( "Undefined fit function.\n" );
        return 0;
    } else if ( !f->create ) {
        paco_error ( "Undefined create function.\n" );
        return 0;
    } else if ( !f->destroy ) {
        paco_error ( "Undefined destroy function.\n" );
        return 0;
    } else if ( !f->eval ) {
        paco_error ( "Undefined eval function.\n" );
        return 0;
    } else if ( !f->prox ) {
        paco_error ( "Undefined prox function." );
        return 0;
    } else  if ( !strlen ( f->name ) ) {
        paco_error ( "Empty function name." );
        return 0;
    }
    return 1;
}

void paco_dummy_destroy() {
    paco_debug ( "(dummy destroy)\n" );
}
