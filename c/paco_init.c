#include <assert.h>
#include <float.h>

#include "paco_init.h"
#include "paco_log.h"
#include "paco_types.h"
#include "paco_mapping.h"
#include "paco_problem.h"

void paco_init_base ( struct paco_problem* problem ) {
    paco_info ( "Common initialization (everything set to 0).\n" );
    paco_iterate_st* iter = &problem->iter;
    paco_config_st* cfg = &problem->config;

    gsl_matrix_set_zero ( iter->A );
    gsl_matrix_set_zero ( iter->B );
    gsl_matrix_set_zero ( iter->U );
    gsl_matrix_set_zero ( iter->X );
    iter->dAB = FLT_MAX;
    iter->f = FLT_MAX;
    iter->k = 0;
    iter->lambda = cfg->admm_penalty;
    iter->nA = 1;
    iter->nB = 1;
    iter->nU = 1;
}

void paco_init_default ( struct paco_problem* problem ) {
    paco_init_base ( problem );
    paco_iterate_st* iter = &problem->iter;
    gsl_matrix_memcpy ( iter->X, problem->data.input );
    problem->mapping.extract ( iter->B, iter->X );
}

void paco_init_missing_with_average (
    struct paco_problem* problem ) {
    assert ( problem != NULL );
    paco_data_st* data = &problem->data;
    paco_iterate_st* iter = &problem->iter;
    paco_mapping_st* map = &problem->mapping;

    paco_init_base ( problem );
    paco_info ( "Initializing missing samples with average of known samples...\n" );
    if ( data->mask == NULL ) {
        paco_warn ( "No mask. Nothing to initialize.\n" );
        return;
    }
    const gsl_matrix* input_samples = data->input;
    const gsl_matrix* initial_samples = data->initial;
    const gsl_matrix* mask_samples = data->mask;
    const index_t M = input_samples->size1;
    const index_t N = input_samples->size2;
    assert ( mask_samples != NULL );
    assert ( input_samples != NULL );

    if ( initial_samples == NULL ) {
        //
        // compute the average value of the known samples
        //
        paco_info ( "Computing average pixel value of known regions.\n" );
        double numpix = 0;
        double sumpix = 0;

        for ( index_t i = 0; i < M; i++ ) {
            for ( index_t j = 0; j < N; j++ ) {
                const sample_t x = gsl_matrix_get ( input_samples, i, j );
                if ( gsl_matrix_get ( mask_samples, i, j ) == 0 ) {
                    sumpix += x;
                    numpix ++;
                    gsl_matrix_set ( iter->X, i, j, x );
                }
            }
        }
        if ( numpix == 0 ) {
            paco_warn ( "No known regions??\n" );
        }
        //
        // fill unknown regions with average value
        //
        sumpix /= numpix;
        paco_info ( "initializing erased regions to value %7.5f\n", sumpix );

        for ( index_t i = 0; i < M; i++ ) {
            for ( index_t j = 0; j < N; j++ ) {
                if ( gsl_matrix_get ( mask_samples, i, j ) != 0 ) {
                    gsl_matrix_set ( iter->X, i, j, sumpix );
                }
            }
        }
    } else {
        paco_info ( "Using provided initial estimation\n" );
        gsl_matrix_memcpy ( iter->X, initial_samples );
    }
    paco_info ( "Extracting initial patches.\n" );
    //
    // extract the patches from the initial signal estimation
    // and compute their DCT: this is the initial value of the ADMM variable B
    //
    map->extract ( iter->B, iter->X );
}
