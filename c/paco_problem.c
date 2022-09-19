#include <assert.h>
#include "paco_problem.h"
#include "paco_log.h"
#include "paco_linalg.h"
#include "paco_image.h"
#include "paco_io.h"
#include "paco_metrics.h"
#include <string.h>

paco_problem_st  paco_problem_factory ( paco_problem_spec_st* spec ) {
    paco_problem_st problem;
    problem.validate_data = spec->validate_data;
    problem.init          = spec->init;
    problem.solve         = spec->solve;
    problem.monitor       = spec->monitor;
    problem.mapping       = spec->mapping;
    problem.cost_function = spec->cost_function;
    problem.constraint_function = spec->constraint_function;
    return problem;
}

//============================================================================

static paco_iterate_st paco_iterate_create ( const paco_problem_st* problem ) {
    paco_iterate_st out;
    const paco_mapping_st* map = &problem->mapping;
    const paco_config_st* cfg = &problem->config;

    const index_t n = map->num_mapped_patches;
    const index_t m = map->patch_dim;
    const index_t p = cfg->D ? cfg->D->size1 : m;
    const index_t M = map->input_nrows;
    const index_t N = map->input_ncols;
    out.X = gsl_matrix_alloc ( M, N );
    out.A = gsl_matrix_alloc ( n, m );
    out.B = gsl_matrix_alloc ( n, p );
    out.U = gsl_matrix_alloc ( n, p );
    return out;
}

static void paco_iterate_destroy ( paco_iterate_st *iter ) {
    gsl_matrix_free ( iter->A );
    gsl_matrix_free ( iter->B );
    gsl_matrix_free ( iter->U );
    gsl_matrix_free ( iter->X );
    memset ( iter, 0, sizeof ( paco_iterate_st ) );
}

void paco_problem_create ( paco_problem_st* problem ) {
    paco_info ( "Creating mapping %s...\n", problem->mapping.name );
    problem->mapping.create ( problem );
    paco_info ( "Creating cost function %s...\n", problem->cost_function.name );
    problem->cost_function.create ( problem );
    paco_info ( "Creating constraint function %s...\n", problem->constraint_function.name );
    problem->constraint_function.create ( problem );
    problem->iter = paco_iterate_create ( problem );
}

void paco_problem_destroy ( paco_problem_st* problem ) {
    paco_info ( "Destroying cost function %s...\n", problem->cost_function.name );
    problem->cost_function.destroy();
    paco_info ( "Destroying constraint function %s...\n", problem->constraint_function.name );
    problem->constraint_function.destroy();
    paco_info ( "Destroying mapping %s...\n", problem->mapping.name );
    problem->mapping.destroy();
    paco_info ( "Destroying iterate ...\n", problem->mapping.name );
    paco_iterate_destroy ( &problem->iter );
}

