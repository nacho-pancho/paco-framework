#include <assert.h>
#include "paco_problem_spec.h"
#include "paco_log.h"
#include <string.h>

/// number of specs defined
static index_t num_specs = 0;

/// unordered list of specs (name is included in the spec itself)
static paco_problem_spec_st specs[128]; // I don't think there will ever be so many specs

index_t paco_problem_spec_num() {
    return num_specs;
}

int paco_problem_spec_validate ( paco_problem_spec_st* spec ) {
    if ( !paco_mapping_validate ( &spec->mapping ) ) {
        paco_error ( "Invalid mapping function.\n" );
        return 0;
    } else if ( !paco_function_validate ( &spec->constraint_function ) ) {
        paco_error ( "Invalid constraint function.\n" );
        return 0;
    } else if ( !paco_function_validate ( &spec->cost_function ) ) {
        paco_error ( "Invalid cost function.\n" );
        return 0;
    } else if ( spec->validate_data == NULL ) {
        paco_error ( "Undefined validate function.\n" );
        return 0;
    } else if ( spec->init == NULL ) {
        paco_error ( "Undefined init function.\n" );
        return 0;
    } else if ( spec->solve == NULL ) {
        paco_error ( "Undefined solve function.\n" );
        return 0;
    } else if ( spec->monitor == NULL ) {
        paco_error ( "Undefined monitor function.\n" );
        return 0;
    } else if ( !strlen ( spec->name ) ) {
        paco_error ( "Empty spec name." );
        return 0;
    }
    return 1;
}

paco_problem_spec_st paco_problem_spec_get ( const char* name ) {
    paco_problem_spec_st ret;
    for ( index_t i = 0; i < num_specs; ++i ) {
        if ( !strcasecmp ( name, specs[i].name ) ) {
            return specs[i];
        }
    }
    memset ( &ret, 0, sizeof ( paco_problem_spec_st ) );
    return ret;
}

const char* paco_problem_spec_name ( index_t i ) {
    assert ( i < num_specs );
    return specs[i].name;
}

const char* paco_problem_spec_description ( index_t i ) {
    assert ( i < num_specs );
    return specs[i].name;
}

void paco_problem_spec_register ( paco_problem_spec_st* spec ) {
    if ( num_specs < 128 ) {
        memcpy ( &specs[num_specs], spec, sizeof ( paco_problem_spec_st ) );
        num_specs++;
    }
}
