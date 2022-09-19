/*
  * Copyright (c) 2019 Ignacio Francisco Ramírez Paulino and Ignacio Hounie
  * This program is free software: you can redistribute it and/or modify it
  * under the terms of the GNU Affero General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or (at
  * your option) any later version.
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
  * General Public License for more details.
  * You should have received a copy of the GNU Affero General Public License
  *  along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef PACO_PROBLEM_SPEC_H
#define PACO_PROBLEM_SPEC_H

#include <argp.h>                     // argument parsing
#include <gsl/gsl_matrix.h>
#include "paco_mapping.h"
#include "paco_config.h"
#include "paco_function.h"
#include "paco_solver.h"
#include "paco_monitor.h"
#include "paco_init.h"
#include "paco_solver.h"
#include "paco_data.h"
#include "paco_mapping.h"

typedef struct paco_problem_spec {
    char name[64];
    char description[192];
    /* functions */
    paco_data_validate_f validate_data;
    paco_init_f init;
    paco_solver_f solve;
    paco_monitor_f monitor;
    /* functors */
    paco_mapping_st mapping;
    paco_function_st cost_function;
    paco_function_st constraint_function;

} paco_problem_spec_st;

int paco_problem_spec_validate ( paco_problem_spec_st* spec );

/**
 * register a problem spec
 */
void paco_problem_spec_register ( paco_problem_spec_st* spec );

/**
 * configure PACO problem using a named spec
 */
paco_problem_spec_st paco_problem_spec_get ( const char* name );

/**
 * get number of available specifications
 */
index_t paco_problem_spec_num();

/**
 * get the i-th spec name in the list of available specs
 */
const char* paco_problem_spec_name ( index_t i );

/**
 * get the i-th spec description
 */
const char* paco_problem_spec_description ( index_t i );

#endif
