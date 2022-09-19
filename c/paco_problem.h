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
/**
 * \file paco.h
 * \brief common interface to all PACO problems
 * The only function that needs to be declared differently for each implementation is the
 * problem initialization.
 * Any new problem needs to declare the initializar function and define all the
 * ones declared here. The configuration is performed at linking time.
 */
#ifndef PACO_PROBLEM_H
#define PACO_PROBLEM_H

#include <argp.h>                     // argument parsing
#include <gsl/gsl_matrix.h>
#include "paco_mapping.h"
#include "paco_config.h"
#include "paco_init.h"
#include "paco_function.h"
#include "paco_solver.h"
#include "paco_problem_spec.h"
#include "paco_data.h"

//
//----------------------------------------------------------------------------
// MAIN PROJECT PAGE
//----------------------------------------------------------------------------
//
/**
 * \mainpage
 *
 * \section summary Summary
 *
 * This is the C implementation of the PACO-DCT algorithm
 * described originally in [arXiv](https://arxiv.org/abs/1808.06942).
 *
 * \section architecture Overall architecture
 *
 * The code is designed so that different PACO-based problems
 * can be implemented by simply writing a few functions.
 *
 * The overall architecture is based on the 'Strategy' pattern,
 * that is, functions are passed as arguments to a generic implementation, paco_admm,
 * which is always the same. These functions are encapsulated along with
 * their related information into structures; such information is function dependent,
 * and thus they are referenced by generic (void*) poiners within the structure.
 *
 * For each particular problem, the user must provide a 'factory' function
 * which creates and populates the above structures with appropriate implementations
 * and function-specific structures, which are opaque to the user.
 *
 * \section more More information
 *
 * For more information see README.md in the root folder of the project.
 */
//
//----------------------------------------------------------------------------
// END MAIN PROJECT PAGE
//----------------------------------------------------------------------------
//

/**
 *  Stores the variables at each iterate of the ADMM algorithm
 */
typedef struct _paco_iterate_ {
    index_t k; ///< iteration number
    double lambda; ///< penalty parameter for this iteration
    gsl_matrix *A; ///< main variable
    gsl_matrix *B; ///< mirror (split) variable
    gsl_matrix *U; ///< (Scaled) Lagrange multiplier
    gsl_matrix *X; ///< intermediate algorithm output

    double nA; ///< Frobenius norm of A
    double nB; ///< Frobenius norm of B
    double nU; ///< Frobenius norm of U
    double dAB; ///< ||A-B||
    double f; ///< problem cost function
} paco_iterate_st;


typedef struct paco_problem {
    //paco_problem_spec_st spec;
    /* functions */
    paco_data_validate_f validate_data;
    paco_init_f init;
    paco_solver_f solve;
    paco_monitor_f monitor;
    /* functors */
    paco_mapping_st mapping;
    paco_function_st cost_function;
    paco_function_st constraint_function;

    paco_data_st data;
    paco_config_st config;
    paco_iterate_st iter;
} paco_problem_st;

paco_problem_st paco_problem_factory ( paco_problem_spec_st* spec );

void paco_problem_create ( paco_problem_st* problem );

void paco_problem_destroy ( paco_problem_st* problem );

#endif
