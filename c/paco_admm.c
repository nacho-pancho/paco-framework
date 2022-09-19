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
 * \file paco_admm.h
 * \brief ADMM implementation
 */

#include <gsl/gsl_cblas.h>
#include <gsl/gsl_linalg.h>
#include <omp.h>
#include <assert.h>

#include "paco_problem.h"
#include "paco_solver.h"
#include "paco_linalg.h"
#include "paco_log.h"
#include "math.h"

/**
 * update ADMM multiplier
 */
static double paco_update_mult ( gsl_matrix *U, const gsl_matrix *A, const gsl_matrix *B );

//============================================================================

void paco_admm ( paco_problem_st* problem ) {
    paco_iterate_st* iterate = &problem->iter;
    paco_function_st* cost = &problem->cost_function;
    paco_function_st* constraint = &problem->constraint_function;
    paco_config_st* cfg = &problem->config;
    paco_monitor_f monitor = problem->monitor;

    double norm_factor = sqrt ( iterate->A->size1 * iterate->A->size2 );
    iterate->f = cost->eval ( iterate->A );
    iterate->nA = gsl_matrix_frobenius ( iterate->A );
    iterate->nB = gsl_matrix_frobenius ( iterate->B );
    iterate->nU = gsl_matrix_frobenius ( iterate->U );
    iterate->dAB = sqrt ( gsl_matrix_dist2_squared ( iterate->A, iterate->B ) ) / norm_factor;
    iterate->k = 0;
    gsl_matrix *aux = gsl_matrix_alloc ( iterate->A->size1, iterate->A->size2 ); // for computing dA and dB
    double lambda = cfg->admm_penalty;
    double lmul = cfg->kappa;
    paco_info ( "%16s%16s%16s%16s%16s%16s%16s%16s%16s%16s%16s%16s%16s%16s\n",
		"iter","lambda","|A|","|B|","|U|","|A-B|","|dA|","|dB|","|dU|","f","df","dF","dt","target");
    paco_info ( "%16lu%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f%16f\n",
	         iterate->k, lambda, iterate->nA, iterate->nB, iterate->nU, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0);
    double rdA, rdB, rdU, rdf;
    double dA, dB, dU;
    //
    // note: target is the objective function in eq. (7) of the manuscript.
    //
    double target = iterate->dAB * norm_factor;
    target = ( 0.5 / lambda ) * target * target + iterate->f;

    do {
        monitor ( problem );
        //
        // run ADMM iteration
        //
        // A(k+1) <- prox_lambda(B(k) + U(k))
        //
        paco_debug ( "A(k+1) = prox_f( B(k)-U(k) )\n" );
        gsl_matrix_memcpy ( aux, iterate->A );
        gsl_matrix_memcpy ( iterate->A, iterate->B );
        gsl_matrix_sub ( iterate->A, iterate->U );
        cost->prox ( iterate->A, iterate->A, lambda );
        dA = sqrt ( gsl_matrix_dist2_squared ( iterate->A, aux ) );
        //
        // B(k+1) <- prox_lambda(A(k+1) - U(k))
        //
        paco_debug ( "B(k+1) = prox_g( A(k)+U(k) )\n" );
        gsl_matrix_memcpy ( aux, iterate->B );
        gsl_matrix_memcpy ( iterate->B, iterate->A );
        gsl_matrix_add ( iterate->B, iterate->U );
        constraint->prox ( iterate->B, iterate->B, lambda );
        dB = sqrt ( gsl_matrix_dist2_squared ( iterate->B, aux ) );
        //
        // U(k+1) <- U(k) + A(k+1) - B(k+1)
        //
        paco_debug ( "U(k+1) = U(k) + A(k+1) - B(K+1)\n" );
        dU = paco_update_mult ( iterate->U, iterate->A, iterate->B );
        //
        // compute algorithm evolution
        //
        const double prevf = iterate->f;
        iterate->f = cost->eval ( iterate->B );
        const double df = prevf - iterate->f;

        iterate->nA = gsl_matrix_frobenius ( iterate->A );
        //iterate->nA = norm_factor;
        rdA = iterate->nA > 0 ? dA / iterate->nA : 1.0;

        iterate->nB = gsl_matrix_frobenius ( iterate->B );
        //iterate->nB = norm_factor;
        rdB = iterate->nB > 0 ? dB / iterate->nB : 1.0;

        iterate->nU = gsl_matrix_frobenius ( iterate->U );
        //iterate->nU = norm_factor;
        rdU = iterate->nU > 0 ? dU / iterate->nU : 1.0;

        iterate->dAB = sqrt ( gsl_matrix_dist2_squared ( iterate->A, iterate->B ) );
        rdf = iterate->f != 0.0 ? fabs ( df ) / fabs ( iterate->f )  : 1.0;
        iterate->k = iterate->k + 1;
        double prev_target = target;
        double penalty =  ( 0.5 / lambda ) *  iterate->dAB * iterate->dAB;
        target = penalty + iterate->f;
        double rdT = target != 0.0 ? ( prev_target - target ) / fabs ( target ) : 1.0;
        paco_info ( "%16lu%16.4f%16.4f%16.4f%16.4f%16.4f%16.8f%16.8f%16.8f%16.4f%16.8f%16.8f%16.2f%16.8f\n",
                    iterate->k, lambda, iterate->nA, iterate->nB, iterate->nU, penalty, rdA, rdB, rdU, iterate->f, rdf, rdT, 0.0, cfg->min_cost_change );

        //
        // continue until convergence is attained to within specified parameters
        //
        if ( fabs(rdT) < cfg->min_cost_change ) { // the last one is a hack: the algorithm shouldn't stop if A and B are far away
            break;
        }
	// non-standard ADMM hack: if augmented Lagrangian increases, reduce "stepsize" lambda
        if (rdT < 0) { // function value went up
	    lambda *= lmul;
	}
    } while ( iterate->k < cfg->max_iter );
    //
    // save final output
    //
    problem->mapping.stitch ( iterate->X, iterate->B );
    //
    // cleanup
    //
    gsl_matrix_free ( aux );
}

//============================================================================

static double paco_update_mult ( gsl_matrix *U, const gsl_matrix *A, const gsl_matrix *B ) {
    assert ( U->size1 == A->size1 );
    assert ( U->size2 == A->size2 );
    assert ( U->tda == A->tda );
    assert ( U->size1 == B->size1 );
    assert ( U->size2 == B->size2 );
    assert ( U->tda == B->tda );

    const index_t tda = U->tda;
    const sample_t *pA = A->data;
    const sample_t *pB = B->data;
    sample_t *pU = U->data;
    double a = 0.0;

    for ( index_t i = 0; i < U->size1; i++ ) {
        for ( index_t j = 0; j < U->size2; j++ ) {
            const double d = pA[j] - pB[j];
            pU[j] += d;
            a += d * d;
        }

        pA += tda;
        pB += tda;
        pU += tda;
    }

    return sqrt ( a );
}
