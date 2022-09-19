#include <math.h>
#include "paco_linalg.h"
#include "paco_log.h"
#include "paco_io.h"
#include "paco_solver.h"
#include "paco_problem.h"

//=============================================================================
void paco_ladmm ( struct paco_problem* problem )  {

    paco_info("Using Linearized ADMM\n");
    paco_iterate_st* iterate = &problem->iter;
    paco_function_st* cost = &problem->cost_function;
    paco_function_st* constraint = &problem->constraint_function;
    paco_config_st* cfg = &problem->config;
    paco_monitor_f monitor = problem->monitor;

    const index_t n = iterate->B->size1;
    const index_t p = iterate->A->size2;

    double norm_factor = sqrt ( n * p );
    const gsl_matrix *dict = cfg->D;
    iterate->f = cost->eval ( iterate->A );
    iterate->nA = gsl_matrix_frobenius ( iterate->A );
    iterate->nB = gsl_matrix_frobenius ( iterate->B );
    iterate->nU = gsl_matrix_frobenius ( iterate->U );
    iterate->dAB = 1e8; // sqrt(gsl_matrix_dist2_squared(iterate->A,iterate->B)) / norm_factor;
    iterate->k = 0;
    //gsl_matrix_set_all(iterate->A,10.0);
    // for computing dA and dB
    gsl_matrix *auxA = gsl_matrix_calloc ( iterate->A->size1, iterate->A->size2 );
    gsl_matrix *auxB = gsl_matrix_calloc ( iterate->B->size1, iterate->B->size2 );
    gsl_matrix *DA   = gsl_matrix_calloc ( iterate->B->size1, iterate->B->size2 );

    double lambda0 = cfg->admm_penalty;
    //const double kappa = cfg->kappa;
    paco_info ( "iter=%d lambda=%f ||A-B||=%f ||A||=%f ||B||=%f ||U||=%f  J=%f |dA|=N/A |dB|=N/A |df|=N/A\n",
                iterate->k, lambda0, iterate->dAB, iterate->nA, iterate->nB, iterate->nU );
    double rdA,  rdf;
    double dA;
    paco_info ( "Running LINEARIZED ADMM...\n" );
    // cheaper, faster and more intuitive than actual relative change...
    double target = iterate->dAB * norm_factor;
    target = ( 0.5 / lambda0 ) * target * target + iterate->f;
    double rdT;
    int stuck = 0;
    const double nD = gsl_matrix_spectral_norm(cfg->D,1);
    double mu = 0.99 *lambda0 / ( nD * nD ); //   mu <- lambda / ||D||_2^2
    paco_info("lambda %f |D| %f mu %f\n",lambda0,nD,mu);
    gsl_blas_dgemm ( CblasNoTrans, CblasTrans, 1.0, iterate->A, dict, 0.0, DA ); 
    //7const double mn2 = iterate->B->size1 * iterate->B->size2;
    //const double mn = iterate->A->size1 * iterate->A->size2;
    double lambda = lambda0;
    do {
        paco_debug("monitor.\n");
        monitor ( problem );
        //
        // run ADMM iteration
        gsl_matrix_memcpy ( auxA, iterate->A );
        //
        // A(k+1) <- prox_muf{ A(k) - (mu/lambda)Dt[DA(k) - B(k) + U(k)] }
        //
        paco_debug ( "A(k+1) <- prox_muf{ A(k) - (mu/lambda)Dt[DA(k) - B(k) + U(k)] }\n" );

        // auxB <- DA(k) - B(k) + U(k)
        // the actual computation is transposed: AD - B + U
        gsl_matrix_memcpy ( auxB, iterate->U );
        gsl_matrix_sub ( auxB, iterate->B );
        gsl_matrix_add ( auxB, DA);

        // A <- A - (mu/lambda) Dt auxB
        // the actual computation is transposed: auxB D
        gsl_blas_dgemm ( CblasNoTrans, CblasNoTrans, -mu / lambda, auxB, dict, 1.0, iterate->A );

        // A(k+1) <- prox_lambdaf(A)
        cost->prox ( iterate->A, iterate->A, mu );
        dA = sqrt ( gsl_matrix_dist2_squared ( iterate->A, auxA ) );
        gsl_blas_dgemm ( CblasNoTrans, CblasTrans, 1.0, iterate->A, dict, 0.0, DA ); 
        //
        // B(k+1) <- prox_lambdag(DA(k+1) - U(k))
        //
        paco_debug ( "B(k+1) <- prox_lambdag( DA(k+1) + U(k) )\n" );
        gsl_matrix_memcpy ( auxB, iterate->B ); //for difference
        gsl_matrix_memcpy ( iterate->B, DA ); // B <- DA 
        gsl_matrix_sub(iterate->B, iterate->U);
        constraint->prox  ( iterate->B, iterate->B, lambda ); // B(t+1) <- prox_mug(B)
        //
        // U(k+1) <- U(k) + DA(k+1) - Z(k+1)
        //
        paco_debug ( "U(k+1) = U(k) + DA(k+1) - B(K+1)\n" );
        gsl_matrix_memcpy ( auxB, DA ); //for difference
        gsl_matrix_sub    ( auxB, iterate->U ); // ... - B(k+1)
        const double lambda_prev = lambda;
        lambda = lambda0 / (iterate->k+1.0);	
        const double kappa = lambda/lambda_prev;	
        mu *= kappa;
        gsl_matrix_scale(auxB, 1.0/kappa);
        gsl_matrix_sub ( iterate->U, iterate->B ); // ... - B(k+1)

        //
        // compute algorithm evolution
        //
        paco_debug("J(k+1).\n");
        const double prevJ = iterate->f;
        iterate->f = cost->eval ( iterate->A );
        const double dJ = prevJ - iterate->f;

        iterate->nA = gsl_matrix_frobenius ( iterate->A );
        //iterate->nA = norm_factor;
        rdA = iterate->nA > 0 ? dA / iterate->nA : 1.0;

        iterate->nB = gsl_matrix_frobenius ( iterate->B );
        //iterate->nB = norm_factor;
        //rdB = iterate->nB > 0 ? dB / iterate->nB : 1.0;

        iterate->nU = gsl_matrix_frobenius ( iterate->U );
        //iterate->nU = norm_factor;
        //rdU = iterate->nU > 0 ? dU / iterate->nU : 1.0;

        // MORE wasteful: DA(k+1) computed THRICE!
        //iterate->dAB = sqrt ( gsl_matrix_dist2_squared ( DA, iterate->B ) ) / norm_factor;
        rdf = prevJ != 0.0 ? fabs ( dJ ) / fabs ( iterate->f )  : 1.0;
        iterate->k = iterate->k + 1;
        double prev_target = target;
        target = iterate->dAB * norm_factor;
        target = ( 0.5 / lambda ) * target * target + iterate->f;
        rdT = fabs ( prev_target - target ) / ( target + 1e-8 );
        paco_info ( "iter=%d lambda=%f  ||A||=%f ||U||=%f J=%f |dA|=%f |df|=%f |dJ|=%f (target=%f)\n",
                    iterate->k, lambda, iterate->nA, iterate->nU, iterate->f, rdA, rdf, rdT, cfg->min_cost_change );

        //
        // continue until convergence is attained to within specified parameters
        //
        if ( rdA < cfg->min_cost_change ) { // the last one is a hack: the algorithm shouldn't stop if A and B are far away
            stuck++;
        } else {
            stuck = 0;
        }
    } while ( ( iterate->k < cfg->max_iter ) && ( stuck < 3 ) );

    //} while ((iterate->k < cfg->max_iter) && (rdA > cfg->min_arg_change) && (rdf > cfg->min_cost_change) );
    gsl_matrix_free ( auxA );
    gsl_matrix_free ( auxB );
}

//=============================================================================
