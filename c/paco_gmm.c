#include <assert.h>
#include <math.h>
#include <string.h>
#include <float.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>

#include "paco_problem.h"
#include "paco_log.h"
#include "paco_io.h"
#include "paco_util.h"
#include "paco_gmm.h"

#define MAX_CLUSTERS 128

//============================================================================

paco_function_st paco_gmm() {
    paco_function_st preset;
    preset.create = paco_gmm_create;
    preset.fit = paco_gmm_fit;
    preset.eval   = paco_gmm_eval;
    preset.prox   = paco_gmm_prox;
    preset.destroy = paco_gmm_destroy;
    strcpy ( preset.name, "gmm" );
    return preset;
}

//============================================================================

typedef struct _gaussian_model_ {
    gsl_vector *mu;
    gsl_matrix *Sigma;
    gsl_vector *D; // singular values of Sigma
    gsl_matrix *Ut; // U in Sigma = UDUt
    index_t effective_rank;
} gaussian_mode_t;

//============================================================================


//============================================================================
// MODULE  VARIABLES
//============================================================================

#define GMM_MODE_L0 0
#define GMM_MODE_L1 1
#define GMM_MODE_L2 2

static gaussian_mode_t modes[MAX_CLUSTERS];
static index_t nmodes;
static double sv_thres = 0;
static index_t sv_order = 10000000;
static index_t kmeans_iter = 16;
static index_t em_iter = 0; // E-M is DISABLED by default
static index_t inference_mode = 2;
static gsl_vector *candidates[MAX_CLUSTERS];
static gsl_vector *cluster_assignement = NULL;
static gsl_vector* ones; // auxiliary all-ones vector
// for denoising
static unsigned char remove_dc = 0;
static const char* workdir = ".";

//============================================================================
// FUNcTIONS
//============================================================================

void paco_gmm_create ( const paco_problem_st* problem ) {
    assert ( problem->mapping.patch_dim > 0 );
    const index_t m = problem->mapping.patch_dim;
    nmodes = problem->config.model_order;
    assert ( nmodes > 0 );
    paco_info ( "Createing GMM model with %d modes\n", nmodes );
    workdir      =problem->config.work_dir;    
    sv_thres    = problem->config.gmm_sv_thres;
    sv_order    = problem->config.gmm_sv_order;
    kmeans_iter = problem->config.gmm_kmeans_iter;
    em_iter     = problem->config.gmm_em_iter;    
    inference_mode = problem->config.gmm_mode;
    remove_dc = problem->config.remove_dc;
    for ( index_t k = 0; k < nmodes; k++ ) {
        gaussian_mode_t *modek = &modes[k];
        modek->mu = gsl_vector_calloc ( m );
        modek->D = gsl_vector_calloc ( m );
        modek->Ut = gsl_matrix_calloc ( m, m );
        modek->Sigma = gsl_matrix_calloc ( m, m );
        //
        // initialize all models with a diagonal
        //
        for ( index_t i = 0; i < m; i++ ) {
            //gsl_matrix_set(modek->Sigma,i,i,1s*s);
            //gsl_vector_set(modek->D,i,s);
            gsl_matrix_set ( modek->Sigma, i, i, 1 );
            gsl_vector_set ( modek->D, i, 1 );
            gsl_matrix_set ( modek->Ut, i, i, 1 );
        }
        //
        // allocate candidates
        //
        candidates[k] = gsl_vector_calloc ( m );
    }
    ones = gsl_vector_alloc ( m );
    gsl_vector_set_all ( ones, 1 );
}

//============================================================================

void paco_gmm_destroy() {
    for ( index_t k = 0; k < nmodes; k++ ) {
        gaussian_mode_t *mode = &modes[k];
        gsl_matrix_free ( mode->Ut );
        mode->Ut = 0;
        gsl_vector_free ( mode->D );
        mode->D = 0;
        gsl_vector_free ( mode->mu );
        mode->mu = 0;
        gsl_vector_free ( candidates[k] );
        candidates[k] = 0;
    }

    if ( cluster_assignement ) {
        gsl_vector_free ( cluster_assignement );
        cluster_assignement = NULL;
    }
    gsl_vector_free ( ones );
}


//============================================================================

static void paco_gmm_reconstruct_sigma ( gaussian_mode_t *mode ) {
    const index_t m = mode->D->size;
    gsl_matrix *aux = gsl_matrix_alloc ( m, m );
    gsl_matrix_memcpy ( aux, mode->Ut );

    for ( index_t i = 0; i < m; i++ ) {
        const double li = gsl_vector_get ( mode->D, i );
        gsl_vector_view auxiview = gsl_matrix_row ( aux, i );
        gsl_vector *auxi = &auxiview.vector;
        gsl_vector_scale ( auxi, li );
    }

    // Sigma = aux*Ut = (UD)*Ut
    if ( mode->Sigma == NULL ) {
        mode->Sigma = gsl_matrix_alloc ( m, m );
    }

    gsl_matrix_set_zero ( mode->Sigma );
    gsl_blas_dgemm ( CblasTrans, CblasNoTrans, 1.0, aux, mode->Ut, 0.0, mode->Sigma );
    gsl_matrix_free ( aux );
}


//============================================================================

static double eval_mode ( const gaussian_mode_t *mode, const gsl_vector *x ) {
    //paco_debug("GMM eval mode \n ");
    const index_t m = x->size;
    gsl_vector *z = gsl_vector_alloc ( m );
    //
    // a <- Ut(x-mu)
    //
    gsl_vector_memcpy ( z, x );
    gsl_vector_sub ( z, mode->mu );
    if ( remove_dc ) {
        // if DC is not taken into account:
        // a <- Ut(x - <x>1 - mu) = Utx - <x>Ut1 - Utmu
        const sample_t xdc = gsl_vector_mean ( x );
        gsl_blas_daxpy ( -xdc, ones, z );
    }
    //
    // cost_k = log |D| + (x-mu)*U*Di*Ut(x-mu)
    //        = log |D| + \sum_i |a_i|/sqrt(D_i)
    //
    sample_t *pD = mode->D->data;
    const sample_t svt = pD[0] * sv_thres;
    const index_t  svo = sv_order;
    double cost = 0;

    for ( index_t i = 0; i < m; i++ ) {
        const double d_i = pD[i];
        if ( i > svo ) {
            break;
        }
        if ( d_i < svt ) {
            break;
        }
        sample_t a_i;
        gsl_vector_const_view U_i_view = gsl_matrix_const_row ( mode->Ut, i );
        const gsl_vector* U_i = &U_i_view.vector;
        gsl_blas_ddot ( U_i, z, &a_i );
        if ( a_i == 0 ) {
            continue;
        }
        switch ( inference_mode ) {
        default:
        case GMM_MODE_L0:
            cost += log ( d_i ) + 1;
            break;
        case GMM_MODE_L1:
            cost += log ( d_i ) + fabs ( a_i );
            break;
        case GMM_MODE_L2:
            cost += log ( d_i ) + a_i * a_i / d_i;
            break;
        }
    }
    gsl_vector_free ( z );
    return cost;
}

//============================================================================
///
/// \todo: SOLVE FOR BEST ORDER WITHIN MODE USING L1 prior on coefs
///
/**
 * solve the proximal operator of a vector y where the target function is the likelihood
 * under a given mode k, and tau is the proximal parameter.
 * x = prox_{tau*f_k}(y)
 */
static double gmm_infer ( const gaussian_mode_t *mode, const sample_t tau, const gsl_vector *y, gsl_vector *x ) {
    const gsl_matrix *Ut = mode->Ut;
    const gsl_vector *D = mode->D;
    const index_t m = y->size;
    gsl_vector *v = gsl_vector_calloc ( m );
    //
    // compute proximal onto mode k using L1 prior on coefficients
    //
    // x* = mu + Ua*,
    // a* = arg min_p \sum_i^p a_i^2/d_i + (1/tau)||b-a||^2_2, b = U^t(y-mu)
    //
    // if DC is removed:
    // x* = <y> + mu + Ua*,
    // a* = arg min_p \sum_i^p a_i^2/d_i + (1/tau)||b-a||^2_2,  b = U^t(y-<y>-mu)
    //
    gsl_vector_memcpy ( v, y );                    // v <- y
    gsl_vector_sub ( v, mode->mu );                // v <- y - mu
    sample_t ydc = 0;
    if ( remove_dc ) {
        ydc = gsl_vector_mean ( y );
        gsl_blas_daxpy ( -ydc, ones, v );                  // v <- y - mu - <y>1
    }

    gsl_vector_memcpy ( x, mode->mu );             // x <- mu
    if ( remove_dc ) {
        gsl_blas_daxpy ( ydc, ones, x );                   // x <- mu + <y>
    }
    //
    // x <- mu + <y> + sum_i a_i U_i
    //
    const sample_t *pD = D->data;
    const sample_t svt = pD[0] * sv_thres;
    const index_t  svo = sv_order;
    double cost = 0;
    for ( index_t i = 0; i < m; i++ ) {
        const double d_i = pD[i];
        if ( i > svo ) {
            break;
        }
        if ( d_i < svt ) {
            break;
        }
        gsl_vector_const_view U_i_view = gsl_matrix_const_row ( Ut, i );
        const gsl_vector* U_i = &U_i_view.vector;
        // note U_i is tue i-th row of Ut, that is, the i-th *column* of U, and that's ok
        sample_t a_i, b_i;
        gsl_blas_ddot ( U_i, v, &b_i );                   // b_i = (U^t v)_i = <U_i,v>
        const sample_t t_i = tau / sqrt ( d_i );
        switch ( inference_mode ) {
        default:
            paco_error ( "GMM: Unknown mode!! Defaulting to L2." );
        //break;
        case GMM_MODE_L0:
            //paco_error("GMM: inference mode L0 not implemented!! Defaulting to L2.");
            a_i = b_i > t_i ? b_i : ( b_i < -t_i ? b_i : 0 );
            break;
        case GMM_MODE_L1:
            //paco_error("GMM: inference mode L1 not implemented!! Defaulting to L2.");
            a_i = b_i > t_i ? b_i - t_i : ( b_i < -t_i ? b_i + t_i : 0 );
            break;
        case GMM_MODE_L2:
            a_i = d_i / ( tau + d_i ) * b_i;   // (self explaining)
            break;
        }
        if ( fabs ( a_i ) > 0 ) {
            gsl_blas_daxpy ( a_i, U_i, x );      // x <- mu + <y> + sum_i a_i U_i
            cost += log ( d_i ) + a_i * a_i / d_i; // fabs ( a_i ) / sqrt ( d_i );
        }
    }
    gsl_vector_free ( v );
    return cost;
}


//============================================================================

double paco_gmm_eval ( const gsl_matrix *X ) {
    const index_t n = X->size1;
    const index_t m = X->size2;

    if ( cluster_assignement == NULL ) {
        cluster_assignement = gsl_vector_calloc ( n );
    }

    double total_cost = 0;

    for ( index_t j = 0; j < n; j++ ) {
        gsl_vector_const_view Xjview = gsl_matrix_const_row ( X, j );
        double best_cost = eval_mode ( &modes[0], &Xjview.vector );
        index_t bestk = 0;
        for ( index_t k = 1; k < nmodes; k++ ) {
            double cost = eval_mode ( &modes[k], &Xjview.vector );

            if ( cost < best_cost ) {
                best_cost = cost;
                bestk = k;
            }
        }
        gsl_vector_set ( cluster_assignement, j, bestk );

        total_cost +=  best_cost;
    }

    return total_cost / ( (double) m * (double) n );
}

//============================================================================

const gsl_vector *get_cluster_assignement () {
    return cluster_assignement;
}

//============================================================================

void paco_gmm_prox ( gsl_matrix *PA, const gsl_matrix *A, const double tau ) {
    const index_t n = A->size1;
    for ( index_t j = 0; j < n; j++ ) {
        double best_cost = GSL_FLT_MAX;
        index_t best_mode = 0;
        gsl_vector_const_view Ajview = gsl_matrix_const_row ( A, j );
        gsl_vector_view PAjview = gsl_matrix_row ( PA, j );
        for ( index_t k = 0; k < nmodes; k++ ) {
            //const double cost = eval_mode ( &modes[k], &Ajview.vector );
            const double cost = gmm_infer ( &modes[k], tau, &Ajview.vector, candidates[k] );
            if ( cost < best_cost ) {
                best_cost = cost;
                best_mode = k;
            }
        }
        gsl_vector_memcpy ( &PAjview.vector, candidates[best_mode] );
    }
}

//============================================================================

/**
 * Setup model from raw model data. For a p-components mixture
 * over a space of dimension m, the matrix model_data is
 * an (m+2)p x m matrix. Each band of size m+2 contains:
 * first row: model mean, mu_k
 * second row: singular values of Sigma_k, a vector lambda_k
 * third onward: orthogonal diagonalization matrix U_k so that Sigma_k = U_k diag(lambda_k) U_k^t
 */
void paco_gmm_set_model ( gsl_matrix *model_data ) {
    const index_t m = model_data->size2;
    const index_t p = model_data->size1 / ( m + 2 );
    //
    // destroy previous model, if any
    //
    paco_gmm_destroy();
    //
    //
    //
    nmodes = p;
    for ( index_t k = 0; k < nmodes; k++ ) {
        gsl_vector_view vview;
        gsl_matrix_view mview;
        gaussian_mode_t *modek = &modes[k];
        modek->mu = gsl_vector_alloc ( m );
        vview = gsl_matrix_row ( model_data, k * ( m + 2 ) );
        gsl_vector_memcpy ( modek->mu, &vview.vector );

        modek->D = gsl_vector_alloc ( m );
        vview = gsl_matrix_row ( model_data, k * ( m + 2 ) + 1 );
        gsl_vector_memcpy ( modek->D, &vview.vector );

        modek->Ut = gsl_matrix_alloc ( m, m );
        mview = gsl_matrix_submatrix ( model_data, k * ( m + 2 ) + 2, 0, m, m );
        gsl_matrix_memcpy ( modek->Ut, &mview.matrix );

        modek->Sigma = gsl_matrix_alloc ( m, m );
        gsl_matrix_set_zero ( modek->Sigma );
        paco_gmm_reconstruct_sigma ( &modes[k] );
        //
        // compute log det(Sigma) and record largest significant dimension
        // according to relative and absolute thresholds
        //
        //paco_info("Largest eigenvalue %f eigenvalue threshold %f\n",largest_eigval, relative_threshold);
        modek->effective_rank = m;
        const double svt = sv_thres * gsl_vector_get ( modek->D, 0 );
        index_t j = 0;

        for ( j = 0; j < m; j++ ) {
            const sample_t Lj = gsl_vector_get ( modek->D, j );

            if ( Lj < svt ) {
                modek->effective_rank = j;
                break;
            }
        }
    }

    //
    // initialize auxiliary variables
    //
    for ( index_t k = 0; k < nmodes; k++ ) {
        candidates[k] =  gsl_vector_alloc ( m );
    }

}

//============================================================================

gsl_matrix  *paco_gmm_get_model ( gsl_matrix **pmodel_data ) {
    const index_t m = modes[0].mu->size;
    const index_t p = nmodes;
    gsl_matrix *model_data;

    if ( pmodel_data == 0 ) {
        model_data = gsl_matrix_alloc ( p * ( m + 2 ), m );
    } else {
        model_data = *pmodel_data;
    }

    for ( index_t k = 0; k < nmodes; k++ ) {
        gaussian_mode_t *modek = &modes[k];
        gsl_vector_view vview;
        gsl_matrix_view mview;

        vview = gsl_matrix_row ( model_data, k * ( m + 2 ) );
        gsl_vector_memcpy ( &vview.vector, modek->mu );

        vview = gsl_matrix_row ( model_data, k * ( m + 2 ) + 1 );
        gsl_vector_memcpy ( &vview.vector, modek->D );

        mview = gsl_matrix_submatrix ( model_data, k * ( m + 2 ) + 2, 0, m, m );
        gsl_matrix_memcpy ( &mview.matrix, modek->Ut );
    }

    return model_data;
}



//============================================================================
// ADAPTATION STUFF
//============================================================================

typedef struct _gaussian_stats_ {
    gsl_vector *sumX;
    gsl_matrix *sumXtX;
    index_t counts;
} gaussian_stats_t;

//============================================================================

static sample_t adapt_rate = 1;
static index_t batch_size = 20000;
static index_t max_iter = 10; // DEBUG
static gaussian_stats_t *stats[MAX_CLUSTERS];

//============================================================================

void paco_gmm_set_adapt_rate ( const sample_t rate ) {
    adapt_rate = rate;
}

//============================================================================

void paco_gmm_set_batch_size ( const index_t bs ) {
    batch_size = bs;
}

//============================================================================

void paco_gmm_set_adapt_max_iter ( const index_t mi ) {
    max_iter = mi;
}

//============================================================================

static gaussian_stats_t  *paco_gmm_stats_alloc ( const index_t nb, const index_t m ) {
    gaussian_stats_t *stats = ( gaussian_stats_t * ) malloc ( sizeof ( gaussian_stats_t ) );
    stats->sumX = gsl_vector_alloc ( m );
    stats->sumXtX = gsl_matrix_alloc ( m, m );
    gsl_vector_set_zero ( stats->sumX );
    gsl_matrix_set_all ( stats->sumXtX, sv_thres );
    return stats;
}

//============================================================================

static void paco_gmm_stats_reset ( gaussian_stats_t *stats ) {
    gsl_vector_set_zero ( stats->sumX );
    gsl_matrix_set_all ( stats->sumXtX, 1.0f );
    stats->counts = 1.0f;
}

//============================================================================

static void paco_gmm_stats_free ( gaussian_stats_t *stats ) {
    gsl_vector_free ( stats->sumX );
    gsl_matrix_free ( stats->sumXtX );
    stats->sumX = 0;
    stats->sumXtX = 0;
}

//============================================================================

static double paco_kmeans_expectation_step ( const gsl_matrix *X ) {
    assert ( nmodes > 0 );
    const index_t n = X->size1;
    const index_t m = X->size2;
    double dist[MAX_CLUSTERS];
    double min_prop = 1e-3;
    double total_loglik = 0;
    paco_debug ( "K-MEANS: EXP." );

    for ( index_t k = 0; k < nmodes; k++ ) {
        paco_gmm_stats_reset ( stats[k] );
        stats[k]->counts = 0;
    }

    for ( index_t j = 0; j < n; j++ ) {
        gsl_vector_const_view Xjview = gsl_matrix_const_row ( X, j );
        const gsl_vector *Xj = &Xjview.vector;
        // assign weight of this sample to each cluster
        dist[0] = l2dist ( modes[0].mu, Xj );
        double best_dist = dist[0];
        int best_k = 0;
        for ( index_t k = 1; k < nmodes; k++ ) {
            dist[k] = l2dist ( modes[k].mu, Xj );
            // best is minimum
            if ( best_dist > dist[k] ) {
                best_dist = dist[k];
                best_k = k;
            }
        }
        stats[best_k]->counts++;
        gsl_vector_set ( cluster_assignement, j, best_k );
        total_loglik += best_dist;
    }
    paco_debug ( "cluster counts:" );
    gsl_rng* rng = gsl_rng_alloc ( gsl_rng_taus );
    gsl_rng_set ( rng, gsl_rng_default_seed );
     for ( index_t k = 0; k < nmodes; k++ ) {
	const double prop = (double)stats[k]->counts / (double)n ;
        paco_debug ( "%7.4%% ", 100.0* prop );
	if (prop < min_prop) {
	    // no one loves this one: reassign it to another random sample
            index_t ri = ( index_t ) ( gsl_rng_uniform ( rng ) * ( double ) n );
	    paco_info("RESET\n");
            for ( index_t i = 0; i < m; i++ ) {
                gsl_vector_set ( stats[k]->sumX, i, gsl_matrix_get ( X, ri, i ) );
            }
	    stats[k]->counts = 1;
        }
    }
    gsl_rng_free ( rng );
    paco_debug ( " cost=%f\n", total_loglik / ( double ) n );
    return total_loglik / ( double ) n;
}

//============================================================================

static void paco_kmeans_maximization_step ( const gsl_matrix *X ) {
    const index_t n = X->size1;
    //
    // compute batch means
    //
    paco_debug ( "K-MEANS: MAX.\n " );

    for ( index_t j = 0; j < n; j++ ) {
        gsl_vector_const_view Xjview = gsl_matrix_const_row ( X, j );
        const int cj = ( int ) gsl_vector_get ( cluster_assignement, j );
        gsl_vector_add ( stats[cj]->sumX, &Xjview.vector );
    }

    //
    // update mode mean
    //
    for ( index_t k = 0; k < nmodes; k++ ) {
        gaussian_stats_t *statsk = stats[k];

        if ( statsk->counts == 0 )
            continue;

        const double normk = adapt_rate / ( double ) statsk->counts;
        gsl_vector *sumXk = statsk->sumX;
        gaussian_mode_t *modek = &modes[k];
        //
        // new mean = (1.0-adapt_rate)*old_mu + adapt_rate*new_mu
        //
        gsl_vector_scale ( modek->mu, 1.0 - adapt_rate );
        gsl_blas_daxpy ( normk, sumXk, modek->mu );
    }
}

//============================================================================


static void paco_kmeans_adjust_covariances ( const gsl_matrix *X ) {
    const index_t n = X->size1;
    const index_t m = X->size2;
    gsl_vector *aux = gsl_vector_alloc ( m );

    for ( index_t k = 0; k < nmodes; k++ ) {
        gaussian_stats_t *statsk = stats[k];
        gaussian_mode_t *modek = &modes[k];

        if ( statsk->counts == 0 )
            continue;

        const double normk = 1.0 / ( double ) statsk->counts;

        for ( index_t j = 0; j < n; j++ ) {
            gsl_vector_const_view Xjview = gsl_matrix_const_row ( X, j );
            const gsl_vector *Xj = &Xjview.vector;
            const int clusterj = ( int ) gsl_vector_get ( cluster_assignement, j );

            if ( k == clusterj ) {
                gsl_vector_memcpy ( aux, Xj );
                gsl_vector_sub ( aux, modek->mu );
                gsl_blas_dger ( normk, aux, aux, statsk->sumXtX );
            }
        }
        //
        // sumXtX <- U, Ut <- V, d <- S
        //
        gsl_linalg_SV_decomp ( statsk->sumXtX, modek->Ut, modek->D, aux );
        gsl_matrix_transpose_memcpy ( modek->Ut, statsk->sumXtX ); // Ut <- sumXtX^t <- U^t
        //
        // we now scale the singular values so that the smallest one is 1.
        // this is to maintain a similar scale w.r.t. other prior models such as L1,
        // whose largest weight is 1
        //
        // in this case we the elements d_i of D are the inverse weights,
        // so that the largest weight w_max
        // will correspond to the smaller coefficient, w_max = 1/d_min.
        //
        // we now want to scale the coefficients so that w_max = 1, so that d*_min = 1
        // thus we need to multiply D by K=1/d_min
        //
        // here we have to be careful because d_min may be very small, which may
        // result in larger values of D being too large
        //
        // thus, we use sv_thres as the minimum divisor: K = 1/max{sv_thres,d_min}
        //
        // threshold singular values, compute logdet
        //
        const sample_t d_max = gsl_vector_get ( modek->D, 0 );
        const sample_t d_min = gsl_vector_get ( modek->D, m - 1 );
        const sample_t K = d_min > ( sv_thres * d_max ) ? d_min : ( sv_thres * d_max );
        paco_info ( "k %lu d_maxk %f d_min %f  K %f\n", k, d_max, d_min, K );
        for ( index_t j = 0; j < m; j++ ) {
            gsl_vector_set ( modek->D, j, gsl_vector_get ( modek->D, j ) / K );
        }

        //
        // re-compute Sigma from possibly thresholded singular values
        //
        paco_gmm_reconstruct_sigma ( modek );
    }

    gsl_vector_free ( aux );
}

//============================================================================



static double paco_gmm_expectation_step ( const gsl_matrix *X ) {
    assert ( nmodes > 0 );
    const index_t n = X->size1;
    double loglik[MAX_CLUSTERS];

    double total_loglik = 0;

    paco_debug ( "E-M: EXP.\n" );
    for ( index_t j = 0; j < n; j++ ) {
        gsl_vector_const_view Xjview = gsl_matrix_const_row ( X, j );
        const gsl_vector *Xj = &Xjview.vector;
        // assign weight of this sample to each cluster
        loglik[0] = eval_mode ( &modes[0], Xj );
        double best_loglik = loglik[0];
        int best_k = 0;
        for ( index_t k = 1; k < nmodes; k++ ) {
            loglik[k] = eval_mode ( &modes[k], Xj );
            // best is minimum
            if ( best_loglik > loglik[k] ) {
                best_loglik = loglik[k];
                best_k = k;
            }
        }
        gsl_vector_set ( cluster_assignement, j, best_k );
        total_loglik += best_loglik;
    }

    return total_loglik / ( double ) n;
}

//============================================================================

static void paco_gmm_maximization_step ( const gsl_matrix *X ) {
    const index_t n = X->size1;
    const index_t m = X->size2;
    gsl_vector *aux = gsl_vector_alloc ( m );

    //
    // compute batch means
    //
    paco_debug ( "E-M: MAX: cluster counts:" );

    for ( index_t k = 0; k < nmodes; k++ ) {
        gaussian_stats_t *statsk = stats[k];
        paco_gmm_stats_reset ( statsk );
        statsk->counts = 0;
        gsl_vector *sumXk = statsk->sumX;

        for ( index_t j = 0; j < n; j++ ) {
            gsl_vector_const_view Xjview = gsl_matrix_const_row ( X, j );
            int clusterj = ( int ) gsl_vector_get ( cluster_assignement, j );

            if ( k == clusterj ) {
                statsk->counts++;
                gsl_vector_add ( sumXk, &Xjview.vector );
            }
        }

        paco_debug ( "%lu%% ", 100 * statsk->counts / n );
    }

    paco_debug ( "\n" );

    //
    // update mode mean
    //
    for ( index_t k = 0; k < nmodes; k++ ) {
        gaussian_stats_t *statsk = stats[k];

        if ( statsk->counts == 0 )
            continue;

        const double normk = adapt_rate / ( double ) statsk->counts;
        gsl_vector *sumXk = statsk->sumX;
        gaussian_mode_t *modek = &modes[k];
        //
        // new mean = (1.0-adapt_rate)*old_mu + adapt_rate*new_mu
        //
        gsl_vector_scale ( modek->mu, 1.0 - adapt_rate );
        gsl_blas_daxpy ( normk, sumXk, modek->mu );
    }

    //
    // update covariance matrix
    //
    for ( index_t k = 0; k < nmodes; k++ ) {
        gaussian_stats_t *statsk = stats[k];
        gaussian_mode_t *modek = &modes[k];

        if ( statsk->counts == 0 )
            continue;

        const double normk = 1.0 / ( double ) statsk->counts;

        for ( index_t j = 0; j < n; j++ ) {
            gsl_vector_const_view Xjview = gsl_matrix_const_row ( X, j );
            const gsl_vector *Xj = &Xjview.vector;
            const int clusterj = ( int ) gsl_vector_get ( cluster_assignement, j );

            if ( k == clusterj ) {
                gsl_vector_memcpy ( aux, Xj );
                gsl_vector_sub ( aux, modek->mu );
                gsl_blas_dger ( normk, aux, aux, statsk->sumXtX );
            }
        }

        gsl_vector_free ( aux );
        //
        // recompute SVD from XtX
        //
        gsl_vector *work = gsl_vector_alloc ( m );
        gsl_vector *S = gsl_vector_alloc ( m );
        gsl_matrix *U = gsl_matrix_alloc ( m, m );
        gsl_matrix *V = gsl_matrix_alloc ( m, m );

        //
        // U <- (1-a)*Sigma + a*XtX
        //
        for ( index_t i = 0; i < m; i++ ) {
            for ( index_t j = 0; j < m; j++ ) {
                const double vold = gsl_matrix_get ( modek->Sigma, i, j );
                const double vnew = gsl_matrix_get ( statsk->sumXtX, i, j );
                const double v = ( 1.0 - adapt_rate ) * vold + adapt_rate * vnew;
                gsl_matrix_set ( U, i, j, v );
            }
        }

        // USVt = XtX
        gsl_linalg_SV_decomp ( U, V, S, work );

        // D <- S
        // Ut <- transp(U)
        // additional term due to noise
        //const sample_t s2 = adapt_rate*noise_sigma*noise_sigma;
        for ( index_t i = 0; i < m; i++ ) {
            const double Sii = gsl_vector_get ( S, i );
            //gsl_vector_set(modek->D,i,Sii > s2 ? Sii-s2: 0);
            gsl_vector_set ( modek->D, i, Sii );

            for ( index_t j = 0; j < m; j++ ) {
                gsl_matrix_set ( modek->Ut, j, i, gsl_matrix_get ( U, i, j ) );
            }
        }
	//
	// rescale weights
	//
        const sample_t d_max = gsl_vector_get ( modek->D, 0 );
        const sample_t d_min = gsl_vector_get ( modek->D, m - 1 );
        const sample_t K = d_min > ( sv_thres * d_max ) ? d_min : ( sv_thres * d_max );
        paco_info ( "k %lu d_maxk %f d_min %f  K %f\n", k, d_max, d_min, K );
        for ( index_t j = 0; j < m; j++ ) {
            gsl_vector_set ( modek->D, j, gsl_vector_get ( modek->D, j ) / K );
        }
        //
        // re-compute Sigma from possibly thresholded singular values
        //
        paco_gmm_reconstruct_sigma ( modek );
    }
}

static void save_cluster_map ( const gsl_vector* c, const index_t w, const index_t h,
                               const gsl_matrix* mask, const int iter, paco_image_st* auximg ) {
    char aux[128];
    gsl_matrix* I = get_channel_samples ( auximg, 0 );
    const index_t mg = I->size1 - h + 1;
    const index_t ng = I->size2 - w  + 1;
    for ( index_t i = 0, li = 0; i < mg; i++ ) {
        for ( index_t j = 0; j < ng; j++, li++ ) {
            if ( !mask || !mask->data[mask->tda * i + j] ) {
                I->data[i * I->tda + j] = c->data[li];
            }
        }
    }
    if ( iter >= 0 ) {
        snprintf ( aux, 127, "%s/gmm_cluster_map_iter_%05d.png", workdir, iter );
    } else {
        snprintf ( aux, 127, "%s/kmeans_cluster_map_iter_%05d.png",workdir, -iter );
    }
    paco_image_norm ( auximg );
    write_png_file ( aux, auximg );
}

//============================================================================

void paco_gmm_fit ( const struct paco_problem* problem ) {

    const gsl_matrix *mask = problem->data.mask;
    const gsl_matrix* input = problem->data.input;
    const gsl_matrix* initial = problem->data.initial;
    const paco_config_st* cfg = &problem->config;
    const index_t w = cfg->patch_width;
    paco_image_st* auximg = paco_image_alloc ( input->size1, input->size2, COLORSPACE_GRAY );
    gsl_matrix* auxpixels = get_channel_samples ( auximg, 0 );
    gsl_matrix_set_zero ( auxpixels );
    gsl_matrix* X;
    gsl_vector* dc = NULL;
    if ( initial ) {
        X = paco_mapping_extract_all ( initial, w, w, NULL );
    } else {
        X = paco_mapping_extract_complete ( input, mask, w, w, NULL );
    }
    if ( problem->config.remove_dc ) {
        dc = gsl_vector_alloc ( X->size1 );
        paco_mapping_remove_dc ( X, dc );
        gsl_vector_free ( dc );
    }
    gsl_matrix *raw_model_data = paco_gmm_get_model ( NULL );
    gsl_matrix *mosaic_samples = NULL;

    paco_debug ( "Adapting model to %ld input patches\n", X->size1 );
    const index_t n = X->size1;
    const index_t m = X->size2;

    //cluster_assignement = gsl_vector_alloc ( batch_size );
    cluster_assignement = gsl_vector_alloc ( n );
    gsl_vector_set_zero ( cluster_assignement );
    //
    // initialize learning-related variables and structures
    //
    gsl_rng* rng = gsl_rng_alloc ( gsl_rng_taus );
    gsl_rng_set ( rng, gsl_rng_default_seed );
    for ( index_t k = 0; k < nmodes; k++ ) {
        stats[k] = paco_gmm_stats_alloc ( n, m );
        gaussian_mode_t *modek = &modes[k];
        index_t ri = ( index_t ) ( gsl_rng_uniform ( rng ) * ( double ) n );
        for ( index_t i = 0; i < m; i++ ) {
            //      gsl_vector_set(modek->mu,i,gsl_rng_uniform(rng));
            gsl_vector_set ( modek->mu, i, gsl_matrix_get ( X, ri, i ) );
        }
        modek->effective_rank = m;
    }
    gsl_rng_free ( rng );
    //
    // K-Means initialization of centroids
    //
    paco_info("K-MEANS for %ld iterations\n",kmeans_iter);
    double J = 0;
    for ( index_t iter = 0; iter < kmeans_iter; iter++ ) {
        double prevJ = J;
        J = paco_kmeans_expectation_step ( X );
        paco_kmeans_maximization_step ( X );
        save_cluster_map ( cluster_assignement, w, w, mask, -iter, auximg );
        double dJ = ( J - prevJ ) / fabs ( J );
        if ( fabs ( dJ ) < 1e-3 ) {
            paco_debug ( "K-MEANS converged after %d iterations\n", iter );
            break;
        }
    }

    //
    // regardless of a full GMM E-M optimization,
    // we learn covariance matrices for each K-Means cluster
    // this is a sort of half-baked GMM, but works quite well
    // and is stable.
    //
    paco_info("Adjust covariances\n");
    paco_kmeans_adjust_covariances ( X );


    //
    // Expectation-Maximization update of Gaussian modes
    //
    paco_info("E-M for %ld iterations\n",em_iter);
    for ( index_t iter = 0; iter < em_iter; iter++ ) {
        char aux[128];
        for ( index_t k = 0; k < nmodes; k++ ) {
            gaussian_mode_t *modek = &modes[k];
            const double dmax = gsl_vector_get ( modek->D, 0 );
            const double dmin = gsl_vector_get ( modek->D, m - 1 );

            paco_debug ( "mode %lu Lmax %f Lmin %f erank %lu \n", k, dmax, dmin, modek->effective_rank );
            mosaic_samples = paco_dict_mosaic ( modek->Ut, mosaic_samples == NULL ? NULL : &mosaic_samples, 1, 2, 0 );
            paco_image_st* Itmp = paco_image_from_samples ( mosaic_samples, COLORSPACE_GRAY );
            snprintf ( aux, 128, "%s/gmm_model_iter%03d_mode%03d.png", workdir, iter, k );
            write_png_file ( aux, Itmp );
            paco_image_free ( Itmp );
        }
        double prevJ = J;
        J = paco_gmm_expectation_step ( X );
        double dJ = ( J - prevJ ) / fabs ( J );

        paco_gmm_maximization_step ( X );
        raw_model_data = paco_gmm_get_model ( &raw_model_data );
        snprintf ( aux, 128, "gmm_model_iter%03d.asc", iter );

        write_ascii_matrix ( aux, raw_model_data );
        paco_debug ( "GMM: E-M iteration %lu cost %f rel %f\n", iter, J, dJ );

        save_cluster_map ( cluster_assignement, w, w, mask, iter, auximg );
        if ( fabs ( dJ ) < 1e-8 ) {
            paco_debug ( "GMM converged after %d iterations\n", iter );
            break;
        }
    }

    raw_model_data = paco_gmm_get_model ( &raw_model_data );
    write_ascii_matrix ( "gmm_model.asc", raw_model_data );

    for ( index_t k = 0; k < nmodes; k++ ) {
        char aux[128];
        gaussian_mode_t *modek = &modes[k];
        const double dmax = gsl_vector_get ( modek->D, 0 );
        const double dmin = gsl_vector_get ( modek->D, m - 1 );

        paco_debug ( "mode %lu Lmax %f Lmin %f erank %lu \n", k, dmax, dmin, modek->effective_rank );
        mosaic_samples = paco_dict_mosaic ( modek->Ut, mosaic_samples == NULL ? NULL : &mosaic_samples, 1, 2, 0 );
        paco_image_st* Itmp = paco_image_from_samples ( mosaic_samples, COLORSPACE_GRAY );
        snprintf ( aux, 128, "%s/gmm_model_mode%03d.png", workdir, k );
        write_png_file ( aux, Itmp );
        paco_image_free ( Itmp );
    }


    //
    // save final cluster map
    //
    //
    // cleanup
    //
    paco_image_free ( auximg );
    gsl_matrix_free ( X );

    //paco_image_free ( Itmp );
    gsl_matrix_free ( mosaic_samples );
    //gsl_matrix_free ( Xsub );

    for ( index_t i = 0; i < nmodes; i++ ) {
        paco_gmm_stats_free ( stats[i] );
    }

    free ( cluster_assignement );
    cluster_assignement = NULL;
    gsl_matrix_free ( raw_model_data );
}
