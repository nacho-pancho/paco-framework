#include "paco_problem.h"
#include "paco_log.h"
#include <assert.h>
#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#define MAX_CLUSTERS 64

//============================================================================

typedef struct _gaussian_stats_ {
    gsl_vector *sumX;
    gsl_matrix *sumXtX;
    double count;
} gaussian_stats_t;

//============================================================================

typedef struct _gaussian_model_ {
    gsl_vector *mu;
    gsl_vector *D;
    gsl_matrix *Ut;
    gsl_vector *Utmu;
    double logdet;
} gaussian_mode_t;

//============================================================================

sample_t svthres = 1e-4;

gsl_vector *aux;
gsl_vector *aux2;
gsl_vector *candidates[MAX_CLUSTERS];
gsl_vector *cluster_assignement = NULL;

index_t nmodes;
long cluster_freq[MAX_CLUSTERS];
gaussian_stats_t stats[MAX_CLUSTERS];
gaussian_mode_t modes[MAX_CLUSTERS];


// for denoising
const gsl_matrix *noisy_reference = NULL;
sample_t noise_sigma = 0;


index_t paco_gmm_get_cluster_freq ( const index_t k ) {
    return cluster_freq[k];
}

index_t paco_gmm_get_num_clusters() {
    return nmodes;
}

//============================================================================

void paco_gmm_set_noisy_reference ( const gsl_matrix *noisy_patches ) {
    noisy_reference = noisy_patches;
}

//============================================================================

void paco_gmm_set_noise_sigma ( const sample_t sigma ) {
    noise_sigma = sigma;
}

//============================================================================

void paco_gmm_set_svthres ( const sample_t svt ) {
    svthres = svt;
}

//============================================================================

static double eval_mode ( const index_t k, const gsl_vector *x ) {
    double cost = modes[k].logdet;
    const index_t m = x->size;
    gsl_vector_memcpy ( aux, x );
    gsl_vector_sub ( aux, modes[k].mu );
    gsl_blas_dgemv ( CblasNoTrans, 1.0, modes[k].Ut, aux, 0.0, aux2 );
    // cost_k = log |D| + (X_j-mu_k).U.invD.Ut(X_j-mu_k)
    //        = log |D| + \sum_i aux2_i^2/D_i
    sample_t *pD = modes[k].D->data;
    sample_t *pa = aux2->data;
    const sample_t svt = svthres * pD[0];

    for ( index_t i = 0; i < m; i++ ) {
        cost += ( 1.0 / ( pD[i] > svt ? pD[i] : svt ) ) * pa[i] * pa[i];
    }

    return cost;
}

//============================================================================

static void solve_for_mode_noiseless ( index_t k, const sample_t tau, const gsl_vector *x, gsl_vector *z ) {
    const gsl_matrix *Ut_k = modes[k].Ut;
    const gsl_vector *Utmu_k = modes[k].Utmu;
    const gsl_vector *D_k = modes[k].D;
    const index_t m = x->size;

    // compute proximal onto  mode k
    // [x_k] = arg min_w (w-mu_k)^tUD_k^{-1}U^t(w-mu_k) + (1/tau)||x-w||^2_2
    //       = U_k [ D_k^{-1} + (1/tau) I ]^{-1} ( D^{-1}U^tmu_k + U^tx/tau )
    // [ D_k^{-1} + (1/tau) I ]^{-1} = diag( s_i*tau/(s_i + tau), i = 1,...,m )
    //       = U_k ( tau/(s_i+tau)Utmu + s_i/(s_i+tau)Utx )
    gsl_vector_set_zero ( aux );
    gsl_blas_dgemv ( CblasNoTrans, 1.0, Ut_k, x, 0.0, aux );

    sample_t *paux = aux->data;
    const sample_t *plambda = D_k->data;
    const sample_t svt = svthres * plambda[0];
    const sample_t *pUtmu = Utmu_k->data;

    for ( index_t i = 0; i < m; i++ ) {
        sample_t li = plambda[i] > svt ? plambda[i] : svt;
        paux[i] = ( tau * pUtmu[i] + li * paux[i] ) / ( li + tau );
    }

    // z <- U_k * aux
    gsl_vector_set_zero ( z );
    gsl_blas_dgemv ( CblasTrans, 1.0, Ut_k, aux, 0.0, z );
}

//============================================================================

static void solve_for_mode_denoising ( index_t k, const sample_t tau, const gsl_vector *x, const gsl_vector *z, gsl_vector *px ) {
    const gsl_matrix *Ut_k = modes[k].Ut;
    const gsl_vector *Utmu_k = modes[k].Utmu;
    const gsl_vector *D_k = modes[k].D;
    const index_t m = x->size;

    // compute proximal onto  mode k
    // [x_k] = arg min_w (w-mu_k)^tUD_k^{-1}U^t(w-mu_k) + (1/tau)||x-w||^2_2
    //       = U_k [ D_k^{-1} + (1/tau) I + (1/sigma^2) ]^{-1} ( D^{-1}U^tmu_k + (1/tau)U^tx  + (1/sigma2)U^tz)
    //
    // [ D_k^{-1} + (1/tau) I + (1/sigma^2) I ]^{-1} = diag( s_i*tau*sigma^2/(tau*sigma^2 + s_i*sigma^2 + s_i*tau), i = 1,...,m )
    //       = U_k ( tau/(s_i+tau)Utmu + s_i/(s_i+tau)Utx )
    gsl_vector_set_zero ( aux );
    const double sigma2 = noise_sigma * noise_sigma;
    gsl_blas_dgemv ( CblasNoTrans, 1.0 / sigma2, Ut_k, z, 0.0, aux );
    gsl_blas_dgemv ( CblasNoTrans, 1.0 / tau, Ut_k, x, 1.0, aux );

    sample_t *paux = aux->data;
    const sample_t *plambda = D_k->data;
    const sample_t svt = svthres * plambda[0];
    const sample_t *pUtmu = Utmu_k->data;

    for ( index_t i = 0; i < m; i++ ) {
        const sample_t li = plambda[i] > svt ? plambda[i] : svt;
        const double nom = li * sigma2 * tau;
        const double den =  li * sigma2 + li * tau + sigma2 * tau;
        paux[i] = ( sample_t ) ( pUtmu[i] / li + paux[i] ) * ( nom / den );
    }

    // z <- U_k * aux
    gsl_vector_set_zero ( px );
    gsl_blas_dgemv ( CblasTrans, 1.0, Ut_k, aux, 0.0, px );
}

//============================================================================

double paco_eval_f ( const gsl_matrix *X ) {
    const index_t n = X->size1;
    const index_t m = X->size2;

    if ( cluster_assignement == NULL ) {
        cluster_assignement = gsl_vector_alloc ( n );
    }

    memset ( cluster_freq, 0, sizeof ( index_t ) *MAX_CLUSTERS );
    double total_cost = 0;

    for ( index_t j = 0; j < n; j++ ) {
        double cost = 0;
        double best = 1e20;
        index_t bestk = 0;
        gsl_vector_const_view Xjview = gsl_matrix_const_row ( X, j );

        for ( index_t k = 0; k < nmodes; k++ ) {
            double cost = eval_mode ( k, &Xjview.vector );

            if ( cost < best ) {
                best = cost;
                bestk = k;
            }
        }

        gsl_vector_set ( cluster_assignement, j, ( double ) bestk / ( double ) nmodes );
        cluster_freq[bestk]++;
        cost = best;

        //
        //
        //
        if ( ( noise_sigma > 0 ) && ( noisy_reference != NULL ) ) {
            double error = 0;
            sample_t *pnoisy = noisy_reference->data;
            sample_t *psignal = Xjview.vector.data;

            for ( index_t i = 0; i < m; i++ ) {
                double tmp = pnoisy[i] - psignal[i];
                error += tmp * tmp;
            }

            error /= ( noise_sigma * noise_sigma );
            cost += error;
            //paco_info("patch %08d prior %f error %f total %f\n",j,best,error,cost);
        } else {
            //paco_info("patch %08d prior  %f total %f\n",j,best,cost);
        }

        total_cost +=  cost;
    }

    return total_cost;
}

//============================================================================

const gsl_vector *get_clustering_map () {
    return cluster_assignement;
}

//============================================================================

void paco_prox_f ( gsl_matrix *PX, const gsl_matrix *X, const double tau ) {
    const index_t n = X->size1;

    for ( index_t j = 0; j < n; j++ ) {
        double best_cost = 1e20;
        index_t best_mode = 0;
        gsl_vector_const_view Xjview = gsl_matrix_const_row ( X, j );
        gsl_vector_view PXjview = gsl_matrix_row ( PX, j );

        for ( index_t k = 0; k < nmodes; k++ ) {
            if ( noise_sigma > 0 ) {
                gsl_vector_const_view Zjview = gsl_matrix_const_row ( noisy_reference, j );
                solve_for_mode_denoising ( k, tau, &Xjview.vector, &Zjview.vector, candidates[k] );
            }  else {
                solve_for_mode_noiseless ( k, tau, &Xjview.vector, candidates[k] );
            }

            double cost = eval_mode ( k, candidates[k] );

            if ( cost < best_cost ) {
                best_cost = cost;
                best_mode = k;
            }
        }

        // copy the best
        gsl_vector_memcpy ( &PXjview.vector, candidates[best_mode] );
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
    nmodes = p;

    for ( index_t i = 0; i < nmodes; i++ ) {
        gsl_vector_view vview;
        gsl_matrix_view mview;

        modes[i].mu = gsl_vector_alloc ( m );
        vview = gsl_matrix_row ( model_data, i * ( m + 2 ) );
        gsl_vector_memcpy ( modes[i].mu, &vview.vector );

        modes[i].D = gsl_vector_alloc ( m );
        vview = gsl_matrix_row ( model_data, i * ( m + 2 ) + 1 );
        gsl_vector_memcpy ( modes[i].D, &vview.vector );

        modes[i].logdet = 0;
        const sample_t svt = gsl_vector_get ( modes[i].D, 0 ) * svthres;

        for ( index_t j = 0; j < m; j++ ) {
            const sample_t lj = gsl_vector_get ( modes[i].D, j );
            modes[i].logdet += log ( lj > svt ? lj : svt );
        }

        modes[i].Ut = gsl_matrix_alloc ( m, m );
        mview = gsl_matrix_submatrix ( model_data, i * ( m + 2 ) + 2, 0, m, m );
        gsl_matrix_memcpy ( modes[i].Ut, &mview.matrix );

        modes[i].Utmu = gsl_vector_alloc ( m );
        gsl_vector_set_zero ( modes[i].Utmu );
        gsl_blas_dgemv ( CblasNoTrans, 1.0, modes[i].Ut, modes[i].mu, 0.0, modes[i].Utmu );
    }

    // the logdet can be very negative and drive the cost function
    //
    sample_t minval = 0;

    for ( index_t k = 0; k < p; k++ ) {
        paco_info ( "k=%d log|S_k|=%f\n", k, modes[k].logdet );

        if ( modes[k].logdet < minval )
            minval = modes[k].logdet;
    }

    //for (index_t k = 0; k < p; k++) {
    //  modes[k].logdet -= minval;
    //}
    aux = gsl_vector_alloc ( m );
    aux2 = gsl_vector_alloc ( m );

    for ( index_t k = 0; k < nmodes; k++ ) {
        candidates[k] =  gsl_vector_alloc ( m );
        cluster_freq[k] = 0;
    }

}


//============================================================================

void paco_gmm_init_model_stats ( const index_t patch_dim, const index_t K ) {
#if 0
    nmodes = K;
    sXtX = malloc ( sizeof ( gsl_matrix * ) *nmodes );
    sX = malloc ( sizeof ( gsl_vector * ) *nmodes );

    for ( index_t k = 0; k < nmodes; k++ ) {
        sXtX[k] = gsl_matrix_alloc ( patch_dim, patch_dim );
        gsl_matrix_set_zero ( sXtX[k] );
        sX[k] = gsl_vector_alloc ( patch_dim );
        gsl_vector_set_zero ( sX[k] );
    }

    counts = ( double * ) calloc ( K, sizeof ( double ) );
#endif
}

//============================================================================

void paco_gmm_update_model_stats ( const gsl_matrix *patches, gsl_vector **pass ) {
#if 0

    if ( nmodes == 0 ) {
        return;
    }

    //
    // init stats if first time
    //
    if ( invS == NULL ) {
        invS = malloc ( sizeof ( gsl_matrix * ) *nmodes );
        mu = malloc ( sizeof ( gsl_vector * ) *nmodes );
        const index_t patch_dim = sX[0]->size;

        for ( index_t k = 0; k < nmodes; k++ ) {
            invS[k] = gsl_matrix_alloc ( patch_dim, patch_dim );
            gsl_matrix_set_zero ( invS[k] );
            mu[k] = gsl_vector_alloc ( patch_dim );
            gsl_vector_set_zero ( mu[k] );
        }
    }

    //
    // allocate the auxiliary vector pass
    //
    const index_t n = patches->size1;

    if ( pass == NULL ) {
        *pass = gsl_vector_alloc ( n );
    } else {
        assert ( ( *pass )->size == n );
    }

    //
    // update statistics
    // \todo
#endif
}

//============================================================================

void paco_gmm_estimate_model ( const gsl_matrix *patches ) {
    if ( nmodes == 0 ) {
        return; // no data!
    }
}

//============================================================================

void paco_gmm_destroy() {
    gsl_vector_free ( aux2 );
    aux2 = 0;
    gsl_vector_free ( aux );
    aux = 0;

    for ( index_t k = 0; k < nmodes; k++ ) {
        gsl_vector_free ( modes[k].Utmu );
        modes[k].Utmu = 0;
        gsl_matrix_free ( modes[k].Ut );
        modes[k].Ut = 0;
        gsl_vector_free ( modes[k].D );
        modes[k].D = 0;
        gsl_vector_free ( modes[k].mu );
        modes[k].mu = 0;
        free ( candidates[k] );
        candidates[k] = 0;
    }

    if ( cluster_assignement ) {
        gsl_vector_free ( cluster_assignement );
        cluster_assignement = NULL;
    }
}


//============================================================================

static sample_t adapt_rate = 0.9;

//============================================================================

void paco_gmm_set_adapt_rate ( const sample_t rate ) {
    adapt_rate = rate;
}

//============================================================================

static void paco_gmm_init_stats() {
    const index_t m = modes[0].mu->size;

    if ( stats[0].sumX == NULL ) {
        for ( index_t k = 0; k < nmodes; k++ ) {
            stats[k].sumX = gsl_vector_alloc ( m );
            stats[k].sumXtX = gsl_matrix_alloc ( m, m );
            stats[k].count = 0;
            gsl_vector_set_zero ( stats[k].sumX );
            gsl_matrix_set_zero ( stats[k].sumXtX );
        }
    }
}

//============================================================================

void paco_gmm_adapt_to ( const gsl_matrix *X ) {
    paco_info ( "Adapting model to %ld input patches\n", X->size1 );
    const index_t n = X->size1;
    const index_t m = X->size2;

    if ( cluster_assignement == NULL ) {
        cluster_assignement = gsl_vector_alloc ( n );
    }

    memset ( cluster_freq, 0, sizeof ( index_t ) *MAX_CLUSTERS );

    //
    // first pass: cluster assignement
    //
    gsl_matrix *scores = gsl_matrix_alloc ( n, MAX_CLUSTERS );
    gsl_vector *xminusmu = gsl_vector_alloc ( m );

    for ( index_t j = 0; j < n; j++ ) {
        gsl_vector_const_view Xjview = gsl_matrix_const_row ( X, j );
        gsl_vector_view Sjview = gsl_matrix_row ( scores, j );
        gsl_vector *Sj = &Sjview.vector;

        double score_sum = 0;

        for ( index_t k = 0; k < nmodes; k++ ) {
            const double Sjk = exp ( ( -0.5 / m ) * eval_mode ( k, &Xjview.vector ) );
            gsl_vector_set ( Sj, k, Sjk );
            score_sum += Sjk;
        }

        gsl_vector_scale ( Sj, 1.0 / ( double ) score_sum );
        index_t best = gsl_vector_min_index ( Sj );
        gsl_vector_set ( cluster_assignement, j, ( double ) best / ( double ) nmodes );
        cluster_freq[best]++;
    }

    //
    // update each mode proportionally to its influence in the current sample
    //

    paco_gmm_init_stats();

    //
    // update means
    //
    for ( index_t j = 0; j < n; j++ ) {
        gsl_vector_const_view Xjview = gsl_matrix_const_row ( X, j );

        for ( index_t k = 0; k < nmodes; k++ ) {
            const double Sjk = gsl_matrix_get ( scores, j, k );
            stats[k].count += Sjk;
            gsl_blas_daxpy ( Sjk, &Xjview.vector, stats[k].sumX );
        }
    }

    for ( index_t k = 0; k < nmodes; k++ ) {
        // normalize batch estimation
        gsl_vector_scale ( stats[k].sumX, 1.0 / stats[k].count );
        // new mean = (1.0-adapt_rate)*old_mu + adapt_rate*new_mu
        gsl_vector_scale ( modes[k].mu, 1.0 - adapt_rate );
        gsl_blas_daxpy ( adapt_rate, stats[k].sumX, modes[k].mu );
    }

    //
    // update covariance matrices
    //
    //
    // in the case of the covariance matrix, we need to reconstruct it
    // from U and D
    //
    gsl_matrix *aux = gsl_matrix_alloc ( m, m );

    for ( index_t k = 0; k < nmodes; k++ ) {
        // aux = D*Ut
        gsl_matrix_memcpy ( aux, modes[k].Ut );

        for ( index_t i = 0; i < m; i++ ) {
            const double li = gsl_vector_get ( modes[k].D, i );
            gsl_vector_view auxiview = gsl_matrix_row ( aux, i );
            gsl_vector *auxi = &auxiview.vector;
            gsl_vector_scale ( auxi, li );
        }

        // sumXtX = (1-a)*U.D.Ut
        gsl_blas_dgemm ( CblasTrans, CblasNoTrans, 1.0 - adapt_rate, aux, modes[k].Ut, 0.0, stats[k].sumXtX );
    }

    gsl_matrix_free ( aux );

    //
    // now add the new data into XtX
    // in this case we have stats[k].count computed, so we can normalize
    // on the go.
    for ( index_t j = 0; j < n; j++ ) {
        gsl_vector_const_view Xjview = gsl_matrix_const_row ( X, j );
        const gsl_vector *Xj = &Xjview.vector;

        for ( index_t k = 0; k < nmodes; k++ ) {
            const double Sjk = gsl_matrix_get ( scores, j, k );
            gsl_vector_memcpy ( xminusmu, Xj );
            gsl_vector_sub ( xminusmu, modes[k].mu );
            // normalize "on the go":
            gsl_blas_dger ( adapt_rate * Sjk / stats[k].count, xminusmu, xminusmu, stats[k].sumXtX );
            // if samples are noisy, remove sigma^2 from diagonal
        }
    }

    //
    // recompute SVD from sumXtX
    //
    gsl_vector *work = gsl_vector_alloc ( m );
    gsl_vector *S = gsl_vector_alloc ( m );
    gsl_matrix *U = gsl_matrix_alloc ( m, m );
    gsl_matrix *V = gsl_matrix_alloc ( m, m );

    for ( index_t k = 0; k < nmodes; k++ ) {
        // U <- XtX
        for ( index_t i = 0; i < m; i++ ) {
            for ( index_t j = 0; j < m; j++ ) {
                gsl_matrix_set ( U, i, j, gsl_matrix_get ( stats[k].sumXtX, i, j ) );
            }
        }

        // USVt = XtX
        gsl_linalg_SV_decomp ( U, V, S, work );
        // D <- S
        // Ut <- transp(U)
        // additional term due to noise
        const sample_t s2 = adapt_rate * noise_sigma * noise_sigma;

        for ( index_t i = 0; i < m; i++ ) {
            const double Sii = gsl_vector_get ( S, i );
            gsl_vector_set ( modes[k].D, i, Sii > s2 ? Sii - s2 : 0 );

            for ( index_t j = 0; j < m; j++ ) {
                gsl_matrix_set ( modes[k].Ut, j, i, gsl_matrix_get ( U, i, j ) );
            }
        }

        // Utmu
        gsl_blas_dgemv ( CblasNoTrans, 1.0, modes[k].Ut, modes[k].mu, 0.0, modes[k].Utmu );
        // log det |S|
        modes[k].logdet = 0;
        const sample_t svt = gsl_vector_get ( modes[k].D, 0 ) * svthres;

        for ( index_t j = 0; j < m; j++ ) {
            const sample_t lj = gsl_vector_get ( modes[k].D, j );
            modes[k].logdet += log ( lj > svt ? lj : svt );
        }

    }

    gsl_vector_free ( work );
    gsl_vector_free ( S );
    gsl_matrix_free ( U );
    gsl_matrix_free ( V );
    gsl_matrix_free ( scores );
}
