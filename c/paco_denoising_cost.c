#include <math.h>
#include <assert.h>
#include <string.h>

#include <gsl/gsl_blas.h>

#include "paco_log.h"
#include "paco_image.h"
#include "paco_mapping.h"
#include "paco_function.h"
#include "paco_denoising.h"
#include "paco_util.h"

//============================================================================

static double noise_sigma = 0;
//static double denoising_const = 1.15;
static gsl_matrix* noisy_patches = 0;
static const gsl_matrix* noisy_image = 0;
static gsl_matrix* aux_image = 0;
static paco_function_st prior; ///< patch prior used as surrogate to denoising prior
static const paco_mapping_st* mapping;

static void patch_map_cost_prox ( gsl_matrix *A, const gsl_matrix *B, const double tau );
//static void signal_ball_cost_prox ( gsl_matrix *A, const gsl_matrix *B, const double tau );
//static void patch_ball_cost_prox ( gsl_matrix *A, const gsl_matrix *B, const double tau );
static double patch_map_cost_eval ( const gsl_matrix *B );
static void paco_denoising_cost_create ( const struct paco_problem* problem );
static void paco_denoising_cost_destroy(  );

//============================================================================

paco_function_st paco_denoising_cost ( paco_function_st* _prior, int mode ) {
    paco_function_st preset;
    prior = *_prior;
    preset.create  = paco_denoising_cost_create;
    paco_debug ( "Using denoising mode %d\n", mode );
    switch ( mode ) {
    case DENOISING_MODE_PATCH_BALL:
        strcpy ( preset.name, "patch-ball-" );
        preset.eval    = prior.eval;
        preset.prox    = prior.prox; // see this...
        break;
    case DENOISING_MODE_SIGNAL_BALL:
        strcpy ( preset.name, "signal-ball-" );
        preset.eval    = prior.eval;
        preset.prox    = prior.prox;
        break;
    case DENOISING_MODE_PATCH_MAP:
        strcpy ( preset.name, "patch-map-" );
        preset.eval    = patch_map_cost_eval;
        preset.prox    = patch_map_cost_prox;
        break;
    case DENOISING_MODE_SIGNAL_MAP:
        strcpy ( preset.name, "signal-map-" );
        preset.eval    = prior.eval;
        preset.prox    = prior.prox;
        break;
    default:
        strcpy ( preset.name, "INVALID-MODE-" );
    }
    preset.destroy = paco_denoising_cost_destroy;
    strcat ( preset.name, prior.name );
    preset.create  = paco_denoising_cost_create;
    preset.fit     = prior.fit;
    return preset;
}

//============================================================================

void paco_denoising_cost_create ( const struct paco_problem* problem ) {
    const index_t n = problem->mapping.num_mapped_patches;
    const index_t m = problem->mapping.patch_dim;
    if ( problem->config.sigma <= 0 ) {
        paco_warn ( "Noise sigma is 0." );
    }
    noise_sigma = problem->config.sigma / 255.0;
    noisy_patches = gsl_matrix_alloc ( n, m );
    mapping = &problem->mapping;
    noisy_image = problem->data.input;
    mapping->extract ( noisy_patches, problem->data.input );
    aux_image = gsl_matrix_alloc ( noisy_image->size1, noisy_image->size2 );
    prior.create ( problem );
}

//============================================================================

void paco_denoising_cost_destroy(  ) {
    gsl_matrix_free ( noisy_patches );
    gsl_matrix_free ( aux_image );
}

//============================================================================

static void patch_map_cost_prox ( gsl_matrix *A, const gsl_matrix *B, const double tau ) {
    const index_t n = B->size1;
    const index_t m = B->size2;
    const gsl_matrix *Z = noisy_patches;
    const double sigma2 = noise_sigma * noise_sigma;
    // arg min f(A) + 1/2sigma2||A-Z|| + 1/2tau ||A-B||
    //
    // A <- tau/(tau+sigma2) Z
    // view matrices as a vectors
    // this allows to use blas 1 operations
    // to compute tau/(tau+sigma2)*Z + sigma2/(tau+sigma2)*B
    //
    gsl_vector_view Aview = gsl_vector_view_array ( A->data, m * n );
    gsl_vector *a = &Aview.vector;
    gsl_vector_const_view Bview = gsl_vector_const_view_array ( B->data, m * n );
    const gsl_vector *b = &Bview.vector;
    gsl_vector_const_view Zview = gsl_vector_const_view_array ( Z->data, m * n );
    const gsl_vector *z = &Zview.vector;
    gsl_vector_set_zero ( a );
    paco_debug ( "patch_map_cost_prox: tau=%f sigma2=%f,  tau/(tau+sigma2)=%f  sigma2/(tau+sigma2)=%f  ", tau, sigma2, tau / ( tau + sigma2 ), sigma2 / ( tau + sigma2 ) );
    paco_debug ( "tau'=tau*sigma2/(tau+sigma2)=%f\n", tau * sigma2 / ( tau + sigma2 ) );
    gsl_blas_daxpy ( tau / ( tau + sigma2 ),    z, a );
    gsl_blas_daxpy ( sigma2 / ( tau + sigma2 ), b, a );
    prior.prox ( A, A, tau * sigma2 / ( tau + sigma2 ) );
}

#if 0

static void patch_ball_cost_prox ( gsl_matrix *A, const gsl_matrix *B, const double tau ) {

    prior.prox ( A, B, tau );

    // compliquetti!
    // if the result of the proximal operator is further than Cmsigma^2 from the samples, a more complex
    // problem needs to be solved. For now we just do the above and mark when that condition is violated
    //(surely always)
    const index_t m = A->size2;
    const index_t n = A->size1;
    paco_info ( "denoising const %f\n", denoising_const );
    const double radius = denoising_const * m * noise_sigma * noise_sigma;
    for ( index_t j = 0; j < n; ++j ) {

        gsl_vector_view Ajv = gsl_matrix_row ( A, j );
        //gsl_vector_const_view Bjv = gsl_matrix_const_row(B,j);
        gsl_vector_const_view Zjv = gsl_matrix_const_row ( noisy_patches, j );

        gsl_vector* Aj = &Ajv.vector;
        const gsl_vector* Zj = &Zjv.vector;

        double dj = l2dist ( Aj, Zj );
        if ( dj > radius ) {
            // PENDING
        }
    }
}


static void signal_ball_cost_prox ( gsl_matrix *A, const gsl_matrix *B, const double tau ) {
    const double MN = noisy_image->size1 * noisy_image->size2;
    const double r = sqrt ( denoising_const * MN * noise_sigma * noise_sigma );
    if ( A != B ) {
        gsl_matrix_memcpy ( A, B );
    }
    mapping->stitch ( aux_image, A );
    const double d = sqrt ( frobenius_squared_dist ( aux_image, noisy_image ) );
    paco_info ( "signal ball: distance=%f radius=%f\n", d, r );
    if ( d > r ) {
        sample_t *pA = aux_image->data;
        const sample_t *pC = noisy_image->data;
        const double a = r / d;
        const double b = 1.0 - a;
        for ( index_t i = 0; i < MN; ++i ) {
            pA[i] = a * pA[i] + b * pC[i];
        }
    }
    mapping->extract ( A, aux_image );
}
#endif

//============================================================================


static double patch_map_cost_eval ( const gsl_matrix *B ) {
    const index_t m = B->size1;
    const index_t n = B->size2;
    return  1.0 / ( 2.0 * noise_sigma * noise_sigma * (double) m * (double) n ) * frobenius_squared_dist ( B, noisy_patches ) + prior.eval ( B );
} 


