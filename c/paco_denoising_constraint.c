/*
  * Copyright (c) 2019 Ignacio Francisco Ramírez Paulino and Ignacio Hounie
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
 * \file paco_denoising_problem.c
 * \brief denoising problem implementation
 *
 */
#include <math.h>
#include <assert.h>
#include <string.h>
#include "paco_log.h"
#include "paco_denoising.h"
#include "paco_mapping.h"
#include "paco_function.h"

//============================================================================

/**
 * variants of the denoising problem: MAP = maximum a posteriori
 */

static const gsl_matrix *input_samples = NULL; ///< input signal samples (one channel)
static gsl_matrix *stitched_samples = NULL; ///< auxiliary signal; contains the result of the algorithm at the end
static double noise_sigma = 0;
static double denoising_const = 0;
static const paco_mapping_st* mapping;
static int remove_dc = 0;
static gsl_vector* DC; //

//============================================================================


static void patch_map_constraint_prox ( gsl_matrix *A, const gsl_matrix *B, const double lambda );
static void patch_ball_constraint_prox ( gsl_matrix *A, const gsl_matrix *B, const double tau );
static void signal_map_constraint_prox ( gsl_matrix *A, const gsl_matrix *B, const double tau );
static void signal_ball_constraint_prox ( gsl_matrix *A, const gsl_matrix *B, const double tau );
static double signal_map_constraint_eval ( const gsl_matrix *B );
static void paco_denoising_constraint_create ( const paco_problem_st* problem );
static void paco_denoising_constraint_destroy();
/**
 * \brief The denoising constraint function depends on a surrogate prior and a denoisig mode.
 */
paco_function_st paco_denoising_constraint ( int mode ) {
    paco_function_st preset;
    switch ( mode ) {
    case DENOISING_MODE_PATCH_BALL:
        strcpy ( preset.name, "patch-ball" );
        preset.eval   = paco_dummy_eval;
        preset.prox   = patch_ball_constraint_prox;
        break;
    case DENOISING_MODE_SIGNAL_BALL:
        strcpy ( preset.name, "signal-ball" );
        preset.eval   = paco_dummy_eval;
        preset.prox   = signal_ball_constraint_prox;
        break;
    case DENOISING_MODE_PATCH_MAP:
        strcpy ( preset.name, "patch-map" );
        preset.eval   = paco_dummy_eval; // ??? check whether this should evaluate to this...
        preset.prox   = patch_map_constraint_prox;
        break;
    case DENOISING_MODE_SIGNAL_MAP:
        strcpy ( preset.name, "signal-map" );
        preset.eval   = signal_map_constraint_eval;
        preset.prox   = signal_map_constraint_prox;
        break;
    default:
        strcpy ( preset.name, "INVALID-MODE" );
    }

    preset.create = paco_denoising_constraint_create;
    preset.fit = paco_dummy_fit;
    preset.destroy = paco_denoising_constraint_destroy;

    return preset;
}


//============================================================================

static void paco_denoising_constraint_create ( const paco_problem_st* problem ) {
    mapping = &problem->mapping;
    input_samples = problem->data.input;
    noise_sigma = problem->config.sigma / 255.0;
    denoising_const = problem->config.denoising_const;
    remove_dc = problem->config.remove_dc;
    const index_t M = input_samples->size1;
    const index_t N = input_samples->size2;
    stitched_samples = gsl_matrix_alloc ( M, N );
    if ( remove_dc ) {
        DC = gsl_vector_alloc ( mapping->num_mapped_patches );
        gsl_matrix* noisy_patches = gsl_matrix_alloc ( mapping->num_mapped_patches, mapping->patch_dim );
        paco_mapping_remove_dc ( noisy_patches, DC );
        gsl_matrix_free ( noisy_patches );
    }
}


//============================================================================

static double frobenius_squared_dist ( const gsl_matrix* A, const gsl_matrix* B ) {
    const index_t n = A->size1 * A->size2;
    const sample_t* pA = A->data;
    const sample_t* pB = B->data;
    double d = 0;
    for ( index_t i = 0; i < n; ++i ) {
        const double t = pA[i] - pB[i];
        d += t * t;
    }
    return d;
}



//============================================================================

static void patch_map_constraint_prox ( gsl_matrix *A, const gsl_matrix *B, const double lambda ) {
    mapping->stitch ( stitched_samples, A );
    mapping->extract ( A, stitched_samples );
}

//============================================================================

static void patch_ball_constraint_prox ( gsl_matrix *A, const gsl_matrix *B, const double tau ) {
    mapping->stitch ( stitched_samples, A );
    mapping->extract ( A, stitched_samples );
}

//============================================================================


static void signal_map_constraint_prox ( gsl_matrix *A, const gsl_matrix *B, const double tau ) {
    // A = arg min (1/2sigma2)||Y - Z||^2_2 + (1/2tau)||Y - S(B)|| s.t. Y = S(A)
    // A = arg min ||Y - W||^2_2  s.t. Y = S(A)
    // where W = tau/(tau+sigma^2)Z + sigma^2/(tau+sigma^2)S(B)
    // A = R(W)
    mapping->stitch ( stitched_samples, A ); // S(A)
    const sample_t a = tau / ( tau + noise_sigma * noise_sigma );
    const sample_t b = 1.0f - a;
    sample_t* pSB = stitched_samples->data;
    const sample_t* pZ = input_samples->data;
    // stitched_samples is modified in place
    sample_t* pSA = pSB;
    const index_t N = stitched_samples->size1 * stitched_samples->size2;
    for ( index_t i = 0; i < N; ++i ) {
        pSA[i] = a * pZ[i] + b * pSB[i];
    }
    mapping->extract ( A, stitched_samples );
}

//============================================================================

static void signal_ball_constraint_prox ( gsl_matrix *A, const gsl_matrix *B, const double tau ) {
    // Y = S(A)
    // A in C
    // ||Y - Z||^2_F <= nCsigma^2
    if ( A != B ) {
        gsl_matrix_memcpy ( A, B );
    }
    //if (remove_dc) {
    //  paco_mapping_add_back_dc(DC,A);
    //}
    mapping->stitch ( stitched_samples, A );
    const index_t N = stitched_samples->size1 * stitched_samples->size2;
    // project onto ball
    double d = sqrt ( frobenius_squared_dist ( stitched_samples, input_samples ) );
    double r = sqrt ( denoising_const * noise_sigma * noise_sigma * ( double ) N );
    paco_debug ( "signal_ball_constraint_prox: d=%f r=%f\n", d, r );
    if ( d > r ) {
        // project onto ball of radius r
        // Y = Z + (Y-Z)*(r/d)
        // Y = (1-r/d)Z + (r/d)Y
        sample_t* pY = stitched_samples->data;
        const sample_t* pZ = input_samples->data;
        const sample_t b = r / d;
        const sample_t a = 1.0f - b;
        for ( index_t i = 0; i < N; ++i ) {
            pY[i] = a * pZ[i] + b * pY[i];
        }
        mapping->extract ( A, stitched_samples );
        //if (remove_dc) {
        //  gsl_vector* auxDC = gsl_vector_alloc(A->size1);
        //  paco_mapping_remove_dc(A,auxDC);
        //  gsl_vector_free(auxDC);
        //}
    }
}


//============================================================================


static double signal_map_constraint_eval ( const gsl_matrix *B ) {
    mapping->stitch ( stitched_samples, B );
    const index_t m = B->size1;
    const index_t n = B->size2;
    return frobenius_squared_dist ( stitched_samples, input_samples ) / ( 2 * noise_sigma * noise_sigma * ( double ) m * ( double ) n  );
    //
}


//============================================================================

static void paco_denoising_constraint_destroy() {
    gsl_matrix_free ( stitched_samples );
    if ( remove_dc ) {
        gsl_vector_free ( DC );
    }
}
