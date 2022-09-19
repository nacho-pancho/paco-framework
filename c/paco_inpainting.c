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
 * \file paco_dct_inpainting_problem.c
 * \brief Implementation of the PACO-DCT inpainting problem
 *
 */
#include <math.h>
#include <assert.h>
#include <string.h>
#include "paco_log.h"
#include "paco_dct.h"
#include "paco_io.h"
#include "paco_metrics.h"
#include "paco_image.h"
#include "paco_inpainting.h"
#include "paco_mapping.h"
#include "paco_problem.h"

static gsl_matrix *stitched_samples; ///< auxiliary signal; contains the result of the algorithm at the end
static const gsl_matrix *input_samples;
static const gsl_matrix *mask_samples;

static const paco_mapping_st* mapping;
static index_t *idx_missing_samples; ///< missing pixels' linear indexes
static index_t num_missing_samples; ///< number of missing pixels; computed.
static index_t *idx_projected_samples;  ///< projected pixels linear indexes: this is the set O in the paper
static index_t num_projected_samples; ///< affected pixels with known values: this is the size of the set O in the paper

//============================================================================

void paco_inpainting_create ( const struct paco_problem* problem );

void paco_inpainting_prox();

void paco_inpainting_destroy();

paco_function_st paco_inpainting() {
    paco_function_st preset;
    strcpy ( preset.name, "inpainting" );
    preset.create = paco_inpainting_create;
    preset.fit = paco_dummy_fit;
    preset.eval   = paco_dummy_eval;
    preset.prox   = paco_inpainting_prox;
    preset.destroy = paco_inpainting_destroy;
    return preset;
}

//============================================================================

/**
 * \brief Proximal operator for the PACO-DCT inpainting problem (ALGORITHM 3 in the paper.)
 * It performs the following steps:
 * 1) convert patches from DCT to sample space using the IDCT
 * 2) stitch the patches, creating an image I
 * 3) Fills in known sample values from the image I, creating PI
 * 4) Extracts the patches from PI
 * 5) Convert back the patches to DCT space by applying the direct DCT to them.DCT_H
 *
 * @param[out] proxA output matrix with the result of the proximal operator
 * @param A input to the proximal operator
 * @param lambda parameter of the proximal operator (not used in this case)
 * @return the result of the above procedure
 */
void paco_inpainting_prox ( gsl_matrix *PB, const gsl_matrix *B, const double lambda ) {
    paco_debug ( "prox_dct_inpainting.\n" );


    const  gsl_matrix *X = input_samples;
    gsl_matrix *V = stitched_samples;

    if ( B != PB ) {
        gsl_matrix_memcpy ( PB, B );
    }

    //const index_t m = B->size2;
    //const index_t w = ( index_t ) sqrt ( m ); // only square patches for now
    //idct2dn ( PB, w, w, PB );
//* 2) stitch the patches, creating an image V
    mapping->stitch ( V, PB );
//* 3) Overwrites samples in V with known samples from O,
    sample_t *pV = V->data;
    const sample_t *pX = X->data;

    for ( int k = 0; k < num_projected_samples; ++k ) {
        pV[idx_projected_samples[k]] = pX[idx_projected_samples[k]];
    }

//* 4) Extracts the patches from V
    mapping->extract ( PB, V );
//* 5) Convert back the patches to DCT space by applying the direct DCT to them.DCT_H
    //dct2dn ( PB, w, w, PB );
}

//============================================================================

/**
 * as this is an indicator function, and B is always feasible, the value is always 0
 */
double paco_inpainting_eval ( const gsl_matrix *B ) {
    return 0;
}

//============================================================================

/**
 * initialize problem structures and state
 */
void paco_inpainting_create ( const paco_problem_st* problem ) {

    paco_info ( "Creating inpainting problem...\n" );
    const paco_data_st* data = &problem->data;
    const gsl_matrix *signal = data->input;
    const gsl_matrix *mask = data->mask;
    mapping = &problem->mapping;

    input_samples = signal;
    mask_samples = mask;
    const index_t M = mask->size1;
    const index_t N = mask->size2;
    const index_t tda = mask->tda;
    stitched_samples = gsl_matrix_alloc ( M, N );

    paco_debug ( "input: %d x %d\n", M, N );
    /// Counting missing pixels:
    ///Non zero values are interpreted as missing pixels
    //
    // gather missing pixels information
    //
    int num_mis = 0;

    for ( int i = 0; i < M; i ++ ) {
        for ( int j = 0; j < N; j ++ ) {
            if ( gsl_matrix_get ( mask, i, j ) != 0 ) {
                num_mis++;
            }
        }
    }

    num_missing_samples = num_mis;
    paco_debug ( "Number of missing pixels :%ld\n", num_mis );

    if ( num_mis == 0 ) {
        paco_warn ( "No missing pixels. Nothing else to do here." );
        idx_missing_samples = 0;
        num_projected_samples = 0;
        idx_projected_samples = 0;
        return;
    }

    idx_missing_samples = ( index_t * ) malloc ( num_mis * sizeof ( index_t ) );
    num_mis = 0;

    for ( int i = 0; i < M; i ++ ) {
        for ( int j = 0; j < N; j ++ ) {
            if ( gsl_matrix_get ( mask, i, j ) ) {
                idx_missing_samples[num_mis++] = i * tda + j;
            }
        }
    }

    //
    // gather projected pixels information
    // these are the affected pixels which are not missing
    //
    {
        const index_t  nmapped = mapping->num_mapped_samples;
        const index_t *pmapped = mapping->idx_mapped_samples;
        const index_t *pmis = idx_missing_samples;
        paco_info ( "Number of missing pixels: %ld  \n", num_mis );
        paco_info ( "Number of mapped pixels: %ld\n", nmapped );
        index_t imis = 0;
        index_t nproj = 0;
        index_t next_missing = pmis[imis];

        //Count projected pixels
        for ( index_t imap = 0; imap < nmapped; imap++ ) {
            const index_t idx_map = pmapped[imap];

            if ( idx_map != next_missing ) {
                nproj++;
            } else { // hit a missing pixel
                imis++;

                if ( imis >= num_mis ) {
                    // zero can never be the 'next' missing pixel
                    // so this here means that all the remaining mapped
                    // pixels will be projected
                    next_missing = 0;
                } else {
                    next_missing = pmis[imis];
                }
            }
        }

        //Allocate storage for indexes
        num_projected_samples = nproj;
        idx_projected_samples = ( index_t * ) malloc ( nproj * sizeof ( index_t ) );
        paco_info ( "Number of projected pixels: %ld\n", nproj );

        if ( imis < num_mis ) {
            paco_info ( "Warning: %d missing pixels were not mapped",  num_mis - imis );
        }

        //Store projected indexes
        index_t iproj = 0;
        imis = 0;
        next_missing = pmis[imis];

        for ( index_t imap = 0; imap < nmapped; imap++ ) {
            const index_t idx_map = pmapped[imap];

            if ( idx_map != next_missing ) {
                idx_projected_samples[iproj++] = idx_map;
            } else { // hit a missing pixel
                imis++;

                if ( imis >= num_mis ) {
                    // zero can never be the 'next' missing pixel
                    // so this here means that all the remaining mapped
                    // pixels will be projected
                    next_missing = 0;
                } else {
                    next_missing = pmis[imis];
                }
            }
        }
    }
}


//============================================================================

/**
 * destroy problem structures and free any allocated space
 */
void paco_inpainting_destroy() {
    paco_info ( "Destroying inpainting problem...\n" );
    free ( idx_missing_samples );
    free ( idx_projected_samples );
    gsl_matrix_free ( stitched_samples );
}
