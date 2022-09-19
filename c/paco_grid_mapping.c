#include <string.h>
#include <assert.h>

#include "paco_image.h"
#include "paco_grid_mapping.h"
#include "paco_log.h"

#define MAX_SCALES 16

static index_t *idx_patch_samples; ///< precomputed linear indexes of each pixel in each patch (size num_patch_samples * num_mapped_samples)
static index_t  num_mapped_samples; ///< number of pixels which are mapped to patch space
static index_t *idx_mapped_samples; ///< linear indexes of pixels which are mapped to patch space
static sample_t *fact_mapped_samples; ///< normalization factor for mapped pixels, used for stitching


paco_mapping_st paco_grid_mapping() {
    paco_mapping_st map;
    memset ( &map, 0, sizeof ( paco_mapping_st ) );
    map.create  = paco_grid_mapping_create;
    map.destroy = paco_grid_mapping_destroy;
    map.stitch  = paco_grid_mapping_stitch;
    map.extract = paco_grid_mapping_extract;
    strcpy ( map.name, "grid" );
    return map;
}

void paco_grid_mapping_destroy() {
    // delete stuff...
    free ( idx_patch_samples );
    idx_patch_samples = 0;
    free ( idx_mapped_samples );
    idx_mapped_samples = 0;
    free ( fact_mapped_samples );
    fact_mapped_samples = 0;
}


void paco_grid_mapping_extract ( gsl_matrix *X, const gsl_matrix *I ) {

    const index_t m = X->size1;
    const index_t n = X->size2;
    const sample_t *pimg = I->data;
    const index_t *pidx = idx_patch_samples;
    const index_t mn = m * n;
    sample_t *px = X->data;

    for ( int k = 0; k < mn; k ++ ) {
        * ( px++ ) = pimg[ * ( pidx++ ) ];
    }
}

//
//============================================================================
//
void paco_grid_mapping_stitch ( gsl_matrix *I, const gsl_matrix *X ) {
    //
    // zero out mapped pixels
    //
    sample_t *pimg = I->data;
    const sample_t *px = X->data;
    const index_t nmapped = num_mapped_samples;
    const index_t n = X->size2;
    const index_t m = X->size1;
    const index_t *pimpi = idx_mapped_samples;
    const index_t *pidx = idx_patch_samples;

    //
    // initialize to zero
    //
    for ( int k = 0; k < nmapped; ++k )  {
        pimg[ pimpi[k] ] = 0;
    }

    //
    // add up all the patch estimates (from X) into their corresponding places in I
    //
    const index_t mn = m * n;

    for ( int k = 0; k < mn; ++k )  {
        pimg[ * ( pidx++ ) ] += * ( px++ );
    }

    //
    // Normalize
    //
    const sample_t *pf = fact_mapped_samples;

    for ( int k = 0; k < nmapped; ++k )  {
        pimg[ * ( pimpi++ ) ] *= * ( pf++ );
    }
}

/**
 * ============================================================================
 *
 * This function creates an instance of a mapping between an image
 * and a set of patches. This is a cheap multiscale variant of the grid
 * mapping initialization where the different 'resolutions' are achieved
 * by skipping pixels. For example, for 8x8 patches, a patci at level 0 would be:
 *
 * . . . . . . . .
 * . . . . . . . .
 * . . x x x x . .
 * . . x x x x . .
 * . . x x x x . .
 * . . x x x x . .
 * . . . . . . . .
 * . . . . . . . .
 *
 * the level 1 corresponding patch:
 * . . . . . . . . .
 * . . . . . . . . .
 * . . x . x . x . x
 * . . . . . . . . .
 * . . x . x . x . x
 * . . . . . . . . .
 * . . x . x . x . x
 * . . . . . . . . .
 * . . x . x . x . x
 *
 * and so on
 *
 * If a mask is provided, then only those patches which contain ones in the mask are extracted.
 * If no mask is provided, then all patches will be extracted.
 *
 * In any case, a number of index sets are precomputed here to speed
 * up the extraction and stitching operations. Namely:
 *
 * - relative linear indexes of a patch w.r.t. the upper-left corner; this
 * enables the use of only one loop for any N-dimensional problem
 *
 * - linear indexes of the upper-left corners of all the mapped patches.
 *
 * - linear indexes of the pixels which are mapped to one (or more) patches.
 *
 * - normalization factors for the mapped pixels. This allows for a very
 *   efficient stitching operation.
 */
#include "paco_problem.h"

void paco_grid_mapping_create ( struct paco_problem* problem ) {

    paco_mapping_st* map = &problem->mapping;

    const gsl_matrix* input = problem->data.input;
    const gsl_matrix* mask = problem->data.mask;
    const paco_config_st* config = &problem->config;
    //
    // store basic information such as the signal dimensions
    // the number of channels, etc.
    //
    //memset(map,0,sizeof(paco_mapping_st));

    const index_t num_scales = config->num_scales;
    map->input_nrows    = input->size1;
    map->input_ncols    = input->size2;
    const index_t s0    = config->grid_stride;
    const index_t w0    = config->patch_width;
    map->patch_dim = w0 * w0;
    const index_t ncols = input->size2;

    gsl_matrix *hits_map = NULL;

    //
    // compute the grid size in both directions.
    // count number of patches to be mapped
    // count number of appearances of each sample in the patches matrix (hits)
    //
    paco_debug ( "input image size %d x %d\n", map->input_nrows, map->input_ncols );
    paco_debug ( "num_scales: %d\n", num_scales );

    hits_map = gsl_matrix_alloc ( map->input_nrows, map->input_ncols );
    gsl_matrix_set_zero ( hits_map );
    if ( mask != NULL ) {
        int nmis = 0;
        for ( index_t i = 0; i < mask->size1 * mask->size2; ++i ) {
            nmis += mask->data[i];
        }
        paco_debug ( "unkown samples %d\n", nmis );
    }
    map->num_mapped_patches = 0;
    for ( int scale = 1; scale <= num_scales; scale++ ) {
        const index_t stride = s0 * scale;
        const index_t width = w0 * scale;
        const index_t ni = ( map->input_nrows - width ) / stride + 1;
        const index_t nj = ( map->input_ncols - width ) / stride + 1;

        const index_t num_patches  = nj * ni;
        paco_debug ( "scale %d: grid size %d x %d = %d patches\n", scale, ni, nj, num_patches );

        if ( mask == NULL ) {
            //
            // no mask: extract all patches from image
            // store patch linear index (leftmost & uppermost patch pixel)
            //
            map->num_mapped_patches += ni * nj;
            paco_debug ( "scale %d mapped patches %d (no mask)\n", scale, ni * nj );
        } else {
            //
            // mask provided: mapped patches are those whose
            // corresponding samples in the mask have at least one
            // non-zero element
            //
            int ninc = 0;
            int nonc = 0;
            for ( int i = 0; i <= map->input_nrows - width; i += stride ) {
                for ( int j = 0; j <= map->input_ncols - width; j += stride ) {
                    if ( is_patch_incomplete ( mask, i, j, w0, w0, scale ) ) {
                        ninc++;
                    } else {
                        nonc++;
                    }
                }
            }
            map->num_mapped_patches += ninc;
            paco_debug ( "scale %d mapped patches %d not mapped %d\n", scale, ninc, nonc );
        }
    }
    assert ( map->num_mapped_patches > 0 );
    paco_debug ( "total number of mapped patches %d\n", map->num_mapped_patches );
    //
    // allocate patch index information
    //
    idx_patch_samples = ( index_t * ) malloc ( map->patch_dim * map->num_mapped_patches * sizeof ( index_t ) );
    //
    // fill in patch index data
    //
    index_t k = 0;

    for ( int scale = 1; scale <= num_scales; scale++ ) {
        const index_t stride = s0 * scale;
        const index_t w = w0 * scale;
        const index_t ni = ( map->input_nrows - w  ) / stride + 1;
        const index_t nj = ( map->input_ncols - w  ) / stride + 1;
        const index_t num_patches  = nj * ni;
        paco_info ( "scale %d: grid size %d x %d = %d patches\n", scale, ni, nj, num_patches );

        index_t num_scale_patches = 0;

        for ( int i = 0; i <= map->input_nrows - w; i += stride ) {
            for ( int j = 0; j <= map->input_ncols - w; j += stride ) {
                if ( !mask || is_patch_incomplete ( mask, i, j, w0, w0, scale ) ) {
                    for ( int ii = 0; ii < w; ii += scale ) {
                        for ( int jj = 0; jj < w; jj += scale ) {
                            const int iii = REFLECT ( i + ii, map->input_nrows );
                            const int jjj = REFLECT ( j + jj, map->input_ncols );
                            idx_patch_samples[k++] = ( iii ) * input->tda + ( jjj );
                            hits_map->data[ ( iii ) * hits_map->tda + ( jjj )] ++;
                        }
                    }
                } // if
                num_scale_patches++;
            }
        }
        assert ( num_scale_patches == ni * nj );
    } // for each scale

    assert ( k == ( map->patch_dim * map->num_mapped_patches ) );
    //
    // Count mapped pixels = any pixel for which hits_map is > 0
    //
    int nmapped = 0;

    for ( int i = 0, k = 0; i < map->input_nrows; i ++ ) {
        for ( int j = 0; j < map->input_ncols; j ++, k++ ) {
            if ( gsl_matrix_get ( hits_map, i, j ) > 0 )  {
                nmapped++;
            }
        }
    }

    paco_debug ( "mapped pixels=%d\n", nmapped );
    idx_mapped_samples = ( index_t * ) malloc ( nmapped * sizeof ( index_t ) );
    num_mapped_samples  = nmapped;
    map->num_mapped_samples = num_mapped_samples;
    map->idx_mapped_samples = idx_mapped_samples;
    //
    // Store mapped and projected linear indexes
    //
    fact_mapped_samples = ( sample_t * ) calloc ( num_mapped_samples, sizeof ( sample_t ) );
    //
    // record mapped pixel indexes
    // compute correspondining normalization factors from hits_map
    //
    nmapped = 0;

    for ( int i = 0, k = 0; i < map->input_nrows; i ++ ) {
        for ( int j = 0; j < map->input_ncols; j ++, k++ ) {
            double hits = gsl_matrix_get ( hits_map, i, j );

            if ( hits > 0 )  {
                idx_mapped_samples[nmapped] = ncols * i + j;
                fact_mapped_samples[nmapped] = 1.0 / ( double ) hits;
                nmapped++;
            }
        }
    }

    if ( hits_map ) gsl_matrix_free ( hits_map );
}

