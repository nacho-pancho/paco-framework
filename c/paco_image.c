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
 * \file paco_image.c
 * \brief simple multichannel image implementation based on GSL matrices
 */
#include <assert.h>
#include <math.h>
#include <string.h>
#include "paco_image.h"
#include "paco_log.h"

//============================================================================

/**
 * simple image structure for PACO
 */
typedef struct _paco_image_st_ {
    index_t bitd; ///< channel bit depth; by default 8
    char colorspace[MAX_CHANNELS + 1]; /// < at most 4 channels
    gsl_matrix *samples[MAX_CHANNELS]; ///< sample data, max three channels
} paco_image_st;

//============================================================================

sample_t get_sample ( const gsl_matrix *im, const int i, const int j ) {
    const int m = im->size1;
    const int n = im->size2;
    return im->data[ REFLECT ( i, m ) * im->tda + REFLECT ( j, n ) ];
}


//============================================================================

/**
 * \return actual maximum sample value
 */
sample_t get_maxval ( const paco_image_st *im ) {
    sample_t maxval = gsl_matrix_max ( im->samples[0] );

    for ( index_t c = 1; c < get_nchannels ( im ); c++ ) {
        sample_t maxc = gsl_matrix_max ( im->samples[c] );

        if ( maxc > maxval ) {
            maxval = maxc;
        }
    }

    return maxval;
}

//============================================================================

/**
 * \return actual minimum sample value in image
 */
sample_t get_minval ( const paco_image_st *im ) {
    sample_t minval = gsl_matrix_min ( im->samples[0] );
    const index_t nchannels = get_nchannels ( im );

    for ( index_t c = 1; c < nchannels; c++ ) {
        sample_t minc = gsl_matrix_min ( im->samples[c] );

        if ( minc < minval ) {
            minval = minc;
        }
    }

    return minval;
}

//============================================================================

void paco_image_clip ( paco_image_st *I ) {
    const index_t n = get_nsamples ( I );
    const index_t nchannels = get_nchannels ( I );

    for ( index_t c = 0; c < nchannels; c++ ) {
        gsl_matrix *X = get_channel_samples ( I, c );

        for ( index_t i = 0; i < n; i++ ) {
            const sample_t x = X->data[i];
            X->data[i] = ( x < 1.0f ? ( x > 0.0f ? x : 0.0f ) : 1.0f );
        }
    }
}

//============================================================================

void paco_image_norm ( paco_image_st *I ) {
    const sample_t minval = get_minval ( I );
    const sample_t maxval = get_maxval ( I );
    const sample_t offset = minval < 0.0f ? minval : 0.0f;
    const sample_t scale = 1.0 / ( maxval - minval );
    const index_t n = get_nsamples ( I );

    for ( index_t c = 0; c < get_nchannels ( I ); c++ ) {
        gsl_matrix *X = get_channel_samples ( I, c );

        for ( index_t i = 0; i < n; i++ ) {
            X->data[i] = scale * ( X->data[i] - offset );
        }
    }
}

//============================================================================

void paco_image_squash ( paco_image_st *I ) {
    assert ( I != NULL );
    const sample_t minval = get_minval ( I );
    const sample_t maxval = get_maxval ( I );
    const sample_t offset = minval < 0.0f ? minval : 0.0f;
    const sample_t scale = maxval > 1.0f ? 1.0 / ( maxval - minval ) : 1.0 / ( 1.0 - minval );
    const index_t n = get_nsamples ( I );

    for ( index_t c = 0; c < get_nchannels ( I ); c++ ) {
        gsl_matrix *X = get_channel_samples ( I, c );

        for ( index_t i = 0; i < n; i++ ) {
            X->data[i] = scale * ( X->data[i] - offset );
        }
    }
}

//============================================================================

paco_image_st *paco_image_from_samples ( const gsl_matrix *samples, const char *colorspace ) {
    assert ( samples != NULL );
    const index_t nchan = strlen ( colorspace );
    const sample_t min = gsl_matrix_min ( samples );
    const sample_t max = gsl_matrix_max ( samples );
    const sample_t offset = min < 0 ? min : 0;
    const sample_t scale = max > 255 ? max - offset : 255 - offset;

    if ( samples->size2 % nchan ) { // size2 should be multiple of nchan
        paco_error ( "Samples matrix size not compatible with %s colorspace.", colorspace );
        return NULL;
    }

    paco_image_st *out = ( paco_image_st * ) calloc ( 1, sizeof ( paco_image_st ) );
    const index_t m = samples->size1;
    const index_t n = samples->size2 / nchan;
    strncpy ( out->colorspace, colorspace, 4 );
    out->bitd = 8;

    if ( nchan == 1 ) { // single channel
        out->samples[0] = gsl_matrix_alloc ( m, n );
        gsl_matrix_memcpy ( out->samples[0], samples );
        out->samples[1] = 0;
        out->samples[2] = 0;
    } else {
        for ( size_t c = 0; c < nchan; ++c ) {
            out->samples[c] = gsl_matrix_alloc ( m, n );

            for ( size_t i = 0; i < m; ++i ) {
                for ( size_t j = 0; j < n; ++j ) {
                    const sample_t v = scale * ( gsl_matrix_get ( samples, i, j * nchan + c ) - offset );
                    gsl_matrix_set ( out->samples[c], i, j, v );
                }
            }
        }
    }

    return out;
}


const double rgb_to_yuv[9] = {
    0.29900, 0.58700, 0.11400,
    -0.14713, -0.28886, 0.43600,
    0.61500, -0.51499, -0.10001
};

const double yuv_to_rgb[9] = {
    1.00000,  0.00000,  1.13983,
    1.00000, -0.39465, -0.58060,
    1.00000,  2.03211,  0.00000
};

void paco_image_convert_colorspace ( paco_image_st *img, const char *to ) {
    assert ( img != NULL );
    const char *from = get_colorspace ( img );
    const double *coefs = NULL;

    if ( strcmp ( from, COLORSPACE_RGB ) == 0 ) {
        if ( strcmp ( to, COLORSPACE_RGB ) == 0 ) {
            paco_warn ( "Requested trivial colorspace mapping %s->%s.", from, to );
            return;
        } else if ( strcmp ( to, COLORSPACE_YUV ) == 0 ) {
            coefs = rgb_to_yuv;
        } else {
            paco_error ( "Colorspace mapping %s->%s not implemented.", from, to );
            return;
        }
    } else if ( strcmp ( from, COLORSPACE_YUV ) == 0 ) {
        if ( strcmp ( to, COLORSPACE_RGB ) == 0 ) {
            coefs = yuv_to_rgb;
        } else if ( strcmp ( to, COLORSPACE_YUV ) == 0 ) {
            paco_warn ( "Requested trivial colorspace mapping %s->%s.", from, to );
            return;
        } else {
            paco_error ( "Colorspace mapping %s->%s not implemented.", from, to );
            return;
        }
    } else {
        paco_error ( "Unsupported  %s->%s.", from, to );
        return;
    }

    const index_t N = get_ncols ( img );
    const index_t M = get_nrows ( img );

    for ( index_t i = 0; i < M; ++i ) {
        for ( index_t j = 0; j < N; ++j ) {
            const double x1 = get_channel_sample ( img, i, j, 0 );
            const double x2 = get_channel_sample ( img, i, j, 1 );
            const double x3 = get_channel_sample ( img, i, j, 2 );
            const double y1 = ( double ) ( coefs[0] * x1 + coefs[1] * x2 + coefs[2] * x3 );
            const double y2 = ( double ) ( coefs[3] * x1 + coefs[4] * x2 + coefs[5] * x3 );
            const double y3 = ( double ) ( coefs[6] * x1 + coefs[7] * x2 + coefs[8] * x3 );
            set_channel_sample ( img, i, j, 0, y1 );
            set_channel_sample ( img, i, j, 1, y2 );
            set_channel_sample ( img, i, j, 2, y3 );
        }
    }

    strncpy ( img->colorspace, to, 4 );
}


//
//============================================================================
//
paco_image_st *paco_image_alloc ( index_t nrows, index_t ncols, const char *colorspace ) {
    paco_image_st *im_st = ( paco_image_st * ) malloc ( sizeof ( paco_image_st ) );
    strncpy ( im_st->colorspace, colorspace, 4 );
    im_st->bitd = 8; // by default

    if ( get_nchannels ( im_st ) == 1 ) {
        im_st->samples[0] = gsl_matrix_alloc ( nrows, ncols );
        im_st->samples[1] = NULL;
        im_st->samples[2] = NULL;
    } else {
        im_st->samples[0] = gsl_matrix_alloc ( nrows, ncols );
        im_st->samples[1] = gsl_matrix_alloc ( nrows, ncols );
        im_st->samples[2] = gsl_matrix_alloc ( nrows, ncols );
    }

    return im_st;
}
//
//============================================================================
//
void paco_image_free ( paco_image_st *im_st ) {
    if ( !im_st ) {
        return;
    }

    if ( im_st->samples[0] ) {
        gsl_matrix_free ( im_st->samples[0] );
        im_st->samples[0] = 0;
    }

    if ( im_st->samples[1] ) {
        gsl_matrix_free ( im_st->samples[1] );
        im_st->samples[1] = 0;
    }

    if ( im_st->samples[2] ) {
        gsl_matrix_free ( im_st->samples[2] );
        im_st->samples[2] = 0;
    }

    im_st->bitd = 0;
    im_st->colorspace[0] = 0;
    free ( im_st );

}
//
//============================================================================
//
sample_t get_channel_sample ( const paco_image_st *im, const index_t i, const index_t j, const index_t c ) {
    assert ( im != NULL );
    return	gsl_matrix_get ( im->samples[c], i, j );
}
//
//============================================================================
//
index_t get_sample_idx ( const paco_image_st *im, const index_t i, const index_t j ) {
    assert ( im != NULL );
    return	im->samples[0]->tda * i + j;
}

//
//============================================================================
//
void get_sample_coords ( const paco_image_st *im, const index_t li, index_t *pi, index_t *pj ) {
    assert ( im != NULL );
    const gsl_matrix *samples = im->samples[0];
    *pi = li / samples->tda;
    *pj = li % samples->tda;
}

//
//============================================================================
//
void set_channel_sample ( paco_image_st *im, const index_t i, const index_t j, const index_t c, const sample_t val ) {
    assert ( im != NULL );
    gsl_matrix_set ( im->samples[c], i, j, val );
}

//
//============================================================================
//
void set_sample_linear ( paco_image_st *im, const index_t idx, const index_t c, const sample_t val ) {
    assert ( im != NULL );
    im->samples[c]->data[idx] = val;
}


//
//============================================================================
//
sample_t get_sample_linear ( const paco_image_st *im, const index_t idx, const index_t c ) {
    assert ( im != NULL );
    return im->samples[c]->data[idx];
}

//
//============================================================================
//
gsl_matrix *get_samples ( const paco_image_st *im, const index_t c ) {
    assert ( im != NULL );
    assert ( c < get_nchannels ( im ) );
    return im->samples[c];
}

//
//============================================================================
//
index_t get_ncols ( const paco_image_st *im ) {
    assert ( im != NULL );
    return im->samples[0]->size2;
}

//
//============================================================================
//
index_t get_nrows ( const paco_image_st *im ) {
    return im->samples[0]->size1;
}

//
//============================================================================
//
index_t get_nsamples ( const paco_image_st *im ) {
    return im->samples[0]->size1 * im->samples[0]->size2;
}

//
//============================================================================
//

index_t get_nchannels ( const paco_image_st *im ) {
    return strlen ( im->colorspace );
}

//
//============================================================================
//

index_t get_bitd ( const paco_image_st *im ) {
    return im->bitd;
}

//
//============================================================================
//

const char *get_colorspace ( const paco_image_st *im ) {
    return im->colorspace;
}

//
//============================================================================
//

/** \return single-letter name of channel c*/
index_t get_channel_name ( const paco_image_st *im, index_t c ) {
    return im->colorspace[c];
}

/** \return samples matrix for channel number c */
gsl_matrix *get_channel_samples ( paco_image_st *im, index_t c ) {
    return im->samples[c];
}
//
//============================================================================
//
//
index_t compute_padded_size ( const index_t n, const index_t w, const index_t s ) {
    // n <= (s*(g-1) + w) -> (n-w)/s <= g-1 -> (n-w)/s + 1 <= g -> g = ceil((n-w)/s + 1) = ceil((n-w)/s) + 1
    // N = s*(g-1) + w = s*[ceil(n-w)/s + 1 - 1] + w = s*ceil((n-w)/s) + w
    if ( ( ( n - w ) % s ) == 0 ) { // fits exactly:
        return n; //no need for padding
    } else { // otherwise
        return s * ( index_t ) ceil ( ( double ) ( n - w ) / ( double ) s ) + w;
    }
}

//
//============================================================================
//
paco_image_st *pad_image_mirror ( paco_image_st *img, const index_t padded_width, const index_t padded_height ) {

    const index_t height = get_nrows ( img );
    const index_t width = get_ncols ( img );

    assert ( padded_height >= height );
    assert ( padded_width >= width );

    if ( ( height == padded_height ) && ( width == padded_width ) ) {
        return img;
    }

    const index_t nchan = get_nchannels ( img );

    paco_image_st *img_pad = paco_image_alloc ( padded_height, padded_width, img->colorspace );

    for ( index_t c = 0; c < nchan; c++ ) {

        gsl_matrix *dst = get_samples ( img_pad, c );
        gsl_matrix *src = get_samples ( img, c );

        //copy image
        for ( int i = 0; i < height; ++i ) {
            for ( int j = 0; j < width; j++ ) {
                gsl_matrix_set ( dst, i, j, gsl_matrix_get ( src, i, j ) );
            }
        }

        //mirror
        //Lower side
        for ( int i = height; i < padded_height; ++i ) {
            for ( int j = 0; j < width; ++j ) {
                gsl_matrix_set ( dst, i, j, gsl_matrix_get ( src, height * 2 - 1 - i, j ) );
            }
        }

        //Right side
        for ( int i = 0; i < height; ++i ) {
            for ( int j = width; j < padded_width; ++j ) {
                gsl_matrix_set ( dst, i, j, gsl_matrix_get ( src, i, width * 2 - 1 - j ) );
            }
        }

        //Corner
        for ( int i = height; i < padded_height; ++i ) {
            for ( int j = width; j < padded_width; ++j ) {
                gsl_matrix_set ( dst, i, j, gsl_matrix_get ( src, height * 2 - 1 - i, width * 2 - 1 - j ) );
            }
        }
    }

    return img_pad;
}
//
//============================================================================
//

paco_image_st *unpad_image ( paco_image_st *img, const index_t original_width, const index_t original_height ) {
    const index_t height = get_nrows ( img );
    const index_t width = get_ncols ( img );
    paco_image_st *img_unpad;
    assert ( original_height <= height );
    assert ( original_width <= width );

    if ( ( height == original_height ) && ( width == original_width ) ) {
        return img;
    }

    img_unpad = paco_image_alloc ( original_height, original_width, img->colorspace );
    paco_info ( "Cropping image to original size \n" );
    const index_t nchan = get_nchannels ( img );

    for ( index_t c = 0; c < nchan; c++ ) {
        gsl_matrix *src = get_samples ( img, c );
        gsl_matrix *dst = get_samples ( img_unpad, c );

        //copy image
        for ( int i = 0; i < original_height; ++i ) {
            for ( int j = 0; j < original_width; ++j ) {
                gsl_matrix_set ( dst, i, j, gsl_matrix_get ( src, i, j ) );
            }
        }
    }

    return img_unpad;
}
//
//============================================================================
//
paco_image_st *pad_image_zeros ( paco_image_st *img, const index_t padded_width, const index_t padded_height ) {

    const index_t height = get_nrows ( img );
    const index_t width = get_ncols ( img );
    assert ( padded_height >= height );
    assert ( padded_width >= width );

    //paco_debug("channels : %d\n",img->nchan);

    if ( ( height == padded_height ) && ( width == padded_width ) ) {
        return img;
    }

    paco_info ( "padding mask %d x %d => padded %d %d\n", height, width,  padded_height, padded_width );
    /* reallocate the memory of the block */
    paco_image_st *img_pad = paco_image_alloc ( padded_height, padded_width, img->colorspace );
    paco_image_clear ( img_pad );
    const index_t nchan = get_nchannels ( img );

    for ( index_t c = 0; c < nchan; c++ ) {
        gsl_matrix *aux = get_samples ( img_pad, c );

        //copy image
        for ( int i = 0; i < height; ++i ) {
            for ( int j = 0; j < width; ++j ) {
                gsl_matrix_set ( aux, i, j, gsl_matrix_get ( aux, i, j ) );
            }
        }
    }

    return img_pad;
}

//
//============================================================================
//
paco_image_st *paco_create_compatible ( const paco_image_st *src ) {
    assert ( src != NULL );
    paco_image_st *out = paco_image_alloc ( get_nrows ( src ), get_ncols ( src ), get_colorspace ( src ) );
    out->bitd = src->bitd;
    strncpy ( out->colorspace, src->colorspace, 5 );
    return out;
}

//
//============================================================================
//
paco_image_st *paco_create_copy ( const paco_image_st *src ) {
    assert ( src != NULL );
    paco_image_st *out = paco_image_alloc ( get_nrows ( src ), get_ncols ( src ), get_colorspace ( src ) );
    const index_t nchan = get_nchannels ( src );

    for ( index_t c = 0; c < nchan; c++ ) {
        gsl_matrix_memcpy ( out->samples[c], src->samples[c] );
    }

    strncpy ( out->colorspace, src->colorspace, 5 );
    out->bitd = src->bitd;
    return out;
}

//
//============================================================================
//
void paco_image_clear ( paco_image_st *img ) {
    const index_t nchan = get_nchannels ( img );

    for ( index_t c = 0; c < nchan; c++ ) {
        gsl_matrix_set_zero ( get_samples ( img, c ) );
    }
}

