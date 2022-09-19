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
#include "paco_types.h"
#include "paco_metrics.h"
#include<math.h>

double rmse_partial ( const sample_t *ref, const sample_t *cmp, const sample_t *mask, const index_t n ) {
    double mse = 0;
    double mis = 0;

    for ( index_t i = 0; i < n; ++i ) {
        if ( !mask || mask[i] ) {
            const double dif = ref[i] - cmp[i];
            mse += dif * dif;
            mis++;
        }
    }

    mse /= mis;
    return sqrt ( mse );
}

//--------------------------------------------------------------

#ifdef WIN32
#include <windows.h>
#ifndef INFINITY
#define INFINITY (double)HUGE_VAL /**< Defined in C99 (Windows is C89) */
#endif /*INFINITY*/
#ifndef NAN
static const unsigned long __nan[2] = {0xffffffff, 0x7fffffff};
#define NAN (*(const double *) __nan) /**< Defined in C99 (Windows is C99) */
#endif
#endif

//--------------------------------------------------------------
// MATH UTILS
//--------------------------------------------------------------

//static  int _round(double a);
//static  int _max(int x, int y);
//static  int _min(int x, int y);
/**
 * Compares 2 doubles to the specified digit of precision.
 * @return 0 if equal, 1 otherwise.
 */
//static  int _cmp_double(double a, double b, int digits);
/**
 * Compares 2 matrices with the specified precision. 'b' is assumed to be the
 * same size as 'a' or smaller.
 * @return 0 if equal, 1 otherwise
 */
//static  int _matrix_cmp(const double *a, const double *b, int w, int h, int digits);


//--------------------------------------------------------------
// CONVOLVE
//--------------------------------------------------------------

typedef double ( *_iqa_get_pixel ) ( const double *img, int w, int h, int x, int y, double bnd_const );

/** Out-of-bounds array values are a mirrored reflection of the border values*/
static double KBND_SYMMETRIC ( const double *img, int w, int h, int x, int y, double bnd_const );
/** Out-of-bounds array values are set to the nearest border value */
//static double KBND_REPLICATE(const double *img, int w, int h, int x, int y, double bnd_const);
/** Out-of-bounds array values are set to 'bnd_const' */
//static double KBND_CONSTANT(const double *img, int w, int h, int x, int y, double bnd_const);


/** Defines a convolution kernel */
struct _kernel {
    double *kernel;          /**< Pointer to the kernel values */
    int w;                  /**< The kernel width */
    int h;                  /**< The kernel height */
    int normalized;         /**< 1 if the kernel values add up to 1. 0 otherwise */
    _iqa_get_pixel bnd_opt; /**< Defines how out-of-bounds image values are handled */
    double bnd_const;        /**< If 'bnd_opt' is KBND_CONSTANT, this specifies the out-of-bounds value */
};

/**
 * @brief Applies the specified kernel to the image.
 * The kernel will be applied to all areas where it fits completely within
 * the image. The resulting image will be smaller by half the kernel width
 * and height (w - kw/2 and h - kh/2).
 *
 * @param img Image to modify
 * @param w Image width
 * @param h Image height
 * @param k The kernel to apply
 * @param result Buffer to hold the resulting image ((w-kw)*(h-kh), where kw
 *               and kh are the kernel width and height). If 0, the result
 *               will be written to the original image buffer.
 * @param rw Optional. The width of the resulting image will be stored here.
 * @param rh Optional. The height of the resulting image will be stored here.
 */
static void _iqa_convolve ( double *img, int w, int h, const struct _kernel *k, double *result, int *rw, int *rh );

/**
 * The same as _iqa_convolve() except the kernel is applied to the entire image.
 * In other words, the kernel is applied to all areas where the top-left corner
 * of the kernel is in the image. Out-of-bound pixel value (off the right and
 * bottom edges) are chosen based on the 'bnd_opt' and 'bnd_const' members of
 * the kernel structure. The resulting array is the same size as the input
 * image.
 *
 * @param img Image to modify
 * @param w Image width
 * @param h Image height
 * @param k The kernel to apply
 * @param result Buffer to hold the resulting image ((w-kw)*(h-kh), where kw
 *               and kh are the kernel width and height). If 0, the result
 *               will be written to the original image buffer.
 * @return 0 if successful. Non-zero otherwise.
 */
//static int _iqa_img_filter(double *img, int w, int h, const struct _kernel *k, double *result);

/**
 * Returns the filtered version of the specified pixel. If no kernel is given,
 * the raw pixel value is returned.
 *
 * @param img Source image
 * @param w Image width
 * @param h Image height
 * @param x The x location of the pixel to filter
 * @param y The y location of the pixel to filter
 * @param k Optional. The convolution kernel to apply to the pixel.
 * @param kscale The scale of the kernel (for normalization). 1 for normalized
 *               kernels. Required if 'k' is not null.
 * @return The filtered pixel value.
 */
static double _iqa_filter_pixel ( const double *img, int w, int h, int x, int y, const struct _kernel *k, const double kscale );


//--------------------------------------------------------------
// DECIMATE
//--------------------------------------------------------------

/**
 * @brief Downsamples (decimates) an image.
 *
 * @param img Image to modify
 * @param w Image width
 * @param h Image height
 * @param factor Decimation factor
 * @param k The kernel to apply (e.g. low-pass filter). Can be 0.
 * @param result Buffer to hold the resulting image (w/factor*h/factor). If 0,
 *               the result will be written to the original image buffer.
 * @param rw Optional. The width of the resulting image will be stored here.
 * @param rh Optional. The height of the resulting image will be stored here.
 * @return 0 on success.
 */
static int _iqa_decimate ( double *img, int w, int h, int factor, const struct _kernel *k, double *result, int *rw, int *rh );


//--------------------------------------------------------------
// SSIM
//--------------------------------------------------------------

/* Holds intermediate SSIM values for map-reduce operation. */
struct _ssim_int {
    double l;
    double c;
    double s;
    double m; // mask
};

/* Defines the pointers to the map-reduce functions. */
typedef int ( *_map ) ( const struct _ssim_int *, void * );
typedef double ( *_reduce ) ( int, int, void * );

/* Arguments for map-reduce. The 'context' is user-defined. */
struct _map_reduce {
    _map map;
    _reduce reduce;
    void *context;
};

/**
 * Private method that calculates the SSIM value on a pre-processed image.
 *
 * The input images must have stride==width. This method does not scale.
 *
 * @note Image buffers are modified.
 *
 * Map-reduce is used for doing the final SSIM calculation. The map function is
 * called for every pixel, and the reduce is called at the end. The context is
 * caller-defined and *not* modified by this method.
 *
 * @param ref Original reference image
 * @param cmp Distorted image
 * @param w Width of the images
 * @param h Height of the images
 * @param k The kernel used as the window function
 * @param mr Optional map-reduce functions to use to calculate SSIM. Required
 *           if 'args' is not null. Ignored if 'args' is null.
 * @param args Optional SSIM arguments for fine control of the algorithm. 0 for defaults.
 *             Defaults are a=b=g=1.0, L=255, K1=0.01, K2=0.03
 * @return The mean SSIM over the entire image (MSSIM), or NAN if error.
 */
static double _iqa_ssim ( double *ref, double *cmp, double *mask, int w, int h, const struct _kernel *k, const struct _map_reduce *mr, const struct iqa_ssim_args *args );

//--------------------------------------------------------------
// MS-SSIM
//--------------------------------------------------------------
//
// (nothing to declare)
//

//
//=======================================================================
// IMPLEMENTATION
//=======================================================================
//
//--------------------------------------------------------------
// CONVOLVE
//--------------------------------------------------------------

static double KBND_SYMMETRIC ( const double *img, int w, int h, int x, int y, double bnd_const ) {
    if ( x < 0 ) {
        x = -1 - x;
    } else if ( x >= w ) {
        x = ( w - ( x - w ) ) - 1;
    }

    if ( y < 0 ) {
        y = -1 - y;
    } else if ( y >= h ) {
        y = ( h - ( y - h ) ) - 1;
    }

    return img[y * w + x];
}

//--------------------------------------------------------------

static double _calc_scale ( const struct _kernel *k ) {
    int ii, k_len;
    double sum = 0.0;

    if ( k->normalized ) {
        return 1.0f;
    } else {
        k_len = k->w * k->h;

        for ( ii = 0; ii < k_len; ++ii ) {
            sum += k->kernel[ii];
        }

        if ( sum != 0.0 ) {
            return ( double ) ( 1.0 / sum );
        }

        return 1.0f;
    }
}

//--------------------------------------------------------------

static void _iqa_convolve ( double *img, int w, int h, const struct _kernel *k, double *result, int *rw, int *rh ) {
    int x, y, kx, ky, u, v;
    int uc = k->w / 2;
    int vc = k->h / 2;
    int kw_even = ( k->w & 1 ) ? 0 : 1;
    int kh_even = ( k->h & 1 ) ? 0 : 1;
    int dst_w = w - k->w + 1;
    int dst_h = h - k->h + 1;
    int img_offset, k_offset;
    double sum;
    double scale, *dst = result;

    if ( !dst ) {
        dst = img;  /* Convolve in-place */
    }

    /* Kernel is applied to all positions where the kernel is fully contained
     * in the image */
    scale = _calc_scale ( k );

    for ( y = 0; y < dst_h; ++y ) {
        for ( x = 0; x < dst_w; ++x ) {
            sum = 0.0;
            k_offset = 0;
            ky = y + vc;
            kx = x + uc;

            for ( v = -vc; v <= vc - kh_even; ++v ) {
                img_offset = ( ky + v ) * w + kx;

                for ( u = -uc; u <= uc - kw_even; ++u, ++k_offset ) {
                    sum += img[img_offset + u] * k->kernel[k_offset];
                }
            }

            dst[y * dst_w + x] = ( double ) ( sum * scale );
        }
    }

    if ( rw ) {
        *rw = dst_w;
    }

    if ( rh ) {
        *rh = dst_h;
    }
}


//--------------------------------------------------------------

static double _iqa_filter_pixel ( const double *img, int w, int h, int x, int y, const struct _kernel *k, const double kscale ) {
    int u, v, uc, vc;
    int kw_even, kh_even;
    int x_edge_left, x_edge_right, y_edge_top, y_edge_bottom;
    int edge, img_offset, k_offset;
    double sum;

    if ( !k ) {
        return img[y * w + x];
    }

    uc = k->w / 2;
    vc = k->h / 2;
    kw_even = ( k->w & 1 ) ? 0 : 1;
    kh_even = ( k->h & 1 ) ? 0 : 1;
    x_edge_left  = uc;
    x_edge_right = w - uc;
    y_edge_top = vc;
    y_edge_bottom = h - vc;

    edge = 0;

    if ( x < x_edge_left || y < y_edge_top || x >= x_edge_right || y >= y_edge_bottom ) {
        edge = 1;
    }

    sum = 0.0;
    k_offset = 0;

    for ( v = -vc; v <= vc - kh_even; ++v ) {
        img_offset = ( y + v ) * w + x;

        for ( u = -uc; u <= uc - kw_even; ++u, ++k_offset ) {
            if ( !edge ) {
                sum += img[img_offset + u] * k->kernel[k_offset];
            } else {
                sum += k->bnd_opt ( img, w, h, x + u, y + v, k->bnd_const ) * k->kernel[k_offset];
            }
        }
    }

    return ( double ) ( sum * kscale );
}


//
//--------------------------------------------------------------
// DECIMATE
//--------------------------------------------------------------
//
static int _iqa_decimate ( double *img, int w, int h, int factor, const struct _kernel *k, double *result, int *rw, int *rh ) {
    int x, y;
    int sw = w / factor + ( w & 1 );
    int sh = h / factor + ( h & 1 );
    int dst_offset;
    double *dst = img;

    if ( result ) {
        dst = result;
    }

    /* Downsample */
    for ( y = 0; y < sh; ++y ) {
        dst_offset = y * sw;

        for ( x = 0; x < sw; ++x, ++dst_offset ) {
            dst[dst_offset] = _iqa_filter_pixel ( img, w, h, x * factor, y * factor, k, 1.0f );
        }
    }

    if ( rw ) {
        *rw = sw;
    }

    if ( rh ) {
        *rh = sh;
    }

    return 0;
}

//
//--------------------------------------------------------------
// SSIM
//--------------------------------------------------------------
//
/*
 * Equal weight square window.
 * Each pixel is equally weighted (1/64) so that SUM(x) = 1.0
 */
#define SQUARE_LEN 8
static const double g_square_window[SQUARE_LEN][SQUARE_LEN] = {
    {0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f},
    {0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f},
    {0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f},
    {0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f},
    {0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f},
    {0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f},
    {0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f},
    {0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f, 0.015625f},
};

//--------------------------------------------------------------
/*
 * Circular-symmetric Gaussian weighting.
 * h(x,y) = hg(x,y)/SUM(SUM(hg)) , for normalization to 1.0
 * hg(x,y) = e^( -0.5*( (x^2+y^2)/sigma^2 ) ) , where sigma was 1.5
 */
#define GAUSSIAN_LEN 11
static const double g_gaussian_window[GAUSSIAN_LEN][GAUSSIAN_LEN] = {
    {0.000001f, 0.000008f, 0.000037f, 0.000112f, 0.000219f, 0.000274f, 0.000219f, 0.000112f, 0.000037f, 0.000008f, 0.000001f},
    {0.000008f, 0.000058f, 0.000274f, 0.000831f, 0.001619f, 0.002021f, 0.001619f, 0.000831f, 0.000274f, 0.000058f, 0.000008f},
    {0.000037f, 0.000274f, 0.001296f, 0.003937f, 0.007668f, 0.009577f, 0.007668f, 0.003937f, 0.001296f, 0.000274f, 0.000037f},
    {0.000112f, 0.000831f, 0.003937f, 0.011960f, 0.023294f, 0.029091f, 0.023294f, 0.011960f, 0.003937f, 0.000831f, 0.000112f},
    {0.000219f, 0.001619f, 0.007668f, 0.023294f, 0.045371f, 0.056662f, 0.045371f, 0.023294f, 0.007668f, 0.001619f, 0.000219f},
    {0.000274f, 0.002021f, 0.009577f, 0.029091f, 0.056662f, 0.070762f, 0.056662f, 0.029091f, 0.009577f, 0.002021f, 0.000274f},
    {0.000219f, 0.001619f, 0.007668f, 0.023294f, 0.045371f, 0.056662f, 0.045371f, 0.023294f, 0.007668f, 0.001619f, 0.000219f},
    {0.000112f, 0.000831f, 0.003937f, 0.011960f, 0.023294f, 0.029091f, 0.023294f, 0.011960f, 0.003937f, 0.000831f, 0.000112f},
    {0.000037f, 0.000274f, 0.001296f, 0.003937f, 0.007668f, 0.009577f, 0.007668f, 0.003937f, 0.001296f, 0.000274f, 0.000037f},
    {0.000008f, 0.000058f, 0.000274f, 0.000831f, 0.001619f, 0.002021f, 0.001619f, 0.000831f, 0.000274f, 0.000058f, 0.000008f},
    {0.000001f, 0.000008f, 0.000037f, 0.000112f, 0.000219f, 0.000274f, 0.000219f, 0.000112f, 0.000037f, 0.000008f, 0.000001f},
};


//--------------------------------------------------------------

/* Forward declarations. */
static double _calc_luminance ( double, double, double, double );
static double _calc_contrast ( double, double, double, double, double );
static double _calc_structure ( double, double, double, double, double, double );

/*
 * SSIM(x,y)=(2*ux*uy + C1)*(2sxy + C2) / (ux^2 + uy^2 + C1)*(sx^2 + sy^2 + C2)
 * where,
 *  ux = SUM(w*x)
 *  sx = (SUM(w*(x-ux)^2)^0.5
 *  sxy = SUM(w*(x-ux)*(y-uy))
 *
 * Returns mean SSIM. MSSIM(X,Y) = 1/M * SUM(SSIM(x,y))
 */

//--------------------------------------------------------------

/* _iqa_ssim */
static double _iqa_ssim ( double *ref, double *cmp, double *mask, int w, int h, const struct _kernel *k,
                          const struct _map_reduce *mr, const struct iqa_ssim_args *args ) {
    double alpha = 1.0f, beta = 1.0f, gamma = 1.0f;
    int L = 255;
    double K1 = 0.01f, K2 = 0.03f;
    double C1, C2, C3;
    int x, y, offset;
    double *ref_mu, *cmp_mu, *ref_sigma_sqd, *cmp_sigma_sqd, *sigma_both;
    double ssim_sum, numerator, denominator;
    double luminance_comp, contrast_comp, structure_comp, sigma_root;
    struct _ssim_int sint;

    /* Initialize algorithm parameters */
    if ( args ) {
        if ( !mr ) {
            return INFINITY;
        }

        alpha = args->alpha;
        beta  = args->beta;
        gamma = args->gamma;
        L     = args->L;
        K1    = args->K1;
        K2    = args->K2;
    }

    C1 = ( K1 * L ) * ( K1 * L );
    C2 = ( K2 * L ) * ( K2 * L );
    C3 = C2 / 2.0f;

    ref_mu = ( double * ) malloc ( w * h * sizeof ( double ) );
    cmp_mu = ( double * ) malloc ( w * h * sizeof ( double ) );
    ref_sigma_sqd = ( double * ) malloc ( w * h * sizeof ( double ) );
    cmp_sigma_sqd = ( double * ) malloc ( w * h * sizeof ( double ) );
    sigma_both = ( double * ) malloc ( w * h * sizeof ( double ) );

    if ( !ref_mu || !cmp_mu || !ref_sigma_sqd || !cmp_sigma_sqd || !sigma_both ) {
        if ( ref_mu ) {
            free ( ref_mu );
        }

        if ( cmp_mu ) {
            free ( cmp_mu );
        }

        if ( ref_sigma_sqd ) {
            free ( ref_sigma_sqd );
        }

        if ( cmp_sigma_sqd ) {
            free ( cmp_sigma_sqd );
        }

        if ( sigma_both ) {
            free ( sigma_both );
        }

        return INFINITY;
    }

    /* Calculate mean */
    _iqa_convolve ( ref, w, h, k, ref_mu, 0, 0 );
    _iqa_convolve ( cmp, w, h, k, cmp_mu, 0, 0 );

    for ( y = 0; y < h; ++y ) {
        offset = y * w;

        for ( x = 0; x < w; ++x, ++offset ) {
            ref_sigma_sqd[offset] = ref[offset] * ref[offset];
            cmp_sigma_sqd[offset] = cmp[offset] * cmp[offset];
            sigma_both[offset] = ref[offset] * cmp[offset];
        }
    }

    /* Calculate sigma */
    _iqa_convolve ( ref_sigma_sqd, w, h, k, 0, 0, 0 );
    _iqa_convolve ( cmp_sigma_sqd, w, h, k, 0, 0, 0 );
    _iqa_convolve ( sigma_both, w, h, k, 0, &w, &h ); /* Update the width and height */

    /* The convolution results are smaller by the kernel width and height */
    for ( y = 0; y < h; ++y ) {
        offset = y * w;

        for ( x = 0; x < w; ++x, ++offset ) {
            ref_sigma_sqd[offset] -= ref_mu[offset] * ref_mu[offset];
            cmp_sigma_sqd[offset] -= cmp_mu[offset] * cmp_mu[offset];
            sigma_both[offset] -= ref_mu[offset] * cmp_mu[offset];
        }
    }

    ssim_sum = 0.0;

    for ( y = 0; y < h; ++y ) {
        offset = y * w;

        for ( x = 0; x < w; ++x, ++offset ) {

            if ( !args ) {
                /* The default case */
                numerator   = ( 2.0 * ref_mu[offset] * cmp_mu[offset] + C1 ) * ( 2.0 * sigma_both[offset] + C2 );
                denominator = ( ref_mu[offset] * ref_mu[offset] + cmp_mu[offset] * cmp_mu[offset] + C1 ) *
                              ( ref_sigma_sqd[offset] + cmp_sigma_sqd[offset] + C2 );
                ssim_sum += numerator / denominator;
            } else {
                /* User tweaked alpha, beta, or gamma */

                /* passing a negative number to sqrt() cause a domain error */
                if ( ref_sigma_sqd[offset] < 0.0f ) {
                    ref_sigma_sqd[offset] = 0.0f;
                }

                if ( cmp_sigma_sqd[offset] < 0.0f ) {
                    cmp_sigma_sqd[offset] = 0.0f;
                }

                sigma_root = sqrt ( ref_sigma_sqd[offset] * cmp_sigma_sqd[offset] );

                luminance_comp = _calc_luminance ( ref_mu[offset], cmp_mu[offset], C1, alpha );
                contrast_comp  = _calc_contrast ( sigma_root, ref_sigma_sqd[offset], cmp_sigma_sqd[offset], C2, beta );
                structure_comp = _calc_structure ( sigma_both[offset], sigma_root, ref_sigma_sqd[offset], cmp_sigma_sqd[offset], C3, gamma );

                sint.l = luminance_comp;
                sint.c = contrast_comp;
                sint.s = structure_comp;
                sint.m = mask ? mask[offset] : 1;

                if ( mr->map ( &sint, mr->context ) ) {
                    return INFINITY;
                }
            }
        }
    }

    free ( ref_mu );
    free ( cmp_mu );
    free ( ref_sigma_sqd );
    free ( cmp_sigma_sqd );
    free ( sigma_both );

    if ( !args ) {
        return ( double ) ( ssim_sum / ( double ) ( w * h ) );
    }

    return mr->reduce ( w, h, mr->context );
}

//--------------------------------------------------------------

/* _ssim_map */
#if 0
static int _ssim_map ( const struct _ssim_int *si, void *ctx ) {
    double *ssim_sum = ( sample_t ) ctx;
    *ssim_sum += si->l * si->c * si->s;
    return 0;
}
#endif
//--------------------------------------------------------------

/* _ssim_reduce */
#if 0
static double _ssim_reduce ( int w, int h, void *ctx ) {
    double *ssim_sum = ( sample_t ) ctx;
    return ( double ) ( *ssim_sum / ( double ) ( w * h ) );
}
#endif
//--------------------------------------------------------------

/* _calc_luminance */
static double _calc_luminance ( double mu1, double mu2, double C1, double alpha ) {
    double result;
    double sign;

    /* For MS-SSIM* */
    if ( C1 == 0 && mu1 * mu1 == 0 && mu2 * mu2 == 0 ) {
        return 1.0;
    }

    result = ( 2.0 * mu1 * mu2 + C1 ) / ( mu1 * mu1 + mu2 * mu2 + C1 );

    if ( alpha == 1.0f ) {
        return result;
    }

    sign = result < 0.0 ? -1.0f : 1.0f;
    return sign * pow ( fabs ( result ), ( double ) alpha );
}

//--------------------------------------------------------------

/* _calc_contrast */
static double _calc_contrast ( double sigma_comb_12, double sigma1_sqd, double sigma2_sqd, double C2, double beta ) {
    double result;
    double sign;

    /* For MS-SSIM* */
    if ( C2 == 0 && sigma1_sqd + sigma2_sqd == 0 ) {
        return 1.0;
    }

    result = ( 2.0 * sigma_comb_12 + C2 ) / ( sigma1_sqd + sigma2_sqd + C2 );

    if ( beta == 1.0f ) {
        return result;
    }

    sign = result < 0.0 ? -1.0f : 1.0f;
    return sign * pow ( fabs ( result ), ( double ) beta );
}

//--------------------------------------------------------------

/* _calc_structure */
static double _calc_structure ( double sigma_12, double sigma_comb_12, double sigma1, double sigma2, double C3, double gamma ) {
    double result;
    double sign;

    /* For MS-SSIM* */
    if ( C3 == 0 && sigma_comb_12 == 0 ) {
        if ( sigma1 == 0 && sigma2 == 0 ) {
            return 1.0;
        } else if ( sigma1 == 0 || sigma2 == 0 ) {
            return 0.0;
        }
    }

    result = ( sigma_12 + C3 ) / ( sigma_comb_12 + C3 );

    if ( gamma == 1.0f ) {
        return result;
    }

    sign = result < 0.0 ? -1.0f : 1.0f;
    return sign * pow ( fabs ( result ), ( double ) gamma );
}

//
//--------------------------------------------------------------
// MS-SSIM
//--------------------------------------------------------------
//

/* Default number of scales */
#define SCALES  4

/* Low-pass filter for down-sampling (9/7 biorthogonal wavelet filter) */
#define LPF_LEN 9
static const double g_lpf[LPF_LEN][LPF_LEN] = {
    { 0.000714f, -0.000450f, -0.002090f, 0.007132f, 0.016114f, 0.007132f, -0.002090f, -0.000450f, 0.000714f},
    {-0.000450f, 0.000283f, 0.001316f, -0.004490f, -0.010146f, -0.004490f, 0.001316f, 0.000283f, -0.000450f},
    {-0.002090f, 0.001316f, 0.006115f, -0.020867f, -0.047149f, -0.020867f, 0.006115f, 0.001316f, -0.002090f},
    { 0.007132f, -0.004490f, -0.020867f, 0.071207f, 0.160885f, 0.071207f, -0.020867f, -0.004490f, 0.007132f},
    { 0.016114f, -0.010146f, -0.047149f, 0.160885f, 0.363505f, 0.160885f, -0.047149f, -0.010146f, 0.016114f},
    { 0.007132f, -0.004490f, -0.020867f, 0.071207f, 0.160885f, 0.071207f, -0.020867f, -0.004490f, 0.007132f},
    {-0.002090f, 0.001316f, 0.006115f, -0.020867f, -0.047149f, -0.020867f, 0.006115f, 0.001316f, -0.002090f},
    {-0.000450f, 0.000283f, 0.001316f, -0.004490f, -0.010146f, -0.004490f, 0.001316f, 0.000283f, -0.000450f},
    { 0.000714f, -0.000450f, -0.002090f, 0.007132f, 0.016114f, 0.007132f, -0.002090f, -0.000450f, 0.000714f},
};

/* Alpha, beta, and gamma values for each scale */
static double g_alphas[] = { 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1333f };
static double g_betas[]  = { 0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f };
static double g_gammas[] = { 0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f };

//--------------------------------------------------------------

struct _context {
    double l;  /* Luminance */
    double c;  /* Contrast */
    double s;  /* Structure */
    double count; /* number of pixels counted */
    double alpha;
    double beta;
    double gamma;
};

//--------------------------------------------------------------

/* Called for each pixel */
static int _ms_ssim_map ( const struct _ssim_int *si, void *ctx ) {
    struct _context *ms_ctx = ( struct _context * ) ctx;

    if ( si->m ) { // hard: count every sample that is affected by unknown pixels
        ms_ctx->l += si->l;
        ms_ctx->c += si->c;
        ms_ctx->s += si->s;
        ms_ctx->count ++;
    }

    return 0;
}

//--------------------------------------------------------------

/* Called to calculate the final result */
static double _ms_ssim_reduce ( int w, int h, void *ctx ) {
    //double size = (double)(w*h);
    struct _context *ms_ctx = ( struct _context * ) ctx;
    //printf("count:%f\n",ms_ctx->count);
    ms_ctx->l = pow ( ms_ctx->l / ms_ctx->count, ( double ) ms_ctx->alpha );
    ms_ctx->c = pow ( ms_ctx->c / ms_ctx->count, ( double ) ms_ctx->beta );
    ms_ctx->s = pow ( fabs ( ms_ctx->s / ms_ctx->count ), ( double ) ms_ctx->gamma );
    return ( double ) ( ms_ctx->l * ms_ctx->c * ms_ctx->s );
}

//--------------------------------------------------------------

/* Releases the scaled buffers */
static void _free_buffers ( double **buf, int scales ) {
    int idx;

    for ( idx = 0; idx < scales; ++idx ) {
        free ( buf[idx] );
    }
}

//--------------------------------------------------------------

/* Allocates the scaled buffers. If error, all buffers are free'd */
static int _alloc_buffers ( double **buf, int w, int h, int scales ) {
    int idx;
    int cur_w = w;
    int cur_h = h;

    for ( idx = 0; idx < scales; ++idx ) {
        buf[idx] = ( double * ) malloc ( cur_w * cur_h * sizeof ( double ) );

        if ( !buf[idx] ) {
            _free_buffers ( buf, idx );
            return 1;
        }

        cur_w = cur_w / 2 + ( cur_w & 1 );
        cur_h = cur_h / 2 + ( cur_h & 1 );
    }

    return 0;
}


//--------------------------------------------------------------

/*
 * MS_SSIM(X,Y) = Lm(x,y)^aM * MULT[j=1->M]( Cj(x,y)^bj  *  Sj(x,y)^gj )
 * where,
 *  L = mean
 *  C = variance
 *  S = cross-correlation
 *
 *  b1=g1=0.0448, b2=g2=0.2856, b3=g3=0.3001, b4=g4=0.2363, a5=b5=g5=0.1333
 */
//--------------------------------------------------------------
double iqa_ms_ssim ( sample_t *ref, sample_t *cmp, sample_t *mask, int w, int h,
                     int stride, const struct iqa_ms_ssim_args *args ) {
    int wang = 0;
    int scales = SCALES;
    int gauss = 1;
    const double *alphas = g_alphas, *betas = g_betas, *gammas = g_gammas;
    int idx, x, y, cur_w, cur_h;
    int offset, src_offset;
    double **ref_imgs, **mask_imgs, **cmp_imgs; /* Array of pointers to scaled images */
    double msssim;
    struct _kernel lpf, window;
    struct iqa_ssim_args s_args;
    struct _map_reduce mr;
    struct _context ms_ctx;
    /* Make sure we won't scale below 1x1 */
    cur_w = w;
    cur_h = h;

    for ( idx = 0; idx < scales; ++idx ) {
        if ( gauss ? cur_w < GAUSSIAN_LEN || cur_h < GAUSSIAN_LEN : cur_w < LPF_LEN || cur_h < LPF_LEN ) {
            fprintf ( stderr, "SSIM: invalid image dimensions.\n" );
            return INFINITY;
        }

        cur_w /= 2;
        cur_h /= 2;
    }

    window.kernel = ( double * ) g_square_window;
    window.w = window.h = SQUARE_LEN;
    window.normalized = 1;
    window.bnd_opt = KBND_SYMMETRIC;

    if ( gauss ) {
        window.kernel = ( double * ) g_gaussian_window;
        window.w = window.h = GAUSSIAN_LEN;
    }

    mr.map     = _ms_ssim_map;
    mr.reduce  = _ms_ssim_reduce;

    /* Allocate the scaled image buffers */
    ref_imgs = ( double ** ) malloc ( scales * sizeof ( double * ) );
    cmp_imgs = ( double ** ) malloc ( scales * sizeof ( double * ) );
    mask_imgs = ( double ** ) malloc ( scales * sizeof ( double * ) );

    if ( !ref_imgs || !cmp_imgs || !mask_imgs ) {
        if ( ref_imgs ) {
            free ( ref_imgs );
        }

        if ( cmp_imgs ) {
            free ( cmp_imgs );
        }

        if ( mask_imgs ) {
            free ( mask_imgs );
        }

        fprintf ( stderr, "SSIM: Error allocating scaled images.\n" );
        return INFINITY;
    }

    if ( _alloc_buffers ( ref_imgs, w, h, scales ) ) {
        free ( ref_imgs );
        free ( cmp_imgs );
        free ( mask_imgs );
        fprintf ( stderr, "SSIM: Error allocating ref buffers.\n" );
        return INFINITY;
    }

    if ( _alloc_buffers ( cmp_imgs, w, h, scales ) ) {
        _free_buffers ( ref_imgs, scales );
        free ( ref_imgs );
        free ( cmp_imgs );
        free ( mask_imgs );
        fprintf ( stderr, "SSIM: Error allocating cmp buffers.\n" );
        return INFINITY;
    }

    if ( _alloc_buffers ( mask_imgs, w, h, scales ) ) {
        _free_buffers ( ref_imgs, scales );
        _free_buffers ( cmp_imgs, scales );
        free ( ref_imgs );
        free ( cmp_imgs );
        free ( mask_imgs );
        fprintf ( stderr, "SSIM: Error allocating mask buffers.\n" );
        return INFINITY;
    }

    /* Copy original images into first scale buffer, forcing stride = width. */
    for ( y = 0; y < h; ++y ) {
        src_offset = y * stride;
        offset = y * w;

        for ( x = 0; x < w; ++x, ++offset, ++src_offset ) {
            ref_imgs[0][offset] = ( double ) ref[src_offset];
            cmp_imgs[0][offset] = ( double ) cmp[src_offset];
            mask_imgs[0][offset] = ( double ) ( mask ? mask[src_offset] : 1 );
        }
    }

    /* Create scaled versions of the images */
    cur_w = w;
    cur_h = h;
    lpf.kernel = ( double * ) g_lpf;
    lpf.w = lpf.h = LPF_LEN;
    lpf.normalized = 1;
    lpf.bnd_opt = KBND_SYMMETRIC;

    for ( idx = 1; idx < scales; ++idx ) {
        if ( _iqa_decimate ( ref_imgs[idx - 1], cur_w, cur_h, 2, &lpf, ref_imgs[idx], 0, 0 ) ||
                _iqa_decimate ( mask_imgs[idx - 1], cur_w, cur_h, 2, &lpf, mask_imgs[idx], 0, 0 ) ||
                _iqa_decimate ( cmp_imgs[idx - 1], cur_w, cur_h, 2, &lpf, cmp_imgs[idx], &cur_w, &cur_h ) ) {
            _free_buffers ( ref_imgs, scales );
            _free_buffers ( cmp_imgs, scales );
            _free_buffers ( mask_imgs, scales );
            free ( ref_imgs );
            free ( cmp_imgs );
            free ( mask_imgs );
            fprintf ( stderr, "SSIM: Error constructing downscaled images at scale %d.\n", idx );
            return INFINITY;
        }
    }

    cur_w = w;
    cur_h = h;
    msssim = 1.0;

    for ( idx = 0; idx < scales; ++idx ) {

        ms_ctx.l = 0;
        ms_ctx.c = 0;
        ms_ctx.s = 0;
        ms_ctx.count = 0;
        ms_ctx.alpha = alphas[idx];
        ms_ctx.beta  = betas[idx];
        ms_ctx.gamma = gammas[idx];
        double thisssim;

        if ( !wang ) {
            /* MS-SSIM* (Rouse/Hemami) */
            s_args.alpha = 1.0f;
            s_args.beta  = 1.0f;
            s_args.gamma = 1.0f;
            s_args.K1 = 0.0f; /* Force stabilization constants to 0 */
            s_args.K2 = 0.0f;
            s_args.L  = 255;
            s_args.f  = 1; /* Don't resize */
            mr.context = &ms_ctx;
            thisssim = _iqa_ssim ( ref_imgs[idx], cmp_imgs[idx], mask_imgs[idx], cur_w, cur_h, &window, &mr, &s_args );
            msssim *= thisssim;
        } else {
            /* MS-SSIM (Wang) */
            s_args.alpha = 1.0f;
            s_args.beta  = 1.0f;
            s_args.gamma = 1.0f;
            s_args.K1 = 0.01f;
            s_args.K2 = 0.03f;
            s_args.L  = 255;
            s_args.f  = 1; /* Don't resize */
            mr.context = &ms_ctx;
            const double thisssim = _iqa_ssim ( ref_imgs[idx], cmp_imgs[idx], mask_imgs[idx], cur_w, cur_h, &window, &mr, &s_args );
            msssim *= thisssim;
        }

        //printf("scale %d scale ssim %f total ssim %f\n",idx,thisssim, msssim);

        if ( msssim == INFINITY ) {
            fprintf ( stderr, "Warning infinity in single scale SSI.\n" );
            break;
        }

        cur_w = cur_w / 2 + ( cur_w & 1 );
        cur_h = cur_h / 2 + ( cur_h & 1 );
    }

    _free_buffers ( ref_imgs, scales );
    _free_buffers ( cmp_imgs, scales );
    _free_buffers ( mask_imgs, scales );
    free ( ref_imgs );
    free ( cmp_imgs );
    free ( mask_imgs );

    return msssim;
}
//--------------------------------------------------------------

