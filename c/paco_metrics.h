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
 * \file paco_metrics.h
 * \brief implements some standard image quality metrics
 */
#include "paco_types.h"

/**
 * Allows fine-grain control of the MS-SSIM algorithm.
 */
struct iqa_ms_ssim_args {
    int wang;             /**< 1=original algorithm by Wang, et al. 0=MS-SSIM* by Rouse/Hemami (default). */
    int gaussian;         /**< 1=11x11 Gaussian window (default). 0=8x8 linear window. */
    int scales;           /**< Number of scaled images to use. Default is 5. */
    const double *alphas;  /**< Pointer to array of alpha values for each scale. Required if 'scales' isn't 5. */
    const double *betas;   /**< Pointer to array of beta values for each scale. Required if 'scales' isn't 5. */
    const double *gammas;  /**< Pointer to array of gamma values for each scale. Required if 'scales' isn't 5. */
};
/**
 * Allows fine-grain control of the SSIM algorithm.
 */
struct iqa_ssim_args {
    double alpha;    /**< luminance exponent */
    double beta;     /**< contrast exponent */
    double gamma;    /**< structure exponent */
    int L;          /**< dynamic range (2^8 - 1)*/
    double K1;       /**< stabilization constant 1 */
    double K2;       /**< stabilization constant 2 */
    int f;          /**< scale factor. 0=default scaling, 1=no scaling */
};

/**
 * Calculates the Multi-Scale Structural SIMilarity between 2 equal-sized 8-bit
 * images. The default algorithm is MS-SSIM* proposed by Rouse/Hemami 2008.
 * Restricted to pixels in mask.
 *
 * See https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf and
 * http://foulard.ece.cornell.edu/publications/dmr_hvei2008_paper.pdf
 *
 * @note 1. The images must have the same width, height, and stride.
 * @note 2. The minimum image width or height is 2^(scales-1) * filter, where 'filter' is 11
 * if a Gaussian window is being used, or 9 otherwise.
 * @param ref Original reference image
 * @param cmp inpainted image
 * @param mask inpainting mask
 * @param w Width of the images.
 * @param h Height of the images.
 * @param stride The length (in bytes) of each horizontal line in the image.
 *               This may be different from the image width.
 * @param args Optional MS-SSIM arguments for fine control of the algorithm. 0
 * for defaults. Defaults are wang=0, scales=5, gaussian=1.
 * @return The mean MS-SSIM over the entire image, or INFINITY if error.
 */
double iqa_ms_ssim ( sample_t *ref, sample_t *cmp, sample_t *mask, int w, int h, int stride, const struct iqa_ms_ssim_args *args );

/**
 * Calculates the Mean Square Error (MSE) between 2 equal-sized images
 * considering missing pixels only
 * @param ref Original reference image
 * @param cmp inpainted image
 * @param mask inpainting mask
 * @return MSE between images on masked pixels
 */
double rmse_partial ( const sample_t *ref, const sample_t *cmp, const sample_t *mask, const index_t n );
