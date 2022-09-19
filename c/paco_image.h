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
 * \file paco_image.h
 * \brief Multichannel Image used in PACO.
 * Based on the GNU Scientific Library Matrix structure
 */
#ifndef PACO_IMAGE_H
#define PACO_IMAGE_H

#include "paco_types.h"

#define MAX_CHANNELS 4
#define COLORSPACE_RGB "RGB"
#define COLORSPACE_YUV "YUV"
#define COLORSPACE_GRAY "G"

#define REFLECT(i,n) ( (i) < 0 ? 0 : ( (i) >= (n) ? ((n)-1) : i ) )


typedef struct _paco_image_st_ paco_image_st;
/**
 * allocates a paco image with the given number of rows, columns and channels
 * \param nrows number of rows
 * \param ncols number of columms
 * \param nchan number of channels
 */
paco_image_st *paco_image_alloc ( index_t nrows, index_t ncols, const char *colorspace );

/**
 * frees the data within the structure  passed as an argument (not the structure itself).
 * \param im_st the image struct to be freed.
 */
void paco_image_free ( paco_image_st *im_st );

/**
 * creates an image from given samples and colorspace. For multichannel images,
 * channels should be stored in an interleaved fashion on each row.
 */
paco_image_st *paco_image_from_samples ( const gsl_matrix *samples, const char *colorspace );


/** \return the raw pixel data as a gsl_matrix. DANGEROUS! */
gsl_matrix *get_samples ( const paco_image_st *im, const index_t channel );

/** \return the number of columns in an image */
index_t get_ncols ( const paco_image_st *im );

/** \return the number of rows in an image */
index_t get_nrows ( const paco_image_st *im );

/** \return the total number of samples image */
index_t get_nsamples ( const paco_image_st *im );

/** \return image bit depth*/
index_t get_bitd ( const paco_image_st *im );

/** \return maximum pixel value in image */
sample_t get_maxval ( const paco_image_st *im );

/** \return minimum pixel value in image */
sample_t get_minval ( const paco_image_st *im );

/** \return image channel number*/
const char *get_colorspace ( const paco_image_st *im );

/** \return image channel number*/
index_t get_nchannels ( const paco_image_st *im );

/** \return single-letter name of channel c*/
index_t get_channel_name ( const paco_image_st *im, index_t c );

/** \return samples matrix for channel number c */
gsl_matrix *get_channel_samples ( paco_image_st *im, index_t c );

/** in-place conversion of color samples between color spaces */
void paco_image_convert_colorspace ( paco_image_st *img,  const char *to );

/** normalize image samples to [0,1] */
void paco_image_norm ( paco_image_st *I );

/** squash dynamic range to [0,1], does not normalize */
void paco_image_squash ( paco_image_st *I );

/** clip image samples to [0,1] */
void paco_image_clip ( paco_image_st *I );

//
//============================================================================
//
/**
 * \brief returns the sample value of channel c at index (i,j)
 * \param i the row index
 * \param j the column index
 * \param c the channel index
 * \return the sample value of channel c at index (i,j)
 */
sample_t get_channel_sample ( const paco_image_st *im, const index_t i, const index_t j, const index_t c );


//
//============================================================================
//
/**
 * \brief returns the sample value of a samples matrix at index (i,j).
 * Samples outside of the valid range are reflected frmo the border.
 *
 * \param i the row index
 * \param j the column index
 * \return the sample value at index (i,j)
 */
sample_t get_sample ( const gsl_matrix *im, const int i, const int j );

//
//============================================================================
//
/**
 * \brief returns the sample index corresponding to coordinates (i,j)
 * \param i the row index
 * \param j the column index
 * \return the liner index of  (i,j)
 */
index_t get_linear_index ( const paco_image_st *im, const index_t i, const index_t j );

/**
 * \brief given a linear sample index, return the corresponding row, column and channel index
 * \param li the sample value of channel c at index (i,j)
 * \param[out] i the row index
 * \param[out] j the column index
 * \param[out] c the channel index
 */
void get_sample_coords ( const paco_image_st *im, const index_t li, index_t *pi, index_t *pj );

//
//============================================================================
//
/**
 * \brief returns a pointer to the pixel at index (i,j)
 * \param i the row index
 * \param j the column index
 * \return a pointer to pixel at index (i,j)
 */
sample_t *get_pixel_ptr ( paco_image_st *im, const index_t i, const index_t j, const index_t c );
//
//============================================================================
//
/**
 * \brief returns the linear index of the pixel at index (i,j)
 * \param i the row index
 * \param j the column index
 * \return the linear index of pixel (i,j)
 */
index_t get_pixel_idx ( const paco_image_st *im, const index_t i, const index_t j );
//
//============================================================================
//
/**
 * \brief sets the value of channel c sample at index (i,j)
 * \param i the row index
 * \param j the column index
 * \param c the channel index
 * \param val the new value of the sample
 */
void set_channel_sample ( paco_image_st *im, const index_t i, const index_t j, const index_t c, const sample_t val );

//
//============================================================================
//
/**
 * \brief sets the value of channel c of the pixel indexed linearly by  idx
 * \param idx the linear index of the pixel
 * \param c the channel index
 * \param val the new value of the pixel channel sample c
 */
void set_sample_linear ( paco_image_st *im, const index_t idx, const index_t c, const sample_t val );
/**
 * \brief sets the value of the pixel indexed linearly by  idx
 * \param idx the linear index of the pixel
 * \param val the new value of the pixel channel sample c
 */
void set_pixel_linear ( paco_image_st *im, const index_t idx, const sample_t *val );
//
//============================================================================
//
/**
 * \brief gets the value of channel c of the pixel indexed linearly by  idx
 * \param idx the linear index of the pixel
 * \param c the channel index
 * \param val the new value of the pixel channel sample c
 * return the sample value of channel c at linear idx
 */
sample_t get_sample_linear ( const paco_image_st *im, const index_t idx, const index_t c );
//
//============================================================================
//
/**
 * \brief compute the required padding along a dimension given its size, a patch width and a stride
 * \param size along dimension
 * \param w patch width along this dimension
 * \paran m patch stride along this dimension
 */
index_t compute_padded_size ( const index_t n, const index_t w,  const index_t s );
//
//============================================================================
//
/**
 * \brief resizes the image if necessary to the required padded width and height.
 */
paco_image_st *pad_image_mirror ( paco_image_st *img, const index_t padded_width, const index_t padded_height );
//
//============================================================================
//
/**
 * \brief Applies zero-padding to fit the image (if necessary) to the required width and height.
 */
paco_image_st *pad_image_zeros ( paco_image_st *img, const index_t padded_width, const index_t padded_height );
//
//============================================================================
//
/**
 * \brief undoes the padding of an image.
 */
paco_image_st *unpad_image ( paco_image_st *img, const index_t original_width, const index_t original_height );
//
//=======================================================================================================================================================
//
paco_image_st *paco_create_compatible ( const paco_image_st *src );
//
//============================================================================
//
/// \brief create a new copy of a given image
paco_image_st *paco_create_copy ( const paco_image_st *src );
//
//============================================================================
//
/// \brief copies the contents of an image into another, resizing if necessary
int paco_image_copy ( paco_image_st *dest, const paco_image_st *src );
//
//============================================================================
//

/**
 * set all image samples to 0
 */
void paco_image_clear ( paco_image_st *img );

#endif
