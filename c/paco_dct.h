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
 * \file paco_dct.h
 *\brief Interface to patch models based on linear operators.
 */
#ifndef LINOP_H
#define LINOP_H
#include "paco_types.h"

/**
 * Initialize FFTW structures.
 */
void paco_dct_init ( int size1, int size2 );

/**
 * Destroy FFTW structures.
 */
void paco_dct_destroy();

/**
 * \brief Applies the unnormalized 2D orthogonal DCT Type II direct transform to the input vector x
 * which is treated as an R^{w1 x w2} matrix, where w1 * w2 = m, and stores it in the output y
 */
void dct2d ( gsl_vector *TX, const gsl_vector *X );

/**
 * \brief Applies the unnormalized 2D orthogonal INVERSE DCT Type II direct transform to the input vector x
 * which is treated as an R^{w1 x w2} matrix, where w1 * w2 = m, and stores it in the output y
 */
void idct2d ( gsl_vector *X,  const gsl_vector *TX );

/**
 * \brief Applies the NORMALIZED 2D DCT Type II direct transform to the input vector x
 * which is treated as an R^{w1 x w2} matrix, where w1 * w2 = m, and stores it in the output y
 */
void dct2dn ( gsl_vector *TX,  const gsl_vector *X );

/**
 * \brief Applies the NORMALIZED 2D  INVERSE DCT Type II direct transform to the input vector x
 * which is treated as an R^{w1 x w2} matrix, where w1 * w2 = m, and stores it in the output y
 */
void idct2dn ( gsl_vector *X, const gsl_vector *TX );


/**
 * \brief Applies the 2D orthogonal DCT Type II direct transform to each row of X (n x m)
 * which is treated as an R^{w1 x w2} matrix, where w1 * w2 = m
 */
void dct2d_batch ( gsl_matrix *TX, const gsl_matrix *X );

/**
 * \brief Applies the 2D  orthogonal INVERSE DCT Type II direct transform to each row of X (n x m)
 * which is treated as an R^{w1 x w2} matrix, where w1 * w2 = m
 */
void idct2d_batch ( gsl_matrix *X, const gsl_matrix *TX );

/**
 * \brief Applies the NORMALIZED 2D DCT Type II direct transform to each row of X (n x m)
 * which is treated as an R^{w1 x w2} matrix, where w1 * w2 = m
 */
void dct2dn_batch ( gsl_matrix *TX, const gsl_matrix *X );

/**
 * \brief Applies the NORMALIZED 2D  INVERSE DCT Type II direct transform to each row of X (n x m)
 * which is treated as an R^{w1 x w2} matrix, where w1 * w2 = m
 */
void idct2dn_batch ( gsl_matrix *X, const gsl_matrix *TX );


#endif
