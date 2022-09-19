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
 *  \file paco_io.h
 *  \brief Image I/O interface
 */
#ifndef PACO_IO_H
#define PACO_IO_H
#include <stdlib.h>
#include <stdio.h>
#include <png.h>
#include "paco_image.h"


/**
* Read png file from disk using libpng,
* currently suports RGB and Grayscale 8-bit depth images without interlacing only.
*
* @param filename Name (path) of file to read
*
* @return paco image struicture containing paointer to data, number of channels, bit depth
*/
paco_image_st *read_png_file ( char *filename );

/**
* Write png file from a png image structure using libpng,
*( the image structure must have been previously created using read_png_file, pixels may have been processed)
* Currently matches input info parameters, and thus suports RGB and Grayscale 8-bit depth images without interlacing only
*
* @param filename Name (path) of file to read
* @return pimage structure containing png structure internal pointer (to use with libpng functions),
* info pointer also to be handled by libpng functions,
* and a byte pointer to each row of image data to be written.
*/
void write_png_file ( char *filename, paco_image_st *paco_image );

/**
* Read DCT-weights from disk
* @param filename Name (path) to txt with weights
* @param w patch width
* @param input channel number
* @return DCT-weights vector
*/
gsl_vector *read_weights ( char *filename, size_t w );

int scan_ascii_matrix ( const char *fname, index_t *nrows, index_t *ncols );

int load_ascii_matrix ( const char *fname, gsl_matrix *mat );

void write_ascii_matrix ( const char *fname, const gsl_matrix *mat );

#endif
