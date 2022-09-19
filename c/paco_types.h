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
 * \file paco_types.h
 * \brief data types used throughout PACO
 */
#ifndef PACO_TYPES_H
#define PACO_TYPES_H
#include <gsl/gsl_matrix.h>

typedef double sample_t; ///< type used to store sample values
typedef int index_t; ///< type used to index arrays, pixels, etc.
typedef double aux_t; ///< type used for internal high precision operations
#define SAMPLE_SIZE sizeof(sample_t)
#define INDEX_SIZE sizeof(index_t)
#endif
