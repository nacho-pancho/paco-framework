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
 * \file paco_linalg.h
 * \brief some simple gsl-like utilities not present in gsl_vector.h, gsl_matrix.h or gsl_linalg.h
 */
#ifndef PACO_LINALG_H
#define PACO_LINALG_H
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_linalg.h>

/// sum of vector components
double gsl_vector_sum ( const gsl_vector *v );

/// squared Euclidean distance between two vectors
double gsl_vector_dist2_squared ( const gsl_vector *u, const gsl_vector *v );

/// cumulative difference between two vectors
double gsl_vector_dif_sum ( const gsl_vector *u, const gsl_vector *v );

/// signed sum of all elements in a matrix
double gsl_matrix_sum ( const gsl_matrix *A );

/// sum of the signed difference between two matrices
double gsl_matrix_dif_sum ( const gsl_matrix *A, const gsl_matrix *B );

/// squared Frobenius-norm based distance betwen two matrices
double gsl_matrix_dist2_squared ( const gsl_matrix *A, const gsl_matrix *B );

/// Frobenius norm of a matrix
double gsl_matrix_frobenius ( const gsl_matrix *A );

/// squared Frobenius norm of a matrix
double gsl_matrix_frobenius_squared ( const gsl_matrix *A );

/// squared spectral norm of a matrix
double gsl_matrix_spectral_norm ( const gsl_matrix *A, const char transposed );

#endif
