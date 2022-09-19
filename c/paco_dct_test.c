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
 * \file paco_dct_test.c
 * \brief Tests DCT functions.
 */
#include <stdlib.h>
#include <math.h>

#include "paco.h"
#include "paco_dct.h"
#include "paco_log.h"

int main ( int argc, char *argv[] ) {
    index_t n1 = 8;
    index_t n2 = 8;

    if ( argc > 1 ) {
        n1 = atoi ( argv[1] );
    }

    if ( argc > 2 ) {
        n2 = atoi ( argv[2] );
    }

    index_t l = n1 * n2;
    paco_info ( "testing 2D DCT for n1=%d n2=%d L=%d\n", n1, n2, l );
    paco_info ( "DIRECT\n" );
    gsl_matrix *X = gsl_matrix_calloc ( l, l );
    gsl_matrix *Z = gsl_matrix_calloc ( l, l );

    for ( index_t i = 0; i < l; i++ ) {
        gsl_matrix_set ( X, i, i, 1 );
    }

    dct2d_batch ( Z, n1, n2, X );

    for ( index_t i = 0; i < n1; i++ ) {
        for ( index_t j = 0; j < n2; j++ ) {
            index_t li = i * n2 + j;
            double nx = 0;
            double nz = 0;

            for ( index_t k = 0; k < l; k++ ) {
                double xk = gsl_matrix_get ( X, li, k );
                double zk = gsl_matrix_get ( Z, li, k );
                nx += xk * xk;
                nz += zk * zk;
            }

            paco_info ( "i=%d j=%d in=%f out=%f\n", i, j, sqrt ( nx ), sqrt ( nz ) );
        }
    }

    idct2d_batch ( X, n1, n2, Z );
    paco_info ( "INVERSE\n" );

    for ( index_t i = 0; i < n1; i++ ) {
        for ( index_t j = 0; j < n2; j++ ) {
            index_t li = i * n2 + j;
            double nx = 0;
            double nz = 0;

            for ( index_t k = 0; k < l; k++ ) {
                double xk = gsl_matrix_get ( X, li, k );
                double zk = gsl_matrix_get ( Z, li, k );
                nx += xk * xk;
                nz += zk * zk;
            }

            paco_info ( "i=%d j=%d in=%f out=%f\n", i, j, sqrt ( nz ), sqrt ( nx ) );
        }
    }

    gsl_matrix_free ( X );
    gsl_matrix_free ( Z );
    return 0;
}
