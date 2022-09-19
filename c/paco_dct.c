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
 * \file paco_dct.c
 * \brief linear operators defined on patches
 */
#include <omp.h>
#include <assert.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <fftw3.h>
#include <string.h>

#include "paco_types.h"
#include "paco_log.h"

//----------------------------------------------------------------------------

#define MAX_SIZE
#define MAX_THREADS 64 /// if I ever get to reach this limit I'll be very happy

static int initialized = 0;
static int num_threads = 0;
static index_t size1;
static index_t size2;
static index_t dim;

static sample_t *bufin[MAX_THREADS]; // first, hopefully aligned!
static sample_t *bufout[MAX_THREADS]; // first, hopefully aligned!
static fftw_plan dplan[MAX_THREADS];
static fftw_plan iplan[MAX_THREADS];

void paco_dct_init ( int _size1, int _size2 ) {
    assert ( initialized == 0 );
    size1 = _size1;
    size2 = _size2;
    dim = size1 * size2;
    num_threads = 1;
    const int m = dim;
    initialized = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = MAX_THREADS > omp_get_num_threads() ? omp_get_num_threads() : MAX_THREADS;
        omp_set_num_threads ( num_threads );
    }
#endif
    for ( unsigned T = 0; T < num_threads; T++ ) {
        bufin[T] = ( sample_t * ) fftw_malloc ( m * sizeof ( sample_t ) );
        bufout[T] = ( sample_t * ) fftw_malloc ( m * sizeof ( sample_t ) );
        dplan[T] = fftw_plan_r2r_2d ( size1, size2, bufin[T], bufout[T], FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE );
        iplan[T] = fftw_plan_r2r_2d ( size1, size2, bufin[T], bufout[T], FFTW_REDFT01, FFTW_REDFT01, FFTW_ESTIMATE );
    }
}

//---------------------------------------------------------------------------------------

void paco_dct_destroy() {
    assert ( initialized );
    for ( unsigned T = 0; T < num_threads; T++ ) {
        fftw_free ( bufin[T] );
        fftw_free ( bufout[T] );
        fftw_destroy_plan ( iplan[T] );
        fftw_destroy_plan ( dplan[T] );
    }
    fftw_cleanup();
}

//---------------------------------------------------------------------------------------

static int get_thread_num() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}


//---------------------------------------------------------------------------------------

static void dct2n_prenormalize ( sample_t *inT ) {
}

//---------------------------------------------------------------------------------------

static void dct2n_postnormalize ( sample_t *outT ) {
    const index_t m = dim;
    const index_t w1 = size1;
    const index_t w2 = size2;
    const sample_t k = 1.0 / sqrt ( 2 * w1 * 2 * w2 );
    const sample_t k00 = 0.5;
    const sample_t k01 = ( sample_t ) M_SQRT1_2;
    const sample_t k10 = ( sample_t ) M_SQRT1_2;
    // upper-left element

    for ( index_t i = 0; i < m; ++i ) { // all
        outT[i] *= k;
    }

    outT[0] *= k00;

    for ( index_t j = 1; j < w2; ++j ) { // first row
        outT[j] *= k01;
    }

    for ( index_t i = 1; i < w1; ++i ) { // first column
        outT[w2 * i] *= k10;
    }
}

//---------------------------------------------------------------------------------------

static void idct2n_prenormalize ( sample_t *inT ) {
    const index_t w1 = size1;
    const index_t w2 = size2;
    const sample_t k00 = 2.0f;
    const sample_t k01 = ( sample_t ) M_SQRT2;
    const sample_t k10 = ( sample_t ) M_SQRT2;

    // upper-left element
    inT[0] *= k00;

    for ( index_t j = 1; j < w2; ++j ) { // first row
        inT[j] *= k01;
    }

    for ( index_t i = 1; i < w1; ++i ) { // first column
        inT[w2 * i] *= k10;
    }
}

//---------------------------------------------------------------------------------------

static void idct2n_postnormalize ( sample_t *outT ) {
    const index_t m = dim;
    const sample_t k = ( double ) 0.5 / sqrt ( m );
    for ( index_t i = 0; i < m; ++i ) {
        outT[i] *= k;
    }
}

//---------------------------------------------------------------------------------------

void dct2d ( gsl_vector *TX,  const gsl_vector *X ) {
    assert ( initialized );
    assert ( X->size == dim );
    const index_t T = get_thread_num();
    fftw_plan planT = dplan[T];
    sample_t *inT = bufin[T];
    sample_t *outT = bufout[T];
    memcpy ( inT, X->data, dim * sizeof ( sample_t ) );
    fftw_execute_r2r ( planT, inT, outT );
    memcpy ( TX->data, outT, dim * sizeof ( sample_t ) );
}

//---------------------------------------------------------------------------------------

void dct2d_batch ( gsl_matrix *TX, const gsl_matrix *X ) {
    assert ( initialized );
    const index_t m = TX->size2;
    const index_t n = TX->size1;
    assert ( m == dim );
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for ( index_t j = 0; j < n; j++ ) {
        const index_t T = get_thread_num();
        fftw_plan planT = dplan[T];
        sample_t *inT = bufin[T];
        sample_t *outT = bufout[T];
        gsl_vector_view Xj = gsl_vector_view_array ( inT, m );
        gsl_vector_view TXj = gsl_vector_view_array ( outT, m );
        gsl_matrix_get_row ( &Xj.vector, X, j );
        fftw_execute_r2r ( planT, inT, outT );
        gsl_matrix_set_row ( TX, j, &TXj.vector );
    }
}

//---------------------------------------------------------------------------------------

void idct2d ( gsl_vector *X, const gsl_vector *TX ) {
    assert ( initialized );
    const index_t m = TX->size;
    assert ( m == dim );
    const index_t T = get_thread_num();
    fftw_plan planT = iplan[T];
    sample_t *inT = bufin[T];
    sample_t *outT = bufout[T];
    memcpy ( inT, TX->data, m * sizeof ( sample_t ) );
    fftw_execute_r2r ( planT, inT, outT );
    memcpy ( X->data, outT, m * sizeof ( sample_t ) );
}

//---------------------------------------------------------------------------------------

void idct2d_batch ( gsl_matrix *X, const gsl_matrix *TX ) {
    assert ( initialized );
    const index_t m = TX->size2;
    const index_t n = TX->size1;
    assert ( m == dim );
#ifdef _OPENMP
    #pragma omp parallel for
#endif

    for ( index_t j = 0; j < n; j++ ) {
        const index_t T = get_thread_num();
        fftw_plan planT = iplan[T];
        sample_t *inT = bufin[T];
        sample_t *outT = bufout[T];
        gsl_vector_view TXj = gsl_vector_view_array ( inT, m );
        gsl_vector_view Xj = gsl_vector_view_array ( outT, m );
        gsl_matrix_get_row ( &TXj.vector, TX, j );
        fftw_execute_r2r ( planT, inT, outT );
        gsl_matrix_set_row ( X, j, &Xj.vector );
    }
}

//---------------------------------------------------------------------------------------

void dct2dn ( gsl_vector *TX, const gsl_vector *X ) {
    assert ( initialized );
    const index_t m = X->size;
    assert ( m == dim );
    const index_t T = get_thread_num();
    fftw_plan planT = dplan[T];
    sample_t *inT = bufin[T];
    sample_t *outT = bufout[T];
    memcpy ( inT, X->data, m * sizeof ( sample_t ) );
    dct2n_prenormalize ( inT );
    fftw_execute_r2r ( planT, inT, outT );
    dct2n_postnormalize ( outT );
    memcpy ( TX->data, outT, m * sizeof ( sample_t ) );
}

//---------------------------------------------------------------------------------------

void dct2dn_batch ( gsl_matrix *TX,  const gsl_matrix *X ) {
    assert ( initialized );
    const index_t m = TX->size2;
    const index_t n = TX->size1;
    assert ( m == dim );
#ifdef _OPENMP
    #pragma omp parallel for
#endif

    for ( index_t j = 0; j < n; j++ ) {
        const index_t T = get_thread_num();
        fftw_plan planT = dplan[T];
        sample_t *inT = bufin[T];
        sample_t *outT = bufout[T];
        gsl_vector_view Xj = gsl_vector_view_array ( inT, m );
        gsl_vector_view TXj = gsl_vector_view_array ( outT, m );
        gsl_matrix_get_row ( &Xj.vector, X, j );
        dct2n_prenormalize ( inT );
        fftw_execute_r2r ( planT, inT, outT );
        dct2n_postnormalize ( outT );
        gsl_matrix_set_row ( TX, j, &TXj.vector );
    }
}

//---------------------------------------------------------------------------------------

void idct2dn ( gsl_vector *X,  const gsl_vector *TX ) {
    assert ( initialized );
    const index_t m = TX->size;
    const index_t T = get_thread_num();
    fftw_plan planT = iplan[T];
    sample_t *inT = bufin[T];
    sample_t *outT = bufout[T];

    idct2n_prenormalize ( inT );
    memcpy ( inT, TX->data, m * sizeof ( sample_t ) );
    fftw_execute_r2r ( planT, inT, outT );
    idct2n_postnormalize ( outT );
    memcpy ( X->data, outT, m * sizeof ( sample_t ) );
}

//---------------------------------------------------------------------------------------

void idct2dn_batch ( gsl_matrix *X,  const gsl_matrix *TX ) {
    assert ( initialized );
    const index_t m = TX->size2;
    const index_t n = TX->size1;

#ifdef _OPENMP
    #pragma omp parallel for
#endif

    for ( index_t j = 0; j < n; j++ ) {
        const index_t T = get_thread_num();
        fftw_plan planT = iplan[T];
        sample_t *inT = bufin[T];
        sample_t *outT = bufout[T];
        gsl_vector_view TXj = gsl_vector_view_array ( inT, m );
        gsl_vector_view Xj = gsl_vector_view_array ( outT, m );
        gsl_matrix_get_row ( &TXj.vector, TX, j );
        idct2n_prenormalize ( inT );
        fftw_execute_r2r ( planT, inT, outT );
        idct2n_postnormalize ( outT );
        gsl_matrix_set_row ( X, j, &Xj.vector );
    }
}

//===============================================================================

