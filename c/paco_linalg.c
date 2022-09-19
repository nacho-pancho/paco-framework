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
 * \file paco_linalg.c
 * \brief implementation of some useful linear algebra functions
 */
#include "paco_types.h"
#include "paco_linalg.h"
#include <omp.h>
#include <assert.h>

//
//============================================================================
//
double gsl_vector_sum ( const gsl_vector *v ) {
    double a = 0.0;
    const sample_t *pv = v->data;
    const size_t vs = v->stride;
    const size_t n = v->size;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:a)
#endif

    for ( size_t i = 0; i < n; ++i ) {
        a += *pv;
        pv += vs;
    }

    return a;
}

//
//============================================================================
//
double gsl_vector_dist2_squared ( const gsl_vector *u, const gsl_vector *v ) {
    assert ( u->size == v->size );
    register const size_t n = u->size;
    const sample_t *pu = u->data;
    const sample_t *pv = v->data;
    const size_t us = u->stride;
    const size_t vs = v->stride;
    double a = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:a)
#endif

    for ( size_t i = 0; i < n ; ++i ) {
        const double d = pu[i * us] - pv[i * vs];
        a += d * d;
    }

    return a;
}

//
//============================================================================
//
double gsl_vector_dif_sum ( const gsl_vector *u, const gsl_vector *v ) {
    assert ( u->size == v->size );
    register const size_t n = u->size;
    const sample_t *pu = u->data;
    const sample_t *pv = v->data;
    const size_t us = u->stride;
    const size_t vs = v->stride;
    double d = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:d)
#endif

    for ( size_t i = 0; i < n; ++i ) {
        d += *pu - *pv;
        pu += us;
        pv += vs;
    }

    return d;
}


//
//============================================================================
//
double gsl_matrix_sum ( const gsl_matrix *A ) {
    register const size_t m = A->size1;
    register const size_t n = A->size2;
    double a = 0.0;
    const sample_t *prow = A->data;

    for ( register size_t i = m ; i ; --i, prow += A->tda ) {
        const sample_t *pelem = prow;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:a)
#endif

        for ( size_t i = 0; i < n; ++i ) {
            a += pelem[i];
        }
    }

    return a;
}

//
//============================================================================
//
double gsl_matrix_dif_sum ( const gsl_matrix *A, const gsl_matrix *B ) {
    assert ( A->size1 == B->size1 );
    assert ( A->size2 == B->size2 );
    register const size_t m = A->size1;
    register const size_t n = A->size2;
    double a = 0.0;
    const sample_t *parow = A->data;
    const sample_t *pbrow = B->data;

    for ( register size_t i = m ; i ; --i, parow += A->tda, pbrow += B->tda ) {
        const sample_t *paelem = parow;
        const sample_t *pbelem = pbrow;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:a)
#endif

        for ( size_t i = 0; i < n; ++i ) {
            a += paelem[i] - pbelem[i];
        }
    }

    return a;
}

//
//============================================================================
//
/// Euclidean distance between two matrices as elements of R^{mxn}
/// @param A first matrix
/// @param B second matrix
/// @return the euclidean distance between A and B
double gsl_matrix_dist2_squared ( const gsl_matrix *A, const gsl_matrix *B ) {
    assert ( A->size1 == B->size1 );
    assert ( A->size2 == B->size2 );
    register const size_t n = A->size1*A->size2;
    double a = 0.0;
    const sample_t *pa = A->data;
    const sample_t *pb = B->data;
    for (index_t i = 0; i < n; ++i) {
	    const sample_t d = pb[i] - pa[i];
	    a += d*d;
    }
    return a;
}


//
//============================================================================
//
double gsl_matrix_frobenius_squared ( const gsl_matrix *A ) {
    const size_t n = A->size1;
    double a = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:a)
#endif

    for ( size_t j = 0; j < n; ++j ) {
        gsl_vector_const_view Aj = gsl_matrix_const_row ( A, j );
        const double nAj = gsl_blas_dnrm2 ( &Aj.vector );
        a += nAj * nAj;
    }

    return a;
}

//
//============================================================================
//
double gsl_matrix_frobenius ( const gsl_matrix *A ) {
    return sqrt ( gsl_matrix_frobenius_squared ( A ) );
}

//
//============================================================================
//

double gsl_matrix_spectral_norm ( const gsl_matrix *A, const char transposed ) {
    if ( transposed != 0 ) {
        double res;
        gsl_matrix *At = gsl_matrix_alloc ( A->size2, A->size1 );

        for ( index_t i = 0; i < A->size1; ++i ) {
            for ( index_t j = 0; j < A->size2; ++j ) {
                gsl_matrix_set ( At, j, i, gsl_matrix_get ( A, i, j ) );
            }
        }

        res = gsl_matrix_spectral_norm ( At, 0 );
        gsl_matrix_free ( At );
        return res;
    }

    //
    // we solve the optimization problem:
    //
    // max_v f(v) = (1/2)||Av||_2^2  s.t. ||v||=1
    //
    // using a projected gradient method
    //
    // df/dv = A*v
    //
    // v(k+1) = proj(v(k) -s*A*v(k))
    //
    const size_t m = A->size1;
    const size_t n = A->size2;
    gsl_vector *v, *vprev, *Av;
    gsl_matrix *AtA;
    v = gsl_vector_alloc ( n );
    vprev = gsl_vector_alloc ( n );
    Av = gsl_vector_alloc ( m );
    AtA = gsl_matrix_alloc ( n, n );
    gsl_blas_dgemm ( CblasTrans, CblasNoTrans, 1.0, A, A, 0.0, AtA );
    //
    // initialize v to row with maximum norm
    //
    size_t maxrow = 0;
    double maxrownorm = -1;

    for ( size_t i = 0; i < m; ++i ) {
        double a = 0;

        for ( size_t j = 0; j < n; ++j ) {
            const double Aij = gsl_matrix_get ( A, i, j );
            a += Aij * Aij;
        }

        if ( a > maxrownorm ) {
            maxrownorm = a;
            maxrow = i;
        }
    }

    maxrownorm = sqrt ( maxrownorm );

    for ( size_t j = 0; j < n; ++j ) {
        gsl_vector_set ( v, j, gsl_matrix_get ( A, maxrow, j ) / maxrownorm );
    }

    //
    // initial stepsize is inverse frobenius norm of AtA
    //
    double s0 = 1.0 / sqrt ( gsl_matrix_frobenius ( AtA ) );
    //
    // start iteration
    //
    double odif = 1;
    int k = 0;
    double obj = 1;
    double prevobj = 1;

    while ( fabs ( odif ) / ( obj + 1e-10 ) > 1e-5 ) {
        gsl_blas_dgemv ( CblasNoTrans, 1.0, A, v, 0.0, Av );
        obj = gsl_blas_dnrm2 ( Av );
        odif = obj - prevobj;
        //printf("k=%d obj=%f odif=%f\n",k, obj, odif);
        k++;
        // gradient ascent : v = v + (s0/k)*At*A*v
        prevobj = obj;
        gsl_vector_memcpy ( vprev, v );
        gsl_blas_dgemv ( CblasNoTrans, s0 / sqrt ( k ), AtA, vprev, 1.0, v );
        gsl_vector_scale ( v, 1.0 / gsl_blas_dnrm2 ( v ) );
    }

    gsl_matrix_free ( AtA );
    gsl_vector_free ( Av );
    gsl_vector_free ( vprev );
    gsl_vector_free ( v );
    return obj;
}
