#include <assert.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_spblas.h>
#include <gsl/gsl_spmatrix.h>

#include "paco_log.h"
#include "paco_problem.h"
#include "paco_tv.h"

static gsl_matrix* D;      // should be sparse but support in GSL is quite limited
static gsl_spmatrix* Dcoo;      // should be sparse but support in GSL is quite limited
static gsl_spmatrix* Dcrs;      // should be sparse but support in GSL is quite limited
static gsl_matrix*  DtD;     // useful for now...

//============================================================================

static void init_differential_operator ( const index_t m ) {
    D               = gsl_matrix_calloc ( 2 * m, m );
    const index_t w = ( index_t ) sqrt ( ( double ) m );
    //
    // create differential operator
    //
    // row 2*i contains the horizontal operator for the i-th pixel
    // row 2*i+1 contains the vertical operator for the i-th pixel
    for ( index_t pi = 0, i = 0; pi < w; ++pi ) {
        for ( index_t pj = 0; pj < w; ++pj, ++i ) {
            const index_t ih = 2 * i; // index of horizontal derivative for i-th pix
            const index_t iv = 2 * i + 1; //          vertical
            const index_t jc = i;     // column index of i-th pixel in patch
            //
            // vertical operator
            //
            sample_t cv = 0;
            if ( pi > 0 ) {
                const index_t jn = jc - w; // north
                gsl_matrix_set ( D, iv, jn, -0.5 );
                cv += 0.5;
            }
            if ( pi < ( w - 1 ) ) {
                const index_t js = jc + w; // south
                gsl_matrix_set ( D, iv, js, -0.5 );
                cv += 0.5;
            }
            gsl_matrix_set ( D, iv, jc, cv );
            //
            // horizontal operator
            //
            sample_t ch = 0;
            if ( pj > 0 ) {
                const index_t jw = jc - 1;
                gsl_matrix_set ( D, ih, jw, -0.5 );
                ch += 0.5;
            }
            if ( pj < ( w - 1 ) ) {
                const index_t je = jc + 1;
                gsl_matrix_set ( D, ih, je, -0.5 );
                ch += 0.5;
            }
            gsl_matrix_set ( D, ih, jc, ch );
        }
    }
    //gsl_spmatrix* sDcoo = gsl_spmatrix_alloc_nzmax(2*m,m,6*m,GSL_SPMATRIX_COO); newer version of GSL renamed this to COO
    Dcoo = gsl_spmatrix_alloc_nzmax ( 2 * m, m, 6 * m, GSL_SPMATRIX_TRIPLET );
    gsl_spmatrix_d2sp ( Dcoo, D );
    Dcrs = gsl_spmatrix_crs ( Dcoo );
    //gsl_spmatrix_free(sDcoo);

    DtD = gsl_matrix_calloc ( m, m );
    // only lower half is actually stored!
    gsl_blas_dsyrk ( CblasLower, CblasTrans, 1.0, D, 0.0, DtD );
}

//============================================================================

void paco_tv_create ( const struct paco_problem* problem ) {
    assert ( problem->mapping.patch_dim > 0 );
    const index_t m = problem->mapping.patch_dim;
    init_differential_operator ( m );
}

//============================================================================

void paco_tv_fit ( const struct paco_problem* problem ) {
    // no fitting here
}

//============================================================================

void paco_tv_destroy() {
    gsl_matrix_free ( DtD );
    gsl_matrix_free ( D );
    gsl_spmatrix_free ( Dcrs );
    gsl_spmatrix_free ( Dcoo );
}


//============================================================================

paco_function_st paco_tv() {
    paco_function_st preset;
    preset.create = paco_tv_create;
    preset.fit = paco_tv_fit;
    preset.eval   = paco_tv_eval;
    preset.prox   = paco_tv_prox;
    preset.destroy = paco_tv_destroy;
    strcpy ( preset.name, "tv" );
    return preset;
}

//============================================================================

/**
 * Total Variation cost function: \sum_{j=1}^{n} |L(x_j)|_2
 */
double paco_tv_eval ( const gsl_matrix *X ) {
    const index_t n = X->size1;
    const index_t m2 = 2 * X->size2;
    gsl_vector* g = gsl_vector_calloc ( m2 );
    double val = 0;
    //#ifdef _OPENMP
    //#pragma omp parallel for reduction(+:val)
    //#endif
    for ( index_t j = 0; j < n; j++ ) {
        gsl_vector_const_view Xjv = gsl_matrix_const_row ( X, j );
        const gsl_vector* Xj = &Xjv.vector;
        gsl_blas_dgemv ( CblasNoTrans, 1.0, D, Xj, 0.0, g );
        sample_t* pg = g->data;
        double valj = 0;
        for ( index_t i = 0; i < m2; ++i ) {
            valj += fabs ( pg[i] );
        }
        val += valj;
    }
    return val / ( ( double ) m2 * ( double ) n ); // normalized
}

//============================================================================

/**
 *
 * proximal operator for the TV L1 cost function
 * ||x||_tv1 = \sum_i ||(u_i,v_i)||_2 = \sum_i sqrt(u_i^2 + v_i^2)
 * where u_i = (Dh*x)_i
 *       v_i = (Dv*x)_i
 *
 * Dh is the horizontal gradient operator
 * Dv is the vertical gradient operator
 *
 * we solve the following problem.
 *
 * prox_{\mu tv}(y) = arg min_x \sum_i ||(u_i,v_i)||_2 + 1/2tau ||x - y||^2_2
 *
 * using ADMM and the following splitting
 *
 * prox_{\mu tv}(y) = arg_z min_(x,z) \sum_i||(z_2i,z_{2i+1}||_2 + 1/2tau ||x - y||^2_2 + 1/2mu ||Dx - z||^2_2 s.t Dx = z
 * where Dx = [ u_1, v_1, u_2, v_2, ... ]
 *
 * THIS IS WRONG BUT WORKS
 * The ADMM splittin cannot be Dx=z, it is guaranteed to work if the splitting is x=z
 * The Linearized ADMM (Uzawa's) should be used instead.
 * 
 * @param  A   n x m input matrix,  W  1 x m weights, tau
 * @return SA  n x m matrix (preallocated)
 */

void paco_tv_prox ( gsl_matrix *SA, const gsl_matrix *A, const double tau ) {
    assert ( SA->size1 == A->size1 );
    assert ( SA->size2 == A->size2 );
    const index_t m = A->size2;
    const index_t n = A->size1;
    if ( SA != A ) {
        gsl_matrix_memcpy ( SA, A );
    }

    //const sample_t mu   = tau/10.0; // a guess
    const sample_t mu   = tau / 100.0; // a guess

    const sample_t itau = 1.0 / tau;
    const sample_t imu  = 1.0 / mu;
    gsl_matrix* Hi = gsl_matrix_calloc ( m, m );
    gsl_matrix_memcpy ( Hi, DtD ); // H <- DtD
    gsl_matrix_scale ( Hi, imu ); //  H <- (1/mu)DtD
    for ( index_t i = 0; i < m; i++ ) {
        gsl_matrix_set ( Hi, i, i, gsl_matrix_get ( Hi, i, i ) + itau ); // H <- (1/tau)I + (1/mu)DtD
    }
    gsl_linalg_cholesky_decomp ( Hi );
    gsl_linalg_cholesky_invert ( Hi );
    gsl_vector* x = gsl_vector_calloc ( m );
    gsl_vector* z = gsl_vector_calloc ( 2 * m );
    gsl_vector* u = gsl_vector_calloc ( 2 * m );
    gsl_vector* aux1 = gsl_vector_alloc ( 2 * m );
    gsl_vector* aux2 = gsl_vector_alloc ( m );
    for ( index_t j = 0; j < n; j++ ) {
        gsl_vector_view SAjv = gsl_matrix_row ( SA, j );
        gsl_vector*     SAj  = &SAjv.vector;
        gsl_vector_const_view Ajv = gsl_matrix_const_row ( A, j );
        const  gsl_vector* y  = &Ajv.vector;
        gsl_vector_memcpy ( x, SAj );
        gsl_vector_set_zero ( u );
        gsl_vector_set_zero ( z );
        paco_debug ( "TV:%08lu/%08lu:", j, n );
        for ( index_t admmiter = 0; admmiter < 200; admmiter++ ) {
            //
            // x(k+1) = arg min 1/2tau ||x - y||^2_2 + 1/2mu||Dx-(z-u)||_2^2
            //        = [ 1/tau I + 1/mu DtD ]^{-1} [ 1/tau y + 1/mu Dt(z-u) ]
            //        = H^{-1} [ 1/tau y + 1/mu Dt(z-u) ]
            //
            gsl_vector_memcpy ( aux1, z );
            gsl_blas_daxpy ( -1.0, u, aux1 );                             // aux1 <- z(k) - u(k)
            gsl_vector_memcpy ( aux2, y );                                // aux2 <- y
            gsl_spblas_dgemv ( CblasTrans, imu, Dcrs, aux1, itau, aux2 ); // aux2 <- (1/tau)y + 1(/mu)*Dt*aux1
            gsl_blas_dgemv ( CblasNoTrans, 1.0, Hi, aux2, 0.0, x );        // x    <- Hi*aux2
            //
            // z(k+1) = arg min sum_i=1^m sqrt(z_{2i}^2+z_{2i+1}^2) + 1/2mu ||z - (x+u)||
            // the solution for each pair v_i=(z_{2i},z_{2i+1}) is given by the vector
            // thresholding operator: (1-tau/||v_i||_2) v_i
            //
            gsl_vector_memcpy ( aux1, z ); // aux1 <- z
            gsl_vector_memcpy ( z, u );    // z    <- u
            gsl_spblas_dgemv ( CblasNoTrans, 1.0, Dcrs, x, 1.0, z ); // z <- Dx + u
            sample_t* pz = z->data;
            for ( index_t i = 0; i < m; i++ ) {
                const double z1 = pz[2 * i];
                const double z2 = pz[2 * i + 1];
                const double nz = sqrt ( z1 * z1 + z2 * z2 );
                if ( nz <= mu ) {
                    pz[2 * i] = 0;
                    pz[2 * i + 1] = 0;
                } else {
                    const double s = 1.0 - mu / nz;
                    pz[2 * i] *= s;
                    pz[2 * i + 1] *= s;
                }
            }
            gsl_blas_daxpy ( -1.0, z, aux1 );
            gsl_spblas_dgemv ( CblasNoTrans, 1.0, Dcrs, x, 1.0, u );
            gsl_blas_daxpy ( -1.0, z, u );
            const double ndz = gsl_blas_dnrm2 ( aux1 );
            const double nz  = gsl_blas_dnrm2 ( z );
            const double nx  = gsl_blas_dnrm2 ( x );
            const double nu  = gsl_blas_dnrm2 ( u );
            if ( ! ( admmiter % 10 ) )
                paco_debug ( "\titer %06d nx %7f nz %7f nu %7f ndz %7f\n", admmiter, nx, nz, nu, ndz );
            if ( ndz <= 1e-4 * nz ) {
                break;
            }
        }
        paco_debug ( "\n" );
        gsl_vector_memcpy ( SAj, x );
    }
    gsl_vector_free ( aux2 );
    gsl_vector_free ( aux1 );
    gsl_vector_free ( u );
    gsl_vector_free ( z );
    gsl_vector_free ( x );
}

