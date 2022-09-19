#include "paco_types.h"
#include "paco_linalg.h"
#include "paco_sparse_coding.h"
#include "assert.h"
#include "paco_log.h"

static void paco_basis_pursuit_inner ( const gsl_matrix *D, const gsl_matrix *U, const gsl_vector *S, const gsl_matrix *V,
                                       const gsl_vector *xf, gsl_vector *af, const double eps );

/**
 * Solve a = arg min ||a||_1 subject to aD = x. (here the vectors are ROW vectors)
 *
 * We convert it to an unconstrained problem by defining g(a) = 0 if aD = x, +oo otherwise
 *
 * a = arg min ||a||_1 + g(a)
 *
 * And proceed with the standard splitting strategy by defining an auxiliary variable b
 *
 * a,b = arg min ||a||_1 + g(b) s.t. a = b
 *
 * And write the corresponding augmented lagrangian
 *
 * a,b = arg min L_c(a,b,y) = ||a||_1 + I_{.D=x}(b) + y^t(a-b) + 1/2c ||a-b||^2
 *
 * which we solve iteratively as
 *
 * a(k+1) <- prox_{c||.||_1}( b(k) - u(k) )
 * b(k+1) <- prox_{cg}( a(k+1) + u(k) )
 * u(k+1) <- u(k) + a(k+1) - b(k+1)
 */
void paco_basis_pursuit ( const gsl_matrix *D, const gsl_matrix *X, gsl_matrix *A, const double eps ) {
    const index_t p = A->size2;
    const index_t m = X->size2;
    assert ( p == D->size1 );
    assert ( m == D->size2 );
    //
    // compute SVD of D for pseudoinverse computation
    // note that U and V are reversed because the standard form is Db - x
    // but we have the transposed semantic (Dt)b - x
    // so the actual D for which we solve is Dt = VSUt
    //
    // on entry V contains the target matrix D
    //
    gsl_matrix *V = gsl_matrix_alloc ( p, m );
    gsl_matrix *U = gsl_matrix_alloc ( m, m );
    gsl_vector *S = gsl_vector_alloc ( m );
    gsl_vector *work = gsl_vector_alloc ( m );
    gsl_matrix_memcpy ( V, D );
    gsl_linalg_SV_decomp ( V, U, S, work ); // U <- nxn, S <- n, V <- pxn

    //
    // perform Basis Pursuit on each vector
    //
    for ( int j = 0; j < m; ++j ) {
        gsl_vector_const_view Xj = gsl_matrix_const_row ( X, j );
        gsl_vector_view       Aj = gsl_matrix_row ( A, j );
        paco_basis_pursuit_inner ( D, U, S, V, &Xj.vector, &Aj.vector, eps );
    }

    //
    // cleanup
    //
    gsl_vector_free ( work );
    gsl_vector_free ( S );
    gsl_matrix_free ( U );
    gsl_matrix_free ( V );
}

//=============================================================================

void paco_basis_pursuit_double ( const gsl_matrix *D,
                                 const gsl_matrix *X,
                                 gsl_matrix *A,
                                 const double eps ) {
    const index_t p = A->size2;
    const index_t m = X->size2;
    assert ( p == D->size1 );
    assert ( m == D->size2 );
    gsl_matrix *dD = gsl_matrix_alloc ( p, m );

    for ( int j = 0; j < m; ++j ) {
        for ( int i = 0; i < p; ++i ) {
            gsl_matrix_set ( dD, i, j, gsl_matrix_get ( D, i, j ) );
        }
    }

    gsl_vector *dXj = gsl_vector_alloc ( m );
    gsl_vector *dAj = gsl_vector_alloc ( p );

    //
    // compute SVD of D for pseudoinverse computation
    // note that U and V are reversed because the standard form is Db - x
    // but we have the transposed semantic (Dt)b - x
    // so the actual D for which we solve is Dt = VSUt
    //
    // on entry V contains the target matrix D
    //
    gsl_matrix *V = gsl_matrix_alloc ( p, m );
    gsl_matrix *U = gsl_matrix_alloc ( m, m );
    gsl_vector *S = gsl_vector_alloc ( m );
    gsl_vector *work = gsl_vector_alloc ( m );
    gsl_matrix_memcpy ( V, dD );
    gsl_linalg_SV_decomp ( V, U, S, work ); // U <- nxn, S <- n, V <- pxn

    //
    // perform Basis Pursuit on each vector
    //
    for ( int j = 0; j < m; ++j ) {
        // this conversion using get and set must be very slow, but...
        gsl_vector_const_view Xj = gsl_matrix_const_row ( X, j );

        for ( int i = 0; i < m; ++i ) {
            gsl_vector_set ( dXj, i, gsl_vector_get ( &Xj.vector, i ) );
        }

        gsl_vector_view       Aj = gsl_matrix_row ( A, j );

        for ( int i = 0; i < p; ++i ) {
            gsl_vector_set ( dAj, i, gsl_vector_get ( &Aj.vector, i ) );
        }

        paco_basis_pursuit_inner ( dD, U, S, V, dXj, dAj, eps );
    }

    //
    // cleanup
    //
    gsl_vector_free ( dXj );
    gsl_vector_free ( dAj );
    gsl_matrix_free ( dD );
    gsl_vector_free ( work );
    gsl_vector_free ( S );
    gsl_matrix_free ( U );
    gsl_matrix_free ( V );
}

//=============================================================================

static void paco_basis_pursuit_inner ( const gsl_matrix *D, const gsl_matrix *U, const gsl_vector *S, const gsl_matrix *V,
                                       const gsl_vector *x, gsl_vector *a, const double eps ) {
    const index_t p = a->size;
    const index_t n = x->size;
    //paco_info("paco_basis_pursuit p=%d n=%d\n",p,n);
    gsl_vector *tmp = gsl_vector_alloc ( n );
    gsl_vector *b = gsl_vector_alloc ( p );
    gsl_vector *u = gsl_vector_alloc ( p );
    gsl_vector *z = gsl_vector_alloc ( p );
    //
    // compute SVD of D for pseudoinverse computation
    // note that U and V are reversed because the standard form is Db - x
    // but we have the transposed semantic (Dt)b - x
    // so the actual D for which we solve is Dt = VSUt
    //
    // on entry V contains the target matrix D
    //
    double *pa = a->data;
    double *pu = u->data;
    double *pb = b->data;
    double *pz = z->data;

    for ( int i = 0; i < p; i++ ) {
        pa[i] = 1;
        pu[i] = 0;
        pb[i] = 0;
        pz[i] = 0;
    }

    double darg = 1;
    double dargb = 1;
    double dcost = 1;
    double atol = 1e-3;
    double ctol = 1e-5;
    double lambda = 10.0;
    double f = 1e10;
    int iter = 0;
    gsl_vector_memcpy ( tmp, x );
    gsl_blas_dgemv ( CblasTrans, -1.0, D, b, 1.0, tmp ); // tmp = Db - x
    double diff = gsl_blas_dnrm2 ( tmp ) / ( 1e-10 + gsl_blas_dnrm2 ( x ) );

    //paco_info("iter=%d darg=%f dcost=%f diff=%f\n",iter,darg,dcost,diff);
    while ( ( diff > eps ) && ( darg > atol ) && ( dcost > ctol ) ) {
        gsl_vector_memcpy ( z, a );
        //
        // prox_f( b(k)-u(k) ) -> soft thresholding(b+u,lambda)
        //
        //
        // prox_g( a(k+1) + u(k) )

        // after some computations we arrive at the standard orthogonal projection:
        //
        // b <- z - Dt(DDt)i(Dz-x)
        //
        // after using the SVD decomposition one arrives at the much cheaper
        //
        // b <- z - V(Vtz - SiUtx)
        // z = a(k+1) + u(k)
        gsl_vector_memcpy ( z, a );
        gsl_vector_add ( z, u );
        //paco_info("V: %dx%d, U: %dx%d, x: %d, b:%d\n",V->size1,V->size2,U->size1,U->size2,x->size,b->size);
        //paco_info("tmp <- Utx\n");
        gsl_blas_dgemv ( CblasTrans,   1.0, U, x,  0.0, tmp ); // tmp <- Utx
        //paco_info("tmp <- SiUtx\n");
        gsl_vector_div ( tmp, S );                          // tmp <- SiUtx  (the diagonal matrix S is represented as a vector)
        //paco_info("tmp <- -SiUtx + Vtz\n");
        gsl_blas_dgemv ( CblasTrans,   1.0, V, z, -1.0, tmp ); // tmp <- -SiUtx + Vtz
        //paco_info("z <- z - V( Vtz - SiUtx)\n");
        gsl_blas_dgemv ( CblasNoTrans, -1.0, V, tmp,  1.0, z ); // z <- z - V( Vtz - SiUtx)
        double narg = 0;
        dargb = 0;

        for ( int i = 0; i < p; ++i ) {
            const double d = pb[i] - pz[i];
            dargb += d * d;
            narg = pz[i] * pz[i];
            pb[i] = pz[i];
        }

        dargb = sqrt ( dargb ) / sqrt ( 1e-10 + narg );
        //gsl_vector_memcpy(b,z); // b <- z
        //
        // u <- u + a - z
        //
        double g = 0;

        for ( int i = 0; i < p; ++i ) {
            const double d = pa[i] - pb[i];
            g += d * d;
        }

        g = sqrt ( g );
        gsl_vector_add ( u, a );
        gsl_vector_sub ( u, b );
        //
        // evaluate cost function
        //
        const double prevf = f;
        f = 0;
        darg = 0;
        narg = 0;

        for ( index_t i = 0; i < p; ++i ) {
            const double vi = pb[i] - pu[i];
            const double preva = pa[i];
            pa[i] = vi > lambda ? vi - lambda : ( vi < -lambda ? vi + lambda : 0 );
            f += fabs ( pa[i] );
            const double da = pa[i] - preva;
            darg += da * da;
            narg += pa[i] * pa[i];
        }

        darg = sqrt ( darg ) / ( 1e-10 + sqrt ( narg ) );
        //
        // check approximation to objective
        //
        gsl_vector_memcpy ( tmp, x );
        gsl_blas_dgemv ( CblasTrans, -1.0, D, a, 1.0, tmp ); // tmp = Db - x
        diff = gsl_blas_dnrm2 ( tmp ) / ( 1e-10 + gsl_blas_dnrm2 ( x ) );
        //
        // check stopping condition
        //
        dcost = fabs ( prevf - f ) / ( f + 1e-10 );
        iter++;
        //paco_info("iter=%d darg=%f dargb=%f viol=%f cost=%f dcost=%f diff=%f\n",iter, darg, dargb, g, f, dcost, diff);
    }

    gsl_vector_free ( z );
    gsl_vector_free ( u );
    gsl_vector_free ( b );
    gsl_vector_free ( tmp );
}
