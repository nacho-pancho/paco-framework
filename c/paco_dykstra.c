#include <assert.h>
#include "paco_types.h"
#include "paco_dykstra.h"
#include "paco_log.h"
#include "paco_linalg.h"


void paco_dykstra ( gsl_matrix *Z, const gsl_matrix *X, paco_proj_f projA, void *paramA, paco_proj_f projB, void *paramB ) {
    const size_t n = Z->size1;
    const size_t m = Z->size2;
    assert ( n == X->size1 );
    assert ( m == X->size2 );
    gsl_matrix *Zk, *Yk, *Pk, *Qk, *Zprev;
    Zk = Z;
    Yk = gsl_matrix_alloc ( n, m );
    Pk = gsl_matrix_alloc ( n, m );
    Qk = gsl_matrix_alloc ( n, m );
    Zprev = gsl_matrix_alloc ( n, m );

    //
    // Dykstra's algorithm for intersection of two convex sets
    //
    // 0) Zk := X, Qk := 0, Pk := 0
    //
    if ( Zk != X )
        gsl_matrix_memcpy ( Zk, X );

    gsl_matrix_set_zero ( Pk );
    gsl_matrix_set_zero ( Qk );

    double dif = 1e20;
    size_t k = 0;

    while ( dif > 1e-3 ) {
        ///double dZk = 0,dQk = 0,dPk = 0,dYk = 0;
        //
        // 1) Y(k) := proj_C(Z(k) + P(k))
        //
        gsl_matrix_memcpy ( Yk, Zk );
        gsl_matrix_add ( Yk, Pk );
        projA ( Yk, Yk, paramA );
        //
        // 2) P(k+1) := P(k) + Z(k) - Y(k)
        //
        gsl_matrix_add ( Pk, Zk );
        gsl_matrix_sub ( Pk, Yk );
        //
        // 3) Z(k+1) := proj_D(Y(k) + Q(k))
        //
        gsl_matrix_memcpy ( Zprev, Zk );
        gsl_matrix_memcpy ( Zk, Yk );
        gsl_matrix_add ( Zk, Qk );
        projB ( Zk, Zk, paramB );
        //
        // 4) Q(k+1) := Q(k) + Y(k) - Z(k+1)
        //
        gsl_matrix_add ( Qk, Yk );
        gsl_matrix_sub ( Qk, Zk );
        //
        // measure change, stop if not large enough
        //
        k++;

        if ( k > 1000 ) break;

        //dif = sqrt(gsl_matrix_dist2_squared(Yk,Zk)/gsl_matrix_frobenius(Yk));
        dif = sqrt ( gsl_matrix_dist2_squared ( Zprev, Zk ) / gsl_matrix_frobenius ( Zk ) );
    }

    paco_debug ( "DYKSTRA: iter %d |Yk-Zk|/|Yk| %f\n", k, dif );
    gsl_matrix_free ( Yk );
    gsl_matrix_free ( Pk );
    gsl_matrix_free ( Qk );
    gsl_matrix_free ( Zprev );
}


// generic Dykstra's algorithm for N projections


void paco_dykstra_gen ( gsl_matrix *Z, const gsl_matrix *X, paco_proj_f *proj_fun, void **proj_param, const index_t nproj ) {
    const size_t n = Z->size1;
    const size_t m = Z->size2;
    assert ( n == X->size1 );
    assert ( m == X->size2 );

    gsl_matrix **Zk = ( gsl_matrix ** ) malloc ( sizeof ( gsl_matrix * ) *nproj );
    gsl_matrix **Pk =  ( gsl_matrix ** ) malloc ( sizeof ( gsl_matrix * ) *nproj );

    gsl_matrix *prev = gsl_matrix_alloc ( n, m );

    //
    // Dykstra's algorithm for intersection of N convex sets
    //
    // 0)             Qk[0] := 0, Pk[0] := 0
    // 0) Zk[0] := X
    // 0) Zk[i] := 0, Qk[i] := 0, Pk[i] := 0, 1 =1,...,N-1
    //
    for ( index_t i = 0; i < nproj; i++ ) {
        Zk[i] = gsl_matrix_alloc ( n, m );
        gsl_matrix_set_zero ( Zk[i] );
        Pk[i] = gsl_matrix_alloc ( n, m );
        gsl_matrix_set_zero ( Pk[i] );
    }

    gsl_matrix_memcpy ( Zk[0], X );

    double dif = 1e20;
    size_t k = 0;

    while ( dif > 1e-3 ) {

        gsl_matrix_memcpy ( prev, Zk[0] );

        for ( index_t i = 1; i < nproj; i++ ) {
            //
            // 1) Z_i^(k) := proj_i(Z_{i-1}^(k-1) + P_{i-1}^(k-1))
            //
            gsl_matrix_memcpy ( Zk[i], Zk[i - 1] );
            gsl_matrix_add ( Zk[i], Pk[i - 1] );
            proj_fun[i] ( Zk[i], Zk[i], proj_param[i] );
            //
            // 2) P_{i}^(k+1) := P_{i}^(k) + Z_{i-1}(k) - Z_{i}(k)
            //
            gsl_matrix_add ( Pk[i - 1], Zk[i - 1] );
            gsl_matrix_sub ( Pk[i - 1], Zk[i] );
        }

        k++;

        if ( k > 1000 ) break;

        //dif = sqrt(gsl_matrix_dist2_squared(Yk,Zk)/gsl_matrix_frobenius(Yk));
        dif = sqrt ( gsl_matrix_dist2_squared ( prev, Zk[0] ) / gsl_matrix_frobenius ( Zk[0] ) );
    }

    paco_debug ( "DYKSTRA: iter %d |Yk-Zk|/|Yk| %f\n", k, dif );

    for ( index_t i = 0; i < nproj; i++ ) {
        Zk[i] = gsl_matrix_alloc ( n, m );
        gsl_matrix_set_zero ( Zk[i] );
        Pk[i] = gsl_matrix_alloc ( n, m );
        gsl_matrix_set_zero ( Pk[i] );
    }

    free ( Pk );
    free ( Zk );
    gsl_matrix_free ( prev );
}
