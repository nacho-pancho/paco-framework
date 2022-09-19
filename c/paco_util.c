#include <math.h>

#include <gsl/gsl_rng.h>

#include "paco_util.h"
#include "paco_log.h"

//============================================================================

sample_t gsl_vector_mean ( const gsl_vector* x ) {
    const index_t n = x->size;
    double a = 0;
    for ( index_t i = 0; i < n; ++i ) {
        a += x->data[i];
    }
    return ( sample_t ) ( a / ( double ) n );
}

//============================================================================

gsl_matrix  *subsample_patches ( const gsl_matrix *X, const index_t nsub, gsl_matrix **pXsub ) {

    const index_t n = X->size1;
    const index_t m = X->size2;

    gsl_matrix *Xsub;
    gsl_rng *rng = gsl_rng_alloc ( gsl_rng_default );
    gsl_rng_set ( rng, 18636998 );

    if ( pXsub == NULL ) {
        Xsub = gsl_matrix_alloc ( nsub, m );
    } else {
        Xsub = *pXsub;
    }

    if ( ( Xsub->size1 != nsub ) || ( Xsub->size2 != m ) ) {
        paco_error ( "subsample_patches: destination matrix has wrong size." );
        return NULL;
    }

    for ( index_t j = 0; j < nsub; j++ ) {
        const index_t r = gsl_rng_uniform_int ( rng, n );
        gsl_vector_const_view Xrview = gsl_matrix_const_row ( X, r );
        gsl_vector_view Xsubjview = gsl_matrix_row ( Xsub, j );
        gsl_vector_memcpy ( &Xsubjview.vector, &Xrview.vector );
    }

    gsl_rng_free ( rng );

    return Xsub;
}

//============================================================================

gsl_matrix *paco_dict_mosaic ( const gsl_matrix *D, gsl_matrix **pI,
                               int margin, int mag, double bg ) {
    int K = D->size1;
    int w = mag * ( ( int ) sqrt ( ( double ) D->size2 ) );
    int ng = ( int ) ceil ( sqrt ( ( double ) K ) ); // atom grid width
    int mg = ( int ) ceil ( ( ( double ) K ) / ( ( double )  ng ) ); // atom grid height
    int ni = ng * w + margin * ( ng - 1 ); // output image width
    int mi = mg * w + margin * ( mg - 1 ); // output image height

    gsl_matrix *I;

    if ( pI == NULL ) {
        I = gsl_matrix_alloc ( mi, ni );
    } else {
        I = *pI;
    }

    gsl_matrix_set_all ( I, bg );
    double dmin, dmax;
    gsl_matrix_minmax ( D, &dmin, &dmax );
    const double off = -dmin;
    const double sca = 1.0 / ( dmax - dmin );
    //
    // fill
    //
    int k = 0;

    for ( int ig = 0; ig < mg; ig++ ) {
        for ( int jg = 0; ( jg < ng ) && ( k < K ); jg++, k++ ) {
            int ii0 = ig * ( w + margin );
            int ii1 = ii0 + w;
            int ji0 = jg * ( w + margin );
            int ji1 = ji0 + w;
            // inner 2D loop
            gsl_vector_const_view Dkview = gsl_matrix_const_row ( D, k );
            const gsl_vector *Dk = &Dkview.vector;
            int l = 0;

            for ( int ji = ji0; ji <  ji1; ji += mag ) {
                for ( int ii = ii0; ii <  ii1; ii += mag ) {
                    double a = ( double ) gsl_vector_get ( Dk, l++ );

                    for ( int ji2 = 0; ji2 <  mag; ji2++ ) {
                        for ( int ii2 = 0; ii2 <  mag; ii2++ ) {
                            gsl_matrix_set ( I, ii + ii2, ji + ji2,  sca * ( ( double ) ( a ) ) + off  );
                        }
                    }
                }
            } // inner: paint atom
        }
    } // outer: for each atom

    return I;
} // function

//============================================================================

double l2dist ( const gsl_vector* A, const gsl_vector* B ) {
    const index_t n = A->size;
    const sample_t* pA = A->data;
    const sample_t* pB = B->data;
    double d = 0;
    for ( index_t i = 0; i < n; ++i ) {
        const double t = pA[i] - pB[i];
        d += t * t;
    }
    return d;
}

//============================================================================

double frobenius_squared_dist ( const gsl_matrix* A, const gsl_matrix* B ) {
    const index_t n = A->size1 * A->size2;
    const sample_t* pA = A->data;
    const sample_t* pB = B->data;
    double d = 0;
    for ( index_t i = 0; i < n; ++i ) {
        const double t = pA[i] - pB[i];
        d += t * t;
    }
    return d;
}



#if 0
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
void paco_color_dict_display ( const gsl_matrix *D,
                              gsl_matrix **pIr,
                              gsl_matrix **pIg,
                              gsl_matrix **pIb,
                              int margin,
                              int mag,
                              double bgr,
                              double bgg,
                              double bgb ) {
    int K = D->size2;

    if ( D->size1 % 3 ) {
        std::cerr << "Dictionary does not contain color patches." << std::endl;
    }

    int w = mag * int ( sqrt ( double ( D->size1 / 3 ) ) );
    int ng = ( int ) ceil ( sqrt ( double ( K ) ) ); // atom grid width
    int mg = ( int ) ceil ( double ( K ) / double ( ng ) ); // atom grid height
    int ni = ng * w + margin * ( ng - 1 ); // output image width
    int mi = mg * w + margin * ( mg - 1 ); // output image height


    gsl_matrix *Ir = *pIr = gsl_matrix_alloc ( mi, ni );
    gsl_matrix *Ig = *pIg = gsl_matrix_alloc ( mi, ni );
    gsl_matrix *Ib = *pIb = gsl_matrix_alloc ( mi, ni );
    // eventually, allocate space for output
    //
    gsl_matrix_set_all ( Ir, bgr );
    gsl_matrix_set_all ( Ig, bgg );
    gsl_matrix_set_all ( Ib, bgb );
    double dmin, dmax;
    gsl_matrix_minmax ( D, &dmin, &dmax );
    const double off = -dmin;
    const double sca = 255.0 / ( dmax - dmin );
    //  std::cout <<"dmin=" << dmin << " dmax=" << dmax << " sca=" << sca <<  " off=" << off << std::endl;
    //
    // fill
    //
    int k = 0;

    for ( int ig = 0; ig < mg; ig++ ) {
        for ( int jg = 0; ( jg < ng ) && ( k < K ); jg++, k++ ) {
            int ii0 = ig * ( w + margin );
            int ii1 = ii0 + w;
            int ji0 = jg * ( w + margin );
            int ji1 = ji0 + w;
            // inner 2D loop
            gsl_vector_const_view Dkview = gsl_matrix_const_column ( D, k );
            const gsl_vector *Dk = &Dkview.vector;
            int l = 0;

            for ( int ji = ji0; ji <  ji1; ji += mag ) {
                for ( int ii = ii0; ii <  ii1; ii += mag ) {
                    const double r = ( double ) gsl_vector_get ( Dk, l++ );
                    const double g = ( double ) gsl_vector_get ( Dk, l++ );
                    const double b = ( double ) gsl_vector_get ( Dk, l++ );

                    for ( int ji2 = 0; ji2 <  mag; ji2++ ) {
                        for ( int ii2 = 0; ii2 <  mag; ii2++ ) {
                            gsl_matrix_set ( Ir, ii + ii2, ji + ji2, double ( sca * ( double ( r ) + off ) ) );
                            gsl_matrix_set ( Ig, ii + ii2, ji + ji2, double ( sca * ( double ( g ) + off ) ) );
                            gsl_matrix_set ( Ib, ii + ii2, ji + ji2, double ( sca * ( double ( b ) + off ) ) );
                        }
                    }
                }
            } // inner: paint atom
        }
    } // outer: for each atom

    return;
} // function
#endif
