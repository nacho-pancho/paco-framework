#include <assert.h>
#include <string.h>

#include "paco_mapping.h"
#include "paco_log.h"
#include "paco_image.h"

int paco_mapping_validate ( const paco_mapping_st* map ) {
    if ( !map ) {
        paco_error ( "Null function pointer.\n" );
        return 0;
    } else if ( !map->create ) {
        paco_error ( "Undefined create function.\n" );
        return 0;
    } else if ( !map->destroy ) {
        paco_error ( "Undefined destroy function.\n" );
        return 0;
    } else if ( !map->stitch ) {
        paco_error ( "Undefined stitch function.\n" );
        return 0;
    } else if ( !map->extract ) {
        paco_error ( "Undefined extract function." );
        return 0;
    } else  if ( !strlen ( map->name ) ) {
        paco_error ( "Empty function name." );
        return 0;
    }
    return 1;
}

//============================================================================

char is_patch_complete ( const gsl_matrix *mask,
                         const index_t i0,
                         const index_t j0,
                         const index_t width,
                         const index_t height,
                         const index_t decimate ) {
    return mask == NULL ? 1 : !is_patch_incomplete ( mask, i0, j0, width, height, decimate );
}

//============================================================================

char is_patch_incomplete ( const gsl_matrix *mask,
                           const index_t i0,
                           const index_t j0,
                           const index_t width,
                           const index_t height,
                           const index_t decimate ) {
    if ( mask == NULL ) {
        return 0;
    }

    const index_t ph = width * decimate;
    const index_t pw = height * decimate;
    assert ( i0 <= mask->size1 - ph );
    assert ( j0 <= mask->size2 - pw );

    for ( index_t i = 0; i < ph; i += decimate ) {
        for ( index_t j = 0; j < pw; j += decimate ) {
            if ( get_sample ( mask, i0 + i, j0 + j ) )
                return 1;
        }
    }

    return 0;
}

//============================================================================

/**
 * extracts a patch starting at (i,j)
 */
void paco_mapping_extract_single ( sample_t *patch,
                                   const gsl_matrix *img,
                                   const index_t i0,
                                   const index_t j0,
                                   const index_t width,
                                   const index_t height,
                                   const index_t decimate ) {
    const double *offset = gsl_matrix_const_ptr ( img, i0, j0 );
    const index_t tda = img->tda;
    const index_t ph = width * decimate;
    const index_t pw = height * decimate;
    assert ( i0 <= img->size1 - ph );
    assert ( j0 <= img->size2 - pw );

    for ( index_t i = 0; i < ph; i += decimate ) {
        for ( index_t j = 0; j < pw; j += decimate ) {
            * ( patch++ ) = offset[i * tda + j];
        }
    }
}

//============================================================================

gsl_matrix *paco_mapping_extract_all ( const gsl_matrix *img,
                                       const index_t width,
                                       const index_t height,
                                       gsl_matrix **pX ) {

    const index_t m = width * height;
    const index_t mg = img->size1 - height + 1;
    const index_t ng = img->size2 - width  + 1;
    const index_t n = ng * mg;

    gsl_matrix *X;

    if ( pX == 0 ) {
        X = gsl_matrix_alloc ( n, m );
    } else if ( *pX == 0 ) {
        *pX = X = gsl_matrix_alloc ( n, m );
    } else {
        X = *pX;
    }

    index_t li = 0;

    for ( index_t i = 0; i < mg; i++ ) {
        for ( index_t j = 0; j < ng; j++ ) {
            gsl_vector_view xview = gsl_matrix_row ( X, li );
            gsl_vector *x = &xview.vector;
            paco_mapping_extract_single ( x->data, img, i, j, width, height, 1 );
        }
    }

    return X;
}

//============================================================================

gsl_matrix *paco_mapping_extract_complete (
    const gsl_matrix *img,
    const gsl_matrix *mask,
    const index_t width,
    const index_t height,
    gsl_matrix **pX ) {

    const index_t m = width * height;
    const index_t mg = img->size1 - height + 1;
    const index_t ng = img->size2 - width  + 1;

    index_t n = 0;

    for ( index_t i = 0; i < mg; i++ ) {
        for ( index_t j = 0; j < ng; j++ ) {
            if ( is_patch_complete ( mask, i, j, width, height, 1 ) ) {
                n ++;
            }
        }
    }

    gsl_matrix *X;

    if ( pX == 0 ) {
        X = gsl_matrix_alloc ( n, m );
    } else {
        X = *pX;
    }

    index_t li = 0;

    for ( index_t i = 0; i < mg; i++ ) {
        for ( index_t j = 0; j < ng; j++ ) {
            if ( is_patch_complete ( mask, i, j, width, height, 1 ) ) {
                gsl_vector_view xview = gsl_matrix_row ( X, li );
                gsl_vector *x = &xview.vector;
                paco_mapping_extract_single ( x->data, img, i, j, width, height, 1 );
                li ++;
            }
        }
    }

    return X;
}

//============================================================================

sample_t paco_mapping_remove_dc_single ( gsl_vector *x ) {
    const index_t m = x->size;
    sample_t *px = x->data;
    sample_t dc = 0;

    for ( index_t i = 0; i < m; ++i ) {
        dc += px[i];
    }

    dc /= ( sample_t ) m;

    for ( index_t i = 0; i < m; ++i ) {
        px[i] -= dc;
    }

    return dc;
}

//============================================================================

void paco_mapping_remove_dc ( gsl_matrix *X, gsl_vector *dc ) {
    assert ( X != NULL );
    assert ( dc != NULL );
    assert ( X->size1 == dc->size );
    const index_t n = X->size1;

    for ( index_t j = 0; j < n; j++ ) {
        gsl_vector_view Xjview = gsl_matrix_row ( X, j );
        gsl_vector *Xj = &Xjview.vector;
        const sample_t dcj = paco_mapping_remove_dc_single ( Xj );
        gsl_vector_set ( dc, j, dcj );
    }

    paco_info ( "DC min=%f max=%f\n", gsl_vector_min ( dc ), gsl_vector_max ( dc ) );
}

//============================================================================

void paco_mapping_add_back_dc_single ( const sample_t dc, gsl_vector *x ) {
    const index_t m = x->size;
    sample_t *px = x->data;

    for ( index_t i = 0; i < m; ++i ) {
        px[i] += dc;
    }
}

//============================================================================

/**
 * add back the dc to a given set of patches
 */
void paco_mapping_add_back_dc ( const gsl_vector *dc, gsl_matrix *X ) {
    assert ( X != NULL );

    if ( dc == NULL ) return;

    assert ( X->size1 == dc->size );
    const index_t n = X->size1;

    for ( index_t j = 0; j < n; j++ ) {
        gsl_vector_view Xjview = gsl_matrix_row ( X, j );
        gsl_vector *Xj = &Xjview.vector;
        const sample_t dcj = gsl_vector_get ( dc, j );
        paco_mapping_add_back_dc_single ( dcj, Xj );
    }
}


//============================================================================

void paco_mapping_stitch_single (
    const sample_t *patch,
    const index_t i0,
    const index_t j0,
    const index_t width,
    const index_t height,
    const index_t decimate,
    gsl_matrix *img ) {


    sample_t *offset = gsl_matrix_ptr ( img, i0, j0 );
    const index_t tda = img->tda;
    const index_t ph = width * decimate;
    const index_t pw = height * decimate;
    assert ( i0 <= img->size1 - ph );
    assert ( j0 <= img->size2 - pw );

    for ( index_t i = 0, k = 0; i < ph; i += decimate ) {
        for ( index_t j = 0; j < pw; j += decimate, k++ ) {
            offset[i * tda + j] = patch[k];
        }
    }

}

//============================================================================

void paco_mapping_stitch_all ( const gsl_matrix *X,
                               const index_t width,
                               const index_t height,
                               gsl_matrix *img ) {

    const index_t mg = img->size1 - height + 1;
    const index_t ng = img->size2 - width  + 1;
    assert ( ng * mg == X->size1 );
    assert ( width * height == X->size2 );

    index_t li = 0;

    for ( index_t i = 0; i < mg; i++ ) {
        for ( index_t j = 0; j < ng; j++ ) {
            gsl_vector_const_view xview = gsl_matrix_const_row ( X, li );
            const gsl_vector *x = &xview.vector;
            paco_mapping_stitch_single ( x->data, i, j, width, height, 1, img );
        }
    }
}

//============================================================================

void paco_mapping_stitch_complete (
    const gsl_matrix *X,
    const gsl_matrix* mask,
    const index_t width,
    const index_t height,
    gsl_matrix *img ) {

    const index_t mg = img->size1 - height + 1;
    const index_t ng = img->size2 - width  + 1;
    assert ( width * height == X->size2 );

    for ( index_t i = 0, k = 0; i < mg; i++ ) {
        for ( index_t j = 0; j < ng; j++ ) {
            if ( is_patch_complete ( mask, i, j, width, height, 1 ) ) {
                gsl_vector_const_view xview = gsl_matrix_const_row ( X, k );
                const gsl_vector *x = &xview.vector;
                paco_mapping_stitch_single ( x->data, i, j, width, height, 1, img );
                k ++;
            }
        }
    }
}

//============================================================================
