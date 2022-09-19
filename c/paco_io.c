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
 * \file paco_io.c
 * \brief Image I/O implementation.
 */
#include "paco_io.h"
#include "paco_log.h"
#include <string.h>
#include <ctype.h>

paco_image_st *read_png_file ( char *filename ) {

    FILE *fp = fopen ( filename, "rb" );

    if ( !fp ) {
        paco_error ( "Error opening %s \n", filename );
        paco_error ( "Unable to read image!.\n" );
        return ( NULL );
    }

    //
    // create libng internal reading struct
    //
    png_structp png_ptr = png_create_read_struct ( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );

    if ( !png_ptr ) {
        paco_error ( "Unable to read image!.\n" );
        return ( NULL );
    }

    //
    // create libpng info struct
    //
    png_infop info_ptr = png_create_info_struct ( png_ptr );

    if ( !info_ptr ) {
        paco_error ( "Unable to read image!.\n" );
        return ( NULL );
    }

    if ( setjmp ( png_jmpbuf ( png_ptr ) ) ) {
        paco_error ( "Unable to read image!.\n" );
        return ( NULL );
    }

    png_init_io ( png_ptr, fp );
    //
    // read and retrieve image info from file
    //
    png_read_info ( png_ptr, info_ptr );

    int width, height, bit_depth, color_type, channel_n;

    width      = png_get_image_width ( png_ptr, info_ptr );
    height     = png_get_image_height ( png_ptr, info_ptr );
    color_type = png_get_color_type ( png_ptr, info_ptr );
    bit_depth  = png_get_bit_depth ( png_ptr, info_ptr );
    channel_n  = png_get_channels ( png_ptr, info_ptr );

    const char *colorspace = NULL;

    switch ( color_type ) {
    case PNG_COLOR_TYPE_GRAY:
        colorspace = COLORSPACE_GRAY;
        break;

    case PNG_COLOR_TYPE_RGB:
        colorspace = COLORSPACE_RGB;
        break;

    default:
        paco_error ( "PNG Colorspace not supported." );
        return NULL;
    }

    paco_image_st *img_st = paco_image_alloc ( height, width, colorspace );

    if ( color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8 ) {
        png_set_expand_gray_1_2_4_to_8 ( png_ptr );
        png_read_update_info ( png_ptr, info_ptr );
    }

    //
    // allocate pointers to each row of image data
    //
    png_bytep *row_ptrs = ( png_bytep * ) malloc ( sizeof ( png_bytep ) * height );

    if ( !row_ptrs ) {
        png_destroy_read_struct ( & ( png_ptr ), ( png_infopp ) NULL, ( png_infopp ) NULL );
        paco_error ( "Unable to read image!.\n" );
        return ( NULL );
    }

    //
    // allocate storage for each row, data stored as byte stream
    //
    for ( int y = 0; y < height; y++ ) {
        row_ptrs[y] = ( png_byte * ) malloc ( png_get_rowbytes ( png_ptr, info_ptr ) );

        if ( !row_ptrs[y] ) {
            for ( ; y >= 0; y-- ) {
                free ( row_ptrs[y] );
            }

            free ( row_ptrs );
            png_destroy_read_struct ( & ( png_ptr ), ( png_infopp ) NULL, ( png_infopp ) NULL );
            paco_error ( "Unable to read image!.\n" );
            return ( NULL );
        }
    }

    //
    // read image data into row pointers
    //
    png_read_image ( png_ptr, row_ptrs );
    //
    // allocate GSL matrix
    //
    const sample_t scale = 1.0 / 255.0;

    //
    // copie image into GSL matrix
    //
    for ( int c = 0; c < channel_n; c++ ) {
        gsl_matrix *samples = get_channel_samples ( img_st, c );

        for ( int i = 0; i < height; i++ ) {
            for ( int j = 0; j < width; j++ ) {
                gsl_matrix_set ( samples, i, j, ( double ) * ( row_ptrs[i] + j * channel_n + c ) * scale );
            }
        }
    }

    for ( int i = 0; i < height; i++ ) {
        free ( row_ptrs[i] );
    }

    free ( row_ptrs );

    png_destroy_read_struct ( & ( png_ptr ), & ( info_ptr ), NULL );

    fclose ( fp );
    //
    // normalize image to [0,1]
    //
    return img_st;
}
//
//============================================================================
//
void write_png_file ( char *filename, paco_image_st *img ) {


    FILE *fp = fopen ( filename, "wb" );

    if ( !fp ) {
        paco_error ( "Error opening %s for writing\n", filename );
        paco_error ( "Unable to write image!.\n" );
        return;
    }

    png_structp write_ptr = png_create_write_struct ( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );

    if ( !write_ptr ) {
        paco_error ( "Error while creating PNG write struct for %s\n", filename );
        return;
    }

    png_infop write_info_ptr = png_create_info_struct ( write_ptr );

    if ( !write_info_ptr ) {
        paco_error ( "Error while writing PNG info to %s\n", filename );
        png_destroy_write_struct ( &write_ptr,  NULL );
        return;
    }

    if ( setjmp ( png_jmpbuf ( write_ptr ) ) ) {
        paco_error ( "Error while creating PNG buffer for %s\n", filename );
        png_destroy_write_struct ( &write_ptr,  &write_info_ptr );
        return;
    }

    png_init_io ( write_ptr, fp );
    const char *colorspace = get_colorspace ( img );
    index_t nrows = get_nrows ( img );
    index_t ncols = get_ncols ( img );

    int color_type;

    if ( strcmp ( colorspace, COLORSPACE_GRAY ) == 0 ) {
        color_type = PNG_COLOR_TYPE_GRAY;
    } else if ( strcmp ( colorspace, COLORSPACE_RGB ) == 0 ) {
        color_type = PNG_COLOR_TYPE_RGB;
    } else {
        paco_error ( "Channel Number not supported.\n" );
        return;
    }

    //
    // compression and interlacing options can be changed here
    //
    png_set_IHDR ( write_ptr, write_info_ptr, ncols, nrows, get_bitd ( img ), color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE );
    //
    // Uncomment the following line to ignore alpha channel:
    // png_set_filler(png, 0, PNG_FILLER_AFTER);
    //
    png_write_info ( write_ptr, write_info_ptr );
    //
    // allocate pointers to each row of image data
    //
    png_bytep *row_ptrs = ( png_bytep * ) malloc ( sizeof ( png_bytep ) * nrows );

    if ( !row_ptrs ) {
        paco_error ( "Error while allocating PNG row pointers for %s\n", filename );
        png_destroy_write_struct ( &write_ptr,  &write_info_ptr );
        return;
    }

    //
    // allocate storage for each row, data stored as byte stream
    //
    for ( int y = 0; y < nrows; y++ ) {
        row_ptrs[y] = ( png_byte * ) malloc ( png_get_rowbytes ( write_ptr, write_info_ptr ) );

        if ( !row_ptrs[y] ) {
            for ( ; y >= 0; y-- ) {
                free ( row_ptrs[y] );
            }

            free ( row_ptrs );
            png_destroy_write_struct ( &write_ptr,  &write_info_ptr );
            paco_error ( "Error while allocating PNG row pointers for %s\n", filename );
            return;
        }
    }

    //
    // write paco image into row_ptrs
    //
    index_t nchan = get_nchannels ( img );
    sample_t scale = ( sample_t ) ( 1 << get_bitd ( img ) ) - 1.0f;

    for ( int i = 0; i < nrows; i++ ) {
        for ( int j = 0; j < ncols; j++ ) {
            for ( int c = 0; c < nchan; c++ ) {
                sample_t x = get_channel_sample ( img, i, j, c );
                x = x > 1.0f ? 1.0f : ( x < 0.0f ? 0.0 : x );
                * ( row_ptrs[i] + j * nchan + c ) = ( png_byte ) scale * x;
            }
        }
    }


    png_write_image ( write_ptr, row_ptrs );

    png_write_end ( write_ptr, NULL );

    png_destroy_write_struct ( &write_ptr, &write_info_ptr );

    for ( int i = 0; i < nrows; i++ ) {
        free ( row_ptrs[i] );
    }

    free ( row_ptrs );

    fclose ( fp );

    return;
}

//============================================================================

/**
 * read weight values from ASCII file
 */
gsl_vector *read_weights ( char *filename, size_t w ) {
    FILE *f_w = fopen ( filename, "r" );

    if ( !f_w ) {
        paco_error ( "ERROR: Unable to open weights file %s.\n", filename );
        return NULL;
    }

    gsl_vector *moe_betas = gsl_vector_alloc ( w * w );

    if ( gsl_vector_fscanf ( f_w, moe_betas ) != 0 ) {
        paco_error ( "ERROR: while reading weights file %s\n", filename );
        gsl_vector_free ( moe_betas );
        return NULL;
    }

    fclose ( f_w );
    return moe_betas;
}

//============================================================================

int scan_ascii_matrix ( const char *fname, index_t *nrows, index_t *ncols ) {
    char spc;
    int c;
    FILE *f;
    f = fopen ( fname, "r" );

    if ( f == NULL ) {
        return 1;
    }

    // count rows
    spc = 1;
    *ncols = 0;
    *nrows = 0;

    do {
        c = fgetc ( f );

        if ( isspace ( c ) ) {
            spc = 1;
        } else {
            if ( c == '#' ) { // a comment begins
                if ( *ncols > 0 ) {
                    break;
                } else {
                    // all lines so far have been pure comments, no data
                    while ( ( c = fgetc ( f ) ) != '\n' && c != EOF ) ;

                    if ( c == EOF ) { // file has nothing but comments
                        break;
                    }

                    // data may come in lines to be read next
                    c = ' '; // avoid finishing
                    continue;
                }
            }

            if ( spc ) { // was space, now it isn't
                spc = 0;
                ( *ncols )++;
            }
        }
    } while ( c != '\n' && c != EOF );

    // now count rows
    if ( *ncols == 0 ) {
        fclose ( f );
        return 0;
    }

    *nrows = 1;
    int empty = 1;

    while ( ( c = fgetc ( f ) ) != EOF ) {
        if ( !isspace ( c ) && c != '#' ) {
            empty = 0;
        }

        if ( !empty && ( c == '\n' ) ) {
            ( *nrows )++;
        }
    }

    fclose ( f );
    return 0;
}

//
//======================================================================
//

int load_ascii_matrix ( const char *fname, gsl_matrix *mat ) {
    const size_t M = ( size_t ) mat->size1;
    const size_t N = ( size_t )  mat->size2;
    FILE *f = fopen ( fname, "r" );

    for ( size_t i = 0 ; i < M ; i++ ) {
        for ( size_t j = 0 ; j < N ; j++ ) {
            float a = 0.0;
            int k = fscanf ( f, " %f ", &a );

            if ( k < 1 ) {
                return 1;
            }

            gsl_matrix_set ( mat, i, j, a );
        }
    }

    fclose ( f );
    //
    // cleanup
    //
    return 0;
}

//============================================================================

int load_ascii_matrix_float ( const char *fname, gsl_matrix_float *mat ) {
    const size_t M = ( size_t ) mat->size1;
    const size_t N = ( size_t )  mat->size2;
    FILE *f = fopen ( fname, "r" );

    for ( size_t i = 0 ; i < M ; i++ ) {
        for ( size_t j = 0 ; j < N ; j++ ) {
            float a = 0.0f;
            int k = fscanf ( f, " %f ", &a );

            if ( k < 1 ) {
                return 1;
            }

            gsl_matrix_float_set ( mat, i, j, a );
        }
    }

    fclose ( f );
    //
    // cleanup
    //
    return 0;
}

//============================================================================

void write_ascii_matrix ( const char *fname, const gsl_matrix *mat ) {

    const index_t n = mat->size1;
    const index_t m = mat->size2;
    FILE *fout;

    fout = fopen ( fname, "w" );

    if ( !fout ) {
        paco_error ( "Cannot write matrix to file %s\n", fname );
        return;
    }

    for ( index_t i = 0; i < n; i++ ) {
        for ( index_t j = 0; j < m; j++ ) {
            fprintf ( fout, "%10f\t", gsl_matrix_get ( mat, i, j ) );
        }

        fputc ( '\n', fout );
    }

    fclose ( fout );
}
