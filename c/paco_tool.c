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
 * \file paco_dct_inpaint_tool.c
 * \brief Command line interface to the PACO-DCT inpainting algorithm.
 *
 *  Uses GNU argp to parse arguments
 *  see http://www.gnu.org/software/libc/manual/html_node/Argp.html
 */
#include <stdlib.h>
#include <time.h>                     // basic benchmarking
#include <argp.h>                     // argument parsing
#include <string.h>
#include "paco_log.h"                 // logging utilities
#include "paco_io.h"                  // image I/O
#include "paco_metrics.h"                  //performance metrics
#include "paco_problem.h"

//----------------------------------------------------------------------------

paco_problem_st paco_problem_build ( const char* problem_name, paco_config_st* config );

//----------------------------------------------------------------------------

/**
 * main function
 */
int main ( int argc, char **argv ) {

    /**
     * first argument must be preset name
     */
    if ( argc < 2 ) {
        paco_error ( "General usage: paco problem [problem args...]\n\
                Available problems: paco list\n" );
        exit ( 1 );
    }
    const char* problem_type = argv[1];
    if ( strcasecmp ( problem_type, "list" ) == 0 ) {
        paco_info ( "Available problems: inpaint, denoise, bcs.\n" );
        paco_info ( "Available priors  : dct-l1, dct-moe, gmm, tv-l1, dict-l1.\n" );
        exit ( 0 );
    }


    /*
     * Default program configuration
     */
    set_log_level ( PACO_LOG_INFO );
    set_log_stream ( stdout );
    paco_config_st config = paco_config_default();

    /*
     * consume first argument
     */
    //argc--;
    //argv++;

    paco_info ( "Parsing arguments...\n" );
    paco_parse_args ( argc, argv, &config );
    paco_config_print ( &config );
    if ( !paco_config_validate ( &config ) ) {
        exit ( 1 );
    }

    paco_problem_st problem = paco_problem_build ( problem_type, &config );

    /*
     * load data.
     */
    paco_image_st *input = NULL;
    paco_image_st *init  = NULL;
    paco_image_st *ref = NULL;
    paco_image_st *mask = NULL;

    paco_info ( "Loading input data...\n" );
    input = read_png_file ( problem.config.input_file );
    if ( !input ) {
        paco_error ( "Unable to read input!\n" );
        exit ( 1 );
    }

    /*
     * load mask
     */
    if ( problem.config.mask_file && strlen ( problem.config.mask_file ) ) {
        mask = read_png_file ( problem.config.mask_file );
        if ( !mask ) {
            paco_error ( "Unable to read mask!\n" );
            exit ( 1 );
        }
    }

    /*
     * load initial signal estimation, if available
     */
    if ( problem.config.init_file && strlen ( problem.config.init_file ) ) {
        paco_info ( "Loading initialisation...\n" );
        init = read_png_file ( problem.config.init_file );
    }

    /*
     * load reference (aka ground truth) for quality assessment
     */

    if ( problem.config.ref_file && strlen ( problem.config.ref_file ) ) {
        paco_info ( "Loading reference data ... \n" );
        ref = read_png_file ( problem.config.ref_file );
    } else {
        ref = NULL;
    }

    paco_info ( "Input is %d x %d (x %d), bpp=%d\n", get_nrows ( input ), get_ncols ( input ), get_nchannels ( input ), get_bitd ( input ) );

    /*
     * perform colorspace conversion for color inputs */
    if ( strcmp ( get_colorspace ( input ), COLORSPACE_RGB ) == 0 ) {
        paco_info ( "Converting data to YUV...\n" );
        paco_image_convert_colorspace ( input, COLORSPACE_YUV );

        if ( init ) {
            paco_image_convert_colorspace ( init, COLORSPACE_YUV );
        }

        if ( ref ) {
            paco_image_convert_colorspace ( ref, COLORSPACE_YUV );
        }
    }
    const index_t output_nrows = get_nrows ( input ); // \todo: this should be defined by problem (e.g., zoom)
    const index_t output_ncols = get_ncols ( input ); // \todo: this should be defined by problem (e.g., zoom)

    paco_info ( "Allocating output.\n" );
    paco_image_st* output = paco_image_alloc ( output_nrows, output_ncols, get_colorspace ( input ) );
    paco_image_clear ( output );

    /*
     * run algorithm on each channel
     */
    const index_t nchannels = get_nchannels ( input );

    for ( index_t c = 0; c < nchannels; c++ ) {
        /*
         * channels are identified by a single letter
         */
        const char channel_name = get_channel_name ( input, c );
        paco_info ( "Running algorithm on channel %d (%c)...\n", c, channel_name );
        /*
         * load data for this channel
         */
        problem.data.mask      = mask != NULL ? get_channel_samples ( mask, 0 ) : NULL;
        problem.data.input     = get_channel_samples ( input, c );
        problem.data.reference = ref != NULL ? get_channel_samples ( ref, c ) : NULL;
        problem.data.initial   = init != NULL ? get_channel_samples ( init, c ) : NULL;
        /*
         * make sure that known values in input do not accidentally go through unnoticed
         */
        if ( problem.data.mask ) {
            const index_t numpix = problem.data.mask->size1 * problem.data.mask->size2;
            for ( index_t i = 0; i < numpix ; i++ ) {
                if ( problem.data.mask->data[i] )
                    problem.data.input->data[i] = 0;
            }
        }
        /*
         * initialize mapping (only for the first channel, for now)
         */
        if ( c == 0 ) {
            paco_info ( "Creating problem...\n" );
            paco_problem_create ( &problem );
        }

        /*
         * create an instance of the PACO-DCT inpainting problem
         */
        char aux[128];
        snprintf ( aux, 128, "paco_channel_%c_iter_", channel_name );

        paco_info ( "Validating problem data...\n" );
        if ( !problem.validate_data ( &problem.data ) ) {
            exit ( 1 );
        }
        paco_info ( "Initializing problem...\n" );
        problem.init ( &problem );

        for ( index_t J = 0; J < problem.config.algorithm_iter; ++J ) {
            paco_info ( "Adjusting parameters to data...\n" );
            problem.cost_function.fit ( &problem );
            problem.constraint_function.fit ( &problem );
            paco_info ( "Solving problem...\n" );
            problem.solve ( &problem );
            if ( !problem.data.initial ) {
                problem.data.initial = gsl_matrix_alloc ( problem.data.input->size1, problem.data.input->size2 );
            }
            if ( ( J + 1 ) < problem.config.algorithm_iter ) {
                paco_info ( "One more time!! Use output as initialization now.\n" );
                gsl_matrix_memcpy ( problem.data.initial, problem.iter.X );
            }
        }

        paco_info ( "Computing result...\n" );
        paco_iterate_st* iter = &problem.iter;
        gsl_matrix* output_samples = get_channel_samples ( output, c );
        // PENDING: need to add back DC if patches were estimated without DC
        problem.mapping.stitch ( output_samples, iter->A );

        if ( ref != NULL ) {
            double *ref_data = problem.data.reference->data;
            double *mask_data = problem.data.mask ? problem.data.mask->data : NULL;
            double *out_data = output_samples->data;
            const index_t tda = problem.data.input->tda;
            const index_t ncols = problem.data.input->size2;
            const index_t nrows = problem.data.input->size1;
            double prmse = rmse_partial ( ref_data, out_data, mask_data, nrows * ncols );
            double pssim =  iqa_ms_ssim ( ref_data, out_data, mask_data, ncols, nrows, tda, NULL );
            double rmse = rmse_partial ( ref_data, out_data, NULL, nrows * ncols );
            double ssim =  iqa_ms_ssim ( ref_data, out_data, NULL, ncols, nrows, tda, NULL );
            paco_info ( "CHANNEL %d PRMSE %7.4f PSSIM %6.4f RMSE  %7.4f RMSE  %6.4f\n", c, prmse,  pssim, rmse, ssim );
        }
    }
    /*
     * destroy problem
     */
    paco_problem_destroy ( &problem );
    /*
     * write result
     */
    if ( strcmp ( get_colorspace ( output ), COLORSPACE_YUV ) == 0 ) {
        paco_image_convert_colorspace ( output, COLORSPACE_RGB );
    }

    paco_image_clip ( output );
    write_png_file ( problem.config.output_file, output );

    paco_image_free ( output );
    paco_image_free ( input );
    paco_image_free ( mask );
    paco_image_free ( ref );
    paco_image_free ( init );
    exit ( 0 );
}
//
// available priors
//
#include "paco_dct_l1.h"
#include "paco_gmm.h"
#include "paco_dict.h"
#include "paco_moe.h"
#include "paco_l12.h"
#include "paco_bcs.h"
//
// available problems (and respective constraint functions)
//
#include "paco_inpainting.h"
#include "paco_denoising.h"
//
// available solvers
//
//
// available patch mappers
//
#include "paco_grid_mapping.h"

paco_problem_st paco_problem_build ( const char* problem_name, paco_config_st* config ) {

    paco_problem_st problem;
    problem.config = *config;
    problem.mapping = paco_grid_mapping();
    problem.monitor = paco_default_monitor;

    paco_function_st prior;
    if ( !strcasecmp ( config->prior, "gmm" ) ) {
        prior = paco_gmm();
        problem.solve   = paco_admm;
    } else if ( !strcasecmp ( config->prior, "dct-l1" ) ) {
        prior = paco_dct_l1();
        problem.solve   = paco_admm;
    } else if ( !strcasecmp ( config->prior, "dct-moe" ) ) {
        prior = paco_moe();
        problem.solve   = paco_admm;
    } else if ( !strcasecmp ( config->prior, "dict-l1" ) ) {
        paco_error ( "NOT ADDED YET." );
        exit ( 1 );
        //prior = paco_moe();
        problem.solve   = paco_ladmm;
    //} else if ( !strcasecmp ( config->prior, "tv-l1" ) ) {
    //    prior           = paco_tv();
    //    problem.solve   = paco_admm;
    } else if ( !strcasecmp ( config->prior, "l12" ) ) {
        prior           = paco_l12();
        problem.solve   = paco_admm;
    }  else {
        paco_error ( "Unknown prior %s\n", config->prior );
        exit ( 1 );
    }

    if ( !strncasecmp ( problem_name, "inpaint", 7 ) ) {
        problem.validate_data = paco_data_validate_input_and_mask;
        problem.init     = paco_init_missing_with_average;
        problem.constraint_function = paco_inpainting();
        problem.cost_function = prior;
    } else if ( !strncasecmp ( problem_name, "denois", 6 ) ) {
        problem.validate_data = paco_data_validate_default;
        problem.init     = paco_init_default;

        problem.cost_function = paco_denoising_cost ( &prior, config->denoising_mode );
        problem.constraint_function = paco_denoising_constraint ( config->denoising_mode );

    } else if ( !strncasecmp ( problem_name, "bcs", 3 ) ) {
        problem.solve         = paco_ladmm;
        problem.validate_data = paco_data_validate_default;
        problem.init          = paco_init_bcs;
        problem.monitor       = paco_bcs_monitor;
        // LADMM expects the "constraint" function to be the one with "D*A" as argument
        problem.constraint_function = paco_l12 ( &prior );
        problem.cost_function = paco_bcs ( config->seed );


    } else if ( !strncasecmp ( problem_name, "comp", 4 ) ) {
        problem.validate_data = paco_data_validate_default;
        problem.init          = paco_init_default;
    }
    return problem;
}

