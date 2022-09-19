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
#include <argp.h>                     // argument parsing
#include <string.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "paco_log.h"                 // logging utilities
#include "paco_io.h"                  // image I/O
#include "paco_metrics.h"                  //performance metrics
/**
 * These are the options that we can handle through the command line
 */
static struct argp_option options[] = {
    { 0 } // terminator
};

/**
 * PACO-DCT and ADMM options. These are filled in by the argument parser
 */
typedef struct  {
    char *input_file;  ///< input (degraded) image
    char *output_file;  ///< input (degraded) image
    sample_t sigma;
} paco_config_st;

/**
 * options handler
 */
static error_t parse_opt ( int key, char *arg, struct argp_state *state );

/**
 * General description of what this program does; appears when calling with --help
 */
static char program_doc[] =
    "\n*** GAUSSIAN NOISE TOOL. ***";

/**
 * A general description of the input arguments we accept; appears when calling with --help
 */
static char args_doc[] = "<INPUT> <NOISE-STDEV> <OUT>";

/**
 * argp configuration structure
 */
static struct argp argp = { options, parse_opt, args_doc, program_doc };

/**
 * main function
 */
int main ( int argc, char **argv ) {
    paco_config_st cfg; // command-line program configuration


    /*
     * Default program configuration
     */
    set_log_level ( PACO_LOG_INFO );
    set_log_stream ( stdout );
    /*
     * call parser
     */
    argp_parse ( &argp, argc, argv, 0, 0, &cfg );
    /*
      * load data.
      */
    paco_image_st *input = NULL;

    input = read_png_file ( cfg.input_file );

    if ( !input ) {
        paco_error ( "Unable to read input!\n" );
        exit ( 1 );
    }

    const index_t nchannels = get_nchannels ( input );

    gsl_rng *rng = gsl_rng_alloc ( gsl_rng_taus );
    const sample_t sigma = cfg.sigma / 255.0f;

    for ( index_t c = 0; c < nchannels; c++ ) {
        /*
         * get c-th channel from image
         */
        gsl_matrix *input_samples = get_channel_samples ( input, c );
        const index_t ncols = input_samples->size2;
        const index_t nrows = input_samples->size1;
        const index_t L = ncols * nrows;
        sample_t *px = input_samples->data;

        for ( index_t i = 0; i < L; ++i ) {
            sample_t z = px[i] + gsl_ran_gaussian ( rng, sigma );
            px[i] = z < 255.0f ? ( z > 0.0f ? z : 0.0f ) : 255.0f;
        }

        write_png_file ( cfg.output_file, input );
    }


    /*
      * free local storage.
      */
    paco_image_free ( input );
    exit ( 0 );
}


/*
 * argp callback for parsing a single option.
 */
static error_t parse_opt ( int key, char *arg, struct argp_state *state ) {
    /* Get the input argument from argp_parse,
     * which we know is a pointer to our arguments structure.
     */
    paco_config_st *cfg = ( paco_config_st * ) state->input;

    switch ( key ) {
    case ARGP_KEY_ARG:
        switch ( state->arg_num ) {
        case 0:
            cfg->input_file = arg;
            break;

        case 1:
            cfg->sigma = atof ( arg );
            break;

        case 2:
            cfg->output_file = arg;
            break;

        default:
            /** too many arguments! */
            paco_error ( "Too many arguments!.\n" );
            argp_usage ( state );
            break;
        }

        break;

    case ARGP_KEY_END:
        if ( state->arg_num < 3 ) {
            /* Not enough mandatory arguments! */
            paco_error ( "Too FEW arguments!\n" );
            argp_usage ( state );
        }

        break;

    default:
        return ARGP_ERR_UNKNOWN;
    }

    return 0;
}
