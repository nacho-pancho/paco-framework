#include "paco_config.h"
#include "paco_log.h"
#include "paco_io.h"
#include "paco_denoising.h"
#include "paco_inpainting.h"
#include <stdlib.h>
#include <string.h>

#define KEY_MON_MODE             0x201
#define KEY_MON_PAR              0x202
#define KEY_GMM_SV_THRES         0x101
#define KEY_GMM_SV_ORDER         0x102
#define KEY_GMM_MODE             0x103
#define KEY_GMM_KMEANS_ITER      0x104
#define KEY_GMM_EM_ITER          0x105
#define KEY_MOE_ITER             0x301
#define KEY_MOE_KAPPA            0x302

#define KEY_SEED                 0x501
//
// specific to BCS problem
//
#define KEY_BCS_DIFF_OP          0x401
#define KEY_BCS_MEAS_OP          0x402
#define KEY_BCS_SAMPLES          0x403

/**
 * These are the options that we can handle through the command line
 */
static struct argp_option options[] = {
    //
    // general options
    //
    {"aiter",          'A', "int",       0, "In some cases we might want to run PACO many times and readapt parameters." },
    {"denoising-const", 'c', "double+",       0, "Multiplier constant for L2-ball hard denoising constraint: ||x-z|| <= c*sigma^2" },
    {"workdir",        'd', "directory",       0, "Output directory for intermediate results" },
    {"remove-dc",      'D',                  0,      OPTION_ARG_OPTIONAL, "Remove DC from patches." },
    {"epsilon",        'e', "double+",       0, "Minimum change between iterations." },
    {"infile",         'i', "image",        0, "Input file name." },
    {"init",           'I', "image",          0, "Initial estimate" },
    {"kappa",          'k', "double+",         0, "ADMM penalty parameter multiplier (0 < kappa <= 1)." },
    {"order",          'K', "int",         0, "Model order (e.g., number of clusters)." },
    {"lambda",         'l', "lambda",        0, "ADMM penalty parameter." },
    {"mask",           'M', "image",          0, "Unknown samples mask" },
    {"max-iter",       'm', "int",      0, "Maximum allowed iterations." },
    {"denoising-mode", 'N', "mode", 0, "Denoising mode. 0: patch MAP, 1: global MAP, 2: patch L2 ball, 3: global L2 ball" },
    {"noise-sigma",    'n', "sigma", 0, "standard deviation of noise (0-255 scale)." },
    {"outfile",        'o', "image",       0, "Output file name." },
    {"prior",          'p', "name",      0, "Prior used for probability-based cost functions." },
    {"quiet",          'q', 0,      OPTION_ARG_OPTIONAL, "Don't produce any output" },
    {"ref",            'R', "image",           0, "Reference solution for comparison" },
    {"scales",         'S', "int",        0, "Number of scales for multiscale patch extraction." },
    {"stride",         's', "int",  0, "Patch stride in both directions." },
    {"verbose",        'v', 0,      OPTION_ARG_OPTIONAL, "Produce verbose output" },
    {"width",          'w', "int",   0, "Patch width in both directions" },
    {"mon-mode",       KEY_MON_MODE, "int+",  0, "Monitoring mode: 0: linear, 1: log10, 2:log2." },
    {"mon-par",        KEY_MON_PAR,  "int+",  0, "Monitoring parameter (step)." },
    {"bcs-diff-op",      KEY_BCS_DIFF_OP,  "file",  0, "file pointing to a differential operator matrix in ascii format." },
    {"bcs-meas-op",      KEY_BCS_MEAS_OP,  "file",  0, "file pointing to a sensing matrix in ascii format." },
    {"bcs-samples",      KEY_BCS_SAMPLES,  "file",  0, "file pointing to compressed sample vectors, one per row." },
#define KEY_BCS_PROJ_OP          0x402
#define KEY_BCS_MEAS             0x403
    {"seed",           KEY_SEED,   "file",  42, "random seed (42)." },
    //
    // prior-specific options
    //
    {"gmm-kmeans-iter", KEY_GMM_KMEANS_ITER,"int+",    0, "Number of iterations for K-Means stage of GMM adaptation." },
    {"gmm-em-iter",    KEY_GMM_EM_ITER,     "int+",    0, "Number of iterations in E-M stage of GMM adaptation." },
    {"gmm-mode",       KEY_GMM_MODE,        "int+",    0, "Inference method used in GMM: 2: L2 (Ridge), 1:L1 (Lasso/soft thres.), L0: (hard thres.)" },
    {"gmm-sv-order",   KEY_GMM_SV_ORDER,    "int+",    0, "Components over this number will be ignored during inference." },
    {"gmm-sv-thres",   KEY_GMM_SV_THRES,    "(0,1]",   0, "Components with Singular values below this value are ignored during inference." },
    {"moe-kappa",      KEY_MOE_KAPPA,       "(0,inf)", 0, "Shape parameter of MOE prior." },
    {"moe-iter",       KEY_MOE_ITER,        "int+",    0, "Number of re-weighting iterations in MOE proximal operator." },
    { 0 } // terminator
};

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------

/*
 * argp callback for parsing a single option.
 */
static error_t parse_opt ( int key, char *arg, struct argp_state *state ) {
    /* Get the input argument from argp_parse,
     * which we know is a pointer to our arguments structure.
     */
    paco_config_st *cfg = ( paco_config_st * ) state->input;
    switch ( key ) {
    case 'A':
        cfg->algorithm_iter    = atoi ( arg );
        break;
    case 'c':
        cfg->denoising_const   = atof ( arg );
        break;
    case 'd':
        cfg->work_dir          = arg;
        break;
    case 'D':
        cfg->remove_dc         = 1;
        break;
    case 'e':
        cfg->min_cost_change   = atof ( arg );
        break;
    case 'I':
        cfg->init_file         = arg;
        break;
    case 'i':
        cfg->input_file        = arg;
        break;
    case 'k':
        cfg->kappa             = atof ( arg );
        break;
    case 'K':
        cfg->model_order       = atoi ( arg );
        break;
    case 'l':
        cfg->admm_penalty      = atof ( arg );
        break;
    case 'M':
        cfg->mask_file         = arg;
        break;
    case 'm':
        cfg->max_iter          = atoi ( arg );
        break;
    case 'N':
        cfg->denoising_mode    = atoi ( arg );
        break;
    case 'n':
        cfg->sigma             = atof ( arg );
        break;
    case 'o':
        cfg->output_file       = arg;
        break;
    case 'p':
        cfg->prior             = arg;
        break;
    case 'q':
        set_log_level ( PACO_LOG_ERROR );
        break;
    case 'R':
        cfg->ref_file          = arg;
        break;
    case 's':
        cfg->grid_stride       = atoi ( arg );
        break;
    case 'v':
        set_log_level ( PACO_LOG_DEBUG );
        break;
    case 'w':
        cfg->patch_width       = atoi ( arg );
        break;
    case 'S':
        cfg->num_scales        = atoi ( arg );
        break;
    case KEY_SEED:
        cfg->seed        = atoi ( arg );
        break;

    case KEY_BCS_DIFF_OP:
        {
            index_t nrows = 0, ncols = 0;
            scan_ascii_matrix(arg, &nrows, &ncols);
            cfg->D = gsl_matrix_alloc(nrows,ncols);
            load_ascii_matrix(arg, cfg->D);
        }
        break;

    case KEY_BCS_MEAS_OP:
        {
            index_t nrows = 0, ncols = 0;
            scan_ascii_matrix(arg, &nrows, &ncols);
            cfg->P = gsl_matrix_alloc(nrows,ncols);
            load_ascii_matrix(arg, cfg->P);
        }
        break;

    case KEY_BCS_SAMPLES:
        {
            index_t nrows = 0, ncols = 0;
            scan_ascii_matrix(arg, &nrows, &ncols);
            cfg->B = gsl_matrix_alloc(nrows,ncols);
            load_ascii_matrix(arg, cfg->B);
        }
        break;

    case KEY_MON_PAR:
        set_monitoring_parameter( atoi ( arg ) );
        break;
    
    case KEY_MON_MODE:
        set_monitoring_mode( atoi ( arg ) );
        break;

    //
    // prior specific parameters
    //
    case KEY_GMM_SV_THRES:
        cfg->gmm_sv_thres    = atof ( arg );
        break;
    case KEY_GMM_SV_ORDER:
        cfg->gmm_sv_order    = atoi ( arg );
        break;
    case KEY_GMM_MODE:
        cfg->gmm_mode        = atoi ( arg );
        break;
    case KEY_GMM_KMEANS_ITER:
        cfg->gmm_kmeans_iter = atoi ( arg );
        break;
    case KEY_GMM_EM_ITER:
        cfg->gmm_em_iter     = atoi ( arg );
        break;
    case KEY_MOE_ITER:
        cfg->moe_iter     = atoi ( arg );
        break;
    case KEY_MOE_KAPPA:
        cfg->moe_kappa     = atof ( arg );
        break;


    case ARGP_KEY_ARG:
        paco_info ( "ARGP_KEY_ARG\n" );
        if ( state->arg_num == 0 ) {
            cfg->preset = arg;
        } else {
            argp_usage ( state );
            break;
        }
        break;

    case ARGP_KEY_END:
        if ( state->arg_num == 0 ) {
            argp_usage ( state );
        } else {
            return ARGP_ERR_UNKNOWN;
        }
        break;
    default:
        return ARGP_ERR_UNKNOWN;
    }

    return 0; // OK
}

//----------------------------------------------------

/**
 * General description of what this program does; appears when calling with --help
 */
static char program_doc[] =
    "\n*** PACO: A patch-consensus framework for patch-based signal processing ***";

/**
 * A general description of the input arguments we accept; appears when calling with --help
 */
static char args_doc[] = "[OPTIONS]";

/**
 * argp configuration structure
 */
static struct argp paco_argp = {
    options,
    parse_opt,
    args_doc,
    program_doc
};

/*
 * call parser
 */
void paco_parse_args ( int argc, char** argv, paco_config_st* cfg ) {
    argp_parse ( &paco_argp, argc, argv, 0, 0, cfg );
}

paco_config_st paco_config_default() {
    paco_config_st out;
    paco_config_st* cfg = &out;
    cfg->admm_penalty = 10; //nice number
    cfg->grid_stride = 4;
    cfg->init_file = NULL;
    cfg->kappa = 0.99;
    cfg->mask_file = NULL;
    cfg->max_iter = 256;
    cfg->min_cost_change = 1e-5;
    cfg->num_scales = 1;
    cfg->output_file = NULL;
    cfg->patch_width = 16;
    cfg->ref_file = NULL;
    cfg->weight_file = NULL;
    cfg->work_dir = "./";
    cfg->algorithm_iter = 1;
    cfg->prior = "dct-l1";
    cfg->model_order = 1;
    cfg->sigma = 0.000325521; // quantization noise std. dev. for 8-bit signal in [0,1] range: delta/sqrt(12)
    cfg->remove_dc = 0;
    cfg->denoising_mode = DENOISING_MODE_SIGNAL_BALL;
    cfg->denoising_const = 1.01;
    //
    // prior specific parameters
    //
    cfg->gmm_sv_thres    = 1e-5;
    cfg->gmm_sv_order    = 10000000; // 'infinity'
    cfg->gmm_kmeans_iter = 16;
    cfg->gmm_em_iter     = 0;
    cfg->gmm_mode        = 1;

    cfg->moe_kappa       = 2.0;
    cfg->moe_iter        = 2;

    cfg->D = NULL;
    cfg->P = NULL;
    cfg->B = NULL;
    return out;
}

void paco_config_print ( const paco_config_st* cfg ) {
    paco_info ( "\tinput     %s\n", cfg->input_file  ? cfg->input_file : "(none)" );
    paco_info ( "\tmask      %s\n", cfg->mask_file   ? cfg->mask_file : "(none)" );
    paco_info ( "\toutput    %s\n", cfg->output_file ? cfg->output_file : "(none)" );
    paco_info ( "\tref       %s\n", cfg->ref_file   ? cfg->ref_file : "(none)" );
    paco_info ( "\tinit      %s\n", cfg->init_file   ? cfg->init_file : "(none)" );
    paco_info ( "\tworkdir   %s\n", cfg->work_dir   ? cfg->work_dir : "(none)" );
    paco_info ( "\tprior     %s\n", cfg->prior );
    paco_info ( "\twidth     %d\n", cfg->patch_width );
    paco_info ( "\tstride    %d\n", cfg->grid_stride );
    paco_info ( "\tscales    %d\n", cfg->num_scales );
    paco_info ( "\tadmm pen. %f\n", cfg->admm_penalty ); //nice number
    paco_info ( "\tmax. iter %lu\n", cfg->max_iter );
    paco_info ( "\tmin. cost %f\n", cfg->min_cost_change );
    paco_info ( "\tsigma     %f\n", cfg->sigma );
    paco_info ( "\tkappa     %f\n", cfg->kappa );
    paco_info ( "\torder     %d\n", cfg->model_order );
    paco_info ( "\tden. mode %d\n", cfg->denoising_mode);
    paco_info ( "\tden. const %d\n", cfg->denoising_const);
    //
    // prior specific parameters
    //
    paco_info ( "\tGMM parameters\n");
    paco_info ( "\tsv. thres %d\n", cfg->gmm_sv_thres);
    paco_info ( "\tsv. order %d\n", cfg->gmm_sv_order); // 'infinity'
    paco_info ( "\tkmeans iter %d\n", cfg->gmm_kmeans_iter);
    paco_info ( "\tE-M iter    %d\n", cfg->gmm_em_iter);
    paco_info ( "\tGMM mode    %d\n", cfg->gmm_mode);
}

int paco_config_validate ( const paco_config_st* cfg ) {
    if ( cfg->grid_stride < 1 ) {
        paco_error ( "Stride must be a positive integer." );
        return 0;
    }
    if ( cfg->patch_width < 1 ) {
        paco_error ( "Patch width must be a positive integer." );
        return 0;
    }
    if ( cfg->grid_stride > cfg->patch_width ) {
        paco_error ( "Grid stride must not exceed patch width." );
        return 0;
    }
    if ( !cfg->input_file || !strlen ( cfg->input_file ) ) {
        paco_error ( "No input specified!.\n" );
        return 0;
    }
    /// \todo: many many checks to go
    return 1;
}
