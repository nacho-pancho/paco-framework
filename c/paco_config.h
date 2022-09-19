#ifndef PACO_CONFIG_H
#define PACO_CONFIG_H
#include <argp.h>
#include "paco_types.h"

/**
 * PACO-DCT and ADMM options. These are filled in by the argument parser
 */
typedef struct  {
    char *preset;      ///< problem preset
    char *input_file;  ///< input (degraded) image
    char *mask_file;   ///< inpainting mask: nonzero values are considered erased
    char *output_file; ///< output of the algorithm
    char *ref_file;    ///< if available, this will be used to assess the quality of the result
    char *work_dir;    ///< directory where intermediate results will be written
    char *weight_file; ///< if defined and readable, DCT weights will be loaded from this ASCII file
    char *init_file;   ///< if defined and readable, the initial output estimate will be loaded from this file
    int  patch_width;     ///< patch width (both vertical and horizontal)
    int grid_stride;    ///< distance between adjacent patches
    int seed;           ///< random seed, for all that is random
    int num_scales;    ///< number of scales for multi-resolution processing
    double admm_penalty;     ///< initial value of the ADMM parameter
    int max_iter;   ///< maximum number of ADMM iterations allowed
    double min_cost_change;  ///< stop ADMM if the change in cost function f() drops below this value between iterations
    double min_arg_change;  ///< stop ADMM if the change in cost function f() drops below this value between iterations
    char* prior; ///< Prior model used in probability-based cost functions
    double kappa;      ///< if the cost function f() increases instead of decreasing, lambda is shrank by this amount
    double mu; ///< linearization penalty term for Linearized ADMM
    index_t algorithm_iter; ///< if larger than 1, the whole PACO algorithm is solved this many times, re-fitting the parameters of the functions to the solutin of the previous run
    double sigma; ///< standard deviation of noise in input
    index_t model_order; ///< for model families which have an order, such as the number of centroids in a k-means or GMM model, or the number of non-zero elements in a sparse model
    gsl_matrix* D; ///< linearization operator for Linearized ADMM
    int remove_dc; ///< flag: if true/nonzero, remove DC from patches when extracting; add back when stitching
    //
    // problem specific parameters
    //
    int denoising_mode;  ///< different forms of the general denoising problem: MAP (a.k.a. Lagrangian), L2 Ball (hard constraint), on individual patches, on the whole signal.
    double denoising_const; ///< constant 'c' for L2 ball radius in hard constraint: ||x - z || <= c*sigma^2

    gsl_matrix* P; ///< compressive sensing measurement matrix (m x k, where m is the dimension and k is the number of samples)
    gsl_matrix* B; ///< compressive sensing compressed samples (n x k, where n is the number of patches and k the number of samples)

    //
    // prior specific parameters
    //
    double gmm_sv_thres ;
    double gmm_sv_order ;
    int gmm_mode  ;
    int gmm_kmeans_iter ;
    int gmm_em_iter     ;
   
    int moe_iter ;
    double moe_kappa;

} paco_config_st;


paco_config_st paco_config_default();

void paco_parse_args ( int argc, char** argv, paco_config_st* cfg );

void paco_config_print ( const paco_config_st* cfg );

int paco_config_validate ( const paco_config_st* cfg );

#endif
