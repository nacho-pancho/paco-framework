#include <math.h>
#include "paco_config.h"
#include "paco_io.h"
#include "paco_metrics.h"
#include "paco_log.h"
#include "paco_problem.h"

static int mon_par = 10;
static int mon_mode = PACO_MONITOR_LINEAR;

void set_monitoring_parameter ( int mp ) {
    mon_par = mp;
}

void set_monitoring_mode ( int mm ) {
    mon_mode = mm;
}

int is_monitoring_time ( int iter ) {
    if ( iter == 0 ) return 0;
    if ( mon_mode == PACO_MONITOR_LINEAR ) {
        return  ( mon_par * ( int ) ( iter / mon_par ) ) == iter ? 1 : 0;
    } else if ( mon_mode == PACO_MONITOR_LOG2 ) {
        iter *= mon_par; // subdivide log scale
        return  pow( 2, (int) log2 ( iter ) ) == iter ? 1 : 0;
    } else if ( mon_mode == PACO_MONITOR_LOG10 ) {
        iter *= mon_par; // subdivide log scale
        return pow ( 10, ( int ) log10 ( iter ) ) == iter ? 1 : 0;
    } else {
        return 0;
    }
}

/**
 * writes the intermediate images corresponding to the main (A) and split (B)
 * iterates of the algorithm.
 */
void paco_default_monitor ( struct paco_problem* problem )  {

    paco_iterate_st* iter = &problem->iter;
    const gsl_matrix* mask = problem->data.mask;
    paco_config_st* cfg = &problem->config;
    paco_mapping_st* map = &problem->mapping;
    const gsl_matrix* ref = problem->data.reference;

    if ( !is_monitoring_time ( iter->k ) ) {
        return;
    }
    char aux[128];
    paco_image_st *iter_img;

    //
    // pseudo-image for multiplier U
    //
    snprintf ( aux, 128, "%s/%05d_U.png", cfg->work_dir, iter->k );
    map->stitch ( iter->X, iter->U );

    iter_img = paco_image_from_samples ( iter->X, COLORSPACE_GRAY );
    paco_image_norm ( iter_img );
    write_png_file ( aux, iter_img );
    paco_image_free ( iter_img );

    //
    // partial result for main variable A
    //
    snprintf ( aux, 128, "%s/%05d_A.png", cfg->work_dir, iter->k );
    map->stitch ( iter->X, iter->A );

    iter_img = paco_image_from_samples ( iter->X, COLORSPACE_GRAY );
    //paco_image_norm ( iter_img );
    write_png_file ( aux, iter_img );
    paco_image_free ( iter_img );

    //
    // partial result for split variable B
    //
    snprintf ( aux, 128, "%s/%05d_B.png", cfg->work_dir, iter->k );
    map->stitch ( iter->X, iter->B );

    iter_img = paco_image_from_samples ( iter->X, COLORSPACE_GRAY );
    //paco_image_norm ( iter_img );
    write_png_file ( aux, iter_img );
    paco_image_free ( iter_img );

    //
    // the partial result from B is also the
    // partial output of the algorithm
    // thus, we compute partial stats from it.
    //
    if ( ref ) {
        gsl_matrix* out = iter->X;
        double *out_data = out->data;
        double *mask_data = mask ? mask->data : NULL;
        double *ref_data = ref->data;
        const index_t tda = out->tda;
        const index_t ncols = out->size2;
        const index_t nrows = out->size1;
        double rmse = rmse_partial ( ref_data, out_data, NULL, nrows * ncols );
        double ssim =  iqa_ms_ssim ( ref_data, out_data, NULL, ncols, nrows, tda, NULL );
        double prmse = rmse_partial ( ref_data, out_data, mask_data, nrows * ncols );
        double pssim =  iqa_ms_ssim ( ref_data, out_data, mask_data, ncols, nrows, tda, NULL );
        double ppsnr = -20.0 * log10 ( prmse );
        double psnr = -20.0 * log10 ( rmse );
        paco_info ( "PRMSE %7.4f PSSIM %6.4f PPSNR %6.4f RMSE  %7.4f SSIM  %6.4f PSNR %6.4f\n", prmse,  pssim, ppsnr, rmse, ssim, psnr );

        // difference image
        iter_img = paco_image_from_samples ( iter->X, COLORSPACE_GRAY );
        sample_t* diff_data = get_channel_samples ( iter_img, 0 )->data;
        const index_t nsamples = ncols * nrows;
        for ( index_t i = 0; i < nsamples; ++i ) {
            diff_data[i] = ref_data[i] > diff_data[i] ? ref_data[i] - diff_data[i] : diff_data[i] - ref_data[i];
        }
        snprintf ( aux, 128, "%s%05d_DIF.png", cfg->work_dir, iter->k );
        paco_image_norm ( iter_img );
        write_png_file ( aux, iter_img );
        paco_image_free ( iter_img );
    }

}
