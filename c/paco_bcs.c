/*
  * Copyright (c) 2019 Ignacio Francisco Ramírez Paulino and Ignacio Hounie
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
 * \file paco_dct_bcs_problem.c
 * \brief Implementation of the PACO-DCT bcs problem
 *
 */
#include <math.h>
#include <assert.h>
#include <string.h>
#include "paco_log.h"
#include "paco_dct.h"
#include "paco_io.h"
#include "paco_metrics.h"
#include "paco_image.h"
#include "paco_bcs.h"
#include "paco_mapping.h"
#include "paco_problem.h"
#include "paco_linalg.h"

static const paco_mapping_st* mapping;

//============================================================================

void paco_bcs_create ( const struct paco_problem* problem );

void paco_bcs_prox();

void paco_bcs_destroy();

paco_function_st paco_bcs() {
    paco_function_st preset;
    strcpy ( preset.name, "bcs" );
    preset.create  = paco_bcs_create;
    preset.fit     = paco_dummy_fit;
    preset.eval    = paco_dummy_eval;
    preset.prox    = paco_bcs_prox;
    preset.destroy = paco_bcs_destroy;
    return preset;
}

//============================================================================
static gsl_matrix *V = NULL; ///< auxiliary signal; contains the result of the algorithm at the end
static gsl_matrix* P = NULL;
static gsl_matrix* B = NULL;
static gsl_matrix* D    = NULL;
static gsl_matrix* PtPPti    = NULL;
static gsl_matrix* ImPtPPtiP = NULL;
static gsl_matrix* aux       = NULL;


/**
 * \brief Proximal operator for the PACO-BCS problem.
 * This is an alternate projection onto two convex sets:
 * a) the consensus set, CA = 0
 * b) the compressive sensing set PA = B
 *
 * @param[out] proxA output matrix with the result of the proximal operator
 * @param A input to the proximal operator
 * @param lambda parameter of the proximal operator (not used in this case)
 * @return prox(A)
 */
void paco_bcs_prox ( gsl_matrix *PA, const gsl_matrix *A, const double lambda ) {
    paco_debug ( "prox_bcs: alternate projection.\n" );

    if ( A != PA ) {
        gsl_matrix_memcpy ( PA, A );
    }
    gsl_matrix* A1 = aux;
    gsl_matrix* A2 = PA;
    gsl_matrix* prevA = gsl_matrix_calloc(A1->size1,A1->size2);
    //
    // repeat until convergence:
    //
    for (index_t iter = 0; iter < 100; ++iter) {
        //
        // 1) project onto consensus set
        //
        gsl_matrix_memcpy(prevA,A2);
        gsl_matrix_memcpy(A1,A2);
        mapping->stitch ( V, A1 );
        mapping->extract ( A1, V );
        //
        // 2) project onto BCS set: PA = B
        // A <- A - Pt(PPt)i(PA - B)
        //
        gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,A1,ImPtPPtiP,0.0,A2);
        gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,B,PtPPti,1.0,A2);
        //
        // check convergence
        //
        const double dif = sqrt(gsl_matrix_dist2_squared(prevA,A2)) / (A1->size1*A1->size2);
        paco_debug("iter %d dif %f\n",iter,dif);
        if (dif < 5e-4) {
            break;
        }
    }
    gsl_matrix_free(prevA);
}


//============================================================================

/**
 * as this is an indicator function, and B is always feasible, the value is always 0
 */
double paco_bcs_eval ( const gsl_matrix *B ) {
    return 0;
}

//============================================================================

/**
 * initialize problem structures and state
 */
void paco_bcs_create ( const paco_problem_st* problem ) {

    paco_info ( "Creating bcs problem...\n" );

    const paco_data_st* data = &problem->data;
    const gsl_matrix *input = data->input;
    mapping = &problem->mapping;

    const index_t M = input->size1;
    const index_t N = input->size2;

    P = problem->config.P;
    B = problem->config.B;
    D = problem->config.D;

    paco_info("BCS: patches matrix     X: number of patches %d, patch dim %d\n",mapping->num_mapped_patches, mapping->patch_dim);

    if (P==NULL) {
        paco_error("Invalid BCS problem. Missing projection matrix P.\n");
        exit(1);
    }
    paco_info("BCS: measurement matrix P: %d measures of dimension %d each\n",P->size1,P->size2);
    if (B == NULL) {
        paco_error("Invalid BCS problem. Missing samples matrix B.\n");
        exit(1);
    }
    paco_info("BCS: measures matrix    B: %d measurement vectors of dimension %d each\n",B->size1,B->size2);
    if (D == NULL) {
        paco_error("Invalid BCS problem. Missing differential op. matrix D.\n");
        exit(1);
    }
    paco_info("BCS: differential o.    D: input dimension %d, output dimension %d\n",D->size2,D->size1);
    if (D->size2 != mapping->patch_dim) {
        paco_error("Invalid BCS problem!\nDifferential operator input dimension (%d) does not match patch dimension (%d).\n",
        D->size2,mapping->patch_dim);
        exit(1);
    }
    if (P->size2 != mapping->patch_dim) {
        paco_error("Invalid BCS problem!\nProjection operator input dimension (%d) does not match patch dimension (%d).\n",
        P->size2,mapping->patch_dim);
        exit(1);
    }
    if (P->size1 != B->size2) {
        paco_error("Invalid BCS problem!\nProjection operator output dimension (%d) does not match samples dimension (%d).\n",
        P->size1,B->size2);
        exit(1);
    }
    if (B->size1 != mapping->num_mapped_patches) {
        paco_error("Invalid BCS problem!\nNumber of measurements (%d) does not match number of patches (%d).\n",
        B->size1,mapping->num_mapped_patches);
        exit(1);
    }

    V = gsl_matrix_calloc ( M, N );
    aux       = gsl_matrix_alloc(mapping->num_mapped_patches, mapping->patch_dim);

    //
    // everything legit
    //
    const index_t m = P->size2;
    const index_t k = P->size1;
    gsl_matrix* PPti = gsl_matrix_calloc(k,k);
    ImPtPPtiP = gsl_matrix_calloc(m,m);
    write_ascii_matrix("P_c.txt",P);
    gsl_blas_dsyrk ( CblasLower, CblasNoTrans, 1.0, P, 0.0, PPti );
    write_ascii_matrix("PPt_c.txt",PPti);
    gsl_linalg_cholesky_decomp ( PPti );
    gsl_linalg_cholesky_invert ( PPti );
    for (index_t i = 0; i < k; ++i) {
        for (index_t j = (i+1); j < k; ++j) {
            gsl_matrix_set(PPti,i,j,gsl_matrix_get(PPti,j,i));
        }
    }
    write_ascii_matrix("PPti_c.txt",PPti);
    PtPPti = gsl_matrix_calloc(m,k);

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, P, PPti, 0.0, PtPPti);
    write_ascii_matrix("PtPPti_c.txt",PtPPti);
    // this one is symmetric so it is the same 
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, PtPPti, P, 0.0, ImPtPPtiP);
    for (index_t i = 0; i < m; ++i) {
        ImPtPPtiP->data[i*(m+1)] += 1.0; // I - PtPPtiP 
    }
    write_ascii_matrix("ImPtPPtiP_c.txt",ImPtPPtiP);
    gsl_matrix_free(PPti);
}

//============================================================================
#include "paco_init.h"

void paco_init_bcs ( struct paco_problem* problem ) {
    paco_init_base ( problem );
    paco_iterate_st* iter = &problem->iter;
    gsl_matrix_memcpy ( iter->X, problem->data.input );
    const size_t m = P->size2;
    const size_t n = B->size1;
    gsl_matrix* PtPi = gsl_matrix_calloc(m,m);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, P, P, 0.0, PtPi); // B is stored transposed, so this is actually BtP
    write_ascii_matrix("PtP_c.txt",PtPi);
    for (index_t i = 0; i < m; ++i) {
        PtPi->data[i*(m+1)] += 0.1; //  some ridge (same as Python)
    }
    gsl_linalg_cholesky_decomp ( PtPi );
    gsl_linalg_cholesky_invert ( PtPi );
    for (index_t i = 0; i < m; ++i) {
        for (index_t j = (i+1); j < m; ++j) {
            gsl_matrix_set(PtPi,i,j,gsl_matrix_get(PtPi,j,i));
        }
    }
    write_ascii_matrix("PtPi_c.txt",PtPi);

    gsl_matrix* PtB = gsl_matrix_calloc(n,m); // store transposed
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, B, P, 0.0, PtB); // B is stored transposed, so this is actually BtP
    write_ascii_matrix("PtB_c.txt",PtB);

    // no sirve porque tiene que ser de a columnas
    //gsl_linalg_cholesky_solve_mat(PtP, PtB, iter->A);
    
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0, PtB, PtPi, 0.0, iter->A);
}

//============================================================================

/**
 * destroy problem structures and free any allocated space
 */
void paco_bcs_destroy() {
    paco_info ( "Destroying bcs problem...\n" );
    gsl_matrix_free ( V );
    gsl_matrix_free (aux);
    gsl_matrix_free( D );
    gsl_matrix_free ( PtPPti );
    gsl_matrix_free ( ImPtPPtiP );
}

//============================================================================


void paco_bcs_monitor ( struct paco_problem* problem )  {

    paco_iterate_st* iter = &problem->iter;
    if ((iter->k % 10)) {
        return;
    }
    const gsl_matrix* mask = problem->data.mask;
    paco_config_st* cfg = &problem->config;
    paco_mapping_st* map = &problem->mapping;
    // this problem has no input image, so if one is specified, we use it as reference
    const gsl_matrix* ref = problem->data.reference ? problem->data.reference : problem->data.input;

    char aux[128];
    paco_image_st *iter_img;

    //
    // partial result for main variable A
    //
    //paco_info("min(A)=%f max(A)=%f\n",gsl_matrix_min(iter->A),gsl_matrix_max(iter->A));
    snprintf ( aux, 128, "%s/%05d_A.png", cfg->work_dir, iter->k );
    map->stitch ( iter->X, iter->A );
    const double n = iter->X->size1 * iter->X->size2;
    for (index_t i = 0; i < n; ++i) {
	if (iter->X->data[i] < 0) {
		iter->X->data[i] = 0;
	} else if (iter->X->data[i] > 1) {
		iter->X->data[i] = 1;
	}	
    }
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

#if 0
static void init_differential_operator ( const index_t m ) {
    D               = gsl_matrix_calloc ( 2 * m, m );
    const index_t w = ( index_t ) sqrt ( ( double ) m );
    //
    // create differential operator
    //
    // row 2*i contains the horizontal operator for the i-th pixel
    // row 2*i+1 contains the vertical operator for the i-th pixel
    for ( index_t pi = 0, i = 0; pi < w; ++pi ) {
        for ( index_t pj = 0; pj < w; ++pj, ++i ) {
            const index_t ih = 2 * i; // index of horizontal derivative for i-th pix
            const index_t iv = 2 * i + 1; //          vertical
            const index_t jc = i;     // column index of i-th pixel in patch
            //
            // vertical operator
            //
            sample_t cv = 0;
            if ( pi > 0 ) {
                const index_t jn = jc - w; // north
                gsl_matrix_set ( D, iv, jn, -0.5 );
                cv += 0.5;
            }
            if ( pi < ( w - 1 ) ) {
                const index_t js = jc + w; // south
                gsl_matrix_set ( D, iv, js, -0.5 );
                cv += 0.5;
            }
            gsl_matrix_set ( D, iv, jc, cv );
            //
            // horizontal operator
            //
            sample_t ch = 0;
            if ( pj > 0 ) {
                const index_t jw = jc - 1;
                gsl_matrix_set ( D, ih, jw, -0.5 );
                ch += 0.5;
            }
            if ( pj < ( w - 1 ) ) {
                const index_t je = jc + 1;
                gsl_matrix_set ( D, ih, je, -0.5 );
                ch += 0.5;
            }
            gsl_matrix_set ( D, ih, jc, ch );
        }
    }
    //Dcoo = gsl_spmatrix_alloc_nzmax ( 2 * m, m, 6 * m, GSL_SPMATRIX_TRIPLET );
    //gsl_spmatrix_d2sp ( Dcoo, D );
    //Dcrs = gsl_spmatrix_crs ( Dcoo );

    DtD = gsl_matrix_calloc ( m, m );
    gsl_blas_dsyrk ( CblasLower, CblasTrans, 1.0, D, 0.0, DtD );
}

//============================================================================

static void init_sensing_matrix ( const index_t m , const index_t k) {
}
#endif
