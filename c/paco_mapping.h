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
 * \file paco_mapping.h
 * \brief Generic interface to signal<->patch mapping methods
 */
#ifndef PACO_MAPPING_H
#define PACO_MAPPING_H
#include "paco_types.h"
#include "paco_config.h"

//============================================================================

struct paco_problem;

typedef void ( *paco_mapping_create_f ) ( struct paco_problem* problem );

typedef void ( *paco_mapping_destroy_f ) ();

typedef void ( *paco_mapping_stitch_f ) ( gsl_matrix *I, const gsl_matrix *X );

typedef void ( *paco_mapping_extract_f ) ( gsl_matrix *X, const gsl_matrix *I );

//============================================================================

typedef struct paco_mapping {
    char name[64];
    index_t input_ncols;
    index_t input_nrows;
    index_t patch_dim;
    index_t  num_mapped_patches; ///< number of patches produced by extract
    index_t  num_mapped_samples; ///< number of signal samples which are effectively mapped
    index_t *idx_mapped_samples; ///< this is a SHALLOW reference to data stored in the particular mapping used
    paco_mapping_create_f create;
    paco_mapping_destroy_f destroy;
    paco_mapping_stitch_f stitch;
    paco_mapping_extract_f extract;
} paco_mapping_st;

//============================================================================

int paco_mapping_validate ( const paco_mapping_st* map );

//============================================================================
//
// COMMON FUNCTIONS
//
char is_patch_incomplete ( const gsl_matrix *mask,
                           const index_t i0,
                           const index_t j0,
                           const index_t height,
                           const index_t width,
                           const index_t scale );
//============================================================================

/** @return true if all the samples in a patch are known, i.e., it has no missing samples.  */
char is_patch_complete ( const gsl_matrix *mask,
                         const index_t i0,
                         const index_t j0,
                         const index_t height,
                         const index_t width,
                         const index_t scale );


//============================================================================

void paco_mapping_extract_single ( sample_t *patch,
                                   const gsl_matrix *img,
                                   const index_t i0,
                                   const index_t j0,
                                   const index_t width,
                                   const index_t height,
                                   const index_t decimate );

//============================================================================

gsl_matrix *paco_mapping_extract_all (
    const gsl_matrix *img,
    const index_t width,
    const index_t height,
    gsl_matrix **pX );

//============================================================================

gsl_matrix *paco_mapping_extract_complete (
    const gsl_matrix *img,
    const gsl_matrix *mask,
    const index_t width,
    const index_t height,
    gsl_matrix **pX );


//============================================================================

/**
 * subtract average from patches and save them in the dc vector.
 */
void paco_mapping_remove_dc ( gsl_matrix *X, gsl_vector *dc );

//============================================================================

sample_t paco_mapping_remove_dc_single ( gsl_vector *x );

//============================================================================

/**
 * add back the dc to a given set of patches
 */
void paco_mapping_add_back_dc ( const gsl_vector *dc, gsl_matrix *X );

//============================================================================

void paco_mapping_add_back_dc_single ( const sample_t dc, gsl_vector *x );

//============================================================================

void paco_mapping_stitch_single (
    const sample_t *patch,
    const index_t i0,
    const index_t j0,
    const index_t width,
    const index_t height,
    const index_t decimate,
    gsl_matrix *img );

//============================================================================

void paco_mapping_stitch_all ( const gsl_matrix *X,
                               const index_t width,
                               const index_t height,
                               gsl_matrix *img );

//============================================================================

void paco_mapping_stitch_complete (
    const gsl_matrix *X,
    const gsl_matrix* mask,
    const index_t width,
    const index_t height,
    gsl_matrix *img );

//============================================================================

#endif
