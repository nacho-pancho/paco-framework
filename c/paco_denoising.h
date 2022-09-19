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
 * \file paco_denoising.h
 * \brief denoising problem definition.
 *
 */
#ifndef PACO_DENOISING_H
#define PACO_DENOISING_H

#include "paco_problem.h"

#define DENOISING_MODE_PATCH_MAP   0
#define DENOISING_MODE_SIGNAL_MAP  1
#define DENOISING_MODE_PATCH_BALL  2
#define DENOISING_MODE_SIGNAL_BALL 3

/**
 * \brief The denoising cost function depends on a surrogate prior and a denoisig mode.
 */
paco_function_st paco_denoising_cost ( paco_function_st* prior, int mode );

/**
 * \brief The denoising constraint function depends on a surrogate prior and a denoisig mode.
 */
paco_function_st paco_denoising_constraint ( int mode );

#endif
