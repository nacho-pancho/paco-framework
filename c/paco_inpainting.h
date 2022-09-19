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
 * \file paco_inpainting.h
 * \brief inpainting problem definition.
 *
 */
#ifndef PACO_INPAINTING_H
#define PACO_INPAINTING_H


#include "paco_problem.h"

/**
 * \brief Creates an instance of a PACO inpainting problem.
 * The mask suffices for defining all problem-related initialization.
 * Input data, etc, is loaded afterwards before processing each channel
 */
paco_function_st paco_inpainting();


#endif
