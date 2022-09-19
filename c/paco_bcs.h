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
 * \file paco_bcs.h
 * \brief Block Compressed Sensing problem definition.
 *
 */
#ifndef PACO_BCS_H
#define PACO_BCS_H


#include "paco_problem.h"

/**
 * \brief Creates an instance of a PACO bcs problem.
 * The mask suffices for defining all problem-related initialization.
 * Input data, etc, is loaded afterwards before processing each channel
 */
paco_function_st paco_bcs();

void paco_init_bcs ( struct paco_problem* );

void paco_bcs_monitor ( struct paco_problem* problem );

#endif
