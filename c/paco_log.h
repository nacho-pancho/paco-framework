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
 * \file paco_log.h
 * \brief Bare bones logger
 */
#ifndef PACO_LOG
#define PACO_LOG
#include <stdarg.h>
#include <stdio.h>

#define PACO_LOG_NONE    100000
#define PACO_LOG_ERROR   100
#define PACO_LOG_WARNING 90
#define PACO_LOG_QUIET   80
#define PACO_LOG_INFO    70
#define PACO_LOG_XINFO   65
#define PACO_LOG_DEBUG   60
#define PACO_LOG_XDEBUG  50
#define PACO_LOG_ALL     0

void set_log_level ( int level );
void set_log_stream ( FILE *stream );

void paco_warn ( const char *format, ... );
void paco_info ( const char *format, ... );
void paco_xinfo ( const char *format, ... );
void paco_error ( const char *format, ... );
void paco_debug ( const char *format, ... );
void paco_xdebug ( const char *format, ... );

char level_warn ( void );
char level_info ( void );
char level_error ( void );
char level_debug ( void );

#endif
