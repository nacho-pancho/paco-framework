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
 * \file paco_log.c
 * \brief Bare bones logger
 */
#include "paco_log.h"
#include <stdio.h>

static int log_level = PACO_LOG_INFO;
static FILE *log_stream = NULL;

void set_log_level ( int l ) {
    log_level = l;
}

void set_log_stream ( FILE *stream ) {
    log_stream = stream;
}

void  paco_error ( const char *format, ... ) {
    va_list args;
    va_start ( args, format );

    if ( log_level <= PACO_LOG_ERROR ) {
        vfprintf ( stderr, format, args );
    }

    va_end ( args );
}

void paco_warn ( const char *format, ... ) {
    va_list args;
    va_start ( args, format );

    if ( log_level <= PACO_LOG_WARNING ) {
        vfprintf ( stderr, format, args );
    }

    va_end ( args );
}

void paco_info ( const char *format, ... ) {
    va_list args;

    if ( !log_stream ) {
        log_stream = stdout;
    }

    va_start ( args, format );

    if ( log_level <= PACO_LOG_INFO ) {
        vfprintf ( log_stream, format, args );
    }

    va_end ( args );
}

void paco_xinfo ( const char *format, ... ) {
    va_list args;

    if ( !log_stream ) {
        log_stream = stdout;
    }

    va_start ( args, format );

    if ( log_level <= PACO_LOG_XINFO ) {
        vfprintf ( log_stream, format, args );
    }

    va_end ( args );
}

void paco_debug ( const char *format, ... ) {
    va_list args;

    if ( !log_stream ) {
        log_stream = stdout;
    }

    va_start ( args, format );

    if ( log_level <= PACO_LOG_DEBUG ) {
        vfprintf ( log_stream, format, args );
    }

    va_end ( args );
}

void paco_xdebug ( const char *format, ... ) {
    va_list args;

    if ( !log_stream ) {
        log_stream = stdout;
    }

    va_start ( args, format );

    if ( log_level <= PACO_LOG_XDEBUG ) {
        vfprintf ( log_stream, format, args );
    }

    va_end ( args );
}



char level_warn ( void ) {
    return log_level <= PACO_LOG_WARNING;
}
char level_info ( void ) {
    return log_level <= PACO_LOG_INFO;
}
char level_error ( void ) {
    return log_level <= PACO_LOG_ERROR;
}
char level_debug ( void ) {
    return log_level <= PACO_LOG_DEBUG;
}
