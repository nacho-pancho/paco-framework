#ifndef PACO_INIT_H
#define PACO_INIT_H
#include "paco_types.h"
#include "paco_config.h"
#include "paco_data.h"

struct paco_problem;

typedef void ( *paco_init_f ) ( struct paco_problem* problem );

void paco_init_base ( struct paco_problem* problem );

void paco_init_default ( struct paco_problem* problem );

void paco_init_missing_with_average ( struct paco_problem* problem );


#endif