#ifndef PACO_SOLVER_H
#define PACO_SOLVER_H
#include <gsl/gsl_matrix.h>
#include "paco_types.h"
#include "paco_monitor.h"

struct paco_problem;
/**
 * set solver for PACO problem; ADMM or LADMM are implemented
 */
typedef void ( *paco_solver_f ) ( struct paco_problem* problem );

/**
 * Given an instance of a PACO problem and a set of optimization parameters,
 * solve the problem using the ADMM method.
 */
void paco_admm ( struct paco_problem* problem );
/**
 * Given an instance of a PACO problem and a set of optimization parameters,
 * solve the problem using the Linearized ADMM method.
 */
void paco_ladmm ( struct paco_problem* problem );
#endif