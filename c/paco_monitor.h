#ifndef PACO_MONITOR_H
#define PACO_MONITOR_H

#include "paco_types.h"
#include "paco_config.h"

#define PACO_MONITOR_NONE   0
#define PACO_MONITOR_LINEAR 10
#define PACO_MONITOR_LOG2   20
#define PACO_MONITOR_LOG10  100

void set_monitoring_mode (int mode );

void set_monitoring_parameter (int par );

struct paco_problem;
/**
 * Prototipe for _monitor_ functions. These report intermediate results
 * during PACO iterations such as the current status of the optimizer,
 * partial outputs such as images or audio files, etc.
 */
typedef void ( *paco_monitor_f ) ( struct paco_problem* problem );

void paco_default_monitor ( struct paco_problem* problem );

#endif
