CC=gcc

CFLAGS=-Wall -march=native
LFLAGS=-lfftw3 -lfftw3f -lpng -lm -lgsl -lgslcblas
ifdef DEBUG
  CFLAGS += -Werror -DDEBUG -O1 -Wall -g -fstack-protector
else
  CFLAGS+= -DNDEBUG -O3 # -fopenmp
  LFLAGS+= #-lgomp
endif

ifdef PROFILE
  CFLAGS += -pg
endif

HEADERS=$(wildcard *.h)
TOOL_SRC=$(wildcard *_tool.c)
TOOL_TARGETS=$(patsubst %.c,%,$(TOOL_SRC))

CORE=paco_config.o \
	paco_data.o \
	paco_mapping.o\
	paco_grid_mapping.o \
	paco_function.o \
	paco_dykstra.o \
	paco_monitor.o \
	paco_problem.o\
	paco_problem_spec.o \
	paco_admm.o \
	paco_ladmm.o \
	paco_io.o \
	paco_image.o \
	paco_log.o \
	paco_linalg.o \
	paco_metrics.o \
	paco_util.o \
	paco_init.o \
	paco_dct.o 


COST_FUNCTIONS=\
	paco_denoising_cost.o \
	paco_dct_l1.o \
	paco_gmm.o \
	paco_dict.o \
	paco_moe.o \
	paco_sparse_l1.o \
    paco_l12.o 


CONSTRAINTS=\
	paco_inpainting.o\
	paco_denoising_constraint.o \
	paco_bcs.o


OBJ=$(CORE) $(COST_FUNCTIONS) $(CONSTRAINTS)

all: $(OBJ) $(TOOL_TARGETS) 

%.o: %.c $(HEADERS) $(MAKEFILE_LIST)
	$(CC) $(CFLAGS) -c $< 

%_test: %_test.c $(OBJ) $(HEADERS) $(MAKEFILE)
	$(CC) -o $@ $< $(OBJ) $(CFLAGS) $(LFLAGS)

%_tool: %_tool.c $(OBJ) $(HEADERS) $(MAKEFILE)
	$(CC) -o $@ $< $(OBJ) $(CFLAGS) $(LFLAGS)

clean:
	rm -f *.o *test *tool

pull:
	git pull
	gtags .


