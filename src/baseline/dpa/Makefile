#CPLEX system and lib format
SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic

# CPLEX directory
CPLEX_DIRECTORY ?= /opt/ibm/ILOG/CPLEX_Studio1210
CPLEXDIR      = $(CPLEX_DIRECTORY)/cplex
CONCERTDIR    = $(CPLEX_DIRECTORY)/concert

# ---------------------------------------------------------------------
# Compiler selection 
# ---------------------------------------------------------------------

CCC ?= g++
OPT ?= -O3

# ---------------------------------------------------------------------
# Compiler options 
# ---------------------------------------------------------------------

CCOPT = -m64 -Wall -fPIC $(OPT) -fexceptions -DIL_STD

#  ---------------------------------------------------------------------
# Link options and libraries
# ---------------------------------------------------------------------
#libraries
CPLEXLIBDIR   = $(CPLEXDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
CONCERTLIBDIR = $(CONCERTDIR)/lib/$(SYSTEM)/$(LIBFORMAT)

#includes
CONCERTINCDIR = $(CONCERTDIR)/include
CPLEXINCDIR   = $(CPLEXDIR)/include

# ---------------------------------------------------------------------
# Source and include file directory
# ---------------------------------------------------------------------

ROOTDIR       = .
EXDIR         = $(ROOTDIR)/src


EXINC         = $(EXDIR)
EXOBJ	       = ./obj
EXSRC         = $(EXDIR)
BIN_DIR       = ../../../resources/bin/baseline/dpa
EXECUTABLE    = $(BIN_DIR)/main

# ---------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------
#include flags
CCFLAGS = $(CCOPT) -isystem $(CPLEXINCDIR) -isystem $(CONCERTINCDIR) -I$(EXINC)


#library flags
CCLNFLAGS = -L$(CPLEXLIBDIR) -lilocplex -lcplex -L$(CONCERTLIBDIR) -lconcert -lm -pthread -ldl

$(info $$CPLEX_DIRECTORY is [${CPLEX_DIRECTORY}])
$(info $$CPLEXDIR is [${CPLEXDIR}])
$(info $$CPLEXINCDIR is [${CPLEXINCDIR}])
$(info $$CPLEXLIBDIR is [${CPLEXLIBDIR}])

all: $(EXECUTABLE)

main: $(EXECUTABLE)

$(EXECUTABLE): $(EXSRC)/main.cpp $(EXOBJ)/DefiningPoint.o | $(BIN_DIR)
	${CCC} ${CCFLAGS} $(EXSRC)/main.cpp $(EXOBJ)/DefiningPoint.o ${CCLNFLAGS} -o $(EXECUTABLE)

$(EXOBJ)/DefiningPoint.o: $(EXSRC)/DefiningPoint.cpp $(EXINC)/DefiningPoint.h $(EXINC)/Box.h $(EXINC)/Efficient.h
	mkdir -p $(EXOBJ)
	$(CCC) -c $(CCFLAGS) $(EXSRC)/DefiningPoint.cpp -o $(EXOBJ)/DefiningPoint.o

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(EXOBJ)
	rm -f $(EXECUTABLE)



