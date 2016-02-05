# Makefile for compilation fo complete project
# Source code files should be in folder src/
# Header files should be in folder include/


# C compiler
CC = g++

# MPI C++ compiler
MPICC = mpic++

# OpenMP compiler option
OMP = -fopenmp -O3 -Wall
#OMP = -fopenmp -O3 -xW
#OMP = -fopenmp -g
#OMP = -O0

# Folder for header files
head = -I include/
# Additional option
#OPT = -c -Wall
#OPT = -O3 -Wall -c 
OPT = -c

VPATH = src

PROG = multigrid
#PROG = test
OBJECTS = array3d.o communication.o matvecAv.o misc.o prolongation3d.o restriction3d.o smoother7p.o timer.o

$(PROG).exe: $(OBJECTS) $(PROG).o
	@echo "Link" $@
	@$(MPICC) $(OMP) $(head) $^ -o $@ 

%.o: %.cpp
	@echo "Compile" $@
	@$(MPICC) $(OMP) $(head) $^ $(OPT)

clean:
	rm -f *.o *.exe *.py
	
.PHONY: clean, run 
