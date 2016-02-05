# Makefile for compilation fo complete project
# Source code files should be in folder src/
# Header files should be in folder include/


# C compiler
CC = g++

# MPI C++ compiler
MPICC = mpi++.mpich2

# OpenMP compiler option
OMP = -fopenmp

# Folder for header files
head = -I include/
# Additional option
OPT = -c 
#OPT = -O3 -c 

VPATH = src

PROG = multigrid
OBJECTS = array3d.o boundarycondition3d.o matvecAv.o misc.o prolongation3d.o restriction3d.o smoother7p.o timer.o

$(PROG).exe: $(OBJECTS) $(PROG).cpp
	@echo "Compile" $@
	@$(CC) $(OMP) $(head) $^ -o $@ 

%.o: %.cpp
	@echo "Compile" $^
	@$(CC) $(OMP) $(head) $^ $(OPT)

clean:
	rm -f *.o *.exe *.py
	
.PHONY: clean, run 
