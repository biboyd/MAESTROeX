PRECISION = DOUBLE
PROFILE = FALSE

DEBUG = FALSE

COMP = gnu

USE_MPI = FALSE
USE_OMP = FALSE

USE_REACT = TRUE
MAESTROEX_HOME  := ../../../..

DEFINES += -DEXTRA_THERMO

# EOS and network
EOS_DIR := helmholtz
CONDUCTIVITY_DIR := stellar

NETWORK_TOP_DIR := $(MICROPHYSICS_HOME)/networks
NETWORK_DIR := ignition_reaclib/URCA-simple

Bpack := ./Make.package
Blocs := . 

EXTERN_SEARCH += . 

include $(MICROPHYSICS_HOME)/Make.Microphysics