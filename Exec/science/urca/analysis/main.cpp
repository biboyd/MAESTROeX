#include <AMReX_CArena.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>


#include <extern_parameters.H>

#include <fstream>

#include <conv_slopes.H>

int
main (int   argc,
      char* argv[])
{

  //
  // Make sure to catch new failures.
  //
  amrex::Initialize(argc,argv);

  // initialize the runtime parameters


  // we use a single file name for the extern name list and
  // the name list used by the initialization

  init_extern_parameters();
  conv_slopes();

  amrex::Finalize();

  return 0;
}
