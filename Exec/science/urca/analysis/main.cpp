#include <AMReX_CArena.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>


#include <extern_parameters.H>

#include <fstream>
#include <eos.H>
#include <conv_slopes.H>
#include <conv_radial.H>
#include <neutrinos.H>

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

  // init eos
  eos_init();
  if (do_neutrinos){
    neutrinos();
  }
  else if (dim == 1){
    conv_slopes();
  }
  else if (dim == 3) {
    conv_radial();
  }
  else {
    amrex::Print() << "set do_nuetrinos=1 or dim=1 for modelfiles, or dim=3 for plotfiles" << std::endl;
  }

  amrex::Finalize();

  return 0;
}
