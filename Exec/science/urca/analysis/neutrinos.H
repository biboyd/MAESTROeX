//
// This program takes a 3-d cartesian plotfile and calculates
// the A=23 beta decay and electron capture rates as well as related nu loss rates
// and thermal nu loss rates
//
//
// Adapted to c++ by Brendan Boyd 2022-05-26 from conv_radial.H
// 
#ifndef NEUTRINOS_H
#define NEUTRINOS_H

#include <sstream>

#include <extern_parameters.H>
#include <eos.H>
#include <table_rates.H>
#include <burn_type.H>
#include <sneut5.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Array.H>
#include <string>

using namespace rate_tables;

void neutrinos(){

   //get plotdata   
   PlotFileData pf(plotfilename);
   const int nlevs = pf.finestLevel() + 1;

   // init vectors needed to writeout
   Vector<std::string> var_names = pf.varNames();
   Vector<int> lev_steps(nlevs);
   Vector<IntVect> ref_ratio(nlevs);
   Vector<MultiFab> new_mfs(nlevs);
   Vector<Geometry> geoms(nlevs);

   // add variable names
   Vector<std::string> weak_var_names;
   weak_var_names.push_back("A23_electron_capture_rate");
   weak_var_names.push_back("A23_beta_decay_rate");   
   weak_var_names.push_back("A23_electron_capture_nu_loss");
   weak_var_names.push_back("A23_beta_decay_nu_loss");
   weak_var_names.push_back("thermal_nu_loss");
   weak_var_names.push_back("cp");
  
   Vector<int> my_spec_idx(NumSpec, -1); //in network order. holds plotfile index.
   //find first spec indice and fill match array
   int ind_FirstSpec (-1);
   for (int n=0; n < var_names.size(); ++n){
      if (var_names[n][0] == 'X'){
         if (ind_FirstSpec < 0){
            ind_FirstSpec = n;
         }
         for (int comp=0; comp < NumSpec; ++comp){
            std::stringstream comp_name;
            comp_name << "X(";
            for (int ic=0; ic < short_spec_names_cxx[comp].size(); ++ic){
               char c = short_spec_names_cxx[comp][ic];
               if (ic == 0){
                  comp_name << static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
               } else {
                  comp_name << c;
               }
            }
            comp_name << ")";
            if (var_names[n] == comp_name.str()){
               my_spec_idx[comp] = n - ind_FirstSpec;
            }
         }
      }
   }

   // find number of discluded species
   int n_missing{0};
   for (int n = 0; n < NumSpec; ++n){
      if (my_spec_idx[n] < 0){
         n_missing++;
      }
   }

   // init the rhs. reaction stuff
   init_tabular();

   for (int ilev = pf.finestLevel(); ilev >= 0; --ilev) {
      // read plotfile data
      lev_steps[ilev] = pf.levelStep(ilev);
      ref_ratio[ilev] = IntVect(pf.refRatio(ilev), 
                                pf.refRatio(ilev), 
                                pf.refRatio(ilev));

      const std::string temp_name = "tfromp";
      const std::string pres_name = "p0pluspi";
      const std::string rho_name = "rho";

      const MultiFab mf = pf.get(ilev);
      const MultiFab temp = pf.get(ilev, temp_name);
      const MultiFab pi = pf.get(ilev, pres_name);
      const MultiFab rho = pf.get(ilev, rho_name);

      // define new mf with 5 componenets
      // 2 rates, 2 nu energy rate, 1 thermal energy rate
      new_mfs[ilev].define(mf.boxArray(),
                      mf.DistributionMap(),
                      weak_var_names.size(),
                      mf.nGrow());

      // copy data into new_mfs[ilev]
      //MultiFab::Copy(new_mfs[ilev], mf, 0, 0, mf.nComp(), mf.nGrow());
      
      // construct geometry from plotfile
      const amrex::RealBox rb(pf.probLo(), pf.probHi());
      const Array<int,3> is_per({0, 0, 0});
      geoms[ilev].define(pf.probDomain(ilev), rb, pf.coordSys(), is_per);

      //find first spec indice
      int ind_FirstSpec;
      for (int n=0; n < var_names.size(); ++n){
         if (var_names[n][0] == 'X'){
            ind_FirstSpec = n;
            break;
         }
      }
      auto prob_lo = pf.probLo();

      // assume the center is just half the total size plus probLo
      auto center_p = pf.probSize();
      for (int n=0; n < 3; ++n){
         center_p[n] = center_p[n] * 0.5_rt + prob_lo[n];
      }

      // define cell widths
      auto cellsize = pf.cellSize(ilev);
      Real dx = cellsize[0];
      Real dy = cellsize[1];
      Real dz = cellsize[2];
   
      for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi){
         // find bounds
         const auto tileBox = mfi.tilebox();
         const auto lo = amrex::lbound(tileBox);
         const auto hi = amrex::ubound(tileBox);
         
         // load arrays
         Array4<const Real> const& temp_arr = temp.array(mfi);
         Array4<const Real> const& pres_arr = pi.array(mfi);
         Array4<const Real> const& rho_arr = rho.array(mfi);
         Array4<const Real> const& X_arr = mf.array(mfi, ind_FirstSpec);
         Array4<Real> const& new_arr = new_mfs[ilev].array(mfi);

         // loop over all cells
         ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

            // initialize EOS
            eos_t eos_state;
            burn_t burn_state;

            eos_state.T    = temp_arr(i, j, k);
            eos_state.p    = pres_arr(i, j, k);
            eos_state.rho  = rho_arr(i, j, k);

            //calc frac missing
            Real frac_miss{1.0_rt};
            for (auto comp = 0; comp < NumSpec; ++comp) {
               if (my_spec_idx[comp] != -1){
                  frac_miss -= X_arr(i, j, k, my_spec_idx[comp]);
               }
            }
            // fill in composition
            for (auto comp = 0; comp < NumSpec; ++comp) {
               if (my_spec_idx[comp] != -1){
                  eos_state.xn[comp] = X_arr(i, j, k, my_spec_idx[comp]);
               } 
               else{
                  eos_state.xn[comp] = frac_miss/static_cast<Real>(n_missing);
               }
            }

            eos(eos_input_rt, eos_state); // use rt instead?
    
            int ine23 = network_spec_index("neon-23");
            int ina23 = network_spec_index("sodium-23");
            
            composition(eos_state);
            eos_to_burn(eos_state, burn_state);
            
            amrex::Real rate, drate_dt, edot_nu, edot_gamma;
            amrex::Real rhoy = burn_state.rho * burn_state.y_e;


            tabular_evaluate(j_Na23_Ne23_meta, j_Na23_Ne23_rhoy, j_Na23_Ne23_temp, j_Na23_Ne23_data,
                              rhoy, burn_state.T, rate, drate_dt, edot_nu, edot_gamma);
            Real r_ecap = rate;
            Real specific_energy_ecap = C::Legacy::n_A * burn_state.xn[ina23]/23 * (edot_nu + edot_gamma);

            tabular_evaluate(j_Ne23_Na23_meta, j_Ne23_Na23_rhoy, j_Ne23_Na23_temp, j_Ne23_Na23_data,
                              rhoy, burn_state.T, rate, drate_dt, edot_nu, edot_gamma);
            Real r_beta = rate;
            Real specific_energy_beta = C::Legacy::n_A * burn_state.xn[ine23]/23 * (edot_nu + edot_gamma);

            Real xr_ecap = burn_state.xn[ina23] * r_ecap;
            Real xr_beta = burn_state.xn[ine23] * r_beta;


            //thermal neutrino loss   
            Real sneut, dsneutdt, dsneutdd, dsnuda, dsnudz;

            constexpr int do_T_derivatives = 0;
            sneut5<do_T_derivatives>(burn_state.T, burn_state.rho, burn_state.abar, burn_state.zbar, sneut, dsneutdt, dsneutdd, dsnuda, dsnudz);

            //save values
            new_arr(i, j, k, 0) = xr_ecap;
            new_arr(i, j, k, 1) = xr_beta;
            new_arr(i, j, k, 2) = specific_energy_ecap;
            new_arr(i, j, k, 3) = specific_energy_beta;
            new_arr(i, j, k, 4) = sneut;
            new_arr(i, j, k, 5) = eos_state.cp;
         });
      }
   }
   const Vector<Geometry> write_geoms = geoms;

   //write out file
   if (writeplot){
      // check if single level
      if (nlevs == 1){

         WriteSingleLevelPlotfile(outfile, 
                                 new_mfs[0], 
                                 weak_var_names, 
                                 geoms[0], 
                                 pf.time(),
                                 lev_steps[0]);
      }
      else{
         // write out with gradient fields
         WriteMultiLevelPlotfile(outfile,
                                 nlevs,
                                 GetVecOfConstPtrs(new_mfs),
                                 weak_var_names,
                                 write_geoms,
                                 pf.time(),
                                 lev_steps,
                                 ref_ratio);

      }
   }
}
#endif
