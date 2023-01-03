//
// This program takes a 3-d cartesian plotfile and calculates
// the actual, adiabatic, and ledoux thermodynamic gradients.
//
// The actual thermodynamic gradient is computed along the
// radial direction, constructed from gradients along
// the cartesian coordinate directions.
//
// See MAESTRO/docs/thermo_notes for details.
//
// Adapted to c++ by Brendan Boyd 2022-12-13 from original fortran code
// 
#ifndef CONV_RADIAL_H
#define CONV_RADIAL_H

#include <sstream>

#include <extern_parameters.H>
#include <eos.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Array.H>
#include <string>

void conv_radial(){

   //get plotdata   
   PlotFileData pf(plotfilename);
   const int nlevs = pf.finestLevel() + 1;

   // init vectors needed to writeout
   Vector<std::string> var_names = pf.varNames();
   Vector<int> lev_steps(nlevs);
   Vector<IntVect> ref_ratio(nlevs);
   Vector<MultiFab> new_mfs(nlevs);
   Vector<Geometry> geoms(nlevs);

   // add conv names
   Vector<std::string> conv_var_names;
   conv_var_names.push_back("conv_actual");
   conv_var_names.push_back("conv_adiabatic");
   conv_var_names.push_back("conv_ledoux");
   
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

      // define new mf with 3 extra componenets
      new_mfs[ilev].define(mf.boxArray(),
                      mf.DistributionMap(),
                      3,
                      mf.nGrow());

      // copy data into new_mfs[ilev]
      //MultiFab::Copy(new_mfs[ilev], mf, 0, 0, mf.nComp(), mf.nGrow());
      
      // construct geometry from plotfile
      const amrex::RealBox rb(pf.probLo(), pf.probHi());
      const Array<int,3> is_per({0, 0, 0});
      geoms[ilev].define(pf.probDomain(ilev), rb, pf.coordSys(), is_per);

      // index for first convective gradient
      int ind_FirstSpec;

      //find first spec indice
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
         Array4<Real> const& conv_arr = new_mfs[ilev].array(mfi);
         

         // loop over all cells
         ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

            // initialize EOS
            eos_t eos_state;
            eos_state.T    = temp_arr(i, j, k);
            eos_state.p    = pres_arr(i, j, k);
            eos_state.rho  = rho_arr(i, j, k);
            for (auto comp = 0; comp < NumSpec; ++comp) {
               eos_state.xn[comp] = X_arr(i, j, k, comp);
            }
            eos(eos_input_rp, eos_state); // use rt instead?
            eos_xderivs_t eos_xderivs = composition_derivatives(eos_state);

            // calculate thermo variables. dpdr = dpdrho here
            Real chi_rho = eos_state.rho * eos_state.dpdr / eos_state.p;
            Real chi_t = eos_state.T * eos_state.dpdT / eos_state.p;
            
            Real chi_X[NumSpec];
            for (int comp=0; comp < NumSpec; ++comp){
               chi_X[comp] = eos_state.xn[comp] * eos_xderivs.dpdX[comp]/ (eos_state.p * chi_t);
            }


            Real dtdx, dpdx, dXdx[NumSpec];
            Real dtdy, dpdy, dXdy[NumSpec];
            Real dtdz, dpdz, dXdz[NumSpec]; 
            Real dtdr, dpdr, dXdr[NumSpec];
            Real actual, adiabatic, ledoux, dXdp[NumSpec];

            // construct actual gradient 
            // x-direction
            // forward diff
            if (i == lo.x){
               dtdx = (temp_arr(i+1, j, k) - temp_arr(i, j, k)) / dx;
               dpdx = (pres_arr(i+1, j, k) - pres_arr(i, j, k)) / dx;

               for (int comp=0; comp < NumSpec; ++comp){
                  dXdx[comp] = (X_arr(i+1, j, k, comp) - X_arr(i, j, k, comp))/dx;
               }
            } 
            // backward diff
            else if (i == hi.x){
               dtdx = (temp_arr(i, j, k) - temp_arr(i-1, j, k)) / dx;
               dpdx = (pres_arr(i, j, k) - pres_arr(i-1, j, k)) / dx;

               for (int comp=0; comp < NumSpec; ++comp){
                  dXdx[comp] = (X_arr(i, j, k, comp) - X_arr(i-1, j, k, comp))/dx;
               }
            } 
            // centered diff
            else {
               dtdx = 0.5_rt * (temp_arr(i+1, j, k) - temp_arr(i-1, j, k)) / dx;
               dpdx = 0.5_rt * (pres_arr(i+1, j, k) - pres_arr(i-1, j, k)) / dx;

               for (int comp=0; comp < NumSpec; ++comp){
                  dXdx[comp] = 0.5_rt * (X_arr(i+1, j, k, comp) - X_arr(i-1, j, k, comp))/dx;
               }
            }

            // y-direction
            // forward diff
            if (j == lo.y){
               dtdy = (temp_arr(i, j+1, k) - temp_arr(i, j, k)) / dy;
               dpdy = (pres_arr(i, j+1, k) - pres_arr(i, j, k)) / dy;

               for (int comp=0; comp < NumSpec; ++comp){
                  dXdy[comp] = (X_arr(i, j+1, k, comp) - X_arr(i, j, k, comp)) / dy;
               }
            } 
            // backward diff
            else if (j == hi.y){
               dtdy = (temp_arr(i, j, k) - temp_arr(i, j-1, k)) / dy;
               dpdy = (pres_arr(i, j, k) - pres_arr(i, j-1, k)) / dy;

               for (int comp=0; comp < NumSpec; ++comp){
                  dXdy[comp] = (X_arr(i, j, k, comp) - X_arr(i, j-1, k, comp)) / dy;
               }
            } 
            // centered diff
            else {
               dtdy = 0.5_rt * (temp_arr(i, j+1, k) - temp_arr(i, j-1, k)) / dy;
               dpdy = 0.5_rt * (pres_arr(i, j+1, k) - pres_arr(i, j-1, k)) / dy;

               for (int comp=0; comp < NumSpec; ++comp){
                  dXdy[comp] = 0.5_rt * (X_arr(i, j+1, k, comp) - X_arr(i, j-1, k, comp)) / dy;
               }
            }

            // z-direction
            // forward diff
            if (k == lo.z){
               dtdz = (temp_arr(i, j, k+1) - temp_arr(i, j, k)) / dz;
               dpdz = (pres_arr(i, j, k+1) - pres_arr(i, j, k)) / dz;

               for (int comp=0; comp < NumSpec; ++comp){
                  dXdz[comp] = (X_arr(i, j, k+1, comp) - X_arr(i, j, k, comp)) / dz;
               }
            } 
            // backward diff
            else if (k == hi.z){
               dtdz = (temp_arr(i, j, k) - temp_arr(i, j, k-1)) / dz;
               dpdz = (pres_arr(i, j, k) - pres_arr(i, j, k-1)) / dz;

               for (int comp=0; comp < NumSpec; ++comp){
                  dXdz[comp] = (X_arr(i, j, k, comp) - X_arr(i, j, k-1, comp)) / dz;
               }
            } 
            // centered diff
            else {
               dtdz = 0.5_rt * (temp_arr(i, j, k+1) - temp_arr(i, j, k-1)) / dz;
               dpdz = 0.5_rt * (pres_arr(i, j, k+1) - pres_arr(i, j, k-1)) / dz;

               for (int comp=0; comp < NumSpec; ++comp){
                  dXdz[comp] = 0.5_rt * (X_arr(i, j, k+1, comp) - X_arr(i, j, k-1, comp)) / dz;
               }
            }

            //calculate position relative to center of star
            const Real xpos = prob_lo[0] + (Real(i) + 0.5) * dx - center_p[0];
            const Real ypos = prob_lo[1] + (Real(j) + 0.5) * dy - center_p[1];
            const Real zpos = prob_lo[2] + (Real(k) + 0.5) * dz - center_p[2];

            Real rpos = std::sqrt(xpos*xpos + ypos*ypos + zpos*zpos);
            Real dr = std::sqrt(dx*dx + dy*dy + dz*dz);

            // now calculate radial gradients
            dtdr = (dtdx * xpos + dtdy * ypos + dtdz * zpos) / rpos;
            dpdr = (dpdx * xpos + dpdy * ypos + dpdz * zpos) / rpos;

            for (int comp=0; comp < NumSpec; ++comp){
               dXdr[comp] = (dXdx[comp] * xpos + dXdy[comp] * ypos + dXdz[comp] * zpos) / rpos;
            }

            //finally, actual gradient
            if (rho_arr(i, j, k) <= low_cutoff || std::abs(dpdr*dr) <= small){
               actual = 0.0_rt;
               for (int comp=0; comp < NumSpec; ++comp){
                  dXdp[comp] = 0.0_rt;
               }
            } 
            else {
               actual = pres_arr(i, j, k) * dtdr / (dpdr * temp_arr(i, j, k));
               for (int comp=0; comp < NumSpec; ++comp){
                  dXdp[comp] = dXdr[comp]/dpdr;
               }
            }

            // construct adiabatic gradient
            adiabatic = (eos_state.gam1 - chi_rho) / (eos_state.gam1 * chi_t);

            //ledoux gradient. adiabat with correction
            ledoux = adiabatic;

            for (int comp=0; comp < NumSpec; ++comp){
               if (X_arr(i, j, k, comp) > small) { 
                  ledoux -= chi_X[comp] *pres_arr(i, j, k)*dXdp[comp] / X_arr(i, j, k, comp);
               }
            }

            conv_arr(i, j, k, 0) = actual;
            conv_arr(i, j, k, 1) = adiabatic;
            conv_arr(i, j, k, 2) = ledoux;
            
         });
      }
   }

   //const Vector<const MultiFab* > write_mfs = new_mfs;
   const Vector<Geometry> write_geoms = geoms;


   // check if single level
   if (nlevs == 1){

      WriteSingleLevelPlotfile(outfile, 
                               new_mfs[0], 
                               conv_var_names, 
                               geoms[0], 
                               pf.time(),
                               lev_steps[0]);
   }
   else{
      // write out with gradient fields
      WriteMultiLevelPlotfile(outfile,
                              nlevs,
                              GetVecOfConstPtrs(new_mfs),
                              conv_var_names,
                              write_geoms,
                              pf.time(),
                              lev_steps,
                              ref_ratio);

   }
}
#endif
