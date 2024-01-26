//
// This program takes a 3-d cartesian plotfile and calculates
// reduces teh number of saved variables
//
// 
#ifndef shrink_pfile_H
#define shrink_pfile_H

#include <sstream>

#include <extern_parameters.H>
#include <eos.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Array.H>
#include <string>

void shrink_pfile(){

   //get plotdata   
   PlotFileData pf(plotfilename);
   const int nlevs = pf.finestLevel() + 1;

   // init vectors needed to writeout
   Vector<std::string> var_names = pf.varNames();
   Vector<int> lev_steps(nlevs);
   Vector<IntVect> ref_ratio(nlevs);
   Vector<MultiFab> new_mfs(nlevs);
   Vector<Geometry> geoms(nlevs);

   // important indices
   int ine23 = network_spec_index("neon-23");
   int ina23 = network_spec_index("sodium-23");

   // add conv names
   Vector<std::string> small_var_names;
   small_var_names.push_back("rho");
   small_var_names.push_back("entropy");
   small_var_names.push_back("tfromp");
   small_var_names.push_back("p0pluspi");
   small_var_names.push_back("vort");
   small_var_names.push_back("velx");
   small_var_names.push_back("vely");
   small_var_names.push_back("velz");
   small_var_names.push_back("Hnuc");
   small_var_names.push_back("X(c12)");
   small_var_names.push_back("X(h1)");
   small_var_names.push_back("X(he4)");
   small_var_names.push_back("X(mg23)");
   small_var_names.push_back("X(n)");
   small_var_names.push_back("X(na23)");
   small_var_names.push_back("X(ne20)");
   small_var_names.push_back("X(ne23)");
   small_var_names.push_back("X(o16)");
   small_var_names.push_back("omegadot(na23)");
   small_var_names.push_back("omegadot(ne23)");
   small_var_names.push_back("omegadot(c12)");
  
   for (int ilev = pf.finestLevel(); ilev >= 0; --ilev) {
      // read plotfile data
      lev_steps[ilev] = pf.levelStep(ilev);
      ref_ratio[ilev] = IntVect(pf.refRatio(ilev), 
                                pf.refRatio(ilev), 
                                pf.refRatio(ilev));

      const MultiFab mf = pf.get(ilev);

      // define new mf with as many componenets as vector
      new_mfs[ilev].define(mf.boxArray(),
                      mf.DistributionMap(),
                      small_var_names.size(),
                      mf.nGrow());

      // copy data into new_mfs[ilev]
      //MultiFab::Copy(new_mfs[ilev], mf, 0, 0, mf.nComp(), mf.nGrow());
      
      // construct geometry from plotfile
      const amrex::RealBox rb(pf.probLo(), pf.probHi());
      const Array<int,3> is_per({0, 0, 0});
      geoms[ilev].define(pf.probDomain(ilev), rb, pf.coordSys(), is_per);

      // index for first convective gradient
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
               comp_name << "X(" << short_spec_names_cxx[comp] << ")";
               if (var_names[n] == comp_name.str()){
                  my_spec_idx[comp] = n - ind_FirstSpec;
               }
            }
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
         Array4<const Real> const& mf_arr = mf.array(mfi);
         Array4<Real> const& conv_arr = new_mfs[ilev].array(mfi);

         // loop over all cells
         ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

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
                  if (my_spec_idx[comp] != -1)
                     dXdx[comp] = (X_arr(i+1, j, k, comp) - X_arr(i, j, k, comp))/dx;
               }
            } 
            // backward diff
            else if (i == hi.x){
               dtdx = (temp_arr(i, j, k) - temp_arr(i-1, j, k)) / dx;
               dpdx = (pres_arr(i, j, k) - pres_arr(i-1, j, k)) / dx;

               for (int comp=0; comp < NumSpec; ++comp){
                  if (my_spec_idx[comp] != -1)
                     dXdx[comp] = (X_arr(i, j, k, comp) - X_arr(i-1, j, k, comp))/dx;
               }
            } 
            // centered diff
            else {
               dtdx = 0.5_rt * (temp_arr(i+1, j, k) - temp_arr(i-1, j, k)) / dx;
               dpdx = 0.5_rt * (pres_arr(i+1, j, k) - pres_arr(i-1, j, k)) / dx;

               for (int comp=0; comp < NumSpec; ++comp){
                  if (my_spec_idx[comp] != -1)
                     dXdx[comp] = 0.5_rt * (X_arr(i+1, j, k, comp) - X_arr(i-1, j, k, comp))/dx;
               }
            }

            // y-direction
            // forward diff
            if (j == lo.y){
               dtdy = (temp_arr(i, j+1, k) - temp_arr(i, j, k)) / dy;
               dpdy = (pres_arr(i, j+1, k) - pres_arr(i, j, k)) / dy;

               for (int comp=0; comp < NumSpec; ++comp){
                  if (my_spec_idx[comp] != -1)
                     dXdy[comp] = (X_arr(i, j+1, k, comp) - X_arr(i, j, k, comp)) / dy;
               }
            } 
            // backward diff
            else if (j == hi.y){
               dtdy = (temp_arr(i, j, k) - temp_arr(i, j-1, k)) / dy;
               dpdy = (pres_arr(i, j, k) - pres_arr(i, j-1, k)) / dy;

               for (int comp=0; comp < NumSpec; ++comp){
                  if (my_spec_idx[comp] != -1)
                     dXdy[comp] = (X_arr(i, j, k, comp) - X_arr(i, j-1, k, comp)) / dy;
               }
            } 
            // centered diff
            else {
               dtdy = 0.5_rt * (temp_arr(i, j+1, k) - temp_arr(i, j-1, k)) / dy;
               dpdy = 0.5_rt * (pres_arr(i, j+1, k) - pres_arr(i, j-1, k)) / dy;

               for (int comp=0; comp < NumSpec; ++comp){
                  if (my_spec_idx[comp] != -1)
                     dXdy[comp] = 0.5_rt * (X_arr(i, j+1, k, comp) - X_arr(i, j-1, k, comp)) / dy;
               }
            }

            // z-direction
            // forward diff
            if (k == lo.z){
               dtdz = (temp_arr(i, j, k+1) - temp_arr(i, j, k)) / dz;
               dpdz = (pres_arr(i, j, k+1) - pres_arr(i, j, k)) / dz;

               for (int comp=0; comp < NumSpec; ++comp){
                  if (my_spec_idx[comp] != -1)
                     dXdz[comp] = (X_arr(i, j, k+1, comp) - X_arr(i, j, k, comp)) / dz;
               }
            } 
            // backward diff
            else if (k == hi.z){
               dtdz = (temp_arr(i, j, k) - temp_arr(i, j, k-1)) / dz;
               dpdz = (pres_arr(i, j, k) - pres_arr(i, j, k-1)) / dz;

               for (int comp=0; comp < NumSpec; ++comp){
                  if (my_spec_idx[comp] != -1)
                     dXdz[comp] = (X_arr(i, j, k, comp) - X_arr(i, j, k-1, comp)) / dz;
               }
            } 
            // centered diff
            else {
               dtdz = 0.5_rt * (temp_arr(i, j, k+1) - temp_arr(i, j, k-1)) / dz;
               dpdz = 0.5_rt * (pres_arr(i, j, k+1) - pres_arr(i, j, k-1)) / dz;

               for (int comp=0; comp < NumSpec; ++comp){
                  if (my_spec_idx[comp] != -1)
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
               if (my_spec_idx[comp] != -1)
                  dXdr[comp] = (dXdx[comp] * xpos + dXdy[comp] * ypos + dXdz[comp] * zpos) / rpos;
               else
                  dXdr[comp] = 0.0_rt;
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
               if (eos_state.xn[comp] > small) { 
                  ledoux -= chi_X[comp] *pres_arr(i, j, k)*dXdp[comp] / eos_state.xn[comp];
               }
            }

            conv_arr(i, j, k, 0) = actual;
            conv_arr(i, j, k, 1) = adiabatic;
            conv_arr(i, j, k, 2) = ledoux;
            conv_arr(i, j, k, 3) = -pres_arr(i,j, k) * (dXdp[ine23] + dXdp[ina23]) / eos_state.y_e;
         });
                  
                  
         if (do_project1d){

            for (int i=lo.x; i <= hi.x; ++i){
               for (int j=lo.y; j <= hi.y; ++j){
                  for (int k=lo.z; k <=hi.z; ++k){
                     // calc radius
                     const Real xpos = prob_lo[0] + (Real(i) + 0.5) * dx - center_p[0];
                     const Real ypos = prob_lo[1] + (Real(j) + 0.5) * dy - center_p[1];
                     const Real zpos = prob_lo[2] + (Real(k) + 0.5) * dz - center_p[2];

                     const Real rpos = std::sqrt(xpos*xpos + ypos*ypos + zpos*zpos);
                     
                     //calc mass
                     Real mass = dx*dy*dz*rho_arr(i, j, k);

                     // find bin to place
                     for (int bin=0; bin < N_rad; ++bin){
                        if (proj_rad[bin] < rpos && proj_rad[bin + 1] > rpos){
                           proj_actual[bin]    += mass * conv_arr(i, j, k, 0);
                           proj_adiabatic[bin] += mass * conv_arr(i, j, k, 1);
                           proj_ledoux[bin]    += mass * conv_arr(i, j, k, 2);
                           proj_mass[bin]      += mass;
                           break;
                        }
                     }
                  }
               }
            }
         }
      }
   }
   ParallelDescriptor::ReduceRealSum(proj_actual.dataPtr(), proj_actual.size(), ParallelDescriptor::IOProcessorNumber());
   ParallelDescriptor::ReduceRealSum(proj_adiabatic.dataPtr(), proj_adiabatic.size(), ParallelDescriptor::IOProcessorNumber());
   ParallelDescriptor::ReduceRealSum(proj_ledoux.dataPtr(), proj_ledoux.size(), ParallelDescriptor::IOProcessorNumber());
   ParallelDescriptor::ReduceRealSum(proj_mass.dataPtr(), proj_mass.size(), ParallelDescriptor::IOProcessorNumber());

   // divide by mass vector to get projection
   if (do_project1d && ParallelDescriptor::IOProcessor()){
      for (int bin=0; bin < N_rad; ++bin){
         if (proj_mass[bin] == 0.){
            std::cout << "poj mass at bin " << bin << " is zero. use fewer bins" << std::endl;
         }
         else{
            proj_actual[bin]    /= proj_mass[bin];
            proj_adiabatic[bin] /= proj_mass[bin];
            proj_ledoux[bin]    /= proj_mass[bin];
         }
      }
      //writeout data stuff
      std::ofstream of;
      of.open(proj_outfile);

      of << "# npts = " << N_rad << std::endl;
      of << "# r, actual, adiabatic, ledoux" << std::endl;

      for (int bin=0; bin < N_rad; ++bin){
            of << std::setprecision(12) << std::setw(20) << proj_rad[bin] << " ";
            of << std::setprecision(12) << std::setw(20) << proj_actual[bin] << " ";
            of << std::setprecision(12) << std::setw(20) << proj_adiabatic[bin] << " ";
            of << std::setprecision(12) << std::setw(20) << proj_ledoux[bin] << " ";
            of << std::endl;
      }
   }
   //const Vector<const MultiFab* > write_mfs = new_mfs;
   const Vector<Geometry> write_geoms = geoms;


   //write out file
   if (writeplot){
      // check if single level
      if (nlevs == 1){

         WriteSingleLevelPlotfile(outfile, 
                                 new_mfs[0], 
                                 small_var_names, 
                                 geoms[0], 
                                 pf.time(),
                                 lev_steps[0]);
      }
      else{
         // write out with gradient fields
         WriteMultiLevelPlotfile(outfile,
                                 nlevs,
                                 GetVecOfConstPtrs(new_mfs),
                                 small_var_names,
                                 write_geoms,
                                 pf.time(),
                                 lev_steps,
                                 ref_ratio);

      }
   }
}
#endif
