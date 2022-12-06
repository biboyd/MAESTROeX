#ifndef CONV_SLOPES_H
#define CONV_SLOPES_H

#include <sstream>

#include <extern_parameters.H>
#include <eos.H>
#include <ModelParser.H>

void conv_slopes() {

  // the chi_* variables are (\partial p / \partial *) at constant everything 
  // else
  Real chi_rho, chi_t, chi_X[NumSpec];
  Real dT, dp, dcomp[NumSpec], dXdp[NumSpec], dpdX[NumSpec];

  eos_t eos_state;
  eos_xderivs_t eos_xderivs;

  bool found;
  Real small = 1e-10;

  // parse the runtime variables
  eos_init();
  
  ModelParser model;
  model.ReadFile(modelfile);
  

  RealVector actual, adiabatic, ledoux;

  actual.resize(model.npts_model);
  adiabatic.resize(model.npts_model);
  ledoux.resize(model.npts_model);

  // build the gradients

  for (int i=0; i < model.npts_model; ++i) {

    eos_state.rho = model.model_state[i][model.idens_model];
    eos_state.T = model.model_state[i][model.itemp_model];

    for (int j=0; j < NumSpec; ++j){
      eos_state.xn[j] = model.model_state[i][model.ispec_model + j];
    }

    eos(eos_input_rt, eos_state);
    eos_xderivs = composition_derivatives(eos_state);

    chi_rho = eos_state.rho * eos_state.dpdr / eos_state.p;
    chi_t = eos_state.T * eos_state.dpdT / eos_state.p;

    for (int j=0; j < NumSpec; ++j){
      chi_X[j] = eos_state.xn[j] * eos_xderivs.dpdX[j]/ (eos_state.p * chi_t);
    }

    // adiabatic gradient
    adiabatic[i] = (eos_state.gam1 - chi_rho) / (eos_state.gam1 * chi_t);

    // forward difference
    if (i == 0) {
      dT = model.model_state[i + 1][model.itemp_model] - model.model_state[i][model.itemp_model];
      dp = model.model_state[i+1][model.ipres_model] - model.model_state[i][model.ipres_model];

      for (int j=0; j < NumSpec; ++j){
        dcomp[j] = model.model_state[i + 1][model.ispec_model + j] - model.model_state[i][model.ispec_model + j];
      }
    }
    // backward difference
    else if (i == model.npts_model-1) {
      dT = model.model_state[i][model.itemp_model] - model.model_state[i - 1][model.itemp_model];
      dp = model.model_state[i][model.ipres_model] - model.model_state[i - 1][model.ipres_model];
      
      for (int j=0; j < NumSpec; ++j){
        dcomp[j] = model.model_state[i][model.ispec_model + j] - model.model_state[i - 1][model.ispec_model + j];
      }
    }
    // centered difference
    else {
      dT = model.model_state[i + 1][model.itemp_model] - model.model_state[i - 1][model.itemp_model];
      dp = model.model_state[i+1][model.ipres_model] - model.model_state[i - 1][model.ipres_model];

      for (int j=0; j < NumSpec; ++j){
        dcomp[j] = model.model_state[i + 1][model.ispec_model + j] - model.model_state[i - 1][model.ispec_model + j];
      }
    }

    // actual gradient; if we are in a constant pressure region, set
    // gradient to ZERO to prevent div by 0
    if (std::abs(dp) < small) {
      actual[i] = 0.0;
      for (int j=0; j < NumSpec; ++j){
        dXdp[j] = 0.0;
      }
    } 
    else {
      actual[i] = model.model_state[i][model.ipres_model] * dT / (dp * model.model_state[i][model.itemp_model]);
            
            
      for (int j=0; j < NumSpec; ++j){
        dXdp[j] = dcomp[j]/dp;
      }
    }

    // ledoux gradient
    ledoux[i] = adiabatic[i];
    for (int j=0; j < NumSpec; ++j){
      if (model.model_state[i][model.ispec_model + j] > 0.0) {
        ledoux[i] -= chi_X[j] * model.model_state[i][model.ipres_model]*dXdp[j] / model.model_state[i][model.ispec_model + j];
      }
    }
  }

  //writeout data stuff
  std::ofstream of;
  of.open(outfile);

  of << "# npts = " << model.npts_model << std::endl;
  of << "# r, actual, adiabatic, ledoux" << std::endl;

  for (int i = 0; i < model.npts_model; ++i) {
      of << std::setprecision(12) << std::setw(20) << model.model_r[i] << " ";
      of << std::setprecision(12) << std::setw(20) << actual[i] << " ";
      of << std::setprecision(12) << std::setw(20) << adiabatic[i] << " ";
      of << std::setprecision(12) << std::setw(20) << ledoux[i] << " ";
      of << std::endl;
  }

  of.close();

}
#endif