#!/usr/bin/env python

# Title:  Reaction Consistency Checker (recon.py)
# Author: Adam Jacobs
# Creation Date: 04/23/2012

# Description: The script analyses the given plotfiles generated by test_react
# for consistency.
#
# The consistency checks are:
#   1) Check that omegadot * dt = (change in mass fraction) for each species.
#   2) Check that h_new = (h_old + H_nuc * dt + H_ext * dt)
#   3) Check that H_nuc and/or H_ext are zero when do_burning and/or do_heating
#      are set to .false.
#
# Revision History
# Programmer            Date                    Change
# ------------------------------------------------------------------------------
# Adam Jacobs         04/23/2012              Code created
# Adam Jacobs         05/11/2012              Implementing consistency data checking
# Adam Jacobs         05/14/2012              Finished implementing consistency checking methods

# TODO:
# 1) Currently relies upon convention of species component names starting with 'X_'
#    and upon the current convention for indexing (1 is the density component, 2
#    enthalpy, etc..).  Ideally, I should alter this so that recon.py is more robust
#    to potential convention changes.
# 2) Check that array access is efficient/proper.


###############
### Imports ###
###############
# TODO: Delete any unused imports.
import fsnapshot
import numpy
import pylab
import matplotlib
import os
import sys
import getopt
import math
import string
import mpl_toolkits.axes_grid1

#################################
### Global Data and Constants ###
#################################

###############
### Classes ###
###############

# Represents a BoxLib plotfile.


class BLPlotfile(object):
    # Static variables
    _MAX_VAR_NAME = 20
    _SPEC_STR = 'X_'

    # Instance variables:
    # self       - Reference to an instance of this object.
    # name       - Name of the plotfile being represented.
    # path       - Path to the plotfile being represented.
    # nx, ny, nz - Number of cells along each axis.
    # dm         - Dimension of plot.
    # ncomp      - The number of data components in the plotfile.
    # comps      - List of the component names.
    # data       - Arrays of x,y,z data for all components.
    # time       - Time of the data snapshot.
    # nspec      - Number of elemental species modeled in the data.
    def __init__(self, filename):
      # Note that helper functions are sensitive to the order they're called in.
      # They expect certain instance variables to be initialized.
        self.name = filename
        self.path = os.getcwd() + os.sep
        (self.nx, self.ny, self.nz) = self._getCellCounts()
        if(self.nz == -1):
            self.dm = 2
        else:
            self.dm = 3
        self.ncomp = self._getNComp()
        self.comps = self._getComps()
        self.data = self._getData()
        self.time = self._getTime()
        self.nspec = self._getNSpec()

    ##Public methods##
    def checkOmegadot(self, tol):
        ret = []

        # Calculate index of omegadot consistency data based on indexing conventions
        oci = 7 + 2 * self.nspec
        for i in range(oci, oci + self.nspec):
            success = True
            err_msg = ''
            max_err = 0.0

            cur_comp = self.comps[i]
            con_dat_arr = self.data[cur_comp]  # numpy array of data
            # TODO: implement 2D
            for ix in range(con_dat_arr.shape[0]):
                for iy in range(con_dat_arr.shape[1]):
                    for iz in range(con_dat_arr.shape[2]):
                        #print('Cur val: ', con_dat_arr[ix, iy, iz])
                        # Check that each cell is 0 to within tolerance
                        if(con_dat_arr[ix, iy, iz] > tol):
                            success = False
                            err_msg = 'ERROR: Cell (' + str(ix) + ', ' + str(
                                iy) + ', ' + str(iz) + ') exceeds tolerance.'
                        if(con_dat_arr[ix, iy, iz] > max_err):
                            max_err = con_dat_arr[ix, iy, iz]
            ret.append((cur_comp, success, err_msg, max_err))
        return ret

    def checkEnthalpy(self, tol):
        success = True
        err_msg = ''
        max_err = 0.0

        # Calculate index of enthalpy consistency data based on indexing conventions
        eci = 7 + 3 * self.nspec

        h_comp = self.comps[eci]
        h_dat_arr = self.data[h_comp]  # numpy array of data
        # TODO: implement 2D
        for ix in range(h_dat_arr.shape[0]):
            for iy in range(h_dat_arr.shape[1]):
                for iz in range(h_dat_arr.shape[2]):
                    # Check that each cell is 0 to within tolerance
                    if(h_dat_arr[ix, iy, iz] > tol):
                        success = False
                        err_msg = 'ERROR: Cell (' + str(ix) + ', ' + \
                            str(iy) + ', ' + str(iz) + ') exceeds tolerance.'
                    if(h_dat_arr[ix, iy, iz] > max_err):
                        max_err = h_dat_arr[ix, iy, iz]
        return (success, err_msg, max_err)

    def checkHeating(self, do_heat, do_burn):
        nuc_success = True
        ext_success = True
        nuc_err_msg = ''
        ext_err_msg = ''
        max_nuc_err = 0.0
        max_ext_err = 0.0

        # Calculate index of Hnuc and Hext data based on indexing conventions
        nuci = 4 + 2 * self.nspec
        exti = 6 + 2 * self.nspec

        nuc_comp = self.comps[nuci]
        nuc_dat_arr = self.data[nuc_comp]  # numpy array of data
        ext_comp = self.comps[exti]
        ext_dat_arr = self.data[ext_comp]  # numpy array of data
        # TODO: implement 2D
        for ix in range(nuc_dat_arr.shape[0]):
            for iy in range(nuc_dat_arr.shape[1]):
                for iz in range(nuc_dat_arr.shape[2]):
                    # Check that each cell is 0 when it should be.
                    if((not do_burn) and (nuc_dat_arr[ix, iy, iz] != 0.0)):
                        nuc_success = False
                        nuc_err_msg = 'BURN ERROR: Cell (' + str(ix) + ', ' + str(
                            iy) + ', ' + str(iz) + ') is not 0.0.'
                    if((not do_heat) and (ext_dat_arr[ix, iy, iz] != 0.0)):
                        ext_success = False
                        ext_err_msg = 'HEAT ERROR: Cell (' + str(ix) + ', ' + str(
                            iy) + ', ' + str(iz) + ') is not 0.0.'
                    if(nuc_dat_arr[ix, iy, iz] > max_nuc_err):
                        max_nuc_err = nuc_dat_arr[ix, iy, iz]
                    if(ext_dat_arr[ix, iy, iz] > max_ext_err):
                        max_ext_err = ext_dat_arr[ix, iy, iz]
        nuc_ret = (nuc_success, nuc_err_msg, max_nuc_err)
        ext_ret = (ext_success, ext_err_msg, max_ext_err)
        return (nuc_ret, ext_ret)

    def blpPrint(self):
        for s in self.comps:
            print(s)

    ##Private methods##
    def _getCellCounts(self):
        return fsnapshot.fplotfile_get_size(self.name)

    def _getNComp(self):
        return fsnapshot.fplotfile_get_ncomp(self.name)

    def _getComps(self):
        # f2py will convert this into a character array
        cstr = numpy.array([0] * BLPlotfile._MAX_VAR_NAME * self.ncomp)
        ret = []
        i = 1
        curword = ''
        cstr = fsnapshot.fplotfile_get_comps(self.name, cstr).reshape((self.ncomp, BLPlotfile._MAX_VAR_NAME))
        for c in cstr:
            ret.append(b"".join(c).decode("utf-8"))

        return ret

    def _getData(self):
        ret = {}
        if(self.dm == 2):
            # Loop over all components
            for cmpstr in self.comps:
                tmpdat = numpy.array([[0.0] * self.ny] * self.nx)
                tmpdat = [[0.0] * self.nx, [0.0] * self.ny]
                (tmpdat, ierr) = fsnapshot.fplotfile_get_data_2d(
                    self.name, cmpstr, tmpdat)
                if(ierr != 0):
                    sys.exit(2)
                ret[cmpstr] = tmpdat
            return ret
        elif(self.dm == 3):
            # Loop over all components
            for cmpstr in self.comps:
                tmpdat = numpy.array([[[0.0] * self.nz] * self.ny] * self.nx)
                (tmpdat, ierr) = fsnapshot.fplotfile_get_data_3d(
                    self.name, cmpstr, tmpdat)
                if(ierr != 0):
                    sys.exit(2)
                ret[cmpstr] = tmpdat
            return ret
        else:
            print('ERROR! Invalid dimension in BLPlotfile._getData()')
            sys.exit(2)

    def _getTime(self):
        return fsnapshot.fplotfile_get_time(self.name)

    def _getNSpec(self):
        ret = 0

        for s in self.comps:
            if(s.startswith(BLPlotfile._SPEC_STR)):
                ret = ret + 1

        return ret

#################
### Functions ###
#################

# ==============================================================================
# usage
# ==============================================================================


def usage():
    usageStr = """
    Usage:

    ./recon.py plotfile1 [plotfile2 ... ]

    Read each cell of plotfiles generated by the test_react
    unit test and print out the maximum error for
      -Omegadot * dt - (change in mass fraction)
        [For each species, should be near 0]
      -Enthalpy: h_new - (h_old + H_nuc * dt + H_ext * dt)
        [Should be small percentage of h_old]
      -Heating: Print the max value for H_nuc and H_ext.
        [Should be exactly 0.0 if do_burning or do_heating are set to false]

    Note: this script requires the fsnapshot.so library

    """
    print(usageStr)


#################
### Execution ###
#################
if __name__ == "__main__":
    if(len(sys.argv) <= 1):
        usage()
        sys.exit(0)

    fileList = []
    first = True
    for arg in sys.argv:
        if(first):
            first = False
            continue
        fileList.append(arg)

    for f in fileList:
        test = BLPlotfile(f)
        print('Results for ' + f + ':')
        print('--------------------')
        # test.blpPrint()
        retO = test.checkOmegadot(1.1e-17)
        retE = test.checkEnthalpy(100)
        retH = test.checkHeating(False, True)
        for i in range(test.nspec):
            print("-Omegadot max error for " +
                  retO[i][0].rsplit()[0] + ": " + str(retO[i][3]))
        print("-Enthalpy max error:         " + str(retE[2]))
        print("-Max nuclear heating value:  " + str(retH[0][2]))
        print("-Max external heating value: " + str(retH[1][2]))
        print(' ')
