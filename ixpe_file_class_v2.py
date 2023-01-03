#!/usr/bin/env python
#
# Copyright (C) 2016--2019, the ixpeobssim team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from __future__ import print_function, division

import numpy
import os
import fnmatch
from scipy.interpolate import RectBivariateSpline
    
ixpe_loaded=False 

def _full_path(file_name):
    """Convenience function to retrieve the relevant files.
    """
    if ixpe_loaded:
        if fnmatch.fnmatch(file_name, '*dgonzalez/persistent/*'):
            return os.path.join(file_name)
        else:
            return os.path.join(IXPEOBSSIM_CONFIG_ASCII_CEDAR, file_name)
    else:
        print(ixpe_loaded)
        #return os.path.join('.', 'ascii', file_name)
        return os.path.join('.', file_name)

def llinterp(xval,xv,yv):
    return numpy.exp(numpy.interp(numpy.log(xval),numpy.log(xv),numpy.log(yv)))

class ixpe_file_class:
## File with the model
# should be in the config/ascii directory
# columns should be named in the first row
# EnergykeV Phirad QI I [UI]
# the rows and columns can be in any order
# UI is optional
#filename="Double_Blackbody.txt"
#filename="10qedon.txt"

## Geometry of dipole
# alpha is the angle between the line of sight and rotation axis
#alpha=numpy.radians(50)
# beta is the angle between the magnetic axis and the rotation axis

## Renormalize the phase-averaged flux
# different renormalizations for the phase averaged flux
#normflux=1e-9 # total flux from 2-8 keV in (counts or erg)/cm2/s
#
#normflux=enerlist**(-2)*1e-2 # normalize by an array
#
#normflux=(lambda x: 1e-3*x**-2) # normalize by a function of energy
#
#normflux='HerX1_NuSTAR.txt' # normalize using data in the file
#  the first row should name the columns including EnergykeV and I# are the intensities or fluxes in the file in energy units? (IXPEObsSim wants photon units)
# this is done after other renormalizations
#intensity_energy_units=True
## Value of NH in per cm2 (either give a value or a value and a file from the config/ascii directory
# ROIModel can also include the absorption
#NH=0.6e22
#NH='0.6e22;tbabs.dat'
#NH=1e24
# NH='1e24;tbabs.dat'
# the first row should name the columns including Energy and sigma
#     sigma is the cross section times (E/keV)^2 / (1e-24 cm^2)
#     ssabs=numpy.interp(enerlist,abarr['Energy'],abarr['sigma']/(enerlist)**3*1e-24
# final band renorm (you can renormalize the phase-average flux in the band 2-8 keV after the absorption
# finalnorm=1e-2 # total flux from 2-8 keV in (counts or erg)/cm2/s after absorption

    def __init__(self,filename,alpha=90.0,beta=80.0,normflux=None,NH=None,intensity_energy_units=True,kev_cm2_kev_s=True,finalnorm=None,doIXPE=ixpe_loaded,enerlist=None,rmf=None):
        self.filename=filename
        self.alpha=numpy.radians(alpha)
        self.beta=numpy.radians(beta)
        self.normflux=normflux
        self.intensity_energy_units=intensity_energy_units
        self.kev_cm2_kev_s= kev_cm2_kev_s
        self.NH=NH
        self.finalnorm=finalnorm
        self.__model__ = filename
        self.doIXPE=doIXPE and ixpe_loaded
        self.narr=None
        self.enerlist=enerlist
        if rmf is not None:
            self.rmf=numpy.transpose(rmf)
        else:
            self.rmf=rmf
        self.load_model()
        self.calc_values()
        self.normalize_flux()
        self.generate_splines()


    def __str__(self):
        return 'ixpe_file_class: ' + filename
    
    def load_model(self):
        self.arr=numpy.genfromtxt(_full_path(self.filename),names=True)
        # sort in the order that we need
        self.arr=numpy.sort(self.arr,order=('EnergykeV','Phirad'))

        self.inclination=self.arr['Phirad']
        self.energy=self.arr['EnergykeV']
        self.qi=self.arr['QI']
        try:
            self.ui=self.arr['UI']
        except:
            self.ui=0*self.qi
        self.ratio=numpy.hypot(self.ui,self.qi)
        self.ang=0.5*numpy.arctan2(self.ui,self.qi)
        self.incllist=numpy.unique(self.inclination)
        self.enerlist_model=numpy.unique(self.energy)
        if self.enerlist is None:
            self.enerlist=self.enerlist_model
        self.fluxmod=self.arr['I'].reshape((len(self.enerlist_model),len(self.incllist)))

        # build the spectrum and polarization as a function of energy and inclination
        if self.doIXPE:
            self.energy_spectrum_inclination=xInterpolatedBivariateSpline(self.enerlist_model,self.incllist,
                                                                  self.fluxmod,kx=3,ky=3)
            self.ratio_inclination=xInterpolatedBivariateSpline(self.enerlist_model,self.incllist,
                                                                      (self.ratio).reshape((len(self.enerlist_model),len(self.incllist))),kx=3,ky=3)

            self.angle_inclination=xInterpolatedBivariateSpline(self.enerlist_model,self.incllist,
                                                                      (self.ang).reshape((len(self.enerlist_model),len(self.incllist))),kx=3,ky=3)
        else:
            self.energy_spectrum_inclination=numpy.vectorize(RectBivariateSpline(self.enerlist_model,self.incllist,
                                                                  self.fluxmod,kx=3,ky=3))
            self.ratio_inclination=numpy.vectorize(RectBivariateSpline(self.enerlist_model,self.incllist,
                                                                      (self.ratio).reshape((len(self.enerlist_model),len(self.incllist))),kx=3,ky=3))

            self.angle_inclination=numpy.vectorize(RectBivariateSpline(self.enerlist_model,self.incllist,
                                                                      (self.ang).reshape((len(self.enerlist_model),len(self.incllist))),kx=3,ky=3))

    # calculate the flux and polarization over a grid (65% of execution time)
    def calc_values(self):
        # set up phase bins
        self.phase=numpy.linspace(0,1,101)
        #    print(phase,inclination(phase))
        #    ee,tt=numpy.meshgrid(enerlist,phase)
        tt,ee=numpy.meshgrid(self.phase,self.enerlist)

        self.flux=self.rawspec(ee,tt)
        self.pol_deg_data=self.pol_deg(ee,tt)
        self.pol_ang_data=numpy.degrees(self.pol_ang(ee,tt))
        self.meanflux_orig=numpy.mean(self.flux,axis=-1)
        
    def normalize_flux(self):

        # perform the renormalization
        if self.normflux is not None:
            if type(self.normflux) is float:
                from scipy.integrate import simps
                # renormalize the total flux over the band 2-8 keV
                ok=(self.enerlist>2) & (self.enerlist<8)
                self.flux=self.flux/simps(self.meanflux_orig[ok],self.enerlist[ok])*self.normflux
            elif type(self.normflux) is str:
                if self.narr is None:
                    self.narr=numpy.genfromtxt(_full_path(self.normflux),names=True)
                    self.narr=numpy.sort(self.narr,order=('EnergykeV'))
                    self.meanval=llinterp(self.enerlist,self.narr['EnergykeV'],self.narr['I'])
                self.flux=numpy.transpose((self.meanval/self.meanflux_orig)*numpy.transpose(self.flux))
            elif callable(self.normflux):
                # assume norm flux is a function of energy
                self.meanval=self.normflux(self.enerlist)
                self.flux=numpy.transpose((self.meanval/self.meanflux_orig)*numpy.transpose(self.flux))
            else:
                # assume norm flux is an array of fluxes at the same energies as enerlist
                self.flux=numpy.transpose((self.normflux/self.meanflux_orig)*numpy.transpose(self.flux))

        if self.NH is not None:
            if type(self.NH) is float:
                abfilename='tbabs.dat'
                #NH = self.NH
            elif type(self.NH) is str:
                aa=self.NH.split(';')
                abfilename=aa[1]
                NH=float(aa[0])
            else: 
                raise ValueError('The value of NH should be a float or a string with value;filename.')
            abarr=numpy.genfromtxt(_full_path(abfilename),names=True)
            abarr=numpy.sort(abarr,order=('Energy'))
            ssabs=llinterp(self.enerlist,abarr['Energy'],abarr['sigma'])/(self.enerlist)**3*1e-24
            #absenergy=numpy.exp(-ssabs)
            absenergy=numpy.exp(-ssabs*NH)
            self.flux=numpy.transpose(absenergy*numpy.transpose(self.flux))
            
            
        if self.finalnorm is not None:
            if type(self.finalnorm) is float:
                from scipy.integrate import simps
                # renormalize the total flux over the band 2-8 keV
                ok=(self.enerlist>2) & (self.enerlist<8)
                ratio=self.finalnorm/simps(self.meanflux_orig[ok],self.enerlist[ok])
                self.flux=self.flux*ratio
            else:
                raise ValueError('The value of finalnorm should be a float.')

        if self.intensity_energy_units:
            if self.kev_cm2_kev_s:
                self.flux=numpy.transpose(numpy.transpose(self.flux)/self.enerlist)
            else:
                self.flux=numpy.transpose(numpy.transpose(self.flux)/(self.enerlist*1.60217662e-9)) # 1.60217662e-9 erg = 1 keV
            
    def generate_splines(self):
        if self.doIXPE:

            # setup the splines which should be faster than doing the geometry everytime
            # also they are useful for the plot and include the spectral renormalization
            fmt = dict(xlabel='Energy [keV]', ylabel='Phase',
                   zlabel='Flux [cm$^{-2}$ s$^{-1}$ keV$^{-1}$]')

            self.spec_spline = xInterpolatedBivariateSpline(self.enerlist, self.phase, self.flux, kx=3, ky=3, **fmt)
        else:
            self.spec_spline = numpy.vectorize(RectBivariateSpline(self.enerlist,self.phase,self.flux, kx=3, ky=3))

        self.spec = self.spec_spline

        if self.doIXPE:
            fmt = dict(xlabel='Energy [keV]', ylabel='Phase', zlabel='Polarization degree')

            self.pol_deg_spline = xInterpolatedBivariateSpline(self.enerlist, self.phase, self.pol_deg_data,kx=3, ky=3, **fmt)
        else:
            self.pol_deg_spline = numpy.vectorize(RectBivariateSpline(self.enerlist, self.phase, self.pol_deg_data,kx=3, ky=3))
            
        if self.doIXPE:
            fmt = dict(xlabel='Energy [keV]', ylabel='Phase',zlabel='Polarization angle [deg]')

            self.pol_ang_spline = xInterpolatedBivariateSpline(self.enerlist, self.phase, self.pol_ang_data,kx=3, ky=3, **fmt)
        else:
            self.pol_ang_spline = numpy.vectorize(RectBivariateSpline(self.enerlist, self.phase, self.pol_ang_data,kx=3, ky=3))

        self.u_data=numpy.sin(2*numpy.radians(self.pol_ang_data))
        self.q_data=numpy.cos(2*numpy.radians(self.pol_ang_data))
        if self.rmf is None:
            self.pol_deg_spline_conv=self.pol_deg_spline
            self.spec_spline_conv=self.spec_spline
            self.flux_conv=self.flux
            self.pol_deg_data_conv=self.pol_deg_data
            
        else:
            from ixpeobssim.irf import xModulationFactor, irf_file_path, DEFAULT_IRF_NAME

            file_path = irf_file_path(DEFAULT_IRF_NAME, 1, 'modf')
            modf_fun = xModulationFactor(file_path)
            self.flux_conv=numpy.dot(self.rmf,self.flux)
            #self.u_data=numpy.dot(self.rmf,self.flux*self.u_data*self.pol_deg_data)
            #self.q_data=numpy.dot(self.rmf,self.flux*self.q_data*self.pol_deg_data)
            #u_data=numpy.dot(matrixt, numpy.transpose(0.5*modf_fun(effae)*numpy.transpose(model.flux*model.u_data*model.pol_deg_data)))
            #q_data=numpy.dot(matrixt, numpy.transpose(0.5*modf_fun(effae)*numpy.transpose(model.flux*model.q_data*model.pol_deg_data)))
            self.u_data=numpy.dot(self.rmf, numpy.transpose(modf_fun(self.enerlist)*numpy.transpose(self.flux*self.u_data*self.pol_deg_data)))
            self.q_data=numpy.dot(self.rmf, numpy.transpose(modf_fun(self.enerlist)*numpy.transpose(self.flux*self.q_data*self.pol_deg_data)))
            polflux=numpy.hypot(self.u_data,self.q_data)
            #self.u_data/=polflux
            #self.q_data/=polflux
            self.pol_deg_data_conv=polflux/self.flux_conv
            self.phalist=numpy.arange(self.rmf.shape[0])
          

            if self.doIXPE:
                fmt = dict(xlabel='Energy [keV]', ylabel='Phase', zlabel='Convolved Flux [cm$^{-2}$ s$^{-1}$ keV$^{-1}$]')
                #self.spec_spline_conv = xInterpolatedBivariateSpline(self.enerlist, self.phase, self.flux_conv, kx=3,ky=3,**fmt)
                self.spec_spline_conv = xInterpolatedBivariateSpline(self.phalist, self.phase, self.flux_conv, kx=3, ky=3, **fmt)
                
            else:
                self.spec_spline_conv = numpy.vectorize(RectBivariateSpline(self.enerlist,self.phase,self.flux_conv, kx=3, ky=3))
            if self.doIXPE:
                fmt = dict(xlabel='Energy [keV]', ylabel='Phase', zlabel='Convolved Polarization degree')
                #self.pol_deg_spline_conv = xInterpolatedBivariateSpline(self.enerlist, self.phase, self.pol_deg_data_conv,kx=3, ky=3, **fmt)
                self.pol_deg_spline_conv = xInterpolatedBivariateSpline(self.phalist, self.phase, self.pol_deg_data_conv,kx=3, ky=3, **fmt)                
                
            else:
                self.pol_deg_spline_conv = numpy.vectorize(RectBivariateSpline(self.enerlist, self.phase, self.pol_deg_data_conv,kx=3, ky=3))

        if self.doIXPE:
            fmt = dict(xlabel='Energy [keV]', ylabel='Phase',zlabel='sin(2 Polarization angle)')
            #self.u_spline = xInterpolatedBivariateSpline(self.enerlist, self.phase, self.u_data,kx=3, ky=3, **fmt)
            #self.u_spline = xInterpolatedBivariateSpline(self.phalist, self.phase, self.u_data,kx=3, ky=3, **fmt)
           
        else:
            self.u_spline = numpy.vectorize(RectBivariateSpline(self.enerlist, self.phase, self.u_data,kx=3, ky=3))

        if self.doIXPE:
            fmt = dict(xlabel='Energy [keV]', ylabel='Phase',zlabel='cos(2 Polarization angle)')
            
            #self.q_spline = xInterpolatedBivariateSpline(self.enerlist, self.phase, self.q_data,kx=3, ky=3, **fmt)
            #self.q_spline = xInterpolatedBivariateSpline(self.phalist, self.phase, self.q_data,kx=3, ky=3, **fmt)
            
        else:
            self.q_spline = numpy.vectorize(RectBivariateSpline(self.enerlist, self.phase, self.q_data,kx=3, ky=3))

            
    def reset_geometry(self,alpha,beta):
        self.alpha=numpy.radians(alpha)
        self.beta=numpy.radians(beta)
        self.calc_values()
        self.normalize_flux()
        self.generate_splines()


    # inclination as a function of phase
    def inclination_funk(self,t):
        phi_phase=numpy.radians(t*360)
        if (self.incllist[-1] > numpy.pi/2):
            return numpy.arccos(numpy.cos(self.alpha)*numpy.cos(self.beta)+
                                      numpy.sin(self.alpha)*numpy.sin(self.beta)*numpy.cos(phi_phase))
        else:
            return numpy.arccos(numpy.abs(numpy.cos(self.alpha)*numpy.cos(self.beta)+
                                      numpy.sin(self.alpha)*numpy.sin(self.beta)*numpy.cos(self.phi_phase)))

    # polarization degree as a function of energy and phase
    def pol_deg(self,E, t, ra=None, dec=None):
        poldeg = numpy.minimum(self.ratio_inclination(E,self.inclination_funk(t)),1.0)
        poldeg = numpy.where(poldeg<0,0.01,poldeg)
        return poldeg


    # polarization angle as a function of energy and phase
    def pol_ang(self,E, t, ra=None, dec=None):
        ii=self.inclination_funk(t)
        phi_phase=numpy.radians(t*360)
        ang=numpy.arcsin(numpy.sin(self.beta)*numpy.sin(phi_phase)/numpy.sin(ii))
        return numpy.mod(ang+self.angle_inclination(E,ii),numpy.pi)

    # energy spectrum as a function of energy and phase
    def rawspec(self,E,t):
        return self.energy_spectrum_inclination(E,self.inclination_funk(t))

    def display_spectrum(self,emin=1.1, emax=12., phase_indices=[10, 40, 60, 80], y_min=1.e-3, y_max=2.e+2):
        """Display the energy spectrum.
        """
        # Full, 2-d energy spectrum.
        plt.figure('%s spectrum' % self.__model__)
        self.spec_spline.plot()

        # Slices of the energy spectrum at different pulse-phase values.
        plt.figure('%s spectrum phase slices' % self.__model__)
        for i in phase_indices:
            _phase = self.phase[i]
            slice_ = self.spec_spline.hslice(_phase, k=3)
            slice_.plot(label='Pulse phase = %.2f' % _phase)
            plt.plot(self.enerlist, self.flux[:,i], 'o', color=last_line_color())
        setup_gca(xmin=emin, xmax=emax, logx=True, logy=True, grids=True,
                  ymin=y_min, ymax=y_max, legend=True)


    def display_pol_deg(self,emin=1.1, emax=12., phase_indices=[10, 40, 60, 80],
                        energy_indices=[4, 7, 10, 13]):
        """Display the polarization degree.
        """
        # Polarization degree 2-d plot.
        plt.figure('%s polarization degree' % self.__model__)
        self.pol_deg_spline.plot()

        # Slices of the polarization degree at different pulse-phase values.
        plt.figure('%s polarization degree phase slices' % self.__model__)
        for i in phase_indices:
            _phase = self.phase[i]
            slice_ = self.pol_deg_spline.hslice(_phase, k=3)
            slice_.plot(label='Pulse phase = %.2f' % _phase)
            plt.plot(self.enerlist, self.pol_deg_data[:,i], 'o', color=last_line_color())
        setup_gca(xmin=emin, xmax=emax, legend=True)

        # Slices of the polarization degree at different energies.
        plt.figure('%s polarization degree energy slices' % self.__model__)
        for i in energy_indices:
            _energy = self.enerlist[i]
            slice_ = self.pol_deg_spline.vslice(_energy, k=3)
            slice_.plot(label='Energy = %.2f keV' % _energy)
            plt.plot(self.phase, self.pol_deg_data[i,:], 'o', color=last_line_color())
        setup_gca(xmin=0., xmax=1., legend=True)


    def display_pol_ang(self,emin=1.1, emax=12., phase_indices=[10, 40, 60, 80],
                        energy_indices=[4, 7, 10, 13]):
        """Display the polarization angle.
        """
        # Polarization angle 2-d plot.
        plt.figure('%s polarization angle' % self.__model__)
        self.pol_ang_spline.plot()

        # Slices of the polarization angle at different pulse-phase values.
        plt.figure('%s polarization angle phase slices' % self.__model__)
        for i in phase_indices:
            _phase = self.phase[i]
            slice_ = self.pol_ang_spline.hslice(_phase, k=3)
            slice_.plot(label='Pulse phase = %.2f' % _phase)
            plt.plot(self.enerlist, self.pol_ang_data[:,i], 'o', color=last_line_color())
        setup_gca(xmin=emin, xmax=emax, legend=True)

        # Slices of the polarization angle at different energies.
        plt.figure('%s polarization angle energy slices' % self.__model__)
        for i in energy_indices:
            _energy = self.enerlist[i]
            slice_ = self.pol_ang_spline.vslice(_energy, k=3)
            slice_.plot(label='Energy = %.2f keV' % _energy)
            plt.plot(self.phase, self.pol_ang_data[i,:], 'o', color=last_line_color())
        setup_gca(xmin=0., xmax=1., legend=True)


    def display(self,emin=1.1, emax=12., phase_indices=[10, 40, 60, 80],
                energy_indices=[4, 7, 10, 13]):
        """Display the source model.
        """
        self.display_spectrum(emin, emax, phase_indices)
        self.display_pol_deg(emin, emax, phase_indices, energy_indices)
        self.display_pol_ang(emin, emax, phase_indices, energy_indices)
