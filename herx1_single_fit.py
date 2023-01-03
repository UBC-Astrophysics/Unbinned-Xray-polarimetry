import os
import sys
import numpy
import matplotlib.pyplot as plt
sys.path.append("/home/dgonzal/jupyter_py3/lib/python3.7/site-packages")
#sys.path.insert(0, "/home/dgonzal/projects/def-heyl/denis_work/ixpesw-ixpeobssim-f340ac459e08")
sys.path.append("/home/dgonzal/jupyter_py3/lib/python3.7/site-packages/ixpeobssim-18.0.0-py3.7.egg")
from astropy.io import fits
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.optimize import curve_fit

import logging
logging.disable(level=logging.CRITICAL)

from ixpeobssim import IXPEOBSSIM_CONFIG, IXPEOBSSIM_DATA, IXPEOBSSIM_DOC
import ixpeobssim.core.pipeline as pipeline
from ixpeobssim.core.spline import xInterpolatedBivariateSpline 

from ixpeobssim.utils.logging_ import logger
from ixpeobssim import PYXSPEC_INSTALLED
from ixpeobssim.utils.misc import pairwise_enum
from ixpeobssim.evt.binning import xEventBinningBase, xBinnedModulationCube, xBinnedPolarizationCube


sys.path.append("/project/6009439/denis_work/config_ixpeobssim")
import Caiazzo_Heyl_Figure_7C as pulsar

from ixpeobssim import IXPEOBSSIM_CONFIG_CEDAR, IXPEOBSSIM_CONFIG_ASCII_CEDAR
from ixpeobssim.irf import xModulationFactor, irf_file_path, DEFAULT_IRF_NAME
import time

head,cfile=os.path.split(pulsar.__file__)
CFG_FILE = os.path.join(IXPEOBSSIM_CONFIG_CEDAR, cfile)
modulename=os.path.splitext(cfile)[0]
OUT_FILE_PATH_BASE = os.path.join(IXPEOBSSIM_DATA, os.path.splitext(cfile)[0])
EVT_FILE_PATH = '%s.fits' % OUT_FILE_PATH_BASE
SIM_DURATION = 100000.
OUTPUT_FOLDER = os.path.join(IXPEOBSSIM_DOC, 'figures', 'showcase')

ENERGY_BINNING = numpy.array([2., 4., 8.])
PHASE_BINNING = numpy.concatenate((numpy.linspace(0,0.2,11),numpy.linspace(0.3,0.7,11),numpy.linspace(0.8,1.0,11)))


def generate_simulation(id,task):

    pid = '_pid{}_'.format(os.getpid())
    OUT_FILE_PATH_BASE = os.path.join(IXPEOBSSIM_DATA, modulename + pid + 'task_%d_run_%d' % (task,id))
    EVT_FILE_PATH = '%s.fits' % OUT_FILE_PATH_BASE
    #---REMOVE OUTPUT FILES FROM PREVIOUS RUN
    #os.system('rm {}*'.format(OUT_FILE_PATH_BASE))
    #--PERFORM SIMULATION
    EVT_FILE_PATH = pipeline.xpobssim(configfile=CFG_FILE, outfile=EVT_FILE_PATH, duration=SIM_DURATION, overwrite=True)
    #--SELECT ENERGY RANGE
    E_EVT_FILE_PATH = pipeline.xpselect(*EVT_FILE_PATH, emin=2.0, emax=8.0, suffix=pipeline.suffix('Energy'),overwrite=True)
    #--PULSE FOLDING
    folded_EVT_FILE_PATH=pipeline.xpphase(*E_EVT_FILE_PATH,**pulsar.ephem.dict(),suffix='folded',overwrite=True) #modified
    #--PHASE BINNING
    for i, (min_, max_) in pairwise_enum(PHASE_BINNING):
                pipeline.xpselect(*folded_EVT_FILE_PATH, phasemin=min_, phasemax=max_, 
                                  suffix=pipeline.suffix('phase', i),overwrite=True)
    #--GENERATE PCUBE (No MCUBE) FILES
    for i, (min_, max_) in pairwise_enum(PHASE_BINNING):
        file_list = ['%s_phase%04d.fits'% (s[:-5],i)   for s in folded_EVT_FILE_PATH]
        pipeline.xpbin(*file_list, algorithm='PCUBE', ebinalg='LIST',ebinning=ENERGY_BINNING,overwrite=True)
    #--LOAD PCUBE FILES
    phase_bins = xEventBinningBase.bin_centers(PHASE_BINNING)
    phase_err  = xEventBinningBase.bin_widths(PHASE_BINNING)*0.5
    shape = (len(ENERGY_BINNING) - 1, len(PHASE_BINNING) - 1)
    pol_deg = numpy.zeros(shape)
    pol_deg_err = numpy.zeros(shape)
    pol_ang = numpy.zeros(shape)
    pol_ang_err = numpy.zeros(shape)
    emean = numpy.zeros(shape)
    
    counts = numpy.zeros(shape)
    stk_Q = numpy.zeros(shape)
    stk_U = numpy.zeros(shape)
    stk_I = numpy.zeros(shape)
    W2    = numpy.zeros(shape)
    for i, (min_, max_) in pairwise_enum(PHASE_BINNING):
        file_list = ['%s_phase%04d_pcube.fits'% (s[:-5],i)   for s in folded_EVT_FILE_PATH]    
        pcube = xBinnedPolarizationCube.from_file_list(file_list)
        pol_deg[:,i] = pcube.POL_DEG
        pol_deg_err[:,i] = pcube.POL_DEG_ERR
        pol_ang[:,i] = pcube.POL_ANG
        pol_ang_err[:,i] = pcube.POL_ANG_ERR
        emean[:,i] = pcube.ENERGY_MEAN
        counts[:,i]= pcube.COUNTS
        stk_Q[:,i] = pcube.Q
        stk_U[:,i] = pcube.U
        stk_I[:,i] = pcube.I
        W2[:,i]    = pcube.W2
    
    #--COMPUTE STOKES ERRORS
    stk_Q_err = numpy.sqrt((numpy.cos(2.*numpy.radians(pol_ang))*pol_deg_err)**2.
                           +(2.*pol_deg*numpy.sin(2.*numpy.radians(pol_ang))*
                             numpy.radians(pol_ang_err))**2.)
    stk_U_err = numpy.sqrt((numpy.sin(2.*numpy.radians(pol_ang))*pol_deg_err)**2.
                           +(2.*pol_deg*numpy.cos(2.*numpy.radians(pol_ang))*
                             numpy.radians(pol_ang_err))**2.)
    
    stk = [stk_Q/stk_I, stk_Q_err, stk_U/stk_I, stk_U_err, PHASE_BINNING, pol_ang, pol_ang_err, W2, stk_I]
    #--LOAD PHOTON BY PHOTON
    qphoton=[]
    uphoton=[]
    ephoton=[]
    pphoton=[]
    phaphoton=[]
    modfphoton=[]
    for i,f in enumerate(folded_EVT_FILE_PATH):
        with fits.open(f) as hdul:
            print(i, f)
            qphoton=numpy.concatenate((qphoton,hdul[1].data['Q']))
            uphoton=numpy.concatenate((uphoton,hdul[1].data['U']))
            ephoton=numpy.concatenate((ephoton,hdul[1].data['ENERGY']))
            pphoton=numpy.concatenate((pphoton,hdul[1].data['PHASE']))
            pp=(hdul[1].data['PHA']).astype(int)
            phaphoton=numpy.concatenate((phaphoton,pp))
            #modfphoton=numpy.concatenate((modfphoton,modf[pp]))
            file_path = irf_file_path(DEFAULT_IRF_NAME,(i+1), 'modf')
            modf_fun = xModulationFactor(file_path)
            modfphoton=numpy.concatenate((modfphoton,modf_fun(hdul[1].data['ENERGY'])))
    
    phaphoton=phaphoton.astype(int)
    modfphoton/=2  # this is required for the likelihood to work ... I don't know why.  Perhaps my idea of MODF is not correct.
    photon = [qphoton,uphoton,ephoton,pphoton,modfphoton,phaphoton]
    #REMOVE OUTPUT FILES FROM PREVIOUS RUN
    #os.system('rm {}*'.format(OUT_FILE_PATH_BASE))
    return stk, photon, folded_EVT_FILE_PATH
 
#============================

class TimedFun:
    def __init__(self, fun, stop_after=10):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after

    def fun(self, x):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(x)
        self.x = x
        return self.fun_value

#============================

def angfunk(alpha,beta,phase):
    tanhalfC=numpy.tan(phase*numpy.pi)
    halfamb=numpy.radians(alpha-beta)/2
    halfapb=numpy.radians(alpha+beta)/2
    halfAmB=numpy.arctan(numpy.sin(halfamb)/numpy.sin(halfapb)*tanhalfC)
    halfApB=numpy.arctan(numpy.cos(halfamb)/numpy.cos(halfapb)*tanhalfC)
    return -(halfApB-halfAmB)

from ixpe_file_class import ixpe_file_class
from scipy.integrate import simps


with fits.open(irf_file_path(DEFAULT_IRF_NAME, 1, 'arf')) as hdu:
    effae=(hdu[1].data['ENERG_LO']+hdu[1].data['ENERG_HI'])/2
    effa=hdu[1].data['SPECRESP']
with fits.open(irf_file_path(DEFAULT_IRF_NAME, 2, 'arf')) as hdu:
    effa+=hdu[1].data['SPECRESP']
with fits.open(irf_file_path(DEFAULT_IRF_NAME, 3, 'arf')) as hdu:
    effa+=hdu[1].data['SPECRESP']

model=ixpe_file_class('Polarization_wQED_7k_2c.txt',alpha=50,beta=42,
                      normflux='HerX1_NuSTAR.txt',intensity_energy_units=False,enerlist=effae)


with fits.open(irf_file_path(DEFAULT_IRF_NAME, 1, 'rmf')) as hdu:
    matrix=hdu[1].data['MATRIX']
    emin=hdu[1].data['ENERG_LO']

cnts=numpy.interp(effae,model.enerlist,numpy.mean(model.flux,axis=-1))*effa
#cnts=numpy.mean(model.flux,axis=-1)*effa
ratepred=simps(cnts,effae)
npred=SIM_DURATION*ratepred
matrixt=numpy.transpose(matrix)
cnt_conv=numpy.dot(matrixt,cnts)

def stkbinnedgeometry(alpha,beta):
    
    model.reset_geometry(alpha,beta)
    
    q_data = model.pol_deg_data * numpy.cos(2*numpy.radians(model.pol_ang_data))
    u_data = model.pol_deg_data * numpy.sin(2*numpy.radians(model.pol_ang_data))
    fmt = dict(xlabel='Energy [keV]', ylabel='Phase',zlabel='Q/I')
    q_spline = xInterpolatedBivariateSpline(model.enerlist, model.phase, q_data,kx=3, ky=3, **fmt)
    fmt = dict(xlabel='Energy [keV]', ylabel='Phase',zlabel='U/I')
    u_spline = xInterpolatedBivariateSpline(model.enerlist, model.phase, u_data,kx=3, ky=3, **fmt)   
    
    shape = (len(ENERGY_BINNING) - 1, len(PHASE_BINNING) - 1)
    
    stkQ_binned = numpy.zeros(shape)
    stkU_binned = numpy.zeros(shape)
    for j, (emin, emax) in pairwise_enum(ENERGY_BINNING):
        for i, (phasemin, phasemax) in pairwise_enum(PHASE_BINNING):
            int_Q = q_spline.integral(emin,emax,phasemin,phasemax)
            int_U = u_spline.integral(emin,emax,phasemin,phasemax)
            int_Q = int_Q/((emax-emin)*(phasemax - phasemin))
            int_U = int_U/((emax-emin)*(phasemax - phasemin))
            stkQ_binned[j,i]=int_Q
            stkU_binned[j,i]=int_U
    return numpy.array(stkU_binned), numpy.array(stkQ_binned)

#============================

def fitting(ii,task):
    # init stopper
    # minimize_stopper = MinimizeStopper()

    #random.seed()
    numpy.random.seed()
    idproc = 'id{}'.format(os.getpid())
    binned, unbinned, folded_EVT_FILE_PATH  = generate_simulation(ii,task)

    #data_Q,error_Q,data_U,error_U,PHASE_BINNING_OUT,pol_ang,pol_ang_err  = binned[0],binned[1],binned[2],binned[3],binned[4],binned[5],binned[6]
    
    Qdata, Qerror  = binned[0],binned[1]
    Udata, Uerror  = binned[2],binned[3]
    PHASE_BINNING_OUT = binned[4]
    pol_ang,pol_ang_err = binned[5],binned[6]
    W2=binned[7]
    Idata=binned[8]

    phase_bins = xEventBinningBase.bin_centers(PHASE_BINNING_OUT)

    #== events 
    qphoton, uphoton  = unbinned[0],unbinned[1]
    ephoton, pphoton, modfphoton = unbinned[2], unbinned[3], unbinned[4]
    phaphoton=unbinned[5]

    #== select energy range
    indok=(ephoton<=4) & (ephoton>=2)
    eok=ephoton[indok]
    pok=pphoton[indok]
    qok=qphoton[indok]
    uok=uphoton[indok]
    modfok=modfphoton[indok]

    #==RVM: MODF + UNBINNED MAXIMUM LIKELIHOOD FIT ==#
    def likelihood_rvm(alpha,beta,pos_ang,phase0,degm):
        # print(alpha,beta,phase0,pos_ang)
        ang=angfunk(alpha,beta,pok-phase0)+numpy.radians(pos_ang)
        qm=numpy.cos(2*ang)
        um=numpy.sin(2*ang)
        s=numpy.sum(numpy.log(1+modfok*degm*(qok*qm+uok*um)))
        # print(s,alpha,beta,phase0,pos_ang)
        return s

    fun_timed = TimedFun(fun=(lambda x: -likelihood_rvm(x[0],x[1],x[2],x[3],numpy.exp(-x[4]*x[4]))), stop_after=200)
    #try:
    #    res=minimize(fun=fun_timed.fun, x0=[45,45,30,0.5,0.5],method='Nelder-Mead')
    #except Exception as e:
    #    res=minimize((lambda x: -likelihood_rvm(x[0],x[1],x[2],x[3],numpy.exp(-x[4]*x[4]))), x0=[45,45,30,0.5,0.5],method='Nelder-Mead', options= {"maxiter":0})
    #print(res)

    #==RVM: BINNED FIT ==#
    def fitfunk(x,alpha,beta,posang,phase0):
        return numpy.mod(numpy.degrees(angfunk(alpha,beta,x-phase0))+posang,180)
    # using just the zero index, which correspond to 2-4 keV bin
    #popt,pcov=curve_fit(fitfunk,phase_bins,numpy.mod(pol_ang[0,:],180),p0=res.x[:-1],sigma=pol_ang_err[0,:],absolute_sigma=True)
    #print(popt)

    #==UNDERLYING MODEL: BINNED FIT ==#
    def chi2_geometry_delta(alpha,beta,angshift,deltadeg):
        U0, Q0 = stkbinnedgeometry(alpha,beta)
        totalchi2Q= []
        totalchi2U= []
        angshift = numpy.radians(angshift)
        Qmodel = (1.+deltadeg) * (Q0[0,:]*numpy.cos(2.*angshift) - U0[0,:]*numpy.sin(2.*angshift))
        Umodel = (1.+deltadeg) * (U0[0,:]*numpy.cos(2.*angshift) + Q0[0,:]*numpy.sin(2.*angshift))
        #chi2Q  = numpy.sum((Qdata[0,:] - Qmodel)**2. / Qerror[0,:]**2)
        #chi2U  = numpy.sum((Udata[0,:] - Umodel)**2. / Uerror[0,:]**2)
        
        QU_COV = -W2[0,:]*Qdata[0,:]*Udata[0,:]/Idata[0,:]**2
        aux1   = 1./(Qerror[0,:]**2*Uerror[0,:]**2 - QU_COV**2)
        aux2   = Uerror[0,:]**2*(Qdata[0,:]-Qmodel)**2 + Qerror[0,:]**2*(Udata[0,:]-Umodel)**2 - 2.*QU_COV*(Qdata[0,:]-Qmodel)*(Udata[0,:]-Umodel)  
        chi2   = numpy.sum(aux1*aux2)
        #return chi2Q + chi2U
        return chi2
    #res2=minimize((lambda x: chi2_geometry_delta(x[0],x[1],x[2],x[3])), x0=[51.0,43.0,0.1,0.1], method='Nelder-Mead')
    #res2=minimize((lambda x: chi2_geometry_delta(x[0],x[1],x[2],x[3])), x0=[45.0,45.0,0.5,0.5], method='Nelder-Mead')
    res2=minimize((lambda x: chi2_geometry_delta(x[0],x[1],x[2],x[3])), x0=[51.0,43.0,0.1,0.1], method='Nelder-Mead')

    print(res2)

    #== UNDERLYING MODEL: UNBINNED FIT ==#
    #phaphoton=phaphoton.astype(int)
    pinphoton=(pphoton*100+0.5).astype(int)
    #modfpha=modf_fun(effae)

    indok=(ephoton<=4) & (ephoton>=2)
    eok=ephoton[indok]
    pok=pphoton[indok]
    qok=qphoton[indok]
    uok=uphoton[indok]
    modfok=modfphoton[indok]
    pinok=pinphoton[indok]
    phaok=phaphoton[indok]
    def likelihoodspin2_noflux_shift(alpha,beta,angshift,deltadeg):
        #def likelihoodspin2_noflux_shift(alpha,beta,angshift,phase0):
        model.reset_geometry(alpha,beta)
        degm=model.pol_deg_spline(eok, pok)*modfok
        um=model.u_spline(eok, pok)
        qm=model.q_spline(eok, pok)
        cos2d=(qok*qm+uok*um)
        sin2d=(qok*um-uok*qm)
        degmloc=degm*(1+deltadeg)
        #degmloc=degm #*(1+deltadeg)
        cos2tot=cos2d*numpy.cos(2*numpy.radians(angshift))-sin2d*numpy.sin(2*numpy.radians(angshift))
        return numpy.sum(numpy.log(1+degmloc*cos2tot)) 
    #res3=minimize((lambda x: -likelihoodspin2_noflux_shift(x[0],x[1],x[2],x[3])), x0=[51.0, 43.0, 0.1,0.1], method='Nelder-Mead')
    #res3=minimize((lambda x: -likelihoodspin2_noflux_shift(x[0],x[1],x[2],x[3])), x0=[45.0, 45.0, 0.5,0.5], method='Nelder-Mead')
    #print(res3)


    #== UNDERLYING MODEL: MDF + RMF + ARF + 3du + UNBINNED FIT  ==#
    # modulation factor for each du
    file_path_du1 = irf_file_path(DEFAULT_IRF_NAME,1, 'modf')
    modf_fun_du1 = xModulationFactor(file_path_du1) 
    file_path_du2 = irf_file_path(DEFAULT_IRF_NAME,2, 'modf')
    modf_fun_du2 = xModulationFactor(file_path_du2)
    file_path_du3 = irf_file_path(DEFAULT_IRF_NAME,3, 'modf')
    modf_fun_du3 = xModulationFactor(file_path_du3)
   
    # events for each du 
    with fits.open(folded_EVT_FILE_PATH[0]) as hdul_du1:
        qphoton_du1   = hdul_du1[1].data['Q']
        uphoton_du1   = hdul_du1[1].data['U']
        ephoton_du1   = hdul_du1[1].data['ENERGY']
        pphoton_du1   = hdul_du1[1].data['PHASE']
        phaphoton_du1 = hdul_du1[1].data['PHA'].astype(int)
    
    with fits.open(folded_EVT_FILE_PATH[1]) as hdul_du2:
        qphoton_du2   = hdul_du2[1].data['Q']
        uphoton_du2   = hdul_du2[1].data['U']
        ephoton_du2   = hdul_du2[1].data['ENERGY']
        pphoton_du2   = hdul_du2[1].data['PHASE']
        phaphoton_du2 = hdul_du2[1].data['PHA'].astype(int)
    
    with fits.open(folded_EVT_FILE_PATH[2]) as hdul_du3:
        qphoton_du3   = hdul_du3[1].data['Q']
        uphoton_du3   = hdul_du3[1].data['U']
        ephoton_du3   = hdul_du3[1].data['ENERGY']
        pphoton_du3   = hdul_du3[1].data['PHASE']
        phaphoton_du3 = hdul_du3[1].data['PHA'].astype(int)
    # remove the fits files with events    
    #os.system('rm {}*'.format(OUT_FILE_PATH_BASE))
    
    # select energy for events in each du
    indxdu1=(ephoton_du1<=4) & (ephoton_du1>=2)
    edu1=ephoton_du1[indxdu1]
    pdu1=pphoton_du1[indxdu1]
    qdu1=qphoton_du1[indxdu1]
    udu1=uphoton_du1[indxdu1]
    phadu1=phaphoton_du1[indxdu1]
    
    indxdu2=(ephoton_du2<=4) & (ephoton_du2>=2)
    edu2=ephoton_du2[indxdu2]
    pdu2=pphoton_du2[indxdu2]
    qdu2=qphoton_du2[indxdu2]
    udu2=uphoton_du2[indxdu2]
    phadu2=phaphoton_du2[indxdu2]
    
    indxdu3=(ephoton_du3<=4) & (ephoton_du3>=2)
    edu3=ephoton_du3[indxdu3]
    pdu3=pphoton_du3[indxdu3]
    qdu3=qphoton_du3[indxdu3]
    udu3=uphoton_du3[indxdu3]
    phadu3=phaphoton_du3[indxdu3]
    
    # RMF for each detector unit. 
    # Notice that in ixpeobssim v18 the 3 RMF are the same
    with fits.open(irf_file_path(DEFAULT_IRF_NAME, 1, 'rmf')) as hdu:
        matrix1=hdu[1].data['MATRIX']
        matrixt1=numpy.transpose(matrix1)
        
    with fits.open(irf_file_path(DEFAULT_IRF_NAME, 2, 'rmf')) as hdu:
        matrix2=hdu[1].data['MATRIX']
        matrixt2=numpy.transpose(matrix2)
        
    with fits.open(irf_file_path(DEFAULT_IRF_NAME, 3, 'rmf')) as hdu:
        matrix3=hdu[1].data['MATRIX']
        matrixt3=numpy.transpose(matrix3)

    # Effective are for each du
    from ixpeobssim.irf.arf import xEffectiveArea 
    file_path_du1 = irf_file_path(DEFAULT_IRF_NAME,1, 'arf')
    arf_du1 = xEffectiveArea(file_path_du1)
    file_path_du2 = irf_file_path(DEFAULT_IRF_NAME,2, 'arf')
    arf_du2 = xEffectiveArea(file_path_du2)
    file_path_du3 = irf_file_path(DEFAULT_IRF_NAME,3, 'arf')
    arf_du3 = xEffectiveArea(file_path_du3)

    # prepare list to perform loop on each in du
    qdulist     = [qdu1,qdu2,qdu3]
    udulist     = [udu1,udu2,udu3]
    edulist     = [edu1,edu2,edu3]
    pdulist     = [pdu1,pdu2,pdu3]
    phadulist   = [phadu1,phadu2,phadu3]
    funmodflist = [modf_fun_du1,modf_fun_du2, modf_fun_du3]
    matrixtlist = [matrixt1, matrixt2, matrixt3]
    arflist     = [arf_du1, arf_du2, arf_du3]    
    pha_list=numpy.arange(matrixt.shape[0])
    
    def likelihoodspin15(alpha,beta,deltadeg):
        #print(alpha,beta,deltadeg) 
        model.reset_geometry(alpha,beta)
        flux_conv=numpy.dot(matrixt,model.flux)
        fdulist = []
        for matrixtn, arfdu, funmodf, qdu, udu, edu, pdu, phadu in zip(matrixtlist,  arflist, funmodflist, qdulist, udulist, edulist, pdulist, phadulist):
            flux_conv=numpy.dot(matrixtn,numpy.transpose(arfdu(effae)*numpy.transpose(model.flux)))
            u_data=numpy.dot(matrixtn, numpy.transpose((arfdu(effae)*funmodf(effae))*numpy.transpose(model.flux*model.u_data*model.pol_deg_data)))
            q_data=numpy.dot(matrixtn, numpy.transpose((arfdu(effae)*funmodf(effae))*numpy.transpose(model.flux*model.q_data*model.pol_deg_data)))
            u_data/=flux_conv
            q_data/=flux_conv
            u_spline = xInterpolatedBivariateSpline(pha_list, model.phase, u_data,kx=3, ky=3)
            q_spline = xInterpolatedBivariateSpline(pha_list, model.phase, q_data,kx=3, ky=3)
            um=u_spline(phadu,pdu)
            qm=q_spline(phadu,pdu)
            fdu=(1+0.5*(1+deltadeg)*(qdu*qm+udu*um))#*fm
            fdulist=numpy.concatenate((fdulist,fdu))
        return numpy.sum(numpy.log(fdulist))-npred

    #res15=minimize((lambda x: -likelihoodspin15(x[0],x[1],x[2])), x0=[51,41,0.1],method='Nelder-Mead')
    #print(res15)


    def likelihoodspin16(alpha,beta,angshift,deltadeg):
        #print('%0.5f %0.5f %0.5f %0.5f' % (alpha,beta,angshift,deltadeg))
        model.reset_geometry(alpha,beta)
        flux_conv=numpy.dot(matrixt,model.flux)
        fdulist = []
        for matrixtn, arfdu, funmodf, qdu, udu, edu, pdu, phadu in zip(matrixtlist,  arflist, funmodflist, qdulist, udulist, edulist, pdulist, phadulist):
            flux_conv=numpy.dot(matrixtn,numpy.transpose(arfdu(effae)*numpy.transpose(model.flux)))
            u_data=numpy.dot(matrixtn, numpy.transpose((arfdu(effae)*funmodf(effae))*numpy.transpose(model.flux*model.u_data*model.pol_deg_data)))
            q_data=numpy.dot(matrixtn, numpy.transpose((arfdu(effae)*funmodf(effae))*numpy.transpose(model.flux*model.q_data*model.pol_deg_data)))
            u_data/=flux_conv
            q_data/=flux_conv
            u_spline = xInterpolatedBivariateSpline(pha_list, model.phase, u_data,kx=3, ky=3)
            q_spline = xInterpolatedBivariateSpline(pha_list, model.phase, q_data,kx=3, ky=3)
            um=u_spline(phadu,pdu)
            qm=q_spline(phadu,pdu)

            pcos2d = (qdu*qm + udu*um)
            psin2d = (qdu*um - udu*qm)       
            pcos2tot = pcos2d*numpy.cos(2*numpy.radians(angshift)) - psin2d*numpy.sin(2*numpy.radians(angshift))

            fdu= 1 + 0.5*(1.+deltadeg)*pcos2tot

            fdulist=numpy.concatenate((fdulist,fdu))

        return numpy.sum(numpy.log(fdulist))-npred

    #res16=minimize((lambda x: -likelihoodspin16(x[0],x[1],x[2],x[3])), x0=[51,41,0.1,0.1],method='Nelder-Mead')
    #print(res16)

    #return res, popt, ii, idproc, res15
    return res2, ii, idproc

if __name__ == '__main__':
    task = int(sys.argv[2])
    filename = 'task_hx1_' + sys.argv[2] + '.txt'
    ii = int(sys.argv[1])
    res, ii, idproc = fitting(ii,task)
    with open(filename,"a") as file:
        file.write('%12.6f'*4 % (res.x[0],res.x[1],res.x[2],res.x[3]) +
                   '%10s'*1 % (res.success) + '%4d'*1 % (res.nit) + '\n')
        #file.write('%12.6f'*9 % (res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],popt[0],popt[1],popt[2],popt[3]) +
        #           '%10s'*2 % (res.success,idproc)+ '%4d'*2 % (res.nit,ii) +
        #           '%12.6f'*3 % (res2.x[0],res2.x[1],res2.x[2]) +
        #           '%10s'*2 % (res.success,res2.success) + '%4d'*2 % (res.nit,res2.nit) + '\n')
        file.flush()
    file.close()
exit()
