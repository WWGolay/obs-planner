import numpy as np
import ccdproc as ccdp
from astropy.io.fits import getdata
from scipy.ndimage.interpolation import rotate
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
from scipy.signal import medfilt, medfilt2d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from datetime import datetime
import matplotlib.pyplot as plt
import sys

class grism_calibrate:
    def __init__(self, grism_image, ref_file):
        
        self.grism_image = grism_image
        self.ref_file = ref_file
        self.balmer = np.array([397.0, 410.17, 434.05, 486.14, 656.45])
        
        ''' Open image, extract header information '''
        im, hdr = getdata(grism_image, 0, header=True)
        self.object    =  hdr['OBJECT']
        self.utdate    =  hdr['DATE-OBS'][:-3].replace('T',' ')
        self.telescope =  hdr['TELESCOP']
        self.instrument = hdr['INSTRUME']
        
        # Flip image so L-> R corresponds to short -> long wavelength                     
        im = np.fliplr(im)
        
        # Translate filter codes
        fil = hdr['FILTER'][0]
        if fil == '8': self.filter = 'R'
        if fil == '9': self.filter = 'B'
        
        # Create default plot title
        self.title = '%s\n%s %s grism: %s' % \
        (self.object, self.telescope, self.utdate, self.filter)
        self.imsize_x = hdr['NAXIS1'] ; self.imsize_y = hdr['NAXIS2']
                                                        
        # Crack Jacoby reference file, extract spectrum
        wave_ref, spec_ref = np.loadtxt(ref_file, unpack=True, comments='#',usecols=(0,1),dtype = float)
        spec_ref /= np.max(spec_ref)
        
        self.image = im
        self.filter = fil
        self.wave_ref = wave_ref
        self.spec_ref= spec_ref
        
    def get_info(self):
        return self.image,self.object, self.utdate,self.telescope, self.instrument, self.filter

    def plot_image(self,image = np.array([]), title='',figsize =(8,4),cmap='gray'):
        '''Plot image: defaults to full image '''
        if len(np.shape(image))==1 : image = self.image
        fig, ax = plt.subplots(figsize=figsize)
        zmean = np.median(image); s = np.std(image)
        vmin = zmean - 2*s; vmax = zmean + 12*s
        myplot = ax.imshow(image,cmap=cmap, vmin= vmin, vmax = vmax)
        if title == '': title = self.title
        plt.title(title)
        return fig     

    def rotate_image(self,box,width):
        '''Fit linear slope to maximum y values in cropped image'''
        xmin,xmax,ymin,ymax = box
        subim = self.image[ymin:ymax,xmin:xmax]
        X = range(subim.shape[1])
        Y = [np.argmax(subim[:,j]) for j in X ]
        angle_rad,b = np.polyfit(X,Y,1)
        angle = np.rad2deg(angle_rad)
        subim_rot = rotate(subim, angle,reshape=False)
        # Crop subimage width centered 
        yc = angle_rad * (xmax-xmin)/2 + b
        ymin = int(yc - width/2); ymax = int(ymin + width)
        subim = subim_rot[ymin:ymax,:]
        self.subim = subim      
        return angle, subim    
  
    def plot_strip(self,cmap='jet', title = ''):
        '''Plot strip image'''
        im = self.subim
        fig, ax = plt.subplots(figsize=(8, 4))
        myplot = ax.imshow(im,cmap=cmap, vmin= np.min(im), vmax = np.max(im))
        if title == '': title = '%s\n Dispersed strip image' % self.title
        plt.title(title)
        return fig

    def calc_channel_signal(self, xpixel, do_plot=False):
        
        ''' Calculates total counts in specified spectral channel xpixel by subtracting background and summing.
        The spectral signal is assumed to be in middle half of the spectrum. '''
        
        yvals = self.subim[:,xpixel]
        yindex = np.arange(len(yvals))
        
        # Choose first, last quartiles for base, fit linear slope
        n1 = int(len(yindex)/4); n2 = 3*n1
        x1 = yindex[0:n1] ; x2 = yindex[n2:]
        y1 = yvals[0:n1]  ; y2 = yvals[n2:]
        X = np.concatenate((x1,x2),axis=0)
        Y = np.concatenate((y1,y2),axis=0)
        c = np.polyfit(X,Y,1) # linear fit  
        p = np.poly1d(c)
        base = p(yindex)
        
        # Calculate signal vs pixel by subtracting baseline, sum and get index of maximum pixel
        signal = yvals - base
        signal_max = np.max(signal)
        ymax = np.argmax(signal)
    
        # Plot
        fig =''
        if do_plot:
            title = 'Channel %i\n ymax: %.1f, Max value: %.1f' % (xpixel,signal_max,ymax)
            fig, ax = plt.subplots(1,figsize=(10,12))
            ax.plot(yindex,base+yvals,'k.',label ='X pixel number %i' % xpixel)
            ax.plot(yindex,base,'r-')
            ax.grid()
            ax.legend()
        
        return(ymax, np.sum(signal),signal_max,fig)  
    
    def calc_spectrum(self):
        '''Calculates raw spectrum by summing pixels in all vertical slices'''
        im = self.subim
        xsize = im.shape[1]
        pixels = np.arange(xsize)
        S = []
        for pixel in pixels:
                ymax,signal,signal_max,_ = self.calc_channel_signal(pixel, do_plot=False)
                S.append(signal)
        pixels = np.array(pixels) ; S = np.array(S)
        self.pixels = pixels
        self.raw_spectrum = S
        return


    def plot_spectrum(self, uncalibrated = True, title='', plot_balmer=True, medavg = 1,xlims =[0,0]):
        '''Plots raw or calibrated spectrum'''
        fig, ax = plt.subplots(1,1,figsize=(8, 4))
        
        
        
        if title == '': title=self.title
        fig.suptitle(title) 
        xmin,xmax = xlims
        
        if uncalibrated:         
            x = self.pixels ; y = self.raw_spectrum
            y = medfilt(y,kernel_size = medavg)   # Median average if requested
            ax.plot(x,y,'k-')
            ax.set_ylabel('Uncalibrated amplitude')
            ax.set_xlabel('Pixels')
            if xlims != [0,0]: 
                ax.set_xlim(xmin,xmax)
            ax.set_ylim(0,np.max(y)*1.1)
            ax.grid()
            
        else:
            x = self.wave ; y = self.calibrated_spectrum
            y = medfilt(y,kernel_size = medavg)   # Median average if requested
            ax.plot(x,y,'k-')
            ax.set_ylabel('Calibrated amplitude')
            ax.set_xlabel('Wavelength [nm]')
            ax.grid()
            if xlims != [0,0]: ax.set_xlim(xmin,xmax)
            if plot_balmer:
                for x in self.balmer: ax.axvline(x=x,linestyle='-.')
        
        return fig
  
    def find_spectral_peaks(self,prominence=0.2,width=3,do_plot=False):
        ''' Find pixel locations of spectral peaks for wavelength calibration'''
        S = self.raw_spectrum
        Snorm = S/np.nanmax(S)
        X = np.arange(len(Snorm))
        S_medavg = medfilt(Snorm,kernel_size=51)
        #S_peaks = -1*(Snorm - S_medavg)
        S_peaks = np.abs(Snorm - S_medavg)
        peaks, _ = find_peaks(S_peaks,prominence=prominence,width=width,distance=3)
        fig = ''
        if do_plot:
            fig = plt.figure(figsize=(12,3))
            plt.grid()
            plt.title(str(peaks))
            plt.plot(X,S_peaks)
            for peak in peaks:
                plt.axvline(x=peak,color='red')
        return peaks,fig
    
    def calc_wave(self,peaks,ref_lines):
        balmer_pix =  np.array(peaks)
        c = np.polyfit(balmer_pix,ref_lines,2)
        f_wave = np.poly1d(c)
        self.wave = f_wave(self.pixels)
        return f_wave,c
    
    def clip_spectrum(self, wave_min, wave_max):
        # Clips raw spectrum to user-specified wavelength range    
        raw_spectrum = self.raw_spectrum
        A = np.array(list(zip(self.wave,self.raw_spectrum)))
        A = A[A[:,0]>=wave_min]; A = A[A[:,0]<=wave_max]
        wave,raw_spectrum = list(zip(*A))
        wave = np.array(wave)
        spectrum = np.array(raw_spectrum)
        self.wave = wave
        self.raw_spectrum = raw_spectrum
        return wave,raw_spectrum  
    
    def calc_gain_curve(self, do_plot = False, do_poly=False, nsmooth=9,npoly=9):
        '''Calculates a gain curve by comparing Jacoby spectrum to observerd spectrum
        Returns calibrated spectrum  and either smoothed gains with same length as raw_spec
        or coefficients of a polynomial fit and ''' 
        
        wave = self.wave
        
        # Interpolate Jacoby reference spectrum so it has same length and wavelength range as observed spectrum
        f_interp = interp1d(self.wave_ref,self.spec_ref)
        spec_ref_interp = f_interp(wave)

        # Loess average
        spec_avg = lowess(self.raw_spectrum, wave, is_sorted=True, return_sorted=False, frac=0.05, it=0)
        
        # Median average both spectra and take ratio to get gain curve; smooth gain curve
        #spec_avg     = medfilt(raw_spec, kernel_size=nsmooth)
        spec_ref_avg = medfilt(spec_ref_interp, kernel_size=nsmooth)
        gain = spec_avg/spec_ref_avg
        gain_smooth = medfilt(gain,kernel_size=51)
       
        if do_poly:
            # Fit a high order polynomial to gains
            c = np.polyfit(wave,gain_smooth,npoly)
            p = np.poly1d(c)
            gain_curve = p(wave)
        else:
            c = None
            gain_curve = gain_smooth
            
        self.calibrated_spectrum = self.raw_spectrum/gain_curve
        self.gain_curve = gain_curve       

        # Plot gain curve and poly fit if requested           
        if do_plot:
            fig, ax = plt.subplots(1,1,figsize=(8, 4))
            plt.plot(wave,gain,'g.',label='Gains')
            plt.plot(wave,gain_smooth,'b.',label='Smooth gains')
            if do_poly: plt.plot(wave,gain_curve,'r-', lw =2, label ='Polynomial fit, n = %i' % npoly)
            plt.legend()
            plt.grid(); plt.title('Gain curve')
            plt.ylim(0)
            plt.xlabel('Wavelength [nm]')
        
        return c, gain_curve,fig   
    
