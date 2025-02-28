# import packages
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.interpolate import griddata
from scipy.fft import fftshift, ifft2


def one_uv_vis_list(uvw_em,uvw_mw,uvw_ew,vis_sun_mw,vis_sun_em,vis_sun_ew):
    '''
    input: uvw coordinates and visibility (amplitude and phase) for all 3 baselines
    first calculate complex visibility of amplitude and phase
    double visbility for each baseline to match mirrored uvw coordinates
    define one u, v, w and visibility list    
    output: u,v,w and visibility list
    '''
    vis_em1 = vis_sun_em[0]*np.exp(vis_sun_em[1]*1j)
    vis_em = np.append(vis_em1,np.conj(vis_em1))
    vis_mw1 = vis_sun_mw[0]*np.exp(vis_sun_mw[1]*1j)
    vis_mw = np.append(vis_mw1,np.conj(vis_mw1))
    vis_ew1 = vis_sun_ew[0]*np.exp(vis_sun_ew[1]*1j)
    vis_ew = np.append(vis_ew1,vis_ew1)
    vis1 = np.append(vis_em, np.conj(vis_mw))
    vis = np.append(vis1,vis_ew)
    u_list1 = np.append(uvw_em[0],uvw_mw[0])
    u_list = np.append(u_list1,uvw_ew[0])
    v_list1 = np.append(uvw_em[1],uvw_mw[1])
    v_list = np.append(v_list1,uvw_ew[1])
    w_list1 = np.append(uvw_em[2],uvw_mw[2])
    w_list = np.append(w_list1,uvw_ew[2])
    
    return u_list, v_list, w_list, vis

def aperture_function(x, y, baselines_x, baselines_y, sigma):
    # aperture function of Cantenna Interferometer
    aperture = np.zeros_like(x)
    
    for i in range(0,len(baselines_x)):
        pos_x = baselines_x[i]  
        pos_y = baselines_y[i]
        
        # Gaussian function
        aperture += np.exp(-((x - pos_x)**2 + (y+pos_y)**2) / (sigma[i]**2))
    return aperture

def PSF(scan_number,wavel, baselines_x, baselines_y, HPBW,grid_size):
    '''
    input: Wavelength, Position of Baselines (x,y), HPBW, Nx and Ny resolution gridding
    create aperture function (of 3 Gaussian sampling Functions) and FFT of it builds PSF
    output: PSF at local noon time
    '''
    b_x =baselines_x*u.m/wavel
    b_y =baselines_y*u.m/wavel
    x = np.linspace(-max(b_x), max(b_x), grid_size)  # x-Coordinates
    y = np.linspace(-max(b_x), max(b_x), grid_size)  # y-Coordinates
    X, Y = np.meshgrid(x, y)
    
    opening_angle_deg = HPBW  # Opening Angle in degrees
    opening_angle_rad = np.radians(opening_angle_deg)  # Conversion to Radians
    # Standard Deviation for Gaussian function
    sigma = np.tan(opening_angle_rad / 2)  # depends on opening angle
    
    # Calculate Aperture Function
    A = aperture_function(X, Y, b_x, b_y, sigma)
    # inverse Fourier Transform of Aperture Function
    PSF = np.fft.fftshift(np.fft.ifft2(A))
    
    # Calculate Intensity of PSF
    PSF_intensity = np.abs(PSF)

    # Normalisation of PSF
    PSF_intensity /= np.max(PSF_intensity)
    
    # plot aperture function
    plt.figure(figsize=(10, 8))
    plt.subplot(121)
    plt.imshow(A, extent=(-max(b_x), max(b_x),-max(b_x), max(b_x)), cmap='inferno')
    plt.colorbar(label='Intensity',shrink=0.4)
    plt.title(f'Aperture Function of Cantenna Interferometer \ngrid resolution:{grid_size}')
    plt.xlabel('x [in wavelength]')
    plt.ylabel('y [in wavelength]')

    # plot PSF
    plt.subplot(122)
    plt.imshow(PSF_intensity, extent=(-max(b_x), max(b_x),-max(b_x), max(b_x)), cmap='inferno')
    plt.colorbar(label='Intensity',shrink=0.4)
    plt.title(f'Point Spread Function (PSF) of Cantenna Interferometer\ngrid resolution:{grid_size}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/PSF_noon_'+str(grid_size)+'.png',dpi=200,bbox_inches='tight')
    
    return PSF_intensity    

def dirty_beam(scan_number,u_list,v_list,grid_size):
    '''
    input: visibility , u and v coordinates, Nx and Ny resolution gridding
    inverse fourier transform of Sampling function [W(u,v)=1 if measured else 0], natural weighting
    output: dirty beam
    '''
    u_grid = np.linspace(min(u_list), max(u_list), grid_size)
    v_grid = np.linspace(min(v_list), max(v_list), grid_size)
    U, V = np.meshgrid(u_grid, v_grid)
    
    vis_grid = griddata((u_list, v_list), np.ones(len(u_list)), (U, V), method='cubic', fill_value=0)
    
    dirty_beam = fftshift(ifft2(vis_grid))
    
    
    plt.figure(figsize=(10,10))
    plt.title(f'Dirty Beam with grid resolution:{grid_size}', fontsize=20)
    img1=plt.imshow((np.abs(dirty_beam)),cmap="inferno")
    plt.xlabel('l',fontsize=20)
    plt.ylabel('m',fontsize=20)
    plt.colorbar(img1,shrink=0.8)
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/dirty_beam_'+str(grid_size)+'.png',dpi=200,bbox_inches='tight')
    return dirty_beam


def dirty_map(scan_number,vis, u_list,v_list,grid_size):
    '''
    input: visibility , u and v coordinates, Nx and Ny resolution gridding
    inverse fourier transform of measured visibility 
    output: dirty map
    '''
    u_grid = np.linspace(min(u_list), max(u_list), grid_size)
    v_grid = np.linspace(min(v_list), max(v_list), grid_size)
    U, V = np.meshgrid(u_grid, v_grid)
    vis_grid = griddata((u_list, v_list), vis, (U, V), method='cubic', fill_value=0)
    
    dirty_image = fftshift(ifft2(vis_grid))
    
    
    plt.figure(figsize=(10,10))
    plt.title(f'Dirty Map with grid resolution:{grid_size}', fontsize=20)
    img1=plt.imshow((np.abs(dirty_image)),cmap="inferno")
    plt.xlabel('l',fontsize=20)
    plt.ylabel('m',fontsize=20)
    plt.colorbar(img1,shrink=0.8)
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/dirty_map_'+str(grid_size)+'.png',dpi=200,bbox_inches='tight')
    return dirty_image

def clean(dirty_image, dirtybeam, threshold, max_iter,gain):
    """Perform the CLEAN algorithm."""
    # Step 1: initialize clean image and residual image
    
    clean_image = np.zeros_like(dirty_image)
    residual = np.copy(dirty_image)
    
    psf_size = dirtybeam.shape[0]
    psf_center = psf_size // 2
    
    # iterate through number of iterations
    for i in range(max_iter):
        # Step 2: strength and position of peak in residual image
        
        # Get the coordinates of the peak 
        peak_coords = np.unravel_index(np.argmax((residual)), residual.shape)
        #print(peak_coords)
        
        # Find the peak in the residual image
        peak_value = residual[peak_coords]
        #print(peak_value)
        
        # Step 3: Repeat from Step 2 on, unless residual image< threshold
        # Stopping criteria if residual image is below threshold
        if (peak_value) < threshold:
            print("Iteration:",i)
            print("Peak Value:",peak_value)
            break
        # Extract the PSF centered on the peak
        y_start = max(0, peak_coords[0] - psf_center)
        y_end = min(dirty_image.shape[0], peak_coords[0] + psf_center + 1)
        x_start = max(0, peak_coords[1] - psf_center)
        x_end = min(dirty_image.shape[1], peak_coords[1] + psf_center + 1)

        psf_slice = dirtybeam[psf_center - (peak_coords[0] - y_start):psf_center + (y_end - peak_coords[0]),
                        psf_center - (peak_coords[1] - x_start):psf_center + (x_end - peak_coords[1])]

        # Ensure the PSF slice and the residual region have the same shape
        psf_slice = np.pad(psf_slice, ((0, max(0, y_end - y_start - psf_slice.shape[0])),
                                        (0, max(0, x_end - x_start - psf_slice.shape[1]))), mode='constant')

        # Subtract the scaled PSF from the residual
        residual[y_start:y_end, x_start:x_end] -= gain*peak_value*psf_slice#.value

        # Step 5: Record peak position and magnitude subtracted in clean image
        # Add the peak to the clean image
        clean_image[peak_coords] += gain*peak_value
          
    return clean_image, residual