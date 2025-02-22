import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.odr as sodr
import astropy.units as u
from astropy import constants as const

def basefit3(p,x):  # third order polynomial for baseline fit
    a,b,c,d=p
    return a*x**3+b*x**2+c*x+d  
def do_basefit3(x,y,x_fit,antenna, scan_number):
    '''
    Baseline fit for 3. order polynomial
    
    Parameter:
    x: x-data set
    y: y-data set
    
    Output:
    y_fit: Baseline fit
    
    '''
    
    lin_model1 = sodr.Model(basefit3)
    fit_data = sodr.RealData(x,y ,sx=None, sy=None)
    odr = sodr.ODR(fit_data, lin_model1, beta0=[0,0,0,0])
    out = odr.run()
    a_n = out.beta[0]
    b_n= out.beta[1]
    c_n = out.beta[2]
    d_n = out.beta[3]
    err_a = out.sd_beta[0]
    err_b = out.sd_beta[1]
    err_c = out.sd_beta[2]
    err_d = out.sd_beta[3]
    res_var=out.res_var
    file = open("./Analysis/LCORR_"+str(scan_number)+"/output.txt", "a")
    print("Results of baseline fit for "+str(antenna)+" antenna:")
    file.write('Results of baseline fit for '+str(antenna)+'  antenna:'"\n")
    print("residuals:",res_var)
    file.write('residuals:'+str(res_var)+"\n")
    print("Parameter 1/a:",(a_n, err_a))
    file.write('Parameter 1/a:'+str(a_n)+";"+str(err_a)+"\n")
    print("Parameter 2/b:",(b_n, err_b))
    file.write('Parameter 2/b:'+str(b_n)+";"+str(err_b)+"\n")
    print("Parameter 3/c:",(c_n, err_c))
    file.write('Parameter 3/c:'+str(c_n)+";"+str(err_c)+"\n")
    print("Parameter 4/d:",(d_n, err_d))
    file.write('Parameter 4/d:'+str(d_n)+";"+str(err_d)+"\n")
    file.close()
    
    y_fit=a_n*x_fit**3+b_n*x_fit**2+c_n*x_fit+d_n
    return y_fit

def fit_func(p,x): # gaussian fit
    a, b, c, d = p
    return a * np.exp(-(((x-b)**2)/(2 * c**2)))+d

def do_gauss(x,y,x_fit, antenna, scan_number, yerr ):
    '''
    
    
    Parameter:
    x: x-data set
    y: y-data set
    
    
    Output:
    y_fit: Baseline fit
    
    '''
    
    lin_model1 = sodr.Model(fit_func)
    # as errors: yerr is uncertainty of antenna temperature, xerr is time resolution about 3s ~0.0008h
    fit_data = sodr.RealData(x,y ,sx=0.0008, sy=yerr)
    odr = sodr.ODR(fit_data, lin_model1, beta0=[10,11.00,2.5,300.0])
    out = odr.run()
    a_n = out.beta[0]
    b_n= out.beta[1]
    c_n = out.beta[2]
    d_n = out.beta[3]
    err_a = out.sd_beta[0]
    err_b = out.sd_beta[1]
    err_c = out.sd_beta[2]
    err_d = out.sd_beta[3]
    res_var=out.res_var
    file = open("./Analysis/LCORR_"+str(scan_number)+"/output.txt", "a")
    print("Results of gaussian fit for "+str(antenna)+" HPBW antenna:")
    file.write('Results of gaussian fit for '+str(antenna)+' HPBW antenna:'"\n")
    print("residuals:",res_var)
    file.write('residuals:'+str(res_var)+"\n")
    print("Parameter 1/a:",(a_n, err_a))
    file.write('Parameter 1/a:'+str(a_n)+";"+str(err_a)+"\n")
    print("Parameter 2/b:",(b_n, err_b))
    file.write('Parameter 2/b:'+str(b_n)+";"+str(err_b)+"\n")
    print("Parameter 3/c:",(c_n, err_c))
    file.write('Parameter 3/c:'+str(c_n)+";"+str(err_c)+"\n")
    print("Parameter 4/d:",(d_n, err_d))
    file.write('Parameter 4/d:'+str(d_n)+";"+str(err_d)+"\n")
    file.close()
    
    y_fit=a_n * np.exp(-(((x_fit-b_n)**2)/(2 * c_n**2)))+d_n
    return y_fit,c_n, err_c

def wavel(freq,bandwidth,scan_number):
    '''
    input: frequency and scan number
    calculate observation wavelength and write into output file
    output: observation wavelength
    '''
    # frequency and wavelength
    freq1=freq *1/u.s
    wavel=const.c/freq1
    # lower end of bandwidth
    freq_l=(freq-bandwidth/2) *1/u.s
    wavel_l=const.c/freq_l
    # upper end of badnwidth
    freq_u=(freq+bandwidth/2) *1/u.s
    wavel_u=const.c/freq_u
    
    file = open("./Analysis/LCORR_"+str(scan_number)+"/output.txt", "a")
    print('Observed wavelength of central frequency:', wavel)
    file.write('Observed wavelength of central frequency'+str(wavel)+"\n")
    print('Observed wavelength of lower end of bandwidth:', wavel_l)
    file.write('Observed wavelength of lower end of bandwidth'+str(wavel_l)+"\n")
    print('Observed wavelength of upper end of baseline bandwidth:', wavel_u)
    file.write('Observed wavelength of upper end of baseline bandwidth'+str(wavel_u)+"\n")
    file.close()
    
    return wavel, wavel_l, wavel_u

def single_antenna(scan_number,resampling,time_h_res,sunrise, sunset, twil_m, twil_e):
    '''
    input: scan number, resampling and resampled time array
    resample continuum signal of each antenna and plot them
    output: single cantenna plot to check for solar flares and RFI
    '''
    r=resampling
    filename_data = './Data/LCORR_'+str(scan_number)+'.csv'
    data = pd.read_csv(filename_data, delimiter=' ',header = None)

    # power single channel data:
    pow_M1 = data[4]# power Middle (Pluto2) Antenna
    pow_M1_res = [np.average(pow_M1[i:i+r]) for i in range(0,len(pow_M1),r)]
    pow_M2 = data[5]# power Middle (Pluto3) Antenna
    pow_M2_res = [np.average(pow_M2[i:i+r]) for i in range(0,len(pow_M2),r)]
    pow_E = data[6]# power East (Pluto2) Antenna
    pow_E_res = [np.average(pow_E[i:i+r]) for i in range(0,len(pow_E),r)]
    pow_W = data[7]# power West (Pluto3) Antenna
    pow_W_res = [np.average(pow_W[i:i+r]) for i in range(0,len(pow_W),r)]
    
    # plot power single channel: to investigate RFI on each antenna and/or solar activity (solar flares)
    plt.figure(figsize=(10,10))
    plt.title("Power Single Channel", fontsize=20)
    plt.axvline(x=sunrise, color="yellow", label="sun above horizon")
    plt.axvline(x=sunset, color="yellow")
    plt.axvline(x=twil_m, color="green", label="astronomical twilight")
    plt.axvline(x=twil_e, color="green")
    plt.plot(time_h_res, pow_M1_res,color="red",label="Middle1")
    plt.plot(time_h_res, pow_M2_res,color="blue",label="Middle2")
    plt.plot(time_h_res, pow_E_res,color="black",label="East")
    plt.plot(time_h_res, pow_W_res,color="brown",label="West")
    plt.xlabel('UTC[h]',fontsize=20)
    plt.ylabel('Amplitude[a.u.]',fontsize=20)
    #plt.xlim(7,14)
    plt.grid()
    plt.legend(loc="best", prop={'size': 10})
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/power_single_channel.png',dpi=200,bbox_inches='tight')
    return pow_M1_res, pow_M2_res,pow_E_res,pow_W_res
    
def RMS_anal(single_antenna, time_h_res, time_sun,scan_number):
    '''
    input: resampled continuum signals of different antennas, resampled time array, time sun above horizon data(sunrise and sunset)
    output: Error of antenna Temperature for each antenna
    '''
    pow_W_res = single_antenna[3]
    pow_E_res = single_antenna[2]
    pow_M2_res = single_antenna[1]
    pow_M1_res = single_antenna[0]
    
    # select time range for offset/night time
    time_1=np.where((np.asarray(time_h_res)>=0.00)&(np.asarray(time_h_res)<=time_sun[1]))[0]
    time_2=np.where((np.asarray(time_h_res)>=time_sun[3])&(np.asarray(time_h_res)<=24))[0]
#print(vel_1,vel_2)
    i_o=time_1[0]
    #print(i_o)
    j_o=time_1[-1]
    #print(j_o)
    k_o=time_2[0]
    #print(k_o)
    l_o=time_2[-1]
    #print(l_o)

    
    # select data range without sun
    x_off=np.hstack((time_h_res[i_o:j_o],time_h_res[k_o:l_o]))
    y_off_W=np.hstack((pow_W_res[i_o:j_o],pow_W_res[k_o:l_o]))
    y_off_E=np.hstack((pow_E_res[i_o:j_o],pow_E_res[k_o:l_o]))
    y_off_M1=np.hstack((pow_M1_res[i_o:j_o],pow_M1_res[k_o:l_o]))
    y_off_M2=np.hstack((pow_M2_res[i_o:j_o],pow_M2_res[k_o:l_o]))
    
    # perform baseline fit
    y_fit_off_W=do_basefit3(x_off,y_off_W,np.asarray(time_h_res),'West', scan_number)
    y_fit_off_E=do_basefit3(x_off,y_off_E,np.asarray(time_h_res),'East', scan_number)
    y_fit_off_M1=do_basefit3(x_off,y_off_M1,np.asarray(time_h_res), 'Middle1', scan_number)
    y_fit_off_M2=do_basefit3(x_off,y_off_M2,np.asarray(time_h_res), 'Middle2', scan_number)
    
    # calculate new Antenna Temperature
    T_A_W=pow_W_res-y_fit_off_W
    T_A_E=pow_E_res-y_fit_off_E
    T_A_M1=pow_M1_res-y_fit_off_M1
    T_A_M2=pow_M2_res-y_fit_off_M2
    
    # calculate and print error of antenna temperature of each antenna
    y2_W=np.hstack((T_A_W[i_o:j_o],T_A_W[k_o:l_o]))
    err_T_A_W=np.sqrt(np.sum((y2_W)**2)/len(y2_W))
    file = open("./Analysis/LCORR_"+str(scan_number)+"/output.txt", "a")
    print("Error of Antenna Temperature of West Measurement[K]:",err_T_A_W)
    file.write('Error of Antenna Temperature of West Measurement[K]:'+str(err_T_A_W)+"\n")
    y2_E=np.hstack((T_A_E[i_o:j_o],T_A_E[k_o:l_o]))
    err_T_A_E=np.sqrt(np.sum((y2_E)**2)/len(y2_E))
    print("Error of Antenna Temperature of East Measurement[K]:",err_T_A_E)
    file.write('Error of Antenna Temperature of East Measurement[K]:'+str(err_T_A_E)+"\n")
    y2_M1=np.hstack((T_A_M1[i_o:j_o],T_A_M1[k_o:l_o]))
    err_T_A_M1=np.sqrt(np.sum((y2_M1)**2)/len(y2_M1))
    print("Error of Antenna Temperature of Middle 1 Measurement[K]:",err_T_A_M1)
    file.write('Error of Antenna Temperature of Middle1 Measurement[K]:'+str(err_T_A_M1)+"\n")
    y2_M2=np.hstack((T_A_M2[i_o:j_o],T_A_M2[k_o:l_o]))
    err_T_A_M2=np.sqrt(np.sum((y2_M2)**2)/len(y2_M2))
    print("Error of Antenna Temperature of Middle 2 Measurement[K]:",err_T_A_M2)
    file.write('Error of Antenna Temperature of Middle2 Measurement[K]:'+str(err_T_A_M2)+"\n")
    file.close()
    
    return T_A_W, T_A_E, T_A_M1, T_A_M2,err_T_A_W, err_T_A_E, err_T_A_M1, err_T_A_M2

def HPBW(RMS_analysis, time_h_res,scan_number):
    '''
    input: baseline reduced continuum antenna signals
    output: HPBW of each antenna assuming a gaussian beam
    '''
    # select data range with sun
    x=time_h_res
    y_M1=RMS_analysis[2]
    y_M2=RMS_analysis[3]
    y_W=RMS_analysis[0]
    y_E=RMS_analysis[1]
    err_M1=RMS_analysis[6]
    err_M2=RMS_analysis[7]
    err_W=RMS_analysis[4]
    err_E=RMS_analysis[5]
    
    # perform gaussian fit
    y_fit_W=do_gauss(x,y_W,np.asarray(time_h_res),'West', scan_number,err_W)
    y_fit_E=do_gauss(x,y_E,np.asarray(time_h_res),'East', scan_number,err_E)
    y_fit_M1=do_gauss(x,y_M1,np.asarray(time_h_res),'Middle1', scan_number,err_M1)
    y_fit_M2=do_gauss(x,y_M2,np.asarray(time_h_res),'Middle2', scan_number,err_M2)
    
    
    fig, axs = plt.subplots(2, 2,figsize=(10, 10),sharex=True, sharey=True)
    axs[0, 0].plot(time_h_res, y_W, lw=2,color="brown",label="data W")
    axs[0, 0].plot(time_h_res, y_fit_W[0], lw=2,color="red",label="fit ")
    axs[0, 0].set_title('West antenna')
    axs[0, 1].plot(time_h_res, y_E, lw=2,color="black",label="data E")
    axs[0, 1].plot(time_h_res, y_fit_E[0], lw=2,color="red",label="fit ")
    axs[0, 1].set_title('East antenna')
    axs[1, 0].plot(time_h_res, y_M2, lw=2,color="blue",label="data M2")
    axs[1, 0].plot(time_h_res, y_fit_M2[0], lw=2,color="red",label="fit ")
    axs[1, 0].set_title('Middle2 antenna')
    axs[1, 1].plot(time_h_res, y_M1, lw=2,color="green",label="data M1")
    axs[1, 1].plot(time_h_res, y_fit_M1[0], lw=2,color="red",label="fit ")
    axs[1, 1].set_title('Middle1 antenna')
    for ax in axs.flat:
        ax.set(xlabel='UTC[h]', ylabel='antenna temperature[K]')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/single_stovepipe_RMS_HPBW.png',dpi=200,bbox_inches='tight')
    
    # calculate HPBW in hours:
    # West:
    HPBW_W=abs(y_fit_W[1])*2*np.sqrt(2*np.log(2))
    HPBW_W_err=abs(y_fit_W[2])*2*np.sqrt(2*np.log(2))
    # East:
    HPBW_E=abs(y_fit_E[1])*2*np.sqrt(2*np.log(2))
    HPBW_E_err=abs(y_fit_E[2])*2*np.sqrt(2*np.log(2))
    # M1:
    HPBW_M1=abs(y_fit_M1[1])*2*np.sqrt(2*np.log(2))
    HPBW_M1_err=abs(y_fit_M1[2])*2*np.sqrt(2*np.log(2))
    # M2:
    HPBW_M2=abs(y_fit_M2[1])*2*np.sqrt(2*np.log(2))
    HPBW_M2_err=abs(y_fit_M2[2])*2*np.sqrt(2*np.log(2))
    
    HPBW_w_deg=HPBW_W*360/24
    HPBW_w_deg_err=HPBW_W_err*360/24                              
    HPBW_e_deg=HPBW_E*360/24
    HPBW_e_deg_err=HPBW_E_err*360/24                                       
    HPBW_m1_deg=HPBW_M1*360/24
    HPBW_m1_deg_err=HPBW_M1_err*360/24
    HPBW_m2_deg=HPBW_M2*360/24
    HPBW_m2_deg_err=HPBW_M2_err*360/24
    
    file = open("./Analysis/LCORR_"+str(scan_number)+"/output.txt", "a")
    print("results for HPBW  for each antenna:")
    file.write('results for HPBW  for each antenna:'"\n")
    print("West:",(HPBW_w_deg, HPBW_w_deg_err))
    file.write('West:'+str(HPBW_w_deg)+";"+str(HPBW_w_deg_err)+"\n")
    print("East:",(HPBW_e_deg, HPBW_e_deg_err))
    file.write('East:'+str(HPBW_e_deg)+";"+str(HPBW_e_deg_err)+"\n")
    print("Middle1:",(HPBW_m1_deg, HPBW_m1_deg_err))
    file.write('Middle1:'+str(HPBW_m1_deg)+";"+str(HPBW_m1_deg_err)+"\n")
    print("Middle2:",(HPBW_m2_deg, HPBW_m2_deg_err))
    file.write('Middle2:'+str(HPBW_m2_deg)+";"+str(HPBW_m2_deg_err)+"\n")
    file.close()
    
    return  HPBW_e_deg, HPBW_m1_deg,HPBW_w_deg