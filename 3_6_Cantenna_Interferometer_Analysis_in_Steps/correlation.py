import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.odr as sodr


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
    print("Results of baseline fit for "+str(antenna)+" correlation:")
    file.write('Results of baseline fit for '+str(antenna)+'  correlation:'"\n")
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


def correlation_mw(scan_number, time_h_res, time_sun,resampling):
    '''
    input: scan number and resampling
    resample correlation array for MW baseline
    correct phase offset in correlation vector for MW baseline
    output: corrected correlation vector
    '''
    r=resampling
    filename_data = './Data/LCORR_'+str(scan_number)+'.csv'
    # read data and define data sets
    data = pd.read_csv(filename_data, delimiter=' ',header = None)
    
    corr_mw_r = data[12]
    corr_mw_r_res = [np.average(corr_mw_r[i:i+r]) for i in range(0,len(corr_mw_r),r)]
    corr_mw_i = data[13]
    corr_mw_i_res = [np.average(corr_mw_i[i:i+r]) for i in range(0,len(corr_mw_i),r)]
    coor_mw=np.vstack([corr_mw_r_res, corr_mw_i_res])
    # -0.28 rad is phase offset between middle and west antenna
    rot_mw = np.array([[np.cos(0.28), -np.sin(0.28)],
                       [np.sin(0.28), np.cos(0.28)]])
    rot_coor_mw=rot_mw @ coor_mw
    
    # split into real and imaginary part
    rot_coor_r=rot_coor_mw[0]
    rot_coor_i=rot_coor_mw[1]
    
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
    y_off_r=np.hstack((rot_coor_r[i_o:j_o],rot_coor_r[k_o:l_o]))
    y_off_i=np.hstack((rot_coor_i[i_o:j_o],rot_coor_i[k_o:l_o]))
    
    y_fit_off_r=do_basefit3(x_off,y_off_r,np.asarray(time_h_res),'WM real', scan_number)
    y_fit_off_i=do_basefit3(x_off,y_off_i,np.asarray(time_h_res),'WM imag', scan_number)
    
    # calculate new Antenna Temperature
    Corr_r=rot_coor_r-y_fit_off_r
    Corr_i=rot_coor_i-y_fit_off_i
    
    # calculate and print error of antenna temperature of each antenna
    y2_r=np.hstack((Corr_r[i_o:j_o],Corr_r[k_o:l_o]))
    err_corr_r=np.sqrt(np.sum((y2_r)**2)/len(y2_r))
    file = open("./Analysis/LCORR_"+str(scan_number)+"/output.txt", "a")
    print("RMS of MW real correlation Measurement:",err_corr_r)
    file.write('RMS of MW real correlation Measurement:'+str(err_corr_r)+"\n")
    y2_i=np.hstack((Corr_i[i_o:j_o],Corr_i[k_o:l_o]))
    err_corr_i=np.sqrt(np.sum((y2_i)**2)/len(y2_i))
    print("RMS of MW imag correlation Measurement:",err_corr_i)
    file.write('RMS of MW imag correlation Measurement:'+str(err_corr_i)+"\n")
    file.close()
    
    
    
    return  Corr_r,Corr_i, err_corr_r, err_corr_i


def correlation_em(scan_number,time_h_res, time_sun,resampling):
    '''
    input: scan number and resampling
    resample correlation array for EM baseline
    correct phase offset in correlation vector for EM baseline
    output: corrected correlation vector
    '''
    r=resampling
    filename_data = './Data/LCORR_'+str(scan_number)+'.csv'
    # read data and define data sets
    data = pd.read_csv(filename_data, delimiter=' ',header = None)
    
    corr_me_r = data[10]
    corr_me_r_res = [np.average(corr_me_r[i:i+r]) for i in range(0,len(corr_me_r),r)]
    corr_me_i = data[11]
    corr_me_i_res = [np.average(corr_me_i[i:i+r]) for i in range(0,len(corr_me_i),r)]
    coor_me=np.vstack([corr_me_r_res, corr_me_i_res])
    # -2.93 rad is phase offset between middle and east antenna
    rot_me = np.array([[np.cos(2.93), -np.sin(2.93)],
                       [np.sin(2.93), np.cos(2.93)]])
    rot_coor_me=rot_me @ coor_me
    
    # split into real and imaginary part
    rot_coor_r=rot_coor_me[0]
    rot_coor_i=rot_coor_me[1]
    
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
    y_off_r=np.hstack((rot_coor_r[i_o:j_o],rot_coor_r[k_o:l_o]))
    y_off_i=np.hstack((rot_coor_i[i_o:j_o],rot_coor_i[k_o:l_o]))
    
    y_fit_off_r=do_basefit3(x_off,y_off_r,np.asarray(time_h_res),'EM real', scan_number)
    y_fit_off_i=do_basefit3(x_off,y_off_i,np.asarray(time_h_res),'EM imag', scan_number)
    
    # calculate new Antenna Temperature
    Corr_r=rot_coor_r-y_fit_off_r
    Corr_i=rot_coor_i-y_fit_off_i
    
    # calculate and print error of antenna temperature of each antenna
    y2_r=np.hstack((Corr_r[i_o:j_o],Corr_r[k_o:l_o]))
    err_corr_r=np.sqrt(np.sum((y2_r)**2)/len(y2_r))
    file = open("./Analysis/LCORR_"+str(scan_number)+"/output.txt", "a")
    print("RMS of EM real correlation Measurement:",err_corr_r)
    file.write('RMS of EM real correlation Measurement:'+str(err_corr_r)+"\n")
    y2_i=np.hstack((Corr_i[i_o:j_o],Corr_i[k_o:l_o]))
    err_corr_i=np.sqrt(np.sum((y2_i)**2)/len(y2_i))
    print("RMS of EM imag correlation Measurement:",err_corr_i)
    file.write('RMS of EM imag correlation Measurement:'+str(err_corr_i)+"\n")
    file.close()
    
    
        
    return  Corr_r,Corr_i, err_corr_r, err_corr_i

def correlation_ew(scan_number,time_h_res, time_sun,resampling):
    '''
    input: scan number and resampling
    resample correlation array for EW baseline
    correct phase offset in correlation vector for EW baseline
    output: corrected correlation vector
    '''
    r=resampling
    filename_data = './Data/LCORR_'+str(scan_number)+'.csv'
    # read data and define data sets
    data = pd.read_csv(filename_data, delimiter=' ',header = None)
    
    corr_ew_r = data[14]
    corr_ew_r_res = [np.average(corr_ew_r[i:i+r]) for i in range(0,len(corr_ew_r),r)]
    corr_ew_i = data[15]
    corr_ew_i_res = [np.average(corr_ew_i[i:i+r]) for i in range(0,len(corr_ew_i),r)]
    coor_ew=np.vstack([corr_ew_r_res, corr_ew_i_res])
    # -3.11 rad is phase offset between east and west antenna
    rot_ew = np.array([[np.cos(3.11), -np.sin(3.11)],
                       [np.sin(3.11), np.cos(3.11)]])
    rot_coor_ew=rot_ew @ coor_ew
    
    # split into real and imaginary part
    rot_coor_r=rot_coor_ew[0]
    rot_coor_i=rot_coor_ew[1]
    
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
    y_off_r=np.hstack((rot_coor_r[i_o:j_o],rot_coor_r[k_o:l_o]))
    y_off_i=np.hstack((rot_coor_i[i_o:j_o],rot_coor_i[k_o:l_o]))
    
    y_fit_off_r=do_basefit3(x_off,y_off_r,np.asarray(time_h_res),'EW real', scan_number)
    y_fit_off_i=do_basefit3(x_off,y_off_i,np.asarray(time_h_res),'EW imag', scan_number)
    
    # calculate new Antenna Temperature
    Corr_r=rot_coor_r-y_fit_off_r
    Corr_i=rot_coor_i-y_fit_off_i
    
    # calculate and print error of antenna temperature of each antenna
    y2_r=np.hstack((Corr_r[i_o:j_o],Corr_r[k_o:l_o]))
    err_corr_r=np.sqrt(np.sum((y2_r)**2)/len(y2_r))
    file = open("./Analysis/LCORR_"+str(scan_number)+"/output.txt", "a")
    print("RMS of EW real correlation Measurement:",err_corr_r)
    file.write('RMS of EW real correlation Measurement:'+str(err_corr_r)+"\n")
    y2_i=np.hstack((Corr_i[i_o:j_o],Corr_i[k_o:l_o]))
    err_corr_i=np.sqrt(np.sum((y2_i)**2)/len(y2_i))
    print("RMS of EW imag correlation Measurement:",err_corr_i)
    file.write('RMS of EW imag correlation Measurement:'+str(err_corr_i)+"\n")
    file.close()
   
    
    return Corr_r,Corr_i, err_corr_r, err_corr_i

def plot_correlation(rot_coor_me,rot_coor_mw,rot_coor_ew,scan_number,time_h_res):
    '''
    input: scan number, resampled time vector and corrected correlation vectors
    output: plot real and imaginary part of correlation
    '''
    # plot of real part of correlation
    plt.figure(figsize=(40,10))
    plt.title("Correlation real part", fontsize=20)
    plt.plot(time_h_res, rot_coor_me[0],color="blue",label="EM")
    plt.plot(time_h_res, rot_coor_mw[0],color="black",label="MW")
    plt.plot(time_h_res,rot_coor_ew[0],color="red",label="EW")
    plt.xlabel('UTC[h]',fontsize=20)
    plt.ylabel('Amplitude[a.u.]',fontsize=20)
    #plt.xlim(10,12)
    #plt.ylim(-20,20)
    plt.grid()
    plt.legend(loc="best", prop={'size': 10},fontsize=20)
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/corr_real.png',dpi=500,bbox_inches='tight')
    # plot of imag part of correlation
    plt.figure(figsize=(40,10))
    plt.title("Correlation imaginary part", fontsize=20)
    plt.plot(time_h_res, rot_coor_me[1],color="blue",label="EM")
    plt.plot(time_h_res, rot_coor_mw[1],color="black",label="MW")
    plt.plot(time_h_res, rot_coor_ew[1],color="red",label="EW")
    plt.xlabel('UTC[h]',fontsize=20)
    plt.ylabel('Amplitude[a.u.]',fontsize=20)
    #plt.xlim(10,12)
    plt.grid()
    plt.legend(loc="best", prop={'size': 10})
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/corr_imag.png',dpi=200,bbox_inches='tight')
    
    