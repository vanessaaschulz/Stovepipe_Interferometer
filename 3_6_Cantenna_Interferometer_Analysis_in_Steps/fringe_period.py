import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as sodr
from scipy.signal import find_peaks



def fit(p,x):
    '''
    input: x_data vector and parameters p
    output: fit function for fringe period (inverse to cosine function)
    '''
    a,b,c=p
    #fringe period inverse to cosine function
    return a/(np.cos(b*x)+c)#) )+d

def perform_fringe_fit(x_data,y_data,baseline,scan_number,yerr, time_sun):
    '''
    input: x_data vector, y_data vector, baseline and scan number
    use scipy.odr to perform fit to fringe period
    write results of fit into output file
    output: x and y fit vectors 
    '''
    model = sodr.Model(fit)
    # as errors: yerr is uncertainty of real part of correlation temperature, xerr is time resolution about 3s ~0.0008h
    fit_data = sodr.RealData(x_data, y_data,sx=0.0008, sy=yerr )
    odr = sodr.ODR(fit_data, model, beta0=[3.0,0.5,1.0])
    out = odr.run()
    a_n = out.beta[0]
    err_a = out.sd_beta[0]
    b_n = out.beta[1]
    err_b = out.sd_beta[1]
    c_n = out.beta[2]
    err_c = out.sd_beta[2]
    res_var=out.res_var
    
    file = open("./Analysis/LCORR_"+str(scan_number)+"/output.txt", "a")
    print("Results of fringe period fit "+str(baseline)+" baseline:")
    file.write('Results of fit '+str(baseline)+' baseline:'"\n")
    print("residuals:",res_var)
    file.write('residuals:'+str(res_var)+"\n")
    print("Parameter 1/a:",(a_n, err_a))
    file.write('Parameter 1/a:'+str(a_n)+";"+str(err_a)+"\n")
    print("Parameter 2/b:",(b_n, err_b))
    file.write('Parameter 2/b:'+str(b_n)+";"+str(err_b)+"\n")
    print("Parameter 3/c:",(c_n, err_c))
    file.write('Parameter 3/c:'+str(c_n)+";"+str(err_c)+"\n")
    file.close()

    x_fit=time_sun
    y_fit=a_n/(np.cos(b_n*x_fit)+c_n)
    
    return x_fit, y_fit

def fringe_period_simulator(u,dec):
    '''
    input: u (baseline in EW direction) and declination of the sun
    calculate fringe rate and then fringe period
    output: fringe period in minutes
    '''
    
    w_earth = 7.3*10**(-5) # in sec
    fringe_rate= u*w_earth*np.cos(dec) # 1/s
    fringe_period = (1/fringe_rate)/60 # in min
    
    return fringe_period

def fringe_period(rot_coor_em,rot_coor_mw,rot_coor_ew,scan_number,time_h_res, time_sun):
    '''
    input: scan number, resampled time vector and correlation vectors of all baselines
    limit resampled time vector to 7 to 17h UTC
    find peaks of fringes on each baseline
    perform fringe period fit and plot it
    output: plot fringe period
    '''
    # set time range of fringes
    t_new=np.array([])
    r_me=np.array([])
    r_mw=np.array([])
    r_ew=np.array([])
    for i in range(0,len(time_h_res)):
        if time_h_res[i]>=6.00 and time_h_res[i]<=17.00:
            t_new=np.append(t_new,time_h_res[i])
            r_me=np.append(r_me,rot_coor_em[0][i])
            r_mw=np.append(r_mw,rot_coor_mw[0][i])
            r_ew=np.append(r_ew,rot_coor_ew[0][i])
# find peaks in real part of correlation(cos modulation) for 3 baselines
# distance must me matched to baseline: short baseline-->big difference
    [peaks1,loc1] = find_peaks(r_me,distance=120, height=2)
    [peaks2,loc2] = find_peaks(r_mw,distance=150, height=2)
    [peaks3,loc3] = find_peaks(r_ew,distance=100, height=2)

    time_me=np.array(t_new)[peaks1]
    time_mw=np.array(t_new)[peaks2]
    time_ew=np.array(t_new)[peaks3]

# calculate fringe period
# calculate time difference between maxima
    t_me =np.array([])
    time_me_diff=np.array([])
    for i in range(1,len(peaks1)):
        t_me=np.append(t_me,(time_me[i]+time_me[i-1])/2)
        time_me_diff=np.append(time_me_diff,(time_me[i]-time_me[i-1])*60)
        
    t_mw =np.array([])
    time_mw_diff=np.array([])
    for i in range(1,len(peaks2)):
        t_mw=np.append(t_mw,(time_mw[i]+time_mw[i-1])/2)
        time_mw_diff=np.append(time_mw_diff,(time_mw[i]-time_mw[i-1])*60)
    
    t_ew =np.array([])
    time_ew_diff=np.array([])
    for i in range(1,len(peaks3)):
        t_ew=np.append(t_ew,(time_ew[i]+time_ew[i-1])/2)
        time_ew_diff=np.append(time_ew_diff,(time_ew[i]-time_ew[i-1])*60)
        
    fringe_me = perform_fringe_fit(t_me,time_me_diff, 'EM',scan_number,rot_coor_em[2], time_sun)
    fringe_mw = perform_fringe_fit(t_mw,time_mw_diff, 'MW',scan_number,rot_coor_mw[2], time_sun)
    fringe_ew = perform_fringe_fit(t_ew,time_ew_diff, 'EW',scan_number,rot_coor_ew[2], time_sun)
    
    plt.figure(figsize=(10,10))
    plt.title("Fringe Periode", fontsize=20)
    plt.plot(t_me,time_me_diff,marker="x", markersize=5,color="blue",ls='None',label="EM")
    plt.plot(fringe_me[0], fringe_me[1], lw=2,color="cyan",label="fit-EM")
    plt.plot(t_mw,time_mw_diff,marker="x", markersize=5,color="black",ls='None',label="MW")
    plt.plot(fringe_mw[0], fringe_mw[1], lw=2,color="green",label="fit-MW")
    plt.plot(t_ew,time_ew_diff,marker="x", markersize=5,color="red",ls='None', label="EW")
    plt.plot(fringe_ew[0], fringe_ew[1], lw=2,color="orange",label="fit-EW")
    plt.xlabel('UTC[h]',fontsize=20)
    plt.ylabel('fringe period[min]',fontsize=20) # time diff between 2 max is fringe period
    plt.grid()
    plt.xlim(6,18)
    plt.ylim(0,45)
    plt.legend(loc="best", prop={'size': 10})
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/fringe_period.png',dpi=200,bbox_inches='tight')
    plt.figure(figsize=(10,10))
    plt.title("Fringe Maxima", fontsize=20)
    plt.plot(t_new, r_me,color="blue",label="EM")
    plt.plot(time_me, np.array(r_me)[peaks1],marker="x", markersize=5,ls='None',color="green",label="EM max")
    plt.plot(t_new, r_mw,color="black",label="MW")
    plt.plot(time_mw, np.array(r_mw)[peaks2],marker="x", markersize=5,ls='None',color="orange",label="MW max")
    plt.plot(t_new, r_ew,color="red",label="EW")
    plt.plot(time_ew, np.array(r_ew)[peaks3],marker="x", markersize=5,ls='None',color="brown",label="EW max")
    plt.xlabel('UTC[h]',fontsize=20)
    plt.ylabel('Amplitude[a.u.]',fontsize=20)
    plt.xlim(6,17)
    plt.ylim(0,45)
    plt.grid()
    plt.legend(loc="best", prop={'size': 20})
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/max_fringe.png',dpi=200,bbox_inches='tight')

    return fringe_mw[1], fringe_ew[1], fringe_me[1]


def plot_sim(fringe_em,fringe_ew,fringe_mw,time_sun, scan_number):
    '''
    input: simulated fringe periods for all 3 baselines and time_sun vector
    
    output: plot simulation
    '''
    plt.figure(figsize=(10,10))
    plt.title("Fringe period simulator", fontsize=20)
    plt.plot(time_sun, abs(fringe_em),color="blue",label="EM")
    plt.plot(time_sun, abs(fringe_mw),color="black",label="MW")
    plt.plot(time_sun, abs(fringe_ew),color="red",label="EW")
    plt.xlabel('UTC[h]',fontsize=20)
    plt.ylabel('fringe period[min]',fontsize=20)
    plt.xlim(6,18)
    plt.ylim(0,45)
    plt.grid()
    plt.legend(loc="best", prop={'size': 20})
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/fringe_period_simulator.png',dpi=200,bbox_inches='tight')

def diff_fit_model(fringe_period_fit, fringe_em, fringe_ew, fringe_mw, time_sun, scan_number):
    '''
    input: simulated fringe periods for all 3 baselines and performed fits to real data
    
    output: difference between fit and simulation
    '''
    diff_em = fringe_period_fit[2]-abs(fringe_em)
    diff_mw = fringe_period_fit[0]-abs(fringe_mw)
    diff_ew = fringe_period_fit[1]-abs(fringe_ew)
    plt.figure(figsize=(10,10))
    plt.title("Comparison between fringe period model and data", fontsize=20)
    plt.plot(time_sun, diff_em,color="blue",label="EM")
    plt.plot(time_sun, diff_mw,color="black",label="MW")
    plt.plot(time_sun, diff_ew,color="red",label="EW")
    plt.xlabel('UTC[h]',fontsize=20)
    plt.ylabel('(fitted fringe period - modeled fringe period)[min]',fontsize=20)
    plt.xlim(6,18)
    #plt.ylim(0,45)
    plt.grid()
    plt.legend(loc="best", prop={'size': 20})
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/comparison_fringe_period_data_model.png',dpi=200,bbox_inches='tight')