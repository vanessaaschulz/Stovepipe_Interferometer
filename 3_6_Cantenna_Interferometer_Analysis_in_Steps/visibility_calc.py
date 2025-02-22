# import packages
import numpy as np
import matplotlib.pyplot as plt


def visibility_em(correlation_em):
    '''
    input: correlation array of EM baseline
    calculcate visibility amplitude and phase for complex correlator 
    output: visibility amplitude and phase 
    '''
    rot_coor_me = correlation_em
    vis_me = np.sqrt(rot_coor_me[0]**2+rot_coor_me[1]**2)
    err_vis = np.sqrt( (rot_coor_me[0]/(np.sqrt(rot_coor_me[0]**2+rot_coor_me[1]**2))*rot_coor_me[2])**2 + (rot_coor_me[1]/(np.sqrt(rot_coor_me[0]**2+rot_coor_me[1]**2))*rot_coor_me[3])**2)
    ph_me = np.arctan2(rot_coor_me[1],rot_coor_me[0])
    err_ph = np.sqrt( (rot_coor_me[1]/(rot_coor_me[0]**2+rot_coor_me[1]**2)* rot_coor_me[2])**2 + (rot_coor_me[0]/(rot_coor_me[0]**2+rot_coor_me[1]**2)* rot_coor_me[3])**2 )
    return vis_me,ph_me, err_vis, err_ph

def visibility_mw(correlation_mw):
    '''
    input: correlation array of MW baseline
    calculcate visibility amplitude and phase for complex correlator 
    output: visibility amplitude and phase 
    '''
    rot_coor_mw = correlation_mw
    vis_mw = np.sqrt(rot_coor_mw[0]**2+rot_coor_mw[1]**2)
    err_vis = np.sqrt( (rot_coor_mw[0]/(np.sqrt(rot_coor_mw[0]**2+rot_coor_mw[1]**2))*rot_coor_mw[2])**2 + (rot_coor_mw[1]/(np.sqrt(rot_coor_mw[0]**2+rot_coor_mw[1]**2))*rot_coor_mw[3])**2)
    ph_mw = np.arctan2(rot_coor_mw[1],rot_coor_mw[0])
    err_ph = np.sqrt( (rot_coor_mw[1]/(rot_coor_mw[0]**2+rot_coor_mw[1]**2)*rot_coor_mw[2])**2 + (rot_coor_mw[0]/(rot_coor_mw[0]**2+rot_coor_mw[1]**2)*rot_coor_mw[3])**2 )
    return vis_mw,ph_mw,err_vis, err_ph

def visibility_ew(correlation_ew):
    '''
    input: correlation array of EW baseline
    calculcate visibility amplitude and phase for complex correlator 
    output: visibility amplitude and phase 
    '''
    rot_coor_ew = correlation_ew
    vis_ew = np.sqrt(rot_coor_ew[0]**2+rot_coor_ew[1]**2)
    err_vis = np.sqrt( (rot_coor_ew[0]/(np.sqrt(rot_coor_ew[0]**2+rot_coor_ew[1]**2))*rot_coor_ew[2])**2 + (rot_coor_ew[1]/(np.sqrt(rot_coor_ew[0]**2+rot_coor_ew[1]**2))*rot_coor_ew[3])**2)
    ph_ew = np.arctan2(rot_coor_ew[1],rot_coor_ew[0])
    err_ph = np.sqrt( (rot_coor_ew[1]/(rot_coor_ew[0]**2+rot_coor_ew[1]**2)*rot_coor_ew[2])**2 + (rot_coor_ew[0]/(rot_coor_ew[0]**2+rot_coor_ew[1]**2)*rot_coor_ew[3])**2 )
    return vis_ew,ph_ew,err_vis, err_ph

def plot_visibility(visibility_em,visibility_mw,visibility_ew,time_h_res,transit,scan_number):
    '''
    input: visibility arrays(amplitude and phase) of all 3 baselines, resampled time arrray, sun transit time and scan number
    output: plot visibility amplitude and phase 
    '''
    # plot of visibility
    plt.figure(figsize=(40,10))
    plt.title("visibility", fontsize=20)
    plt.plot(time_h_res, visibility_em[0],color="blue",label="ME")
    plt.plot(time_h_res, visibility_mw[0],color="black",label="MW")
    plt.plot(time_h_res, visibility_ew[0],color="red",label="EW")
    plt.xlabel('UTC[h]',fontsize=20)
    plt.ylabel('Amplitude[a.u.]',fontsize=20)
    #plt.xlim(6,18)
    plt.grid()
    plt.legend(loc="best", prop={'size': 10})
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/visibility_amplitude.png',dpi=200,bbox_inches='tight')
    
    # plot of phase 
    plt.figure(figsize=(40,10))
    plt.title("phase", fontsize=20)
    plt.plot(time_h_res, visibility_em[1],marker='o',ls='None',color="blue",label="EM")
    plt.plot(time_h_res, visibility_mw[1],marker='o',ls='None',color="black",label="MW")
    plt.plot(time_h_res, visibility_ew[1],marker='o',ls='None',color="red",label="EW")
    plt.xlabel('UTC[h]',fontsize=20)
    plt.ylabel('Phase[rad]',fontsize=20)
    plt.xlim(0,25)
    plt.grid()
    plt.legend(loc="best", prop={'size': 10})
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/phase.png',dpi=200,bbox_inches='tight')
    
    # plot of phase zoomin
    plt.figure(figsize=(40,10))
    plt.title("phase", fontsize=20)
    plt.plot(time_h_res, visibility_em[1],marker='o',ls='None',color="blue",label="EM")
    plt.plot(time_h_res, visibility_mw[1],marker='o',ls='None',color="black",label="MW")
    plt.plot(time_h_res, visibility_ew[1],marker='o',ls='None',color="red",label="EW")
    plt.xlabel('UTC[h]',fontsize=20)
    plt.ylabel('Phase[rad]',fontsize=20)
    plt.xlim(transit-0.5,transit+0.5)
    plt.grid()
    plt.legend(loc="best", prop={'size': 10})
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/phase_zoomin.png',dpi=200,bbox_inches='tight')
    
def vis_sun_em(visibility_em, time_h_res, Rise,Set):
    '''
    input: visibility array (amplitude and phase) of EM baseline, resampled time array and sunrise and sunset times
    limit visibility amplitude and phase for complex correlator for sun being above the horizon
    output: visibility amplitude and phase 
    '''
    vis_em=visibility_em[0]
    ph_em=visibility_em[1]
    err_vis_em=visibility_em[2]
    err_ph_em=visibility_em[3]
    
    vis_sun_em = np.array([])
    ph_sun_em = np.array([])
    err_vis_sun_em = np.array([])
    err_ph_sun_em = np.array([])
    for i in range(0,len(time_h_res)):
        if time_h_res[i]>=Rise and time_h_res[i]<=Set:
            vis_sun_em=np.append(vis_sun_em,vis_em[i])
            ph_sun_em=np.append(ph_sun_em,ph_em[i])
            err_vis_sun_em=np.append(err_vis_sun_em,err_vis_em[i])
            err_ph_sun_em=np.append(err_ph_sun_em,err_ph_em[i])
    return vis_sun_em, ph_sun_em,err_vis_sun_em, err_ph_sun_em

def vis_sun_mw(visibility_mw, time_h_res, Rise,Set):
    '''
    input: visibility array (amplitude and phase) of MW baseline, resampled time array and sunrise and sunset times
    limit visibility amplitude and phase for complex correlator for sun being above the horizon
    output: visibility amplitude and phase 
    '''
    vis_mw=visibility_mw[0]
    ph_mw=visibility_mw[1]
    err_vis_mw=visibility_mw[2]
    err_ph_mw=visibility_mw[3]
    
    vis_sun_mw = np.array([])
    ph_sun_mw = np.array([])
    err_vis_sun_mw = np.array([])
    err_ph_sun_mw = np.array([])
    for i in range(0,len(time_h_res)):
        if time_h_res[i]>=Rise and time_h_res[i]<=Set:
            vis_sun_mw=np.append(vis_sun_mw,vis_mw[i])
            ph_sun_mw=np.append(ph_sun_mw,ph_mw[i])
            err_vis_sun_mw=np.append(err_vis_sun_mw,err_vis_mw[i])
            err_ph_sun_mw=np.append(err_ph_sun_mw,err_ph_mw[i])
    return vis_sun_mw, ph_sun_mw,err_vis_sun_mw, err_ph_sun_mw

def vis_sun_ew(visibility_ew, time_h_res, Rise,Set):
    '''
    input: visibility array (amplitude and phase) of EW baseline, resampled time array and sunrise and sunset times
    limit visibility amplitude and phase for complex correlator for sun being above the horizon
    output: visibility amplitude and phase 
    '''
    vis_ew=visibility_ew[0]
    ph_ew=visibility_ew[1]
    err_vis_ew=visibility_ew[2]
    err_ph_ew=visibility_ew[3]
    
    vis_sun_ew = np.array([])
    ph_sun_ew = np.array([])
    err_vis_sun_ew = np.array([])
    err_ph_sun_ew = np.array([])
    for i in range(0,len(time_h_res)):
        if time_h_res[i]>=Rise and time_h_res[i]<=Set:
            vis_sun_ew=np.append(vis_sun_ew,vis_ew[i])
            ph_sun_ew=np.append(ph_sun_ew,ph_ew[i])
            err_vis_sun_ew=np.append(err_vis_sun_ew,err_vis_ew[i])
            err_ph_sun_ew=np.append(err_ph_sun_ew,err_ph_ew[i])
    return vis_sun_ew, ph_sun_ew,err_vis_sun_ew, err_ph_sun_ew

def plot_vis_sun(vis_sun_em,vis_sun_mw,vis_sun_ew,time_sun,scan_number):
    '''
    input: visibility arrays for sun(amplitude and phase) of all 3 baselines, resampled time sun arrray, sun transit time and scan number
    output: plot visibility amplitude and phase 
    '''
    # plot of visibility
    plt.figure(figsize=(40,10))
    plt.title("visibility Sun", fontsize=20)
    plt.plot(time_sun, vis_sun_em[0],color="blue",label="EM")
    plt.plot(time_sun, vis_sun_mw[0],color="black",label="MW")
    plt.plot(time_sun, vis_sun_ew[0],color="red",label="EW")
    plt.xlabel('UTC[h]',fontsize=20)
    plt.ylabel('Amplitude[a.u.]',fontsize=20)
    #plt.xlim(6,18)
    plt.grid()
    plt.legend(loc="best", prop={'size': 10})
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/visibility_amplitude_sun.png',dpi=200,bbox_inches='tight')
    # plot of phase 
    plt.figure(figsize=(40,10))
    plt.title("phase Sun", fontsize=20)
    plt.plot(time_sun, vis_sun_em[1],marker='o',ls='None',color="blue",label="EM")
    plt.plot(time_sun, vis_sun_mw[1],marker='o',ls='None',color="black",label="MW")
    plt.plot(time_sun, vis_sun_ew[1],marker='o',ls='None',color="red",label="EW")
    plt.xlabel('UTC[h]',fontsize=20)
    plt.ylabel('Phase[rad]',fontsize=20)
    #plt.xlim(0,25)
    plt.grid()
    plt.legend(loc="best", prop={'size': 10})
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/phase_sun.png',dpi=200,bbox_inches='tight')
    