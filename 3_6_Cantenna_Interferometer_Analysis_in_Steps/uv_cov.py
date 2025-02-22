# import packages
import numpy as np
import ephem
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
import math


# define time function:
def get_sec(time_str):
    h,m,s=time_str.split(':')
    return (float(h)*3600+float(m)*60+float(s))/3600

def hour_angle(time_sun,scan_number):
    # from measured data, calculate hour angle and declination of target(sun)
    ha=np.array([])
    ha_rad=np.array([])
    dec=np.array([])
    filename_data = './Data/LCORR_'+str(scan_number)+'.csv'
    data = pd.read_csv(filename_data, delimiter=' ',header = None)
    
    # define Stockert Observatory
    obs=ephem.Observer()
    obs.lat='50.569417'
    obs.lon='6.722'
    obs.elevation=435
    # sunrise, transit, sunset
    obs_day=data[1][0]
    for i in range(0,len(time_sun)):
        #print(time[i])
        obs.date=(''+str(obs_day)+' '+str(time_sun[i])+'')
        #print(obs.date)
        sun=ephem.Sun()
        sun.compute(obs)
        sun_ha=get_sec(str(sun.ha))
        # adjust hourangle when sun is towards East, so there is no jump at noon
        if sun_ha>15: 
            ha=np.append(ha,sun_ha-24)
        else:
            ha=np.append(ha,sun_ha)
        ha_rad=np.append(ha_rad,np.radians(ha[i]*15))
        #print(sun_ha, ha[i], ha_rad[i])
        sun_dec=sun.dec
        dec=np.append(dec,sun_dec)
        
    # plots of hourangle and declination of sun
    plt.figure(figsize=(10,10))
    plt.title("HourAngle Sun Calculations", fontsize=20)
    plt.plot(time_sun, ha_rad,color="green",marker="x", markersize=5,ls='None',label="hourangle-to-rad(ha)")
    plt.grid()
    plt.legend(loc="best", prop={'size': 10})
    plt.xlabel('UTC [h]',fontsize=20)
    plt.ylabel('hourangle sun [rad]',fontsize=20)
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/hour_angle_rad.png',dpi=200,bbox_inches='tight')
    return ha_rad,dec

def baseline_length(baseline,scan_number,B_x,B_y,B_z):
    '''
    input: baseline, scan number and x,y and z coordinates of baseline vector (x is in EW direction, y is in NS direction and z is height difference)
    output: length of baseline vector, hour angle and declination of intersection of baseline with northern hemisphere 
    '''
    
    file = open("./Analysis/LCORR_"+str(scan_number)+"/output.txt", "a")
    D = np.sqrt(B_x**2+B_y**2+B_z**2)
    print("Results of "+str(baseline)+" baseline:")
    file.write('Results of '+str(baseline)+' baseline:'"\n")
    print("maximum baseline [m]:", D)
    file.write('maximum baseline [m]: '+str(D)+"\n")
    alpha = np.arctan2(B_y,B_x)
    print("Hour Angle of intersection of baseline with northern hemisphere (S=0°, W=90°, N=180°, E=270°)[deg]:", math.degrees(alpha))
    file.write('Hour Angle of intersection of baseline with northern hemisphere (S=0°, W=90°, N=180°, E=270°)[deg]: '+str(math.degrees(alpha))+"\n")
    decl = np.arctan2(B_z,D)
    print("declination of intersection of baseline with northern hemisphere[deg]:", math.degrees(decl))
    file.write('declination of intersection of baseline with northern hemisphere[deg]: '+str(math.degrees(decl))+"\n")
    file.close()
    return D, alpha, decl


def uvw(time_sun,scan_number,baseline,wavel,ha_rad,dec,alpha, d,D):
    '''
    input: observation wavelength, hour angle and declination of sun, alpha and declination of baseline vector, and baseline length
    first calculate u,v and w coordinates in meters
    convert u,v and w coordinates to multiples of wavelength    
    output: u,v and w in wavelength
    '''
        
    ha_plus=(90)*np.pi/180 +alpha
    ha_minus=ha_plus-np.pi
    wavel_c=wavel[0]
    u_m= (D*u.m*np.cos(d)*np.sin(ha_rad-ha_plus))/wavel_c
    v_m= (D*u.m*(np.sin(d)*np.cos(dec)-np.cos(d)*np.sin(dec)*np.cos(ha_rad-ha_plus)))/wavel_c 
    w_m= (D*u.m*(np.sin(d)*np.sin(dec)+np.cos(d)*np.cos(dec)*np.cos(ha_rad-ha_plus)))/wavel_c

    # mirrored uvw coordinates since observed sources are symmmetric/real
    u_m_2= (D*u.m*np.cos(d)*np.sin(ha_rad-ha_minus) )/wavel_c
    v_m_2= (D*u.m*(np.sin(d)*np.cos(dec)-np.cos(d)*np.sin(dec)*np.cos(ha_rad-ha_minus)) )/wavel_c
    w_m_2= (D*u.m*(np.sin(d)*np.sin(dec)+np.cos(d)*np.cos(dec)*np.cos(ha_rad-ha_minus)))/wavel_c
    
    
    
    u_w = np.append(u_m,u_m_2)
    v_w = np.append(v_m,v_m_2)
    w_w = np.append(w_m,w_m_2)
    
    # for baseline and fringe frequency simulation
    u_w_b = D*u.m*np.cos(d)*np.sin(ha_rad-ha_plus) /wavel_c
    
    # min and max recoverable scale
    Arr_uv=np.sqrt(u_m**2+v_m**2)
    pos_min_uv=np.argmax(Arr_uv)
    u_min=u_w[pos_min_uv]
    v_min=v_w[pos_min_uv]
    pos_max_uv=np.argmin(Arr_uv)
    u_max=u_w[pos_max_uv]
    v_max=v_w[pos_max_uv]
    
    file = open("./Analysis/LCORR_"+str(scan_number)+"/output.txt", "a")
    print("Results of "+str(baseline)+" baseline:")
    file.write('Results of '+str(baseline)+' baseline:'"\n")
    scale_min=np.degrees(1.0/np.nanmax(Arr_uv)*u.rad)
    print("min scale from uv coverage:", scale_min)
    print("time of min scale:", time_sun[pos_min_uv])
    file.write('Min Scale from uv coverage: '+str(scale_min)+"\n")
    file.write('time of min scale:'+str(time_sun[pos_min_uv])+"\n")
    scale_max=np.degrees(1.0/np.nanmin(Arr_uv)*u.rad)
    print("max scale from uv coverage:", scale_max)
    print("time of max scale:", time_sun[pos_max_uv-1])
    file.write('Max Scale from uv coverage: '+str(scale_max)+"\n")
    file.write('time of max scale:'+str(time_sun[pos_max_uv-1])+"\n")
    file.close()
    
    return u_w,v_w,w_w, u_w_b, u_min, v_min, u_max, v_max
    
def plot_uvw(uvw1,uvw2,uvw3,scan_number):
    '''
    input: uv and w coordinates of each baseline and scan number
    output: plot uv coverage
    '''
    # plot uv coverage
    plt.figure(figsize=(10,10))
    plt.title("UV-Coverage of this observation", fontsize=20)
    plt.plot(uvw3[0], uvw3[1],color="red",marker="x", markersize=5,ls='None',label="EW")
    plt.plot(uvw3[4], uvw3[5],color="green",marker="o", markersize=10,ls='None',label="Min")
    plt.plot(uvw3[6], uvw3[7],color="orange",marker="o", markersize=10,ls='None',label="Max")
    plt.plot(uvw1[0], uvw1[1],color="blue",marker="x", markersize=5,ls='None',label="EM")
    plt.plot(uvw1[4], uvw1[5],color="green",marker="o", markersize=10,ls='None')
    plt.plot(uvw1[6], uvw1[7],color="orange",marker="o", markersize=10,ls='None')
    plt.plot(uvw2[0], uvw2[1],color="black",marker="x", markersize=5,ls='None',label="MW")
    plt.plot(uvw2[4], uvw2[5],color="green",marker="o", markersize=10,ls='None')
    plt.plot(uvw2[6], uvw2[7],color="orange",marker="o", markersize=10,ls='None')
    plt.grid()
    #plt.xlim(-50,50)
    #plt.ylim(-50,50)
    plt.legend(loc="best", prop={'size': 10})
    plt.xlabel('u [in wavelength]',fontsize=20)
    plt.ylabel('v [in wavelength]',fontsize=20)
    plt.savefig('./Analysis/LCORR_'+str(scan_number)+'/uv_coverage.png',dpi=200,bbox_inches='tight')