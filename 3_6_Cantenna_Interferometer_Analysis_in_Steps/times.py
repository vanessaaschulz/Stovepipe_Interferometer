import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ephem


# define time function:
def get_sec(time_str):
    '''
    input: time string in format 00:00:00
    split time string into hours(h), minutes(m) and seconds(s)
    output: converted fractional time
    '''
    h,m,s=time_str.split(':')
    
    return (float(h)*3600+float(m)*60+float(s))/3600

def time_h(scan_number):
    '''
    input: scan number for datafile
    use get_sec function to convert time data to fractional time array
    output: fractional time array
    '''
    filename_data = './Data/LCORR_'+str(scan_number)+'.csv'
    # read data and define data sets
    data = pd.read_csv(filename_data, delimiter=' ',header = None)
    # time:
    time = data[2]
    time_h = [get_sec(x) for x in time]
    
    return time_h

def time_h_res(time_h,resampling):
    '''
    input: fractional time array and resampling
    resample fractional time array to smooth data
    output: fractional time array
    '''
    r=resampling
    time_h_res = [np.average(time_h[i:i+r]) for i in range(0,len(time_h),r)]
    
    return time_h_res

def time_sun(scan_number,time_h_res):
    '''
    input: resampled time array and scan number
    get sunrise, sun transit and sunset for Stockert observatory and write into output file
    filter resampled time array to sun being above horizon
    output: fractional time array, sunrise, transit and sunset
    '''
    
    filename_data = './Data/LCORR_'+str(scan_number)+'.csv'
    data = pd.read_csv(filename_data, delimiter=' ',header = None)
    file = open("./Analysis/LCORR_"+str(scan_number)+"/output.txt", "a")
    
    # define Stockert Observatory
    obs=ephem.Observer()
    obs.lat='50.569417'
    obs.lon='6.722'
    obs.elevation=435
    
    # sunrise, transit, sunset
    obs_day=data[1][0]
    obs.date=obs_day
    print("Observation date:",obs.date)
    file.write('Observation date '+str(obs.date)+"\n")
    sun=ephem.Sun()
    sun.compute(obs)
    sunrise=str(obs.next_rising(sun))
    transit=str(obs.next_transit(sun))
    sunset=str(obs.next_setting(sun))
    print("sunrise at Stockert Observatory will be [UTC]: ",sunrise)
    file.write("sunrise at Stockert Observatory will be [UTC]: "+str(sunrise)+"\n")
    print("transit at Stockert Observatory will be [UTC]: ",transit)
    file.write("transit at Stockert Observatory will be [UTC]: "+str(transit)+"\n")
    print("sunset at Stockert Observatory will be [UTC]: ",sunset)
    file.write("sunset at Stockert Observatory will be [UTC]: "+str(sunset)+"\n")
    Rise = int(sunrise[-8:-6]) + int(sunrise[-5:-3])/60 + int(sunrise[-2:])/3600
    #print(Rise)
    Transit= int(transit[-8:-6]) + int(transit[-5:-3])/60 + int(transit[-2:])/3600
    #print(Transit)
    Set = int(sunset[-8:-6]) + int(sunset[-5:-3])/60 + int(sunset[-2:])/3600
    #print(Set)
    file.close()
    
    # set time_h_res according to sunrise and sunset --> sun above horizon
    time_sun=np.array([])
    for i in range(0,len(time_h_res)):
        if time_h_res[i]>=Rise and time_h_res[i]<=Set:
            time_sun=np.append(time_sun,time_h_res[i])
    #print(time_sun)
    
    return time_sun, Rise, Transit, Set

def twilight(scan_number,time_h_res):
    
    filename_data = './Data/LCORR_'+str(scan_number)+'.csv'
    data = pd.read_csv(filename_data, delimiter=' ',header = None)
    file = open("./Analysis/LCORR_"+str(scan_number)+"/output.txt", "a")
    
    # define Stockert Observatory
    obs=ephem.Observer()
    obs.lat='50.569417'
    obs.lon='6.722'
    obs.elevation=435
    
    # set horizon to -18 degrees to get time of astronomical twilight
    obs.horizon='-18'
    
    # sunrise, transit, sunset
    obs_day=data[1][0]
    obs.date=obs_day
    sun=ephem.Sun()
    sun.compute(obs)
    twil_morn=str(obs.next_rising(sun, use_center=True))
    twil_ev=str(obs.next_setting(sun,use_center=True))
    print("Twilight morning at Stockert Observatory will be [UTC]: ",twil_morn)
    file.write("Twilight morning at Stockert Observatory will be [UTC]: "+str(twil_morn)+"\n")
    
    print("Twilight evening at Stockert Observatory will be [UTC]: ",twil_ev)
    file.write("Twilight evening at Stockert Observatory will be [UTC]: "+str(twil_ev)+"\n")
    T_m = int(twil_morn[-8:-6]) + int(twil_morn[-5:-3])/60 + int(twil_morn[-2:])/3600
    #print(Rise)
    T_e = int(twil_ev[-8:-6]) + int(twil_ev[-5:-3])/60 + int(twil_ev[-2:])/3600
    #print(Set)
    file.close()
    
    # set time_h_res according to sunrise and sunset --> sun above horizon
    twil_sun=np.array([])
    for i in range(0,len(time_h_res)):
        if time_h_res[i]>=T_m and time_h_res[i]<=T_e:
            twil_sun=np.append(twil_sun,time_h_res[i])
    #print(time_sun)
    
    return twil_sun, T_m, T_e