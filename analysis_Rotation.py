import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import quad
import os
import csv
# global variables

R_0 = 8.5  # Sun's radius around center of the Galaxy [kpc]
V_0 = 240  # Sun's velocity around center of the Galaxy [km/s]
G = 4.3 * 10 ** (-6)  # Newtonian Gravitational Constant [kpc*km^2/ SM *s^2]
# FUNCTIONS #


def FWHM_find_std(desired_peak_width):
    fwhm = desired_peak_width
    sigma = float(fwhm/2.355)
    return sigma  # return the sigma value


def radius_calculator(theta):
    # R_0 = 8.5  # Sun's radius around center of the Galaxy [kpc]
    lon = float((theta * np.pi) / 180)
    return abs(R_0*np.sin(lon))


def err_radius_calculator(theta):
    # R_0 = 8.5  # Sun's radius around center of the Galaxy [kpc]
    lon = float((theta * np.pi) / 180)
    d_lon = float((1*np.pi) / 180)
    dR = np.sqrt(float(((R_0*np.cos(lon)) ** 2) * d_lon ** 2))  # calculating radius error
    return dR


def velocity_calculator(v_radial, theta):
    # V_0 = 240  # Sun's velocity around center of the Galaxy [km/s]
    lon = float((theta * np.pi) / 180)
    velocity = v_radial + V_0*np.sin(lon)  # convert theta to radians
    return abs(velocity)


def err_velocity_calculator(v_radial_err, theta):
    # V_0 = 240  # Sun's velocity around center of the Galaxy [km/s]
    dv = np.sqrt(v_radial_err ** 2 + (float((1*np.pi) / 180) ** 2)*(V_0*np.cos(float((theta * np.pi) / 180))) ** 2)
    return dv


def linear_fit(x, a0, a1):
    y = (a0 * x) + a1
    return y


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# MAIN #


quadrant_1_radii = []
radii_1_error = []
quadrant_1_velocity = []
vel_1_error = []
quadrant_4_radii = []
radii_4_error = []
quadrant_4_velocity = []
vel_4_error = []

b = 0  # latitude
l = 0  # if you want to save gaussian figures later

import os

base_dir  = os.path.dirname(__file__)
directory = os.path.join(base_dir, 'data', 'Measurements')

for filename in os.listdir(directory):  # iterating through the files
    if filename.endswith(".csv"):
        with open(os.path.join(directory, filename), 'r') as file:
            temp = np.genfromtxt(file, delimiter=',')  # upload filepath from explore

        peaks_indx, properties = find_peaks(temp[:, 1], height=30, prominence=None, width=None)  # finding the peaks
        results_half = peak_widths(temp[:, 1], peaks_indx, rel_height=0.5)  # calculate peak widths
        widths = results_half[0]  # create array for peaks' widths
        v = temp[:, 0]
        T = temp[:, 1]
        T_peaks = T[peaks_indx]
        v_peaks = v[peaks_indx]

        if filename.startswith('-'):  # finding our desired peak of gaussian

            desired_peak_inx = peaks_indx[0]
            desired_peak_width = widths[0]
            v_radial = v[desired_peak_inx]

        else:
            desired_peak_inx = peaks_indx[-1]
            desired_peak_width = widths[-1]
            v_radial = v[desired_peak_inx]

        t_peak = T[desired_peak_inx]  # getting the peak's temperature
        v_radial_err = FWHM_find_std(desired_peak_width)  # calculating error for radial velocity

        # getting the longitude angle and calculating R
        if filename.endswith("0_0.csv"):
            theta = float(filename.split('_0.csv')[0])  # the 'split' function returns in this case list of ['theta','']

        else:
            theta = float(filename.strip('_0.csv'))

        # calculating radius and velocity + errors

        if theta < 0:
            R = radius_calculator(theta)
            quadrant_4_radii.append(R)
            radii_4_error.append(err_radius_calculator(theta))
            quadrant_4_velocity.append(velocity_calculator(v_radial, theta))
            vel_4_error.append(err_velocity_calculator(v_radial_err, theta))

        elif theta > 0:
            R = radius_calculator(theta)
            quadrant_1_radii.append(R)
            radii_1_error.append(err_radius_calculator(theta))
            quadrant_1_velocity.append(velocity_calculator(v_radial, theta))
            vel_1_error.append(err_velocity_calculator(v_radial_err, theta))

        # fig, ax = plt.subplots()
        # ax.plot(v, T, ".", color='green', markersize=2)
        # ax.plot(v_peaks, T_peaks, "X", color='blue')
        # plt.plot(v_radial, t_peak, "s")
        # plt.title(f'Hydrogen Cloud spectra at longitude {theta}, latitude {b}')
        # plt.show()
        # plt.close(fig)
        # fig.savefig(f"D:\MICHELLE D BACKUP\michelle\documents\מיכל לימודים\אוניברסיטה\שנה ב\סמסטר א\מעבדה ב1\חומר אפל\gaussian figures\_gig{l}") #insert path to location
        # l += 1

# PLOTTING THE INITIAL ROTATION CURVES 1ST & 4TH QUADRANTS #
plt.style.use('seaborn-v0_8-whitegrid')

# plt.errorbar(quadrant_1_radii, quadrant_1_velocity, xerr=radii_1_error, yerr=vel_1_error, ls="None", marker='o', ms=3, mfc='blue', mec='k', label='Quadrant_1 error')
# plt.errorbar(quadrant_4_radii, quadrant_4_velocity, xerr=radii_4_error, yerr=vel_4_error, ls="None", marker='o', ms=3, mfc='orange', mec='k', label='Quadrant_4 error')
plt.plot(quadrant_1_radii, quadrant_1_velocity, ls="None", marker='o', ms=3, label='Quadrant_1')
plt.plot(quadrant_4_radii, quadrant_4_velocity, ls="None", marker='o', ms=3, label='Quadrant_4')
plt.title("Rotation Curve V[km/s] .vs. R[kpc]")
plt.xlabel("R[kpc]")
plt.ylabel("V[km/s]")
plt.legend()
plt.show()

avg_vel = []  # creating array for  weighted average velocities
avg_radii = []  # creating array for radii
err_vel_avg = []  # creating error array for radii
err_rad_avg = []  # creating error array for weighted averaged velocities
for r1, r4, v1, v4, dr1, dr4, dv1, dv4 in zip(quadrant_1_radii, quadrant_4_radii, quadrant_1_velocity, quadrant_4_velocity, radii_1_error, radii_4_error, vel_1_error, vel_4_error):
    r_avg = (r1+r4)/2
    avg_radii.append(r_avg)
    dr_avg = float(1/(np.sqrt((1/dr1) ** 2 + (1/dr4) ** 2)))  # calculate the new radii error
    err_rad_avg.append(dr_avg)
    v_avg = ((v1/dv1 ** 2) + (v4/dv4 ** 2))/((1/dv1 ** 2) + (1/dv4 ** 2))
    avg_vel.append(v_avg)
    dv_avg = float(1/(np.sqrt((1/dv1) ** 2 + (1/dv4) ** 2)))
    err_vel_avg.append(dv_avg)

# PLOTTING ANALYZED ROTATION CURVE #
plt.figure()
model_order = 3  # set model order
p = np.polyfit(avg_radii, avg_vel, model_order)  # creating polynomial fir for the data
x_line = np.linspace(min(avg_radii), max(avg_radii))
y_line = np.polyval(p, x_line)

inbulge = find_nearest(avg_radii, 3.8)
idx_inbulge = avg_radii.index(inbulge)

x_inside = np.array(avg_radii[0:(idx_inbulge + 1)]).reshape((-1, 1))
yin = np.array(avg_vel[0:(idx_inbulge + 1)])
model_in = LinearRegression().fit(x_inside, yin)  # creating linear regression
r_sq = model_in.score(x_inside, yin)
print('coefficient of determination:', r_sq)
print('intercept:', model_in.intercept_)
print('slope:', model_in.coef_)
y_pred_in = model_in.predict(x_inside)

obulge = find_nearest(avg_radii, 2)
idx_obulge = avg_radii.index(obulge)

x_outside = np.array(avg_radii[(idx_obulge+1):-1])  # .reshape((-1, 1))
yout = np.array(avg_vel[(idx_obulge+1):-1])
p_outside = np.polyfit(x_outside, yout, 0)  # outside the bulge
y_outside = np.polyval(p_outside, x_outside)

plt.plot(avg_radii, avg_vel, ls="None", marker='o', ms=3, mfc='k', mec='r')
plt.plot(x_line, y_line, label='polynomial fit', color='black', lw=0.9)  # plotting the polynomial fit
plt.plot(x_inside, y_pred_in, ls='dashdot', label='y = 'r'$a \cdot x + b$', color='grey', lw=0.5)
plt.plot(x_outside, y_outside, ls='dotted', label='y = const', color='grey', lw=0.5)
# plt.errorbar(avg_radii, avg_vel, xerr=err_rad_avg, yerr=err_vel_avg, ls="None", marker='.', ms=3, color='black')
plt.title("Rotation Curve " r'$\langle V \rangle[km/s]$ '".vs. R[kpc]")
plt.xlabel("R[kpc]")
plt.ylabel(r'$\langle V \rangle[km/s]$')
plt.legend()
plt.show()

# PLOTTING EXPECTED MODEL VS OBSERVES DATA #
plt.figure()
v_kepler = []
r_bulge = find_nearest(avg_radii, 3)  # kpc
theo_bar_mass = 5*10**11
for kep in avg_radii:
    if kep <= r_bulge:
        v_kepler.append((np.sqrt((6.7*10**-8)*theo_bar_mass/(r_bulge**3)))*kep)  # G is now in [erg cm g^-2]
    elif kep > r_bulge:
        v_kepler.append(np.sqrt((6.6*10**-8)*theo_bar_mass/kep))

plt.plot(avg_radii, v_kepler, ls='None', marker='.', label='Keplerian Model')  # plot keplerian model
plt.plot(avg_radii, avg_vel, ls="None", marker='o', ms=3, mfc='blue', mec='r', label='Analyzed Data')  # plot the data
plt.title("Rotation Curve " r"$V[km/s]$" ".vs. R[kpc]")
plt.xlabel("R[kpc]")
plt.ylabel(r"$V[km/s]$")
plt.yscale('log')
plt.legend()
plt.show()

# MASS DISTRIBUTION
M = []
dM = []
for r_i, dr_i, v_i, dv_i in zip(avg_radii, err_rad_avg, avg_vel, err_vel_avg):
    m_i = ((v_i ** 2)*r_i)/G
    M.append(m_i)
    dm_i = (1/G) * (np.sqrt((2 * v_i * r_i * dv_i) ** 2 + (dr_i * v_i ** 2) ** 2))
    dM.append(dm_i)

# PLOTTING MASS VS R #
plt.figure()
plt.plot(avg_radii, M, ls="None", label='Data', marker='o', ms=3, color='r')  # plotting the data
plt.errorbar(avg_radii, M, xerr=err_rad_avg, yerr=dM, ls='None', marker='.', ms=3, color='k', label='Data error bar')
plt.title("M [SM]" ".vs. R[kpc]")
plt.xlabel("R[kpc]")
plt.ylabel(r"$M[SM]$")
plt.yscale('log')
plt.legend()
plt.show()

# PLOTTING DIFFERENT DENSITY MODELS FOR BM & DM

# 1: Hernquist profile
a = 3
Hernquist = []
for k, value in enumerate(avg_radii):
    HNQ = a**4/(avg_radii[k]*((avg_radii[k]+a)**3))
    Hernquist.append(HNQ)

plt.plot(avg_radii, Hernquist, ls="None", marker='o', ms=3, mfc='orchid', mec='orchid')
plt.title(" Hernquist profile: " r"$\rho [SM/kpc^3] .vs. R[kpc]$")
plt.xlabel("R[kpc]")
plt.ylabel(r"$\rho [SM/kpc^3]$")
plt.yscale('log')
plt.show()


# 2: Sersic   profile
n = 1
r_e_Sersic = 4
rho_e = 4
Sersic = []
b_n = 2*n - 1/3
for k, value in enumerate(avg_radii):
    SRS = rho_e*np.exp(-b_n*((avg_radii[k]/r_e_Sersic)**1 - 1))
    Sersic.append(SRS)

plt.plot(avg_radii, Sersic, ls="None", marker='o', ms=3, mfc='r', mec='r')
plt.title("Sersic profile: " r"$\rho [SM/kpc^3] .vs. R[kpc]$")
plt.xlabel("R [kpc]")
plt.ylabel(r"$\rho [SM/kpc^3]$")
plt.yscale('log')
plt.show()

# Burkert
rho_Bu = 40000000
r_sBu = 8
Burkert = []

for k, value in enumerate(avg_radii):
    BP = rho_Bu*((1 + avg_radii[k] / r_sBu)*(1 + (avg_radii[k] / r_sBu)**2))**-1
    Burkert.append(BP)

plt.plot(avg_radii, Burkert, ls="None", marker='o', ms=3, mfc='k', mec='k')
plt.title("Burkert: " r"$\rho [SM/kpc^3] .vs. R[kpc]$")
plt.xlabel("R [kpc]")
plt.ylabel(r"$\rho [SM/kpc^3]$")
plt.yscale('log')
# plt.xscale('log')
plt.show()

# NFW
rho_NW = 14000000
r_sNw = 100
NFW = []
for k, value in enumerate(avg_radii):
    NFW_Profile = rho_NW*(avg_radii[k]/r_sNw*(1+(avg_radii[k]/r_sNw))**2)**-1
    NFW.append(NFW_Profile)

plt.plot(avg_radii, NFW, ls="None", marker='o', ms=3, mfc='midnightblue', mec='midnightblue')
plt.title("NFW_Profile: " r"$\rho [SM/kpc^3] .vs. R[kpc]$")
plt.xlabel("R [kpc]")
plt.ylabel(r"$\rho [SM/kpc^3]$")
plt.yscale('log')
plt.show()

# TRYING DIFFERENT COMBINATIONS #
# 1 first Fit : NFW & Sersic
plt.subplots()
NFW_Sersic = np.add(NFW, Sersic)
srs, = plt.plot(avg_radii, Sersic, ls="None", label='Sersic (BM)', marker='o', ms=3, mfc='r', mec='r')
nfw, = plt.plot(avg_radii, NFW, label='NFW (DM)', ls="None", marker='o', ms=3, mfc='midnightblue', mec='midnightblue')
srsnfw, = plt.plot(avg_radii, NFW_Sersic, label='NFW + Sersic', ls="None", marker='o', ms=3, mfc='g', mec='c')
plt.title("NFW & Sersic Profile")
plt.xlabel("R[kpc]")
plt.ylabel(r"$\rho [SM/kpc^3]$")
plt.yscale('log')
plt.legend()
plt.show()

# 2 Second Fit: Burkert & Sersic
plt.subplots()
Burkert_Sersic = np.add(Burkert, Sersic)
plt.plot(avg_radii, Sersic, ls="None", label='Sersic (BM)', marker='o', ms=3, mfc='r', mec='r')
plt.plot(avg_radii, Burkert, ls="None", label='Burkert (DM)', marker='o', ms=3, mfc='k', mec='k')
plt.plot(avg_radii, Burkert_Sersic, ls="None", label='Burkert + Sersic', marker='o', ms=3, mfc='g', mec='c')
plt.title("Burkert & Sersic")
plt.xlabel("R [kpc]")
plt.ylabel(r"$\rho [SM/kpc^3]$")
plt.yscale('log')
plt.legend()
plt.show()

# 3 Third Fit: NFW & HERNQUIST
plt.subplots()
NFW_Hernquist = np.add(NFW, Hernquist)
plt.plot(avg_radii, Hernquist, ls="None", label='Hernquist (BM)', marker='o', ms=3, mfc='orchid', mec='orchid')
plt.plot(avg_radii, NFW, label='NFW (DM)', ls="None", marker='o', ms=3, mfc='midnightblue', mec='midnightblue')
plt.plot(avg_radii, NFW_Hernquist, ls="None", label='NFW + Hernquist', marker='o', ms=3, mfc='g', mec='c')
plt.title("NFW & Hernquist")
plt.xlabel("R [kpc]")
plt.ylabel(r"$\rho [SM/kpc^3]$")
plt.yscale('log')
plt.legend()
plt.show()

# 4 Fourth Fit : Burkert & HERNQUIST
plt.subplots()
Burkert_Hernquist = np.add(Burkert, Hernquist)
plt.plot(avg_radii, Hernquist, ls="None", label='Hernquist (BM)', marker='o', ms=3, mfc='orchid', mec='orchid')
plt.plot(avg_radii, Burkert, ls="None", label='Burkert (DM)', marker='o', ms=3, mfc='k', mec='k')
plt.plot(avg_radii, Burkert_Hernquist, ls="None",label='Burkert + Hernquist', marker='o', ms=3, mfc='g', mec='c')
plt.title("Burkert & Hernquist")
plt.xlabel("R [kpc]")
plt.ylabel(r"$\rho [SM/kpc^3]$")
plt.yscale('log')
plt.legend()
plt.show()

# Baryonic Matter profile(Rho)

# 1: Henquist profile : rho_1(r) = a**4/(r*(r+a)**3)
# M = total mass , a = scale of legnth (for our lab a = 8.5KPC)
# 2: Sersic_profile : rho_2(r) =  rho_e*exp(-b_n*((r/r_e)**1/n  -1))
# rho_e = rho(r=r_effective=r_e)

# Dark Matter Profile(Rho)

# 3: Burkert Profile = BP : rho_3(r) = rho_Bu*( ( 1+r / r_sBu) * (1+ ( r / r_sBu) **2)) **-1
# r_sBu = 1 , rho0 = rho_Bu =rho_3(r=0)

# 4: NFW     Profile:  rho_4(r) = rho_NW*(r / r_sNw * (1+ (r / r_sNw) ) **2) **-1
# rho_3 = rho_Bu * ((1 + r / r_sBu) * (1 + (r / r_sBu) ** 2)) ** -1

# rho_total
rho_total_error = []
rho_total = []
for k, value in enumerate(avg_radii):
    s = 1/(4*np.pi*(avg_radii[k]**2))
    d = 2*avg_radii[k]*avg_vel[k]/G
    dv_dr = 56.93
    t = avg_vel[k]**2/G
    rho_Error_radd_part1 = dv_dr*(-avg_vel[k]/(2*(G*(avg_radii[k]**2)*np.pi)))*err_rad_avg[k]
    rho_Error_radd_part2 = (-avg_vel[k]**2/(2*(np.pi*G*(avg_radii[k]**3)))*err_rad_avg[k])
    # print('rho_Error_radd_part1', rho_Error_radd_part1)
    # print('rho_Error_radd_part2', rho_Error_radd_part2)
    rho_Error_vel_part1 = dv_dr*(1/(2*np.pi*G*avg_radii[k]))*err_vel_avg[k]
    rho_Error_vel_part2 = (avg_vel[k]/(2*np.pi*G*(avg_radii[k]**2))*err_vel_avg[k])
    # print('rho_Error_vel_part1', rho_Error_vel_part1)
    # print('rho_Error_vel_part2', rho_Error_vel_part2)
    rho_error_rad_total = rho_Error_radd_part1 + rho_Error_radd_part2
    rho_error_vel_total = rho_Error_vel_part1 + rho_Error_vel_part2
    rho_total_k_error = ((rho_error_rad_total**2) + (rho_error_vel_total**2))**(1/2)
    # print('rho_total_k_error', rho_total_k_error)
    rho_total_error.append(rho_total_k_error)
    rho_total_k = s*(d*dv_dr+t)
    rho_total.append(rho_total_k)
print('rho_total =', rho_total)
print('rho_total_error =', rho_total_error)

Details = ['avg_radii', 'err_rad_avg', 'rho_total', 'rho_total_error']
rows = [avg_radii, err_rad_avg, rho_total, rho_total_error]
with open('student1.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(Details)
    write.writerows(rows)

# now we will get a Quantitative assessment to the mass of the milky way galaxy


def integrandSersic_Brukert(r):
    return (r**2)*(11100000*np.exp(1.218*((r/6.396)**(1/0.7979) - 1)) + 354000000*((1 + r / 6.563)*(1 + (r / 6.563)**2))**-1)


I = quad(integrandSersic_Brukert, 0, 15)
I_mass = 4*np.pi*I[0]
print('Sersic_Brukert(total mass) = {:.2e}'.format(I_mass), 'SM')

# a[0] = 1.110e+7 ± 5.4793654e+11 (4.936e+6% error)
# a[1] = 1.218 ± 4.9366635e+4 (4.055e+6% error)
# a[2] = 6.396 ± 2.06882598e+5 (3.235e+6% error)
# a[3] = 0.7979 ± 21.8120 (2.734e+3% error)
# a[4] = 3.540e+8 ± 1.662e+8 (46.95% error)
# a[5] = 6.563 ± 3.537 (53.89% error)


def integrandSersic_NFW(r):
    return r**2 * (26760000 * np.exp(11.27 * ((r/11.69)**(-0.05756/1) - 1)) + 8370000 * (r/57.85*(1+(r/57.85))**2)**-1)


I_1 = quad(integrandSersic_NFW, 0, 15)

I_mass_1 = 4*np.pi*I_1[0]
print('Sersic_NFW(total mass) = {:.2e}'.format(I_mass_1), 'SM')

# Initial parameters' values:
# 6000000.0 1.6667 4.0 1.0 14000000.0 8.0
# Fitted parameters' values:
# a[0] = 2.676e+7 ± 3.7978380e+11 (1.419e+6% error)
# a[1] = -11.27 ± 2.151668e+4 (1.908e+5% error)
# a[2] = 11.69 ± 2.5053371e+5 (2.144e+6% error)
# a[3] = -0.05756 ± 7.581495e+1 (1.317e+5% error)
# a[4] = 8.370e+6 ± 8.926635e+9 (1.066e+5% error)
# a[5] = 57.85 ± 8.743974e+4 (1.512e+5% error)


def integrandHernquist_NFW(r):
    return r**2*(3**4/(r*((r+3)**3)) + 4068000*(r/174.54*(1+(r/174.54))**2)**-1)


I_2 = quad(integrandHernquist_NFW, 0, 15)

I_mass_2 = 4*np.pi*I_2[0]
print('Hernquist_NFW(total mass) = {:.2e}'.format(I_mass_2), 'SM')

# Initial parameters' values:
# 3.0 14000000.0 8.0
# Fitted parameters' values:
# a[0] = 3.000 ± 0.000 (0.000% error)
# a[1] = 4.068e+6 ± 1.710e+6 (42.05% error)
# a[2] = 174.54 ± 67.32 (38.57% error)

def integrandHernquist_Burkert(r):
    return r**2*(-0.6921**4/(r*((r-0.6921)**3)) + 383530000*((1 + r / 11.8721)*(1 + r / 11.8721)**2)**-1)


I_3 = quad(integrandHernquist_Burkert, 0, 15)

I_mass_3 = 4*np.pi*I_3[0]
print('Hernquist_Burkert(total mass) = {:.2e}'.format(I_mass_3), 'SM')

# Initial parameters' values:
# 3.0 40000000.0 8.0
# Fitted parameters' values:
# a[0] = -0.6921 ± 2.0676214817e+6 (2.987e+8% error)
# a[1] = 3.8353e+8 ± 1.628e+7 (4.244% error)
# a[2] = 11.8721 ± 0.4100 (3.453% error)


#Ratio Mass between the Dark matter and Baryonic matter
def integrandSersic(r):
    return r**2 * (26760000 * np.exp(11.27 * ((r/11.69)**(-0.05756/1) - 1)))


I_sersic = quad(integrandSersic, 0, 8)
I_mass_sersic = 4*np.pi*I_sersic[0]





#calculate mass ratio
#I_mass_1 = total mass of Sersic and NFW
ratio_mass_Sersic_NFW = I_mass_sersic/(I_mass_1-I_mass_sersic)
print('ratio mass between Sersic and NFW  is =',ratio_mass_Sersic_NFW)




#in this part we will build a graph that show the mass ratio(sersic divided by NFW per point) vs radius till the sun radius


def integrandSersic(r):
    return r**2 * (26760000 * np.exp(11.27 * ((r/11.69)**(-0.05756/1) - 1)))
I_mass_sersic_list = []
for r in range(1,9):
 I_sersic = quad(integrandSersic, 0, r)

 I_mass_sersic = 4*np.pi*I_sersic[0]
 I_mass_sersic_list.append(I_mass_sersic)
print('Sersic(total mass) = {:.2e}'.format(I_mass_sersic), 'SM')
print(I_mass_sersic_list)



def integrandSersic_NFW(r):
    return r**2 * (26760000 * np.exp(11.27 * ((r/11.69)**(-0.05756/1) - 1)) + 8370000 * (r/57.85*(1+(r/57.85))**2)**-1)
I_mass_total_Sersic_NFW_list = []
for r in range(1,9):
 I_mass_total_sersic_NFW = quad(integrandSersic_NFW, 0, r)

 mass_total_sersic_NFW = 4*np.pi*I_mass_total_sersic_NFW[0]
 I_mass_total_Sersic_NFW_list.append(mass_total_sersic_NFW)
print(I_mass_total_Sersic_NFW_list )


ratio_mass_Sersic_NFW_list = []
raddi_list = [1,2,3,4,5,6,7,8]
for k,value in enumerate(I_mass_total_Sersic_NFW_list):
    ratio_mass_per_radius = I_mass_sersic_list[k]/(I_mass_total_Sersic_NFW_list[k] -I_mass_sersic_list[k])
    ratio_mass_Sersic_NFW_list.append(ratio_mass_per_radius)

print(ratio_mass_Sersic_NFW_list)

plt.plot(raddi_list, ratio_mass_Sersic_NFW_list, ls="None", marker='o', ms=3, mfc='midnightblue', mec='midnightblue')
plt.title("ratio_mass_Sersic_NFW")
plt.xlabel("R [kpc]")
plt.ylabel("ratio_mass_Sersic_NFW")
plt.show()
