import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root

#units
OMEGA = 1.0
HGAS  = 1.0
CS    = 1.0
RHOG  = 1.0
R_MU  = 1.0

'''
plotting parameters
'''
fontsize = 24
nlev = 128
nclev = 6
cmap = plt.cm.inferno

# Function to parse the non-standard ini file
def parse_nonstandard_ini(lines):
    parsed_data = {}
    current_section = None

    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue

        # Check for section headers
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1].strip()
            parsed_data[current_section] = {}
            continue

        # Parse key-value pairs, ignoring extra columns
        if current_section:
            parts = line.split()
            key = parts[0]
            value = parts[1:]  # Store remaining columns as list
            if value:
                parsed_data[current_section][key] = value

    return parsed_data


# Read the ini file
with open('idefix.ini', 'r') as file:
    ini_contents = file.readlines()

# Parse the file
parsed_ini = parse_nonstandard_ini(ini_contents)

#gas disk parameters
gamma   = np.float64(parsed_ini['Hydro']['gamma'][0])
Pi      = np.float64(parsed_ini['Setup']['etahat'][0])
Pe      = np.float64(parsed_ini['Setup']['Pe'][0])
kappaT  = R_MU*RHOG*HGAS*HGAS*OMEGA/((gamma-1.0)*Pe)

#entropy profile parameters
NsqInfty = np.float64(parsed_ini['Setup']['NsqInf'][0])
Nsq0     = np.float64(parsed_ini['Setup']['Nsq0'][0])
D        = np.float64(parsed_ini['Setup']['NsqWidth'][0])

#domain size
x_min = np.float64(parsed_ini['Grid']['X1-grid'][1])
x_max = np.float64(parsed_ini['Grid']['X1-grid'][4])

#perturbation parameters
pert_amp = np.float64(parsed_ini['Setup']['pertamp'][0]) # W = P'/rho at inner boundary (normalized by CS^2)
kz       = np.float64(parsed_ini['Setup']['eigen_kz'][0])/HGAS    # vertical wavenumber

def N2(x): #entropy profile
    s = x/HGAS
    return NsqInfty - (NsqInfty - Nsq0)*np.exp(-0.5*s*s/D/D)

def dN2_dx(x):
    s = x/HGAS
    dN2_ds = (s/D/D)*(NsqInfty-Nsq0)*np.exp(-0.5*s*s/D/D)
    return dN2_ds/HGAS

def P(x): #pressure profile
    s = x/HGAS
    P0= CS*CS*RHOG
    return P0*np.exp(-2.0*Pi*s)

def rho(x): #density profile
    s = x/HGAS
    b = 2.0*Pi*(1.0-1.0/gamma)

    fs1 = NsqInfty/b*(np.exp(b*s)-1.0)
    fs2 = np.sqrt(np.pi/2)*(NsqInfty-Nsq0)*D*np.exp(0.5*b*b*D*D)
    fs3 = erf(b*D/np.sqrt(2.0)) - erf((b*D*D-s)/(np.sqrt(2.0)*D))
    fs  = fs1 - fs2*fs3

    result = 2.0*Pi*RHOG*np.exp(-2.0*Pi*s/gamma)
    result/= 2.0*Pi + fs
    return result

def cs2(x):
    return P(x) / rho(x)

def dlnP_dx(x):
    return -2.0*Pi/HGAS

def dP_dx(x):
    return P(x)*dlnP_dx(x)

def gr(x): #dP/dx/rho
    return dP_dx(x)/rho(x)

def dlnrho_dx(x):
    s         = x/HGAS
    dlnrho_ds = -2.0*Pi/gamma - N2(x)*np.exp(2.0*Pi*s)*(rho(x)/RHOG)/(2.0*Pi)
    return dlnrho_ds/HGAS

def d2lnrho_dx2(x):
    s = x/HGAS
    d2lnrho_ds2 = HGAS*dN2_dx(x)*np.exp(2.0*Pi*s)*rho(x)/RHOG
    d2lnrho_ds2+= 2.0*Pi*np.exp(2.0*Pi*s)*N2(x)*rho(x)/RHOG
    d2lnrho_ds2+= np.exp(2.0*Pi*s)*N2(x)*rho(x)/RHOG*HGAS*dlnrho_dx(x)
    d2lnrho_ds2*=-1.0/(2.0*Pi)
    return d2lnrho_ds2/HGAS/HGAS

def dlnT_dx(x):
    return dlnP_dx(x)-dlnrho_dx(x)

def d2lnT_dx2(x):
    return -d2lnrho_dx2(x) #since d2lnP_dx2 = 0

def LapT_T(x):
    return dlnT_dx(x)*dlnT_dx(x) + d2lnT_dx2(x)

def chi(x):
    result = kappaT*(gamma-1.0)/(R_MU*rho(x))
    return result

def duy_dx(x):
    result = (P(x) / rho(x)) / (2.0 * OMEGA) * dlnT_dx(x) * dlnP_dx(x)
    return result

# Update your ODE system to reflect the changes:
def odes(x, y, s):
    # Unpack variables
    W, theta, theta_x, vx = y

    # Define the ODEs
    vy = -(1.0/s) * (OMEGA / 2.0 + duy_dx(x)) * vx
    Q = W / cs2(x) - theta
    dW_dx = 2.0 * OMEGA * vy + Q * gr(x) - W * dlnrho_dx(x) - s * vx

    dtheta_dx = theta_x

    chix = chi(x)
    divv = -s * Q - vx * dlnrho_dx(x)
    dtheta_x_dx = (s / chix - LapT_T(x) + kz**2) * theta + (vx / chix - 2.0 * theta_x) * dlnT_dx(x) + (gamma - 1.0) / chix * divv

    vz = -1j * kz * W / s
    dvx_dx = -(s * Q + vx * dlnrho_dx(x) + 1j * kz * vz)

    return [dW_dx, dtheta_dx, dtheta_x_dx, dvx_dx]

# Residual function for root finding
def residuals(params, x_vals):
    theta0_real, theta0_imag, s_real, s_imag = params
    theta0 = theta0_real + 1j * theta0_imag
    s = s_real + 1j * s_imag

    y_initial_guess = [pert_amp*CS*CS, theta0, 0.0, 0.0] #variables are W, theta, theta_x, vx
    sol = solve_ivp(odes, [x_vals[0], x_vals[-1]], y_initial_guess, t_eval=x_vals, args=(s,), dense_output=True)

    solution = sol.sol(x_vals)
    thetax   = solution[2]
    vx       = solution[3]  # Extract and take the real part of v_

    vx_L, thetax_L = vx[-1], thetax[-1]
    return [vx_L.real, vx_L.imag, thetax_L.real, thetax_L.imag]

# Root finding for complex theta_0 and s
def find_theta0_and_s(x_vals):
    initial_guess = [0.0, 0.0, 0.0, OMEGA]  # Initial guesses for theta0 (real, imag) and s (real, imag)
    solution = root(residuals, initial_guess, args=(x_vals,), method='hybr')
    theta0 = solution.x[0] + 1j * solution.x[1]
    s = solution.x[2] + 1j * solution.x[3]
    return theta0, s

# Function to compute and save the plots of W(x), v_x(x), theta(x), vy(x), Q(x), and vz(x)
def plot_eigenfunctions(x_vals, theta0, s):
    # Initial conditions
    y_initial_guess = [pert_amp*CS*CS, theta0, 0.0, 0.0]  # W(0)=pert_amp*CS*CS, theta(0)=theta0, theta_x(0)=0, vx(0)=0

    # Solve the ODE system with dense output enabled
    sol = solve_ivp(odes, [x_vals[0], x_vals[-1]], y_initial_guess, args=(s,), dense_output=True)

    # Evaluate the solution at all specified x values
    solution = sol.sol(x_vals)
    W = solution[0]    # Extract W(x)
    theta = solution[1]  # Extract theta(x)
    vx = solution[3]  # Extract v_x(x)

    # Compute vy(x), Q(x), and vz(x)
    vy = [-(1.0 / s) * (OMEGA / 2.0 + duy_dx(x)) * vx_val for x, vx_val in zip(x_vals, vx)]
    Q = [W_val / cs2(x) - theta_val for x, W_val, theta_val in zip(x_vals, W, theta)]
    vz = [-1j * kz * W_val / s for x, W_val in zip(x_vals, W)]

    W_real     = W.real
    theta_real = theta.real
    vx_real    = vx.real

    vy_real = [v.real for v in vy]
    Q_real  = [q.real for q in Q]
    vz_real = [v.real for v in vz]

    fig = plt.figure(figsize=(8, 4.5), constrained_layout=True)
    ax = fig.add_subplot()

    plt.plot(x_vals, W_real, linewidth=2, label=r'Re($W$)')
    plt.plot(x_vals, Q_real, linewidth=2, label=r'Re($Q$)')
    plt.plot(x_vals, vx_real, linewidth=2, label=r'Re($v_x$)')
    plt.plot(x_vals, vy_real, linewidth=2, label=r'Re($v_y$)')
    plt.plot(x_vals, vz_real, linewidth=2, label=r'Re($v_z$)')

    plt.rc('font', size=fontsize, weight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend = ax.legend(lines1, labels1, loc='lower right', frameon=False,
                       ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)

    plt.xlim(np.amin(x_vals),np.amax(x_vals))
    plt.xticks(fontsize=fontsize, weight='bold')
    plt.xlabel(r'$x/H_g$', fontsize=fontsize)

    plt.yticks(fontsize=fontsize, weight='bold')
    # plt.ylabel(r'$Eigenfunctions$', fontsize=fontsize)

    plt.savefig('eigenfunctions', dpi=300)

    # Plot all variables
    # plt.figure(figsize=(12, 8))
    # plt.plot(x_vals, W_real, label=r'$W(x)$',linewidth=2)
    # plt.plot(x_vals, Q_real, label=r'$Q(x)$',linewidth=2)
    # # plt.plot(x_vals, theta_real, label=r'$\theta(x)$', color='g', linestyle='-.')
    # plt.plot(x_vals, vx_real, label=r'$v_x(x)$',linewidth=2)
    # plt.plot(x_vals, vy_real, label=r'$v_y(x)$',linewidth=2)
    # plt.plot(x_vals, vz_real, label=r'$v_z(x)$',linewidth=2)

    # data_cpp = np.loadtxt('output.txt')
    # x_vals_cpp = data_cpp[:, 0]
    # W_cpp      = data_cpp[:, 1]
    # Q_cpp      = data_cpp[:, 3]
    # vx_cpp     = data_cpp[:, 5]
    # vy_cpp     = data_cpp[:, 7]
    # vz_cpp     = data_cpp[:, 9]
    #
    # plt.plot(x_vals_cpp, W_cpp, label=r'$W(x)$, C++',marker='*', markersize=16,linestyle='')
    # plt.plot(x_vals_cpp, Q_cpp, label=r'$Q(x)$, C++',marker='*', markersize=16,linestyle='')
    # plt.plot(x_vals_cpp, vx_cpp, label=r'$v_x(x)$, C++',marker='*', markersize=16,linestyle='')
    # plt.plot(x_vals_cpp, vy_cpp, label=r'$v_y(x)$, C++',marker='*', markersize=16,linestyle='')
    # plt.plot(x_vals_cpp, vz_cpp, label=r'$v_z(x)$, C++',marker='*', markersize=16,linestyle='')

    # Customize the plot
    # plt.xlabel(r'$x$', fontsize=14)
    # plt.ylabel('Values', fontsize=14)
    # plt.title('Comparison of $W(x)$, $Q(x)$, $v_x(x)$ ,$v_y(x)$, and $v_z(x)$', fontsize=16)
    # plt.legend(fontsize=12)
    # plt.grid()
    # plt.tight_layout()

    # Save the plot to a file
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    # print(f"Plot saved to {filename}")
    plt.close()


# Function to save the plot of normalized rho(x), P(x), and T(x) to a PNG file
def plot_equilibrium(x_vals):
    # Compute values at x = 0 for normalization
    rho_0 = rho(0)
    P_0 = P(0)
    T_0 = P_0/rho_0

    # Compute normalized values
    rho_normalized = [rho(x) / rho_0 for x in x_vals]
    P_normalized = [P(x) / P_0 for x in x_vals]
    T_normalized = [(P(x)/rho(x)) / T_0 for x in x_vals]

    # Plot the normalized quantities

    fig = plt.figure(figsize=(8, 4.5), constrained_layout=True)
    ax = fig.add_subplot()

    plt.plot(x_vals, P_normalized, linewidth=2, label=r'$P/P_0$')
    plt.plot(x_vals, rho_normalized, linewidth=2, label=r'$\rho/\rho_0$')
    plt.plot(x_vals, T_normalized, linewidth=2, label=r'$T/T_0$')

    plt.rc('font', size=fontsize, weight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend = ax.legend(lines1, labels1, loc='lower left', frameon=False,
                       ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)

    plt.xlim(np.amin(x_vals),np.amax(x_vals))

    plt.xticks(fontsize=fontsize, weight='bold')
    plt.xlabel(r'$x/H_g$', fontsize=fontsize)

    plt.yticks(fontsize=fontsize, weight='bold')
    # plt.ylabel(r'$Equilibrium profiles$', fontsize=fontsize)

    plt.savefig('equilibrium', dpi=300)

    # plt.figure(figsize=(8, 6))
    # plt.plot(x_vals, rho_normalized, label=r'$\rho(x) / \rho(0)$', color='b')
    # plt.plot(x_vals, P_normalized, label=r'$P(x) / P(0)$', color='r')
    # # plt.plot(x_vals, T_normalized, label=r'$T(x) / T(0)$', color='g')
    # plt.axhline(1.0, color='k', linestyle='--', linewidth=0.8, label=r'Normalized Value = 1')
    # plt.xlabel(r'$x$', fontsize=14)
    # plt.ylabel(r'Normalized Values', fontsize=14)
    # plt.title(r'Normalized Quantities: $\rho(x)$, $P(x)$, $T(x)$', fontsize=16)
    # plt.legend(fontsize=12)
    # plt.grid()
    # plt.tight_layout()
    #
    # # Save the plot to a file
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    # print(f"Plot saved to {filename}")
    plt.close()

# Function to compute and save the comparison plot
def plot_N2(x_vals):
    # Compute the derived term
    N2_def = [
        -gr(x) * (dlnP_dx(x) / gamma - dlnrho_dx(x)) for x in x_vals
    ]

    # Compute N2(x)
    N2_prescribed = [N2(x) for x in x_vals]

    fig = plt.figure(figsize=(8, 4.5), constrained_layout=True)
    ax = fig.add_subplot()

    plt.plot(x_vals, N2_prescribed, linewidth=2, label=r'$N^2(x)$')
    plt.plot(x_vals, N2_def, linewidth=2, linestyle='--', label=r'$-\frac{1}{\rho}\frac{dP}{dx} \left(\frac{1}{\gamma}\frac{d\ln P}{dx} - \frac{d\ln \rho}{dx}\right)$')

    plt.rc('font', size=fontsize, weight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend = ax.legend(lines1, labels1, loc='lower right', frameon=False,
                       ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)

    plt.xlim(np.amin(x_vals),np.amax(x_vals))
    plt.xticks(fontsize=fontsize, weight='bold')
    plt.xlabel(r'$x/H_g$', fontsize=fontsize)

    plt.yticks(fontsize=fontsize, weight='bold')
    # plt.ylabel(r'$Buoyancy profile$', fontsize=fontsize)

    plt.savefig('N2', dpi=300)
    plt.close()


    # Plot the comparison
    # plt.figure(figsize=(8, 6))
    # plt.plot(x_vals, N2_values, label=r'$N^2(x)$', color='b')
    # plt.plot(x_vals, comparison_term, label=r'$-g_r(x) \left(\frac{d\ln P}{dx} / \gamma - \frac{d\ln \rho}{dx}\right)$', color='r', linestyle='--')
    # plt.axhline(0, color='k', linestyle='--', linewidth=0.8, label=r'Zero Line')
    # plt.xlabel(r'$x$', fontsize=14)
    # plt.ylabel('Values', fontsize=14)
    # plt.title('Comparison: $N^2(x)$ and the Derived Term', fontsize=16)
    # plt.legend(fontsize=12)
    # plt.grid()

if __name__ == "__main__":
    # Example usage: Define x domain and initial guesses
    Nx = 128
    x_vals = np.linspace(x_min, x_max, Nx)  # Example x range from 0 to Lx

    # Find theta_0 and s
    theta0, s_solution = find_theta0_and_s(x_vals)

    print("Found initial guess for theta0:", theta0)
    print("Found eigenvalue s:", s_solution)

    # Example usage: Save the plot of W(x), v_x(x), theta(x), v_y(x), Q(x), and v_z(x)
    plot_eigenfunctions(x_vals, theta0, s_solution)

    # Example usage: Save the plot of rho(x), P(x), and T(x) normalized
    plot_equilibrium(x_vals)

    # Example usage: Save the plot for N2(x) and the comparison term
    plot_N2(x_vals)
