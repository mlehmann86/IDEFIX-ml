#include <iostream>
#include <vector>
#include <complex>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_spline.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <map>

using namespace std;

// Constants
const double OMEGA  = 1.0;
const double HGAS   = 1.0;
const double CS     = 1.0;
const double RHOG   = 1.0;
const double R_MU   = 1.0;


//Gas parameters
double gamma;
double Pi;
double Pe;
double kappaT; //= R_MU * RHOG * HGAS * HGAS * OMEGA / ((gamma - 1.0) * Pe);

//Buoyancy profile
double NsqInfty;
double Nsq0;
double D;

//Domain 
double x_min, x_max;

// Perturbation Parameters
double pert_amp;     // W = P'/rho at inner boundary (normalized by CS^2)
double kz;           // Vertical wavenumber
double ky;           // Azimuthal wavenumber
double sguess;       // initial guess for the growth rate
double fguess;       // initial guess for the frequency

using IniData = map<string, map<string, vector<string>>>;

IniData parseNonStandardIni(const string& filename) {
    IniData parsedData;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    string line, currentSection;

    while (getline(file, line)) {
        // Remove leading/trailing whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        // Detect section headers
        if (line.front() == '[' && line.back() == ']') {
            currentSection = line.substr(1, line.size() - 2);
            parsedData[currentSection] = {};
        } else if (!currentSection.empty()) {
            // Parse key-value pairs
            istringstream iss(line);
            string key;
            iss >> key;

            vector<string> values;
            string value;
            while (iss >> value) {
                values.push_back(value);
            }

            if (!values.empty()) {
                parsedData[currentSection][key] = values;
            }
        }
    }

    file.close();
    return parsedData;
}

double getParameter(const IniData& data, const string& section, const string& key, int index = 0) {
    try {
        return stod(data.at(section).at(key).at(index));
    } catch (const out_of_range&) {
        cerr << "Error: Parameter [" << section << "][" << key << "] not found or invalid index!" << endl;
        exit(EXIT_FAILURE);
    } catch (const invalid_argument&) {
        cerr << "Error: Invalid value for [" << section << "][" << key << "]!" << endl;
        exit(EXIT_FAILURE);
    }
}

// Define a struct to pass parameters to the ODE solver
struct ODEParams {
    complex<double> s;  // Eigenvalue s
};

// Mathematical Functions
double N2(double x) {
    double s = x / HGAS;
    return NsqInfty - (NsqInfty - Nsq0) * exp(-0.5 * s * s / (D * D));
}

double dN2_dx(double x) {
    double s = x / HGAS;
    double dN2_ds = (s / (D * D)) * (NsqInfty - Nsq0) * exp(-0.5 * s * s / (D * D));
    return dN2_ds / HGAS;
}

double P(double x) {
    double s = x / HGAS;
    double P0 = CS * CS * RHOG;
    return P0 * exp(-2.0 * Pi * s);
}

double rho(double x) {
    double s = x / HGAS;
    double b = 2.0 * Pi * (1.0 - 1.0 / gamma);
    double fs1 = NsqInfty / b * (exp(b * s) - 1.0);
    double fs2 = sqrt(M_PI / 2.0) * (NsqInfty - Nsq0) * D * exp(0.5 * b * b * D * D);
    double fs3 = erf(b * D / sqrt(2.0)) - erf((b * D * D - s) / (sqrt(2.0) * D));
    double fs = fs1 - fs2 * fs3;

    double result = 2.0 * Pi * RHOG * exp(-2.0 * Pi * s / gamma);
    result /= 2.0 * Pi + fs;
    return result;
}

double vy0(double x) {
    return -(3.0/2.0)*OMEGA*x - Pi*P(x)/rho(x)/CS;
}


double cs2(double x) {
    return P(x) / rho(x);
}

double dlnP_dx(double x) {
    return -2.0 * Pi / HGAS;
}

double dP_dx(double x) {
    return P(x) * dlnP_dx(x);
}

double gr(double x) {
    return dP_dx(x) / rho(x);
}

double dlnrho_dx(double x) {
    double s = x / HGAS;
    double dlnrho_ds = -2.0 * Pi / gamma - N2(x) * exp(2.0 * Pi * s) * (rho(x) / RHOG) / (2.0 * Pi);
    return dlnrho_ds / HGAS;
}

double d2lnrho_dx2(double x) {
    double s = x / HGAS;
    double d2lnrho_ds2 = HGAS * dN2_dx(x) * exp(2.0 * Pi * s) * rho(x) / RHOG;
    d2lnrho_ds2 += 2.0 * Pi * exp(2.0 * Pi * s) * N2(x) * rho(x) / RHOG;
    d2lnrho_ds2 += exp(2.0 * Pi * s) * N2(x) * rho(x) / RHOG * HGAS * dlnrho_dx(x);
    d2lnrho_ds2 *= -1.0 / (2.0 * Pi);
    return d2lnrho_ds2 / (HGAS * HGAS);
}

double dlnT_dx(double x) {
    return dlnP_dx(x) - dlnrho_dx(x);
}

double d2lnT_dx2(double x) {
    return -d2lnrho_dx2(x);  // Since d2lnP_dx2 = 0
}

double LapT_T(double x) {
    return dlnT_dx(x) * dlnT_dx(x) + d2lnT_dx2(x);
}

double chi(double x) {
    return kappaT * (gamma - 1.0) / (R_MU * rho(x));
}

double duy_dx(double x) {
    return (P(x) / rho(x)) / (2.0 * OMEGA) * dlnT_dx(x) * dlnP_dx(x);
}

// Define the ODE system for GSL, using real and imaginary parts for complex numbers
int odes(double x, const double y[], double dydx[], void *params) {
    std::complex<double> ii(0.0,1.0);

    ODEParams *p = (ODEParams *)params;
    complex<double> s = p->s;
    complex<double> sbar = s + ii*ky*vy0(x);

    // Unpack the complex numbers from y array
    double W_real = y[0], W_imag = y[1];
    double theta_real = y[2], theta_imag = y[3];
    double theta_x_real = y[4], theta_x_imag = y[5];
    double vx_real = y[6], vx_imag = y[7];

    // Define the ODEs (complex operations split into real and imaginary parts)
    // Example: Calculate vy, Q, etc. (adjust the logic accordingly)

    std::complex<double> W(W_real, W_imag);
    std::complex<double> theta(theta_real, theta_imag);
    std::complex<double> theta_x(theta_x_real, theta_x_imag);
    std::complex<double> vx(vx_real, vx_imag);

    // These are the ODEs from your original code
    std::complex<double> vy = -(1.0 / sbar) * ( (OMEGA / 2.0 + duy_dx(x)) * vx + ii*ky*W ) ;
    std::complex<double> Q = W / cs2(x) - theta;
    std::complex<double> dW_dx = 2.0 * OMEGA * vy + Q * gr(x) - W * dlnrho_dx(x) - sbar * vx;

    std::complex<double> dtheta_dx = theta_x;

    std::complex<double> chix = chi(x);
    std::complex<double> divv = -sbar * Q - vx * dlnrho_dx(x);
    std::complex<double> dtheta_x_dx = (sbar / chix - LapT_T(x) + kz * kz + ky*ky) * theta + (vx / chix - 2.0 * theta_x) * dlnT_dx(x) + (gamma - 1.0) / chix * divv;

    
    std::complex<double> vz = -ii * kz * W / sbar;
    std::complex<double> dvx_dx = -(sbar * Q + vx * dlnrho_dx(x) + ii * kz * vz + ii*ky*vy);

    // Convert the complex derivatives into real and imaginary parts
    dydx[0] = std::real(dW_dx);
    dydx[1] = std::imag(dW_dx);

    dydx[2] = std::real(dtheta_dx);
    dydx[3] = std::imag(dtheta_dx);

    dydx[4] = std::real(dtheta_x_dx);
    dydx[5] = std::imag(dtheta_x_dx);

    dydx[6] = std::real(dvx_dx);
    dydx[7] = std::imag(dvx_dx);

    return GSL_SUCCESS;
}

// Function to solve the ODE system (modified to pass s and dx)
void solve_ode(double x0, double x_max, double dx, std::vector<double>& y, complex<double> s_guess) {
    gsl_odeiv2_system sys;
    ODEParams params = {s_guess};  // Pass the eigenvalue to the ODE solver

    sys.function = &odes;
    sys.jacobian = NULL;
    sys.dimension = 8;
    sys.params = &params;

    double y_init[8] = {y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]};

    // gsl_odeiv2_driver *driver = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, dx, 1e-12, 1e-12);

    // gsl_odeiv2_driver *driver = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk4, dx, 1e-12, 1e-12);
    gsl_odeiv2_driver *driver = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45, dx, 1e-12, 1e-12);

    double x = x0;
    while (x < x_max) {
        int status = gsl_odeiv2_driver_apply(driver, &x, x_max, y_init);
        if (status != GSL_SUCCESS) {
            std::cerr << "Error: " << gsl_strerror(status) << std::endl;
            break;
        }
    }
    for (int i = 0; i < 8; i++) y[i] = y_init[i];
    gsl_odeiv2_driver_free(driver);
}

// System of equations to find the root (no change)
int system_of_equations(const gsl_vector *v, void *params, gsl_vector *f) {
    double s_real = gsl_vector_get(v, 0);
    double s_imag = gsl_vector_get(v, 1);
    double theta_real = gsl_vector_get(v, 2);
    double theta_imag = gsl_vector_get(v, 3);

    // Initial conditions at x = -Lx/2
    std::vector<double> y_init = {pert_amp*CS*CS, 0.0, theta_real, theta_imag, 0.0, 0.0, 0.0, 0.0};

    // Solve the ODE system from x = -Lx/2 to x = Lx/2
    complex<double> s_guess = complex<double>(s_real, s_imag);
    double dx= 1e-4;
    solve_ode(x_min, x_max, dx, y_init, s_guess);

    // Set the system of equations to find the root
    gsl_vector_set(f, 0, y_init[4]); //real(dtheta/dx) at xmax
    gsl_vector_set(f, 1, y_init[5]); //imag(dtheta/dx) at xmax
    gsl_vector_set(f, 2, y_init[6]); //real(vx) at xmax
    gsl_vector_set(f, 3, y_init[7]); //imag(vx) at xmax

    return GSL_SUCCESS;
}

// Root-finding using GSL's multidimensional root-finding routine
std::tuple<std::complex<double>, std::complex<double>> find_root() {
    const gsl_multiroot_fsolver_type *T;
    gsl_multiroot_fsolver *solver;

    // Define the function to solve
    gsl_multiroot_function f = {&system_of_equations, 4, NULL};

    // Initial guesses for s and theta (real and imaginary parts)
    gsl_vector *x_init = gsl_vector_alloc(4);
    gsl_vector_set(x_init, 0, sguess);  // Initial guess for real part of s
    gsl_vector_set(x_init, 1, fguess);   // Initial guess for imaginary part of s
    gsl_vector_set(x_init, 2, 0);  // Initial guess for real part of theta
    gsl_vector_set(x_init, 3, 0);   // Initial guess for imaginary part of theta

    // Use the function-only solver
    T = gsl_multiroot_fsolver_hybrid;
    solver = gsl_multiroot_fsolver_alloc(T, 4);
    gsl_multiroot_fsolver_set(solver, &f, x_init);

    int status, iter = 0;
    do {
        iter++;
        status = gsl_multiroot_fsolver_iterate(solver);

        if (status == GSL_ENOPROG) {  // No progress
          std::cerr << "No progress can be made!" << std::endl;
          break;
        } else if (status != GSL_SUCCESS && status != GSL_CONTINUE) {
          std::cerr << "Root finding error: " << gsl_strerror(status) << std::endl;
          break;
        }

        // std::cout << "Iteration " << iter << ": "
        //           << "s = " << gsl_vector_get(solver->x, 0) << " + i" << gsl_vector_get(solver->x, 1) << ", "
        //           << "theta = " << gsl_vector_get(solver->x, 2) << " + i" << gsl_vector_get(solver->x, 3) << std::endl;

    } while (gsl_multiroot_test_residual(solver->f, 1e-12) == GSL_CONTINUE);

    // Extract s and theta, return as a tuple
    double s_real = gsl_vector_get(solver->x, 0);
    double s_imag = gsl_vector_get(solver->x, 1);
    std::complex<double> s_final(s_real, s_imag);

    double theta_real = gsl_vector_get(solver->x, 2);
    double theta_imag = gsl_vector_get(solver->x, 3);
    std::complex<double> theta_final(theta_real, theta_imag);

    gsl_multiroot_fsolver_free(solver);
    gsl_vector_free(x_init);

    // Return both s_final and theta_final as a tuple
    return std::make_tuple(s_final, theta_final);

}

// Declare global storage for coarse grid results
vector<double> x_coarse, W_re, W_im, Q_re, Q_im, vx_re, vx_im, vy_re, vy_im, vz_re, vz_im;

void solve_on_coarse_grid(double x_min, double x_max, int Nx, double dx) {
    complex<double> ii(0.0,1.0);
  //Solve the eigenvalue problem
    std::complex<double> s_final, theta_init;
    std::tie(s_final, theta_init) = find_root();

  //Output the final eigenvalue and theta
    std::cout << "Eigenvalue s = " << s_final << std::endl;
    std::cout << "Initial theta = " << theta_init << std::endl;

  //Solve the IVP to get variables on the coarse grid, solve iteratively from one x to the next
    double x0 = x_min;
    std::vector<double> y0 = {pert_amp*CS*CS, 0.0, real(theta_init), imag(theta_init), 0.0, 0.0, 0.0, 0.0};

    for (int i = 0; i < Nx; ++i) {
        double x = x_min + i * (x_max - x_min) / (Nx - 1);

        // Solve the ODE system at x
        solve_ode(x0, x, dx, y0, s_final);

        // Extract variables
        complex<double> W(y0[0], y0[1]);
        complex<double> theta(y0[2], y0[3]);
        complex<double> vx(y0[6], y0[7]);
        complex<double> sbar = s_final + ii*ky*vy0(x);
        complex<double> vy = -(1.0 / sbar) *( (OMEGA / 2.0 + duy_dx(x)) * vx + ii*ky*W );
        complex<double> Q = W / cs2(x) - theta;
        complex<double> vz = -ii* kz * W / sbar;

        // Save results
        x_coarse.push_back(x);
        W_re.push_back(real(W));
        W_im.push_back(imag(W));
        Q_re.push_back(real(Q));
        Q_im.push_back(imag(Q));
        vx_re.push_back(real(vx));
        vx_im.push_back(imag(vx));
        vy_re.push_back(real(vy));
        vy_im.push_back(imag(vy));
        vz_re.push_back(real(vz));
        vz_im.push_back(imag(vz));

        //Update old x and solve for the next one
        x0 = x;
    }
}

complex<double> interpolate_variable(double x, const vector<double>& re, const vector<double>& im) {
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline_re = gsl_spline_alloc(gsl_interp_cspline, x_coarse.size());
    gsl_spline *spline_im = gsl_spline_alloc(gsl_interp_cspline, x_coarse.size());

    gsl_spline_init(spline_re, x_coarse.data(), re.data(), x_coarse.size());
    gsl_spline_init(spline_im, x_coarse.data(), im.data(), x_coarse.size());

    complex<double> result(gsl_spline_eval(spline_re, x, acc), gsl_spline_eval(spline_im, x, acc));

    // Free memory
    gsl_spline_free(spline_re);
    gsl_spline_free(spline_im);
    gsl_interp_accel_free(acc);

    return result;
}

// Main function
int main() {
    //read idefix parameter file
    IniData iniData = parseNonStandardIni("idefix.ini");

    gamma  = getParameter(iniData, "Hydro", "gamma");
    Pi     = getParameter(iniData, "Setup", "etahat");
    Pe     = getParameter(iniData, "Setup", "Pe");
    kappaT = R_MU * RHOG * HGAS * HGAS * OMEGA / ((gamma - 1.0) * Pe);

    // Buoyancy profile
    NsqInfty = getParameter(iniData, "Setup", "NsqInf");
    Nsq0     = getParameter(iniData, "Setup", "Nsq0");
    D        = getParameter(iniData, "Setup", "NsqWidth");

    // Domain parameters
    x_min = getParameter(iniData, "Grid", "X1-grid", 1);  // 2nd value
    x_max = getParameter(iniData, "Grid", "X1-grid", 4);  // 5th value

    // Perturbation parameters
    pert_amp = getParameter(iniData, "Setup", "pertamp");
    kz = getParameter(iniData, "Setup", "eigen_kz")/HGAS;
    ky = getParameter(iniData, "Setup", "eigen_ky")/HGAS;
    sguess = getParameter(iniData, "Setup", "eigen_guess")*OMEGA;
    fguess = getParameter(iniData, "Setup", "eigen_guess", 1)*OMEGA;

    // Step 1: Solve on the coarse grid, including getting the eigenvalue
    int Nx_coarse = 128;
    double    dx  = 1e-4*HGAS;
    solve_on_coarse_grid(x_min, x_max, Nx_coarse, dx);

    // Step 2: Interpolate onto fine grid
    int Nx_fine = 32;

    ofstream outfile("output.txt");
    if (!outfile) {
        cerr << "Error opening output file!" << endl;
        return -1;
    }

    for (int i = 0; i < Nx_fine; ++i) {
        double x  = x_min + i * (x_max - x_min) / (Nx_fine - 1);

        complex<double> W_fine = interpolate_variable(x, W_re, W_im);
        complex<double> Q_fine = interpolate_variable(x, Q_re, Q_im);
        complex<double> vx_fine = interpolate_variable(x, vx_re, vx_im);
        complex<double> vy_fine = interpolate_variable(x, vy_re, vy_im);
        complex<double> vz_fine = interpolate_variable(x, vz_re, vz_im);

        outfile << x << " "
                     << real(W_fine) << " " << imag(W_fine) << " "
                     << real(Q_fine) << " " << imag(Q_fine) << " "
                     << real(vx_fine) << " " << imag(vx_fine) << " "
                     << real(vy_fine) << " " << imag(vy_fine) << " "
                     << real(vz_fine) << " " << imag(vz_fine) << endl;
    }

    outfile.close();
    cout << "Simulation completed. Results saved in output.txt" << endl;

    return 0;
}
