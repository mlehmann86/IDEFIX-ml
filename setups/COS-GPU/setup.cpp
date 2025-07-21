#include "idefix.hpp"
#include "setup.hpp"
#include "analysis.hpp"
#include "cos.hpp"

//Headers for eigenvalue and IVP solver for initialization
#include <vector>
#include <complex>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_spline.h>

Analysis *analysis;

real gammaIdeal;
real omega;
real qshear;

real etahat;
real NsqInf;
real Nsq0;
real NsqWidth;
real tcool;
real Pe;
real pertamp;
real Re;

real damping_time;
real damping_Ltrans;
real damping_dtrans;

std::string eigen_pert;
real eigen_kz;
real eigen_ky;
real eigen_sguess;
real eigen_fguess;
real x_min; //needed in eigenvalue solver
real x_max; //needed in eigenvalue solver

struct ODEParams {
    std::complex<double> eigens;  // Eigenvalue s
};
std::vector<double> x_coarse, W_re, W_im, Q_re, Q_im, vx_re, vx_im, vy_re, vy_im, vz_re, vz_im; // Declare global storage for coarse grid results

// Define the ODE system for GSL, using real and imaginary parts for complex numbers
int odes(double x, const double y[], double dydx[], void *params) {
    std::complex<double> ii(0.0,1.0);

    ODEParams *p = (ODEParams *)params;
    std::complex<double> sbar = p->eigens + ii*eigen_ky*vy0(x);

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
    std::complex<double> vy = -(1.0 / sbar) *( (OMEGA / 2.0 + duy_dx(x)) * vx + ii*eigen_ky*W );
    std::complex<double> Q = W / cs2(x) - theta;
    std::complex<double> dW_dx = 2.0 * OMEGA * vy + Q * gr(x) - W * dlnrho_dx(x) - sbar * vx;

    std::complex<double> dtheta_dx = theta_x;

    std::complex<double> chix = chi(x);
    std::complex<double> divv = -sbar * Q - vx * dlnrho_dx(x);
    std::complex<double> dtheta_x_dx = (sbar / chix - LapT_T(x) + eigen_kz *eigen_kz + eigen_ky*eigen_ky) * theta + (vx / chix - 2.0 * theta_x) * dlnT_dx(x) + (gammaIdeal - 1.0) / chix * divv;

    std::complex<double> vz = -ii * eigen_kz * W / sbar;
    std::complex<double> dvx_dx = -(sbar * Q + vx * dlnrho_dx(x) + ii * eigen_kz * vz + ii*eigen_ky*vy);

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
void solve_ode(double x0, double xtarget, double dx, std::vector<double>& y, std::complex<double> s_guess) {
    gsl_odeiv2_system sys;
    ODEParams params = {s_guess};  // Pass the eigenvalue to the ODE solver

    sys.function = &odes;
    sys.jacobian = NULL;
    sys.dimension = 8;
    sys.params = &params;

    double y_init[8] = {y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]};

    // gsl_odeiv2_driver *driver = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, dx, 1e-12, 1e-12);
    gsl_odeiv2_driver *driver = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45, dx, 1e-12, 1e-12);

    double x = x0;
    while (x < xtarget) {
        int status = gsl_odeiv2_driver_apply(driver, &x, xtarget, y_init);
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
    std::vector<double> y_init = {pertamp*CS*CS, 0.0, theta_real, theta_imag, 0.0, 0.0, 0.0, 0.0};

    // Solve the ODE system from x = -Lx/2 to x = Lx/2
    std::complex<double> s_guess = std::complex<double>(s_real, s_imag);
    double dx= 1e-4*HGAS;
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
    gsl_vector_set(x_init, 0, eigen_sguess);  // Initial guess for real part of s
    gsl_vector_set(x_init, 1, eigen_fguess);   // Initial guess for imaginary part of s
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
          idfx::cout << "No progress can be made!" << std::endl;
          break;
        } else if (status != GSL_SUCCESS && status != GSL_CONTINUE) {
          idfx::cout << "Root finding error: " << gsl_strerror(status) << std::endl;
          break;
        }
        // idfx::cout << "Iteration " << iter << ": "
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

std::complex<double> interpolate_variable(double x, const std::vector<double>& re, const std::vector<double>& im) {
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline_re = gsl_spline_alloc(gsl_interp_cspline, x_coarse.size());
    gsl_spline *spline_im = gsl_spline_alloc(gsl_interp_cspline, x_coarse.size());

    gsl_spline_init(spline_re, x_coarse.data(), re.data(), x_coarse.size());
    gsl_spline_init(spline_im, x_coarse.data(), im.data(), x_coarse.size());

    std::complex<double> result; // Declare result here
    if ((x > x_min) && (x < x_max)) { // Interpolate inside domain
        result = std::complex<double>(gsl_spline_eval(spline_re, x, acc), gsl_spline_eval(spline_im, x, acc));
    } else { // Outside domain, use zero perturbations (=main code boundary conditions)
        result = std::complex<double>(0.0, 0.0);
    }

    // Free memory
    gsl_spline_free(spline_re);
    gsl_spline_free(spline_im);
    gsl_interp_accel_free(acc);

    return result;
}

// void FargoVelocity(DataBlock &data, IdefixArray2D<real> &Vphi) { //user defined FARGO velocity not compatible with shearing box
//   IdefixArray1D<real> x = data.x[IDIR];
//   IdefixArray1D<real> z = data.x[KDIR];

//   real gammaIdealLoc = gammaIdeal;
//   real qshearLoc     = qshear;

//   real etahatLoc     = etahat;
//   real NsqInfLoc     = NsqInf;
//   real Nsq0Loc       = Nsq0;
//   real NsqWidthLoc   = NsqWidth;

//   idefix_for("FargoVphi",0,data.np_tot[KDIR], 0, data.np_tot[IDIR],
//       KOKKOS_LAMBDA (int k, int i) {
//       Vphi(k,i) = VyEqm(x(i), z(k), qshearLoc, etahatLoc, NsqInfLoc, Nsq0Loc, NsqWidthLoc, gammaIdealLoc);
//   });
// }

void ConstantKinematicViscosity(DataBlock &data, const real t, IdefixArray3D<real> &eta1, IdefixArray3D<real> &eta2) {
  IdefixArray4D<real> Vc=data.hydro->Vc;

  real nuvisc = HGAS*HGAS*OMEGA/Re;

  idefix_for("ConstantKinematicViscosity",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                eta1(k,j,i) = nuvisc*Vc(RHO,k,j,i);
                eta2(k,j,i) = ZERO_F;
              });

}

void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray1D<real> x  = data->x[IDIR];
  IdefixArray1D<real> z  = data->x[KDIR];

  real gammaIdealLoc = gammaIdeal;
  real qshearLoc     = qshear;

  real etahatLoc     = etahat;
  real NsqInfLoc     = NsqInf;
  real Nsq0Loc       = Nsq0;
  real NsqWidthLoc   = NsqWidth;

  //Customized radial boundaries, set to equilibrium solution (inviscid)
    // if(dir==IDIR) {
    //     hydro->boundary->BoundaryFor("UserDefBoundaryX", dir, side,
    //       KOKKOS_LAMBDA (int k, int j, int i) {
    //           Vc(RHO,k,j,i) = DensityEqm(x(i), z(k), etahatLoc, NsqInfLoc, Nsq0Loc, NsqWidthLoc, gammaIdealLoc);
    //           Vc(PRS,k,j,i) = PressureEqm(x(i), z(k), etahatLoc);
    //           Vc(VX1,k,j,i) = ZERO_F;
    //           Vc(VX2,k,j,i) = VyEqm(x(i), z(k), qshearLoc, etahatLoc, NsqInfLoc, Nsq0Loc, NsqWidthLoc, gammaIdealLoc);
    //           Vc(VX3,k,j,i) = ZERO_F;
    //       });
    // }

  //Customized radial boundaries, no velocity pert, no gradient in PERTURBED P and rho
    if(dir==IDIR) {
      int ighost;
      if(side == left) {
        ighost = hydro->data->beg[IDIR];
      } else if (side == right) {
        ighost = hydro->data->end[IDIR];
      }
        hydro->boundary->BoundaryFor("UserDefBoundaryX", dir, side,
          KOKKOS_LAMBDA (int k, int j, int i) {
            real rho_act  =  DensityEqm(x(2*ighost-i-1), z(k), etahatLoc, NsqInfLoc, Nsq0Loc, NsqWidthLoc, gammaIdealLoc);
            real drho_act =  (Vc(RHO,k,j,2*ighost-i-1) - rho_act)/rho_act;
            real p_act    =  PressureEqm(x(2*ighost-i-1), z(k), etahatLoc);
            real dp_act   =  (Vc(PRS,k,j,2*ighost-i-1) - p_act)/p_act;

            real rho_gst       = DensityEqm(x(i), z(k), etahatLoc, NsqInfLoc, Nsq0Loc, NsqWidthLoc, gammaIdealLoc);
            real p_gst         = PressureEqm(x(i), z(k), etahatLoc);

            Vc(RHO,k,j,i) =  rho_gst*(1 + drho_act);
            Vc(VX1,k,j,i) =  ZERO_F;
            Vc(VX2,k,j,i) =  VyEqm(x(i), z(k), qshearLoc, etahatLoc, NsqInfLoc, Nsq0Loc, NsqWidthLoc, gammaIdealLoc);
            Vc(VX3,k,j,i) =  ZERO_F;
            Vc(PRS,k,j,i) =  p_gst*(1 + dp_act);
          });
    }

    //Customized vertical boundaries, reflect vz and zero-gradient in others
    if(dir==KDIR) {
      int kghost;
      if(side == left) {
        kghost = hydro->data->beg[KDIR];
      } else if (side == right) {
        kghost = hydro->data->end[KDIR];
      }
      hydro->boundary->BoundaryFor("UserDefBoundaryZ", dir, side,
                       KOKKOS_LAMBDA (int k, int j, int i) {
                            Vc(RHO,k,j,i) =  Vc(RHO,2*kghost-k-1,j,i);
                            Vc(VX1,k,j,i) =  Vc(VX1,2*kghost-k-1,j,i);
                            Vc(VX2,k,j,i) =  Vc(VX2,2*kghost-k-1,j,i);
                            Vc(VX3,k,j,i) = -Vc(VX3,2*kghost-k-1,j,i);
                            Vc(PRS,k,j,i) =  Vc(PRS,2*kghost-k-1,j,i);
                           });
    }
}

void CoolingDamping(Hydro *hydro, const real t, const real dtin) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray4D<real> Uc = hydro->Uc;
  IdefixArray1D<real> xaxis  = data->x[IDIR];
  IdefixArray1D<real> zaxis  = data->x[KDIR];

  real gammaIdealLoc = gammaIdeal;
  real qshearLoc     = qshear;

  real etahatLoc     = etahat;
  real NsqInfLoc     = NsqInf;
  real Nsq0Loc       = Nsq0;
  real NsqWidthLoc   = NsqWidth;
  // real tcoolLoc      = tcool;
  real dt            = dtin;

  bool isFargo = data->haveFargo;

  real xmin{data->mygrid->xbeg[0]};
  real xmax{data->mygrid->xend[0]};

  real Lx = xmax - xmin;
  real damping_LtransLoc = damping_Ltrans*(Lx/2);       //convert to physical distance, assume box is symmetric (|xmax| = |xmin|)
  real damping_timeLoc   = damping_time*2.0*M_PI/OMEGA; //convert to physical time
  real damping_dtransLoc = damping_dtrans;

  idefix_for("CoolingDamping",
    0, data->np_tot[KDIR],
    0, data->np_tot[JDIR],
    0, data->np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                //optically thin cooling
                real x = xaxis(i);
                real z = zaxis(k);

                real Pinit      = PressureEqm(x, z, etahatLoc);
                real rhoinit    = DensityEqm(x, z, etahatLoc, NsqInfLoc, Nsq0Loc, NsqWidthLoc, gammaIdealLoc);
                // real Ptarget    = Vc(RHO,k,j,i)*Pinit/rhoinit;
                // Uc(ENG,k,j,i)  += -dt*(Vc(PRS,k,j,i)-Ptarget)/(tcoolLoc*(gammaIdealLoc-1.0));

                //damping boundaries (damp primitive variables, but update conservative variables)
                if (damping_timeLoc > 0.0){
                  real vyinit = VyEqm(x, z, qshearLoc, etahatLoc, NsqInfLoc, Nsq0Loc, NsqWidthLoc, gammaIdealLoc); //includes background shear flow
                  if(isFargo) {//with FARGO, we take out the shear from the target velocity
                    vyinit   -= qshearLoc*x;
                  }

                  real drho = Vc(RHO,k,j,i) - rhoinit;
                  real dvx1 = Vc(VX1,k,j,i);          //vx=0 in eqm
                  real dvx2 = Vc(VX2,k,j,i) - vyinit;
                  real dvx3 = Vc(VX3,k,j,i);          //vz=0 in eqm
                  real dprs = Vc(PRS,k,j,i) - Pinit;

                  real dmx1 = Vc(RHO,k,j,i)*dvx1 + Vc(VX1,k,j,i) * drho;
                  real dmx2 = Vc(RHO,k,j,i)*dvx2 + Vc(VX2,k,j,i) * drho;
                  real dmx3 = Vc(RHO,k,j,i)*dvx3 + Vc(VX3,k,j,i) * drho;

                  // Kinetic energy fluctuations due to above damping
                  // must be compensated in total energy conservation
                  real deng = 0.5*(Vc(VX1,k,j,i)*Vc(VX1,k,j,i)
                                  +Vc(VX2,k,j,i)*Vc(VX2,k,j,i)
                                  +Vc(VX3,k,j,i)*Vc(VX3,k,j,i))*drho
                              + Vc(RHO,k,j,i) * (
                                  Vc(VX1,k,j,i)*dvx1
                                + Vc(VX2,k,j,i)*dvx2
                                + Vc(VX3,k,j,i)*dvx3
                              )
                              + dprs/(gammaIdealLoc-1.0);

                  real lambda = 0.0;
                  lambda = dt*(1.0 + 0.5 * (tanh(damping_dtransLoc * (x - damping_LtransLoc)) - tanh(damping_dtransLoc * (x + damping_LtransLoc))))/damping_timeLoc;

                  Uc(RHO,k,j,i) += -drho*lambda;
                  Uc(MX1,k,j,i) += -dmx1*lambda;
                  Uc(MX2,k,j,i) += -dmx2*lambda;
                  Uc(MX3,k,j,i) += -dmx3*lambda;
                  Uc(ENG,k,j,i) += -deng*lambda;

                }

});
}

void BodyForce(DataBlock &data, const real t, IdefixArray4D<real> &force) {
  idfx::pushRegion("BodyForce");
  IdefixArray1D<real> x = data.x[IDIR];
  IdefixArray1D<real> z = data.x[KDIR];

  real omegaLoc      = omega;
  real qshearLoc     = qshear;

  idefix_for("BodyForce",
              data.beg[KDIR] , data.end[KDIR],
              data.beg[JDIR] , data.end[JDIR],
              data.beg[IDIR] , data.end[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {

                force(IDIR,k,j,i) = -2.0*omegaLoc*qshearLoc*x(i);
                force(JDIR,k,j,i) = ZERO_F;
                // #ifdef STRATIFIED
                //   force(KDIR,k,j,i) = - omegaLocal*omegaLocal*z(k);
                // #else
                force(KDIR,k,j,i) = ZERO_F;
                // #endif
      });


  idfx::popRegion();
}

void AnalysisFunction(DataBlock &data) {
  analysis->PerformAnalysis(data);
}

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {

  //Read problem input parameters, for use in initial conditions
  gammaIdeal = data.hydro->eos->GetGamma();
  omega      = input.Get<real>("Hydro","rotation",0)*OMEGA;
  qshear     = input.Get<real>("Hydro","shearingBox",0)*OMEGA;

  etahat     = input.Get<real>("Setup","etahat",0);
  NsqInf     = input.Get<real>("Setup","NsqInf",0);
  Nsq0       = input.Get<real>("Setup","Nsq0",0);
  NsqWidth   = input.Get<real>("Setup","NsqWidth",0);
  tcool      = input.Get<real>("Setup","tcool",0)/OMEGA;
  Pe         = input.Get<real>("Setup","Pe",0);
  pertamp    = input.Get<real>("Setup","pertamp",0);
  Re         = input.Get<real>("Setup","Re",0);

  damping_time   = input.Get<real>("Setup","damping_time",0);
  damping_Ltrans = input.Get<real>("Setup","damping_Ltrans",0);
  damping_dtrans = input.Get<real>("Setup","damping_dtrans",0);

  eigen_pert     = input.Get<std::string>("Setup","eigen_pert",0);
  eigen_kz       = input.Get<real>("Setup","eigen_kz",0)/HGAS;
  eigen_ky       = input.Get<real>("Setup","eigen_ky",0)/HGAS;
  eigen_sguess   = input.Get<real>("Setup","eigen_guess",0)*OMEGA;
  eigen_fguess   = input.Get<real>("Setup","eigen_guess",1)*OMEGA;

  // Add our userstep to the timeintegrator
  data.gravity->EnrollBodyForce(BodyForce);
  data.hydro->EnrollUserSourceTerm(&CoolingDamping);
  // data.hydro->thermalDiffusion->EnrollThermalDiffusivity(&MyThermalDiffusivity);

  // if (damping_time > 0.0){
  //   data.hydro->EnrollUserSourceTerm(&Damping);
  // }

  //Enroll customzied boundary conditions
  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);

  //Enroll constant kinematic viscosity
  //data.hydro->viscosity->EnrollViscousDiffusivity(&ConstantKinematicViscosity);

  //Enroll FARGO velocity if needed
  // if(data.haveFargo)
  //   data.fargo->EnrollVelocity(&FargoVelocity);

  analysis = new Analysis(input, grid, data, output,std::string("analysis.dat"));
  output.EnrollAnalysis(&AnalysisFunction);
  // Reset analysis if required
  if(!input.restartRequested) {
    analysis->ResetAnalysis();
  }

  //Solve eigenvalue problem if needed, and get eigensolution on a coarse grid
  if(eigen_pert.compare("yes") == 0){
    idfx::cout << "******************************" << std::endl;
    idfx::cout << "***Eigenmode initialization***" << std::endl;
    idfx::cout << "******************************" << std::endl;

    x_min = grid.xbeg[0];
    x_max = grid.xend[0];

    std::complex<double> s_final, theta_init;
    std::tie(s_final, theta_init) = find_root();

    std::complex<double> ii(0.0,1.0);

    //Output the final eigenvalue and theta
    idfx::cout << "Eigenvalue s = " << s_final << std::endl;

    //Solve the IVP to get variables on the coarse grid, solve iteratively from one x to the next
    double x0 = x_min;
    std::vector<double> y0 = {pertamp*CS*CS, 0.0, std::real(theta_init), std::imag(theta_init), 0.0, 0.0, 0.0, 0.0};

    int Nx_coarse = 128;
    double dx = 1e-4*HGAS;
    for (int i = 0; i < Nx_coarse; ++i) {
        double x = x_min + i * (x_max - x_min) / (Nx_coarse - 1);

        // Solve the ODE system at x
        solve_ode(x0, x, dx, y0, s_final);

        // Extract variables
        std::complex<double> sbar = s_final + ii*eigen_ky*vy0(x);
        std::complex<double> W(y0[0], y0[1]);
        std::complex<double> theta(y0[2], y0[3]);
        std::complex<double> vx(y0[6], y0[7]);
        std::complex<double> vy = -(1.0 / sbar) *( (OMEGA / 2.0 + duy_dx(x)) * vx + ii*eigen_ky*W);
        std::complex<double> Q = W / cs2(x) - theta;
        std::complex<double> vz = -ii * eigen_kz * W / sbar;

        // Save results
        x_coarse.push_back(x);
        W_re.push_back(std::real(W));
        W_im.push_back(std::imag(W));
        Q_re.push_back(std::real(Q));
        Q_im.push_back(std::imag(Q));
        vx_re.push_back(std::real(vx));
        vx_im.push_back(std::imag(vx));
        vy_re.push_back(std::real(vy));
        vy_im.push_back(std::imag(vy));
        vz_re.push_back(std::real(vz));
        vz_im.push_back(std::imag(vz));

        //Update old x and solve for the next one
        x0 = x;
      }
      idfx::cout << "Got eigenfunctions on coarse grid"<< std::endl;
  }
}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
    // Create a host copy
    DataBlockHost d(data);
    real x, y, z, drho, dprs, dvx, dvy, dvz;

    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {
                x=d.x[IDIR](i);
                y=d.x[JDIR](j);
                z=d.x[KDIR](k);

                //exact steady state
                d.Vc(RHO,k,j,i) = DensityEqm(x, z, etahat, NsqInf, Nsq0, NsqWidth, gammaIdeal);
                d.Vc(PRS,k,j,i) = PressureEqm(x, z, etahat);
                d.Vc(VX1,k,j,i) = ZERO_F;
                d.Vc(VX2,k,j,i) = VyEqm(x, z, qshear, etahat, NsqInf, Nsq0, NsqWidth, gammaIdeal);
                d.Vc(VX3,k,j,i) = ZERO_F;

                if(eigen_pert.compare("yes") == 0){//simple eigenmode pert in horizontal velocities only
                  // real xi    = HGAS*HGAS/(OMEGA*Pe);
                  // real beta  = xi*eigen_kz*eigen_kz;
                  // real sgrow = -0.5*beta*Nsq0/(OMEGA*OMEGA + beta*beta);
                  //
                  // dvy = pertamp*CS*cos(eigen_kz*z);
                  // dvx = -2.0*sgrow/OMEGA*dvy*cos(eigen_kz*z);

                  // real dtheta = (sgrow*dvx - 2.0*OMEGA*dvy)/(-Nsq0)*cos(eigen_kz*z);
                  // drho   = d.Vc(RHO,k,j,i)*dtheta/HGAS;

                  std::complex<double> W = interpolate_variable(x, W_re, W_im); // P'/rho
                  std::complex<double> Q = interpolate_variable(x, Q_re, Q_im); // rho'/rho
                  std::complex<double> vx = interpolate_variable(x, vx_re, vx_im);
                  std::complex<double> vy = interpolate_variable(x, vy_re, vy_im);
                  std::complex<double> vz = interpolate_variable(x, vz_re, vz_im);

                  dprs = (std::real(W)*cos(eigen_kz*z + eigen_ky*y) - std::imag(W)*sin(eigen_kz*z + eigen_ky*y))*d.Vc(RHO,k,j,i);
                  drho = (std::real(Q)*cos(eigen_kz*z + eigen_ky*y) - std::imag(Q)*sin(eigen_kz*z + eigen_ky*y))*d.Vc(RHO,k,j,i);
                  dvx  = std::real(vx)*cos(eigen_kz*z + eigen_ky*y) - std::imag(vx)*sin(eigen_kz*z + eigen_ky*y);
                  dvy  = std::real(vy)*cos(eigen_kz*z + eigen_ky*y) - std::imag(vy)*sin(eigen_kz*z + eigen_ky*y);
                  dvz  = std::real(vz)*cos(eigen_kz*z + eigen_ky*y) - std::imag(vz)*sin(eigen_kz*z + eigen_ky*y);

                } else {//random pert in vy
                  drho  = ZERO_F;
                  dprs  = ZERO_F;
                  dvx   = ZERO_F;
                  dvy   = (pertamp*CS)*(2.0*idfx::randm()-1.0);
                  dvz   = ZERO_F;
                }

                d.Vc(RHO,k,j,i) += drho;
                d.Vc(PRS,k,j,i) += dprs;
                d.Vc(VX1,k,j,i) += dvx;
                d.Vc(VX2,k,j,i) += dvy;
                d.Vc(VX3,k,j,i) += dvz;

            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
}

// Analyse data to produce an output

void MakeAnalysis(DataBlock & data) {
}
