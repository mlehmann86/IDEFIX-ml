#include <algorithm>
#include <math.h>
#include "idefix.hpp"
#include "setup.hpp"
#include "planet.hpp"
// NOTE: Make sure to add the following include at the top of setup.cpp
#include "boundary.hpp"

real wkzMinGlob;
real wkzMaxGlob;
real sgTaperMinGlob;
real sgTaperMaxGlob;
real wkzDampingGlob;
real sigma0Glob;
real sigmaSlopeGlob;
real h0Glob;
real flaringIndexGlob;
real alphaGlob;
real densityFloorGlob;
real masstaperGlob;
real omegaGlob;

real nuGlob;
real betaGlob;
real gammaGlob;
real chiGlob;

real kappa0Glob;
real kslopeGlob;

// NEW global variables for viscosity
real betaViscGlob;
real zetaOverEtaGlob;

bool coolResidualsGlob;


std::string outpathGlob;


void MyThermalDiffusivity(DataBlock &data, const real t, IdefixArray3D<real> &kappa) {
  const real kappa0 = kappa0Glob;
  const real kslope = kslopeGlob;
  IdefixArray1D<real> x1 = data.x[IDIR];


  idefix_for("MyThermalDiffusivity",
             0, data.np_tot[KDIR],
             0, data.np_tot[JDIR],
             0, data.np_tot[IDIR],
    KOKKOS_LAMBDA(int k, int j, int i) {
      real R = x1(i);
      kappa(k,j,i) = kappa0 * pow(R, kslope);
  });
}

void MySoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {
  real h0 = h0Glob;
  real flaringIndex = flaringIndexGlob;
  IdefixArray1D<real> x1=data.x[IDIR];
  idefix_for("MySoundSpeed",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                real R = x1(i);
                cs(k,j,i) = h0*pow(R,flaringIndex-0.5);
              });
}


// Complete MyViscosity function in setup.cpp

void MyViscosity(DataBlock &data, const real t,
                 IdefixArray3D<real> &eta1,
                 IdefixArray3D<real> &eta2) {
  // This function implements a density-dependent shear and bulk viscosity
  // based on Lehmann et al. 2019, Equations (6) and (7).
  // Shear Viscosity: eta = nu_0 * sigma_0 * (sigma/sigma_0)^(beta+1)
  // Bulk Viscosity: zeta = zeta_over_eta * eta

  IdefixArray4D<real> Vc = data.hydro->Vc;

  // Get parameters from global scope
  const real nu0_ref = nuGlob;            // Kinematic viscosity at sigma_ref, in code units
  const real sigma0 = sigma0Glob;      // Reference surface density sigma_0, in code units
  const real beta = betaViscGlob;         // Power-law index beta
  const real zeta_ratio = zetaOverEtaGlob;// Bulk to shear viscosity ratio

  idefix_for("MyViscosity",
             0, data.np_tot[KDIR],
             0, data.np_tot[JDIR],
             0, data.np_tot[IDIR],
    KOKKOS_LAMBDA(int k, int j, int i) {
      // Get local surface density
      real sigma_local = Vc(RHO, k, j, i);

      // Avoid issues at the density floor
      if (sigma_local < sigma0) {
          sigma_local = sigma0;
      }

      // Calculate the dynamic shear viscosity (eta1)
      // This is equivalent to eta = nu_0 * sigma * (sigma/sigma_0)^beta
      eta1(k,j,i) = sigma_local * nu0_ref * pow(sigma_local / sigma0, beta);

      // Calculate the dynamic bulk viscosity (eta2) as proportional to shear
      eta2(k,j,i) = zeta_ratio * eta1(k,j,i);
  });
}

// The complete, final Damping function
void Damping(Hydro *hydro, const real t, const real dtin) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray4D<real> Uc = hydro->Uc;
  IdefixArray1D<real> x1 = data->x[IDIR];

  // Get all required values and array views on the HOST
  const real dt = dtin;
  const bool isPeriodicX2 = true;

  // Capture boundary indices on the HOST
  const int beg_i = data->beg[IDIR];
  const int end_i = data->end[IDIR];
  const int beg_j = data->beg[JDIR];
  const int end_j = data->end[JDIR];

  // Get device-side views of the dx arrays on the HOST
  IdefixArray1D<real> dx1 = data->dx[IDIR];
  IdefixArray1D<real> dx2 = data->dx[JDIR];

  // Global variables needed for various terms, captured by the lambda.
  const real nu0            = nuGlob;
  const real sigmaSlope     = sigmaSlopeGlob;
  const real h0             = h0Glob;
  const real flaringIndex   = flaringIndexGlob;
  const real omega          = omegaGlob;
  const real wkzMin         = wkzMinGlob;
  const real rmin           = data->mygrid->xbeg[0];
  const real wkzMax         = wkzMaxGlob;
  const real rmax           = data->mygrid->xend[0];
  const real wkzDamping     = wkzDampingGlob;
  const real sigma0         = sigma0Glob;
  const real beta           = betaGlob;
  const real gamma          = gammaGlob;
  const bool coolResiduals  = coolResidualsGlob; // Capture cooling option

  // --- Fargo-related variables ---
  [[maybe_unused]] IdefixArray2D<real> fargoVelocity;
  [[maybe_unused]] Fargo::FargoType fargoType;
  bool haveFargo;

  haveFargo = hydro->data->haveFargo;
  if(haveFargo) {
    fargoVelocity = hydro->data->fargo->meanVelocity;
    fargoType = hydro->data->fargo->type;
  }

  idefix_for("MySourceTerm",
    0, data->np_tot[KDIR],
    0, data->np_tot[JDIR],
    0, data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      real R = x1(i);

      // =====================================================================
      // Custom Viscous Cooling Term to counteract viscous heating
      // =====================================================================
      #ifndef ISOTHERMAL
        real total_dissipation = 0.0;
        const real nuslope = sigmaSlope - 0.5;
        const real kinematic_nu = nu0 * pow(R, nuslope);
        const real eta1 = kinematic_nu * Vc(RHO, k, j, i);

        // --- Part 1: Analytical background dissipation (our stable "mode 5") ---
        real dissipation_bg = 0.0;
        const real A_const = (1.0 + sigmaSlope - 2.0 * flaringIndex) * h0 * h0;
        const real B_const = 2.0 * flaringIndex;
        real pressure_term = A_const * pow(R, B_const);

        if (pressure_term >= 1.0 || pressure_term < 0.0) {
            const real Omega_K = pow(R, -1.5); // Fallback to Keplerian
            dissipation_bg = (9.0/4.0) * eta1 * Omega_K * Omega_K;
        } else {
            const real sqrt_pressure_corr = sqrt(1.0 - pressure_term);
            const real term1 = -1.5 * pow(R, -1.5) * sqrt_pressure_corr;
            const real term2_numerator = -A_const * B_const * pow(R, B_const - 1.5);
            const real term2 = term2_numerator / (2.0 * sqrt_pressure_corr);
            const real r_dOmega_dr = term1 + term2;
            dissipation_bg = eta1 * r_dOmega_dr * r_dOmega_dr;
        }
        total_dissipation += dissipation_bg;

        // --- Part 2: Optionally add dissipation from the residual velocity field ---
        if (coolResiduals) {
            if (haveFargo && i > beg_i && i < end_i-1 && j > beg_j && j < end_j-1) {
                const real eta2 = 0.0;

                // --- Calculate gradients of the RESIDUAL velocity field ---
                // v_r is already a residual velocity. v_phi residual is Vc(VX2) - fargo.
                real Vr_res_ip1   = Vc(VX1, k, j, i+1);
                real Vphi_res_ip1_rot = Vc(VX2, k, j, i+1) - fargoVelocity(k,i+1);
                real Vr_res_im1   = Vc(VX1, k, j, i-1);
                real Vphi_res_im1_rot = Vc(VX2, k, j, i-1) - fargoVelocity(k,i-1);

                real Vr_res_center = Vc(VX1, k, j, i);
                real Vphi_res_center_rot = Vc(VX2, k, j, i) - fargoVelocity(k,i);

                int jp = j + 1;
                int jm = j - 1;
                if (isPeriodicX2) {
                    if (jp >= end_j) jp = beg_j;
                    if (jm < beg_j) jm = end_j - 1;
                }

                real dr = dx1(i);
                real dphi = dx2(j);
                real dVr_res_dr   = (Vr_res_ip1 - Vr_res_im1) / (2.0 * dr);
                real dVphi_res_dr = (Vphi_res_ip1_rot - Vphi_res_im1_rot) / (2.0 * dr);
                real dVr_res_dphi   = (Vc(VX1, k, jp, i) - Vc(VX1, k, jm, i)) / (2.0 * dphi); // V_r is already residual
                real dVphi_res_dphi = ( (Vc(VX2, k, jp, i) - fargoVelocity(k,i)) - (Vc(VX2, k, jm, i) - fargoVelocity(k,i)) ) / (2.0 * dphi);

                // --- Components of the Rate-of-Strain Tensor for the residual field ---
                real S_rr_res = dVr_res_dr;
                real S_phiphi_res = (1.0 / R) * dVphi_res_dphi + Vr_res_center / R;
                real S_rphi_res = 0.5 * ( dVphi_res_dr - Vphi_res_center_rot / R + (1.0 / R) * dVr_res_dphi );

                // --- Viscous Dissipation Function for the residual field ---
                real S_ij_S_ij_res = S_rr_res*S_rr_res + S_phiphi_res*S_phiphi_res + 2.0 * S_rphi_res*S_rphi_res;
                real div_v_res = S_rr_res + S_phiphi_res;
                real dissipation_res = 2.0 * eta1 * S_ij_S_ij_res + (eta2 - (2.0/3.0) * eta1) * div_v_res*div_v_res;

                total_dissipation += dissipation_res;
            }
        }

        // --- Apply the total calculated cooling term ---
        if(total_dissipation > 0.0) {
            if(Uc(ENG, k, j, i) > total_dissipation * dt) {
                Uc(ENG, k, j, i) -= total_dissipation * dt;
            }
        }
      #endif // NOT ISOTHERMAL
      // =====================================================================


      // --- Other source terms (Beta cooling, relaxation zone) can go here ---
      #ifdef BETACOOLING
        #ifndef ISOTHERMAL
          // Use the captured const variables, do not redeclare them
          real cs2 = h0*h0*pow(R,2*flaringIndex-1.0);
          real tau = beta*pow(R,1.5);
          real Ptarget = cs2*Vc(RHO,k,j,i);
          real Eint = Uc(ENG,k,j,i);
          real Eeq = Ptarget / (gamma - 1.0);
          real factor = 1.0 / (1.0 + dt / tau);
          Uc(ENG,k,j,i) = Eint * factor + Eeq * (1.0 - factor);
        #endif
      #endif

      // === Original IDEFIX damping zone ===
      real lambda = 0.0;
      if (R<wkzMin) {
        lambda = 1.0/(wkzDamping*2.0*M_PI*pow(rmin,1.5))*(1.0 - pow(sin(M_PI*( (R-rmin) / (wkzMin-rmin) )/2.0),2.0));
      }
      if (R>wkzMax) {
        lambda = 1.0/(wkzDamping*2.0*M_PI*pow(rmax,1.5))*pow(sin(M_PI*( (R-wkzMax) / (rmax-wkzMax) )/2.0),2.0);
      }

      #ifdef STOCKHOLM
        if(lambda > 0.0) {
            real Vk = 1.0/sqrt(R);
            bool isFargo = haveFargo; // Make sure isFargo is available

            real rhoTarget = sigma0*pow(R,-sigmaSlope) ;
            real vx2Target = 0.0;
            if(!isFargo) {
              // Use the captured const variables, do not redeclare them
              vx2Target = Vk*sqrt(1.0-(1.0+sigmaSlope-2*flaringIndex)*h0*h0*pow(R,2*flaringIndex)) - omega * R;
            }
            real drho = lambda * (Vc(RHO,k,j,i) - rhoTarget);
            real dvx1 = lambda * Vc(RHO,k,j,i) * Vc(VX1,k,j,i);
            real dvx2 = lambda * Vc(RHO,k,j,i) * (Vc(VX2,k,j,i) - vx2Target);
            real dmx1 = dvx1 + Vc(VX1,k,j,i) * drho;
            real dmx2 = dvx2 + Vc(VX2,k,j,i) * drho;
            //real dmx3 = 0.0; // This variable is unused, but we'll leave it to match original structure
            real deng = 0.5 * (Vc(VX1,k,j,i)*Vc(VX1,k,j,i) + Vc(VX2,k,j,i)*Vc(VX2,k,j,i) + Vc(VX3,k,j,i)*Vc(VX3,k,j,i)) * drho
                      + Vc(RHO,k,j,i) * (Vc(VX1,k,j,i)*dvx1 + Vc(VX2,k,j,i)*dvx2);

            #ifndef ISOTHERMAL
              real P       = Vc(PRS,k,j,i);
              real cs2     = h0*h0*pow(R, 2*flaringIndex - 1.0);
              real Ptarget = cs2*rhoTarget;
              //deng += lambda * (P - Ptarget) / (gamma - 1.0);
            #endif
            Uc(RHO,k,j,i) -= drho * dt;
            Uc(MX1,k,j,i) -= dmx1 * dt;
            Uc(MX2,k,j,i) -= dmx2 * dt;
            #ifndef ISOTHERMAL
            Uc(ENG,k,j,i) -= deng * dt;
            #endif
   
        }
      #endif



      // ===================================================================
      // SCHEME 2: Simpler Rayleigh Friction Damping (Recommended)
      // Enable with -DRAYLEIGH=ON
      // ===================================================================
      #ifdef RAYLEIGH
        if (lambda > 0.0) {
          // This simpler form damps radial and azimuthal momentum towards zero.
          real relaxation_factor = exp(-lambda * dt);

          Uc(MX1, k, j, i) *= relaxation_factor;
          Uc(MX2, k, j, i) *= relaxation_factor;
        }
      #endif // RAYLEIGH


    });
}




// User-defined boundaries
void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray1D<real> x1 = data->x[IDIR];
  real sigmaSlope=sigmaSlopeGlob;
  real omega = omegaGlob;
  real h0 = h0Glob;
  real sigma0 = sigma0Glob;
  real flaringIndex = flaringIndexGlob;

    if(dir==IDIR) {
        int ighost,ibeg,iend;
        if(side == left) {
            ighost = data->beg[IDIR];
            ibeg = 0;
            iend = data->beg[IDIR];
        }
        else if(side==right) {
            ighost = data->end[IDIR]-1;
            ibeg=data->end[IDIR];
            iend=data->np_tot[IDIR];
        }


        idefix_for("UserDefBoundary",
          0, data->np_tot[KDIR],
          0, data->np_tot[JDIR],
          ibeg, iend,
                    KOKKOS_LAMBDA (int k, int j, int i) {
                        real R=x1(i);
                        real Vk = 1.0/sqrt(R);

                        Vc(RHO,k,j,i) = sigma0 * pow(R, -sigmaSlope);

                        // Keplerian + pressure gradient
                        real pressure_correction = 1.0 - (1.0 + sigmaSlope - 2.0 * flaringIndex) * h0 * h0 * pow(R, 2.0 * flaringIndex);
                        if(pressure_correction < 0.0) pressure_correction = 0.0; // Safety check
                        Vc(VX2,k,j,i) = Vk * sqrt(pressure_correction) - omega * R;

                        // Strict zero radial flow
                        Vc(VX1,k,j,i) = 0.0;

                        // Leave vertical velocity untouched in 2D (or set to 0 for symmetry):
                        Vc(VX3,k,j,i) = 0.0;

                        #ifndef ISOTHERMAL
                              real cs2 = h0 * h0 * pow(R, 2 * flaringIndex - 1.0);
                              Vc(PRS, k, j, i) = cs2 * Vc(RHO, k, j, i);
                        #endif
                    });
    }
}

void FargoVelocity(DataBlock &data, IdefixArray2D<real> &Vphi) {
  IdefixArray1D<real> x1 = data.x[IDIR];
  real sigmaSlope=sigmaSlopeGlob;
  real h0=h0Glob;
  real flaringIndex=flaringIndexGlob;
  real omega = omegaGlob;

  idefix_for("FargoVphi",0,data.np_tot[KDIR], 0, data.np_tot[IDIR],
      KOKKOS_LAMBDA (int k, int i) {
        real R = x1(i);
        real Vk = 1.0/sqrt(R);
        real pressure_correction = 1.0-(1.0+sigmaSlope-2*flaringIndex)*h0*h0*pow(R,2*flaringIndex);
        if(pressure_correction < 0.0) pressure_correction = 0.0; // Safety check
        Vphi(k,i) = Vk*sqrt(pressure_correction) - omega * R;
  });
}

// Analyse data to produce an ascii output
void Analysis(DataBlock & data) {
  // Mirror data on Host
  DataBlockHost d(data);

  // Sync it
  d.SyncFromDevice();

  for(int ip=0; ip < data.planetarySystem->nbp ; ip++) {
    // Get the orbital parameters
    real timeStep = data.dt;
    real xp = data.planetarySystem->planet[ip].getXp();
    real yp = data.planetarySystem->planet[ip].getYp();
    real zp = data.planetarySystem->planet[ip].getZp();
    real vxp = data.planetarySystem->planet[ip].getVxp();
    real vyp = data.planetarySystem->planet[ip].getVyp();
    real vzp = data.planetarySystem->planet[ip].getVzp();
    real qp = data.planetarySystem->planet[ip].getMp();
    real time = data.t;

    std::string planetName, tqwkName;
    std::stringstream pName, tName;
    pName << outpathGlob << "/planet" << ip << ".dat";
    tName << outpathGlob << "/tqwk" << ip << ".dat";
    planetName = pName.str();
    tqwkName = tName.str();
    // Write the data in ascii to our file
    if(idfx::prank==0) {
      std::ofstream f;
      f.open(planetName,std::ios::app);
      f.precision(10);
      f << std::scientific << timeStep << "    " << xp << "    " << yp << "    " << zp << "    " << vxp << "    " << vyp << "    " << vzp << "    " << qp << "    " << time << std::endl;
      f.close();
    }

    Force &force = data.planetarySystem->planet[ip].m_force;
    bool isp = true;
    data.planetarySystem->planet[ip].computeForce(data,isp);

    real tq_inner = xp*force.f_inner[1]-yp*force.f_inner[0];
    real tq_outer = xp*force.f_outer[1]-yp*force.f_outer[0];
    real tq_ex_inner = xp*force.f_ex_inner[1]-yp*force.f_ex_inner[0];
    real tq_ex_outer = xp*force.f_ex_outer[1]-yp*force.f_ex_outer[0];
    real wk_inner = vxp*force.f_inner[0]+vyp*force.f_inner[1];
    real wk_outer = vxp*force.f_outer[0]+vyp*force.f_outer[1];
    real wk_ex_inner = vxp*force.f_ex_inner[0]+vyp*force.f_ex_inner[1];
    real wk_ex_outer = vxp*force.f_ex_outer[0]+vyp*force.f_ex_outer[1];

    // Write the data in ascii to our file
    if(idfx::prank==0) {
      std::ofstream ft;
      ft.open(tqwkName,std::ios::app);
      ft.precision(10);
      ft << std::scientific << timeStep << "    " << tq_inner << "    " << tq_outer << "    " << tq_ex_inner << "    " << tq_ex_outer << "    " << wk_inner << "    " << wk_outer << "    " << wk_ex_inner << "    " << wk_ex_outer << "    " << time << std::endl;
      ft.close();
    }
  }
}

// Default constructor
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Set the function for userdefboundary
  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
  data.hydro->EnrollUserSourceTerm(&Damping);

#ifdef VISCOSITY
    data.hydro->viscosity->EnrollViscousDiffusivity(&MyViscosity);
#endif

#ifdef ISOTHERMAL
  data.hydro->EnrollIsoSoundSpeed(&MySoundSpeed);
#endif

#ifndef ISOTHERMAL
  #ifdef THERMALDIFFUSION
    data.hydro->EnrollUserDefThermalDiffusivity(MyThermalDiffusivity);
  #endif
#endif

  if(data.haveFargo) {
    std::cout << "✅ data.haveFargo = true — enrolling FargoVelocity" << std::endl;
    data.fargo->EnrollVelocity(&FargoVelocity);
  } else {
    std::cout << "❌ data.haveFargo = false — not enrolling FargoVelocity" << std::endl;
  }

  if(data.hydro->haveRotation) {
    omegaGlob = data.hydro->OmegaZ;
  } else {
    omegaGlob = 0.0;
  }

  // Enroll the analysis function
  output.EnrollAnalysis(&Analysis);

  // Global parameters
  wkzMinGlob = input.Get<real>("Setup","wkzMin",0);
  wkzMaxGlob = input.Get<real>("Setup","wkzMax",0);
  wkzDampingGlob = input.Get<real>("Setup","wkzDamping",0);
  sgTaperMinGlob = input.GetOrSet<real>("Setup", "sg_taper_min", 0, 0.0);
  sgTaperMaxGlob = input.GetOrSet<real>("Setup", "sg_taper_max", 0, 0.0);
  sigma0Glob = input.Get<real>("Setup","sigma0",0);
  sigmaSlopeGlob = input.Get<real>("Setup","sigmaSlope",0);
  h0Glob = input.Get<real>("Setup","h0",0);
  flaringIndexGlob = input.Get<real>("Setup","flaringIndex",0);
  densityFloorGlob = input.Get<real>("Setup","densityFloor",0);
  masstaperGlob = input.Get<real>("Planet","masstaper",0);
  nuGlob = input.Get<real>("Setup", "nu", 0);
  betaGlob = input.Get<real>("Hydro", "beta", 0);
  gammaGlob = input.Get<real>("Hydro", "gamma", 0);
  coolResidualsGlob = input.Get<bool>("Hydro", "coolResiduals", false);
  chiGlob = input.Get<real>("Hydro", "chi", 0);
  kappa0Glob = input.Get<real>("Hydro", "kappa0", 0);
  kslopeGlob = input.Get<real>("Hydro", "kslope", 0);

  // NEW parameter reads for our viscosity model
  betaViscGlob = input.Get<real>("Hydro", "beta_visc", 0);
  zetaOverEtaGlob = input.Get<real>("Hydro", "zeta_over_eta", 0);

  // Output path
  outpathGlob = input.Get<std::string>("Output", "path", 0);

  // Create planet & torque/work output files in correct path
  for(int ip = 0; ip < data.planetarySystem->nbp; ip++) {
    std::stringstream pName, tName;
    pName << outpathGlob << "/planet" << ip << ".dat";
    tName << outpathGlob << "/tqwk" << ip << ".dat";
    std::string planetName = pName.str();
    std::string tqwkName = tName.str();

    if (!(input.restartRequested)) {
        if(idfx::prank == 0) {
            std::ofstream f(planetName, std::ios::out);
            f.close();
            std::ofstream ft(tqwkName, std::ios::out);
            ft.close();
        }
    }
  }
}

// In setup.cpp, replace the InitFlow function
// Corrected InitFlow function
void Setup::InitFlow(DataBlock &data) {
    // Create a host copy
    DataBlockHost d(data);
    real h0           = h0Glob;
    real flaringIndex = flaringIndexGlob;
    real sigma0       = sigma0Glob;
    real sigmaSlope   = sigmaSlopeGlob;
    real omega        = omegaGlob;
    real densityFloor = densityFloorGlob;

    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {
                real R = d.x[IDIR](i);
                real Vk = 1.0/sqrt(R);

                // Start with the smooth power-law profile
                d.Vc(RHO,k,j,i) = sigma0*pow(R,-sigmaSlope);

                //==========================================================
                // Add a sinusoidal perturbation to the density
                //==========================================================
                // CRITICAL FIX: Use 'data.mygrid' instead of 'd.mygrid'
                real r_min = data.mygrid->xbeg[IDIR];
                real r_max = data.mygrid->xend[IDIR];
                real k_r   = 4.0 * M_PI / (r_max - r_min);
                real amp   = 0.0; // 1% amplitude perturbation

                d.Vc(RHO,k,j,i) *= (1.0 + amp * sin(k_r * (R - r_min)));
                //==========================================================


                #ifndef ISOTHERMAL
                    real cs2 = h0 * h0 * pow(R, 2 * flaringIndex - 1.0);
                    d.Vc(PRS, k, j, i) = cs2 * d.Vc(RHO, k, j, i);
                #endif

                d.Vc(VX1,k,j,i) = 0.0;
                real pressure_correction = 1.0-(1.0+sigmaSlope-2*flaringIndex)*h0*h0*pow(R,2*flaringIndex);
                if(pressure_correction < 0.0) pressure_correction = 0.0;
                d.Vc(VX2,k,j,i) = Vk*sqrt(pressure_correction) - omega * R;
                d.Vc(VX3,k,j,i) = 0.0;

                if(d.Vc(RHO,k,j,i) < densityFloor) {
                  d.Vc(RHO,k,j,i) = densityFloor;
                }
            }
        }
    }

    // Send it all to the device
    d.SyncToDevice();
}
// Destructor (required to avoid linker error)
Setup::~Setup() {
  // Nothing to clean up
}
