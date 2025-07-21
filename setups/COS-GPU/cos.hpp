#ifndef COS_HPP_
#define COS_HPP_

//Units
#define HGAS   1.0
#define OMEGA  1.0
#define CS     (HGAS*OMEGA)
#define RHOG   1.0
#define R_MU   1.0 //Gas constant over mu

#include <string>
#include "idefix.hpp"
#include "input.hpp"

extern real gammaIdeal;
extern real omega;
extern real qshear;

extern real etahat;
extern real NsqInf;
extern real Nsq0;
extern real NsqWidth;
extern real tcool;
extern real Pe;
// extern real pertamp;
extern real Re;
// extern real damping_time;
// extern real damping_Ltrans;
// extern real damping_dtrans;
// extern std::string eigen_pert;
// extern real eigen_kz;

KOKKOS_INLINE_FUNCTION
real PressureEqm(real x, real z, real etahat1){
	real P0 = CS*CS*RHOG;

	return P0*exp(-2.0*etahat1*x/HGAS);
}

KOKKOS_INLINE_FUNCTION
real DensityEqm(real x, real z, real etahat1, real NsqInf1, real Nsq01, real NsqWidth1, real gamma1){
	real b, fx1, fx2, fx, result;

  b   = 2.0*etahat1*(1.0-1.0/gamma1);
  fx1 = NsqInf1*(exp(b*x/HGAS)-1.0)/b;
  fx2 = (NsqInf1 - Nsq01)*sqrt(M_PI/2.0)*NsqWidth1*exp(b*b*NsqWidth1*NsqWidth1/2.0);
  fx2*= erf(b*NsqWidth1/sqrt(2.0)) - erf((b*NsqWidth1*NsqWidth1 - x/HGAS)/(sqrt(2.0)*NsqWidth1));
  fx  = fx1 - fx2;

  result = 2.0*etahat1*exp(-2.0*etahat1*(x/HGAS)/gamma1);
  result/= 2.0*etahat1 + fx;

  return RHOG*result;
  }

KOKKOS_INLINE_FUNCTION
real VyEqm(real x, real z, real qshear1, real etahat1, real NsqInf1, real Nsq01, real NsqWidth1, real gamma1) {
	real prs = PressureEqm(x, z, etahat1);
	real rho = DensityEqm(x, z, etahat1, NsqInf1, Nsq01, NsqWidth1, gamma1);
   return qshear1*x - etahat1*(prs/rho)/CS;
  }

//Routines for eigenvalue and IVP solver for initialization. Since initialization is only done on CPU, we don't need to use Kokkos inline functions.
inline double N2(double x) {
    double s = x / HGAS;
    return NsqInf - (NsqInf - Nsq0) * exp(-0.5 * s * s / (NsqWidth * NsqWidth));
}

inline double dN2_dx(double x) {
    double s = x / HGAS;
    double dN2_ds = (s / (NsqWidth * NsqWidth)) * (NsqInf - Nsq0) * exp(-0.5 * s * s / (NsqWidth * NsqWidth));
    return dN2_ds / HGAS;
}

inline double P(double x) {//this is duplicated from the Kokkos function, consider eliminating
    double s = x / HGAS;
    double P0 = CS * CS * RHOG;
    return P0 * exp(-2.0 * etahat * s);
}

inline double rho(double x) {//this is duplicated from the Kokkos function, consider eliminating
    double s = x / HGAS;
    double b = 2.0 * etahat * (1.0 - 1.0 / gammaIdeal);
    double fs1 = NsqInf / b * (exp(b * s) - 1.0);
    double fs2 = sqrt(M_PI / 2.0) * (NsqInf - Nsq0) * NsqWidth * exp(0.5 * b * b * NsqWidth * NsqWidth);
    double fs3 = erf(b * NsqWidth / sqrt(2.0)) - erf((b * NsqWidth * NsqWidth - s) / (sqrt(2.0) * NsqWidth));
    double fs = fs1 - fs2 * fs3;

    double result = 2.0 * etahat * RHOG * exp(-2.0 * etahat * s / gammaIdeal);
    result /= 2.0 * etahat + fs;
    return result;
}

inline double vy0(double x){
    return qshear*x - etahat*P(x)/rho(x)/CS;
}

inline double cs2(double x) {
    return P(x) / rho(x);
}

inline double dlnP_dx(double x) {
    return -2.0 * etahat / HGAS;
}

inline double dP_dx(double x) {
    return P(x) * dlnP_dx(x);
}

inline double gr(double x) {
    return dP_dx(x) / rho(x);
}

inline double dlnrho_dx(double x) {
    double s = x / HGAS;
    double dlnrho_ds = -2.0 * etahat / gammaIdeal - N2(x) * exp(2.0 * etahat * s) * (rho(x) / RHOG) / (2.0 * etahat);
    return dlnrho_ds / HGAS;
}

inline double d2lnrho_dx2(double x) {
    double s = x / HGAS;
    double d2lnrho_ds2 = HGAS * dN2_dx(x) * exp(2.0 * etahat * s) * rho(x) / RHOG;
    d2lnrho_ds2 += 2.0 * etahat * exp(2.0 * etahat * s) * N2(x) * rho(x) / RHOG;
    d2lnrho_ds2 += exp(2.0 * etahat * s) * N2(x) * rho(x) / RHOG * HGAS * dlnrho_dx(x);
    d2lnrho_ds2 *= -1.0 / (2.0 * etahat);
    return d2lnrho_ds2 / (HGAS * HGAS);
}

inline double dlnT_dx(double x) {
    return dlnP_dx(x) - dlnrho_dx(x);
}

inline double d2lnT_dx2(double x) {
    return -d2lnrho_dx2(x);  // Since d2lnP_dx2 = 0
}

inline double LapT_T(double x) {
    return dlnT_dx(x) * dlnT_dx(x) + d2lnT_dx2(x);
}

inline double chi(double x) {
		double kappaT = R_MU * RHOG * HGAS * HGAS * OMEGA / ((gammaIdeal - 1.0) * Pe);;
    return kappaT * (gammaIdeal - 1.0) / (R_MU * rho(x));
}

inline double duy_dx(double x) {
    return (P(x) / rho(x)) / (2.0 * OMEGA) * dlnT_dx(x) * dlnP_dx(x);
}

#endif //COS_HPP_
