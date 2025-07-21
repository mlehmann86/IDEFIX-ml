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

class COSParams {
	public:
 		real gammaIdeal;
		real omega;
		real shear;

		real etahat;
		real NsqInf;
		real Nsq0;
		real NsqWidth;
		real tcool;
		real Pe;
		real pertamp;
		real Re;

	static COSParams& getInstance() {
        static COSParams instance;
        return instance;
    }

    // Delete copy constructor and assignment operator to ensure single instance
    COSParams(const COSParams&) = delete;
    COSParams& operator=(const COSParams&) = delete;

	private:
    COSParams() = default;  // Private constructor

  };

KOKKOS_INLINE_FUNCTION
real PressureEqm(real x, real z){
	real P0 = CS*CS*RHOG;
	const COSParams& params = COSParams::getInstance();
	real etahat1 = params.etahat;

	return P0*exp(-2.0*etahat1*x/HGAS); 
}

KOKKOS_INLINE_FUNCTION
real DensityEqm(real x, real z){
	real b, fx1, fx2, fx, result;
	const COSParams& params = COSParams::getInstance();
	real etahat1 	= params.etahat;
	real NsqInf1 	= params.NsqInf;
	real Nsq01   	= params.Nsq0;
	real NsqWidth1 = params.NsqWidth;
	real gamma1		= params.gammaIdeal;

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
real VyEqm(real x, real z) {
	const COSParams& params = COSParams::getInstance();
	real shear1    = params.shear;
	real etahat1 	= params.etahat;
	real NsqInf1 	= params.NsqInf;
	real Nsq01   	= params.Nsq0;
	real NsqWidth1 = params.NsqWidth;
	real gamma1		= params.gammaIdeal;

   return shear1*x - etahat1*PressureEqm(x, z)/DensityEqm(x, z)/CS;
  }

#endif //COS_HPP_
