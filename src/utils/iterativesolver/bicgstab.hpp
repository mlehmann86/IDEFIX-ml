// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef UTILS_ITERATIVESOLVER_BICGSTAB_HPP_
#define UTILS_ITERATIVESOLVER_BICGSTAB_HPP_
#include <vector>
#include "idefix.hpp"
#include "vector.hpp"
#include "iterativesolver.hpp"


// The bicgstab derives from the iterativesolver class
template <class T>
class Bicgstab : public IterativeSolver<T> {
 public:
  Bicgstab(T &op, real error, int maxIter,
           std::array<int,3> ntot, std::array<int,3> beg, std::array<int,3> end);

  int Solve(IdefixArray3D<real> &guess, IdefixArray3D<real> &rhs);

  void PerformIter();
  void InitSolver();
  void ShowConfig();

 private:
  real rho;           // BICSTAB parameter
  real alpha;         // BICGSTAB parameter
  real omega;         // BICGSTAB parameter


  IdefixArray3D<real> res0; // Reference (initial) residual
  IdefixArray3D<real> dir; // Search direction for gradient descent
  IdefixArray3D<real> work1; // work array
  IdefixArray3D<real> work2; // work array
  IdefixArray3D<real> work3; // work array
};

template <class T>
Bicgstab<T>::Bicgstab(T &op, real error, int maxiter,
            std::array<int,3> ntot, std::array<int,3> beg, std::array<int,3> end) :
            IterativeSolver<T>(op, error, maxiter, ntot, beg, end) {
  // BICGSTAB scalars initialisation
  this->rho = 1.0;
  this->alpha = 1.0;
  this->omega = 1.0;



  this->dir = IdefixArray3D<real> ("Direction", this->ntot[KDIR],
                                                this->ntot[JDIR],
                                                this->ntot[IDIR]);

  this->res0 = IdefixArray3D<real> ("InitialResidual", this->ntot[KDIR],
                                                        this->ntot[JDIR],
                                                        this->ntot[IDIR]);

  this->work1 = IdefixArray3D<real> ("WorkingArray1", this->ntot[KDIR],
                                                      this->ntot[JDIR],
                                                      this->ntot[IDIR]);

  this->work2 = IdefixArray3D<real> ("WorkingArray2", this->ntot[KDIR],
                                                      this->ntot[JDIR],
                                                      this->ntot[IDIR]);

  this->work3 = IdefixArray3D<real> ("WorkingArray3", this->ntot[KDIR],
                                                      this->ntot[JDIR],
                                                      this->ntot[IDIR]);
}

template <class T>
int Bicgstab<T>::Solve(IdefixArray3D<real> &guess, IdefixArray3D<real> &rhs) {
  idfx::pushRegion("Bicgstab::Solve");
  this->solution = guess;
  this->rhs = rhs;

  // Re-initialise convStatus
  this->convStatus = false;

  this->InitSolver();

  int n = 0;
  while(this->convStatus != true && n < this->maxiter) {
    this->PerformIter();
    if(this->restart) {
      this->restart=false;
      // Resetting parameters
      this->rho = 1.0;
      this->alpha = 1.0;
      this->omega = 1.0;
      n = -1;
      idfx::popRegion();
      return(n);
    }
    n++;
  }

  if(n == this->maxiter) {
    idfx::cout << "Bicgstab:: Reached max iter." << std::endl;
    IDEFIX_WARNING("Bicgstab:: Failed to converge before reaching max iter."
                    "You should consider to use a preconditionner.");
  }

  idfx::popRegion();
  return(n);
}

template <class T>
void Bicgstab<T>::InitSolver() {
  idfx::pushRegion("Bicgstab::InitSolver");
  // Residual initialisation
  this->SetRes();

  Kokkos::deep_copy(this->res0, this->res); // (Re)setting reference residual
  Kokkos::deep_copy(this->dir, this->res); // (Re)setting initial searching direction
  this->linearOperator(this->dir, this->work1); // (Re)setting associated laplacian

  // // Resetting parameters
  // this->rho = 1.0;
  // this->alpha = 1.0;
  // this->omega = 1.0;

  idfx::popRegion();
}

// Replace the PerformIter function in bicgstab.hpp with this debug version
template <class T>
void Bicgstab<T>::PerformIter() {
  idfx::pushRegion("Bicgstab::PerformIter");

  // Loading needed attributes
  IdefixArray3D<real> solution = this->solution;
  IdefixArray3D<real> res = this->res;
  IdefixArray3D<real> dir = this->dir;
  IdefixArray3D<real> res0 = this->res0;

  IdefixArray3D<real> v = this->work1;
  IdefixArray3D<real> s = this->work2;
  IdefixArray3D<real> t = this->work3;
  real omega;
  real &alpha = this->alpha;
  real &rhoOld = this->rho;
  real &omegaOld = this->omega;

  int ibeg = this->beg[IDIR]; int iend = this->end[IDIR];
  int jbeg = this->beg[JDIR]; int jend = this->end[JDIR];
  int kbeg = this->beg[KDIR]; int kend = this->end[KDIR];

  // ***** Step 1.
  real rho = this->ComputeDotProduct(res0, res);
  #ifdef DEBUG_BICGSTAB
    if(idfx::prank == 0) idfx::cout << "Bicgstab Step 1: rho = " << rho << std::endl;
  #endif
  if(std::isnan(rho)) { this->restart = true; idfx::popRegion(); return; }

  // ***** Step 2.
  real beta = rho / rhoOld * alpha / omegaOld;
  #ifdef DEBUG_BICGSTAB
    if(idfx::prank == 0) idfx::cout << "Bicgstab Step 2: beta = " << beta << std::endl;
  #endif
  if(std::isnan(beta)) { this->restart = true; idfx::popRegion(); return; }

  // ***** Step 3.
  idefix_for("UpdateDir", kbeg, kend, jbeg, jend, ibeg, iend,
    KOKKOS_LAMBDA (int k, int j, int i) {
      dir(k,j,i) = res(k,j,i) + beta * (dir(k,j,i) - omegaOld * v(k,j,i));
    });

  // ***** Step 4.
  IdefixArray3D<real> temp_dir = this->work3;
  Kokkos::deep_copy(temp_dir, dir);
  if(this->linearOperator.havePreconditioner) {
      IdefixArray3D<real> P = this->linearOperator.precond;
      idefix_for("PrecondDir", kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA (int k, int j, int i) { temp_dir(k,j,i) = dir(k,j,i) / P(k,j,i); });
  }
  this->linearOperator(temp_dir, v);

  // ******* Step 5.
  real dot_res0_v = this->ComputeDotProduct(res0, v);
  #ifdef DEBUG_BICGSTAB
    if(idfx::prank == 0) idfx::cout << "Bicgstab Step 5: Denominator for alpha (dot_res0_v) = " << dot_res0_v << std::endl;
  #endif
  alpha = rho / dot_res0_v;
  if(std::isnan(alpha)) { this->restart = true; idfx::popRegion(); return; }

  // *********** Step 6.
  idefix_for("FirstUpdatePot", kbeg, kend, jbeg, jend, ibeg, iend,
    KOKKOS_LAMBDA (int k, int j, int i) {
      solution(k,j,i) = solution(k,j,i) + alpha * temp_dir(k,j,i);
    });

  // ********** Step.7.
  this->SetRes();
  this->TestErrorL2();

  if(this->convStatus == false) {
    // ***************** Step. 8.
    idefix_for("FillIntermediateDir", kbeg, kend, jbeg, jend, ibeg, iend,
      KOKKOS_LAMBDA (int k, int j, int i) { s(k,j,i) = res(k,j,i); });

    // ************** Step 9.
    IdefixArray3D<real> temp_s = this->work3;
    Kokkos::deep_copy(temp_s, s);
    if(this->linearOperator.havePreconditioner) {
        IdefixArray3D<real> P = this->linearOperator.precond;
        idefix_for("PrecondS", kbeg, kend, jbeg, jend, ibeg, iend,
          KOKKOS_LAMBDA (int k, int j, int i) { temp_s(k,j,i) = s(k,j,i) / P(k,j,i); });
    }
    this->linearOperator(temp_s, t);

    // ************* Step 10.
    real dot_t_s = this->ComputeDotProduct(t, s);
    real dot_t_t = this->ComputeDotProduct(t, t);
    #ifdef DEBUG_BICGSTAB
      if(idfx::prank == 0) idfx::cout << "Bicgstab Step 10: Denominator for omega (dot_t_t) = " << dot_t_t << std::endl;
    #endif
    omega = dot_t_s / dot_t_t;
    if(std::isnan(omega)) { this->restart = true; idfx::popRegion(); return; }

    // ************ Step 11.
    idefix_for("SecondUpdatePot", kbeg, kend, jbeg, jend, ibeg, iend,
      KOKKOS_LAMBDA (int k, int j, int i) {
        solution(k,j,i) = solution(k,j,i) + omega * temp_s(k,j,i);
      });

    // *********** Step 12.
    this->SetRes();
    this->TestErrorL2();

    if(this->convStatus == false) {
      idefix_for("UpdateRes", kbeg, kend, jbeg, jend, ibeg, iend,
        KOKKOS_LAMBDA (int k, int j, int i) {
          res(k,j,i) = res(k,j,i) - omega * t(k,j,i);
        });
      rhoOld = rho;
      omegaOld = omega;
    }
  }
  idfx::popRegion();
}


template <class T>
void Bicgstab<T>::ShowConfig() {
  idfx::pushRegion("Bicgstab::ShowConfig");
  idfx::cout << "Bicgstab: TargetError: " << this->targetError << std::endl;
  idfx::cout << "Bicgstab: Maximum iterations: " << this->maxiter << std::endl;
  idfx::popRegion();
  return;
}


#endif // UTILS_ITERATIVESOLVER_BICGSTAB_HPP_
