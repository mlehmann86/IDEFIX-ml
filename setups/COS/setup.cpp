#include "idefix.hpp"
#include "setup.hpp"
#include "analysis.hpp"
#include "cos.hpp"

Analysis *analysis;

void ConstantKinematicViscosity(DataBlock &data, const real t, IdefixArray3D<real> &eta1, IdefixArray3D<real> &eta2) {
  IdefixArray4D<real> Vc=data.hydro->Vc;
 
  COSParams& params = COSParams::getInstance();
  real nuvisc = HGAS*HGAS*OMEGA/params.Re;

  idefix_for("ConstantKinematicViscosity",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                eta1(k,j,i) = nuvisc*Vc(RHO,k,j,i);
                eta2(k,j,i) = ZERO_F;
              });

}

//Customized radial boundaries, set to equilibrium solution (inviscid)
void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray1D<real> x  = data->x[IDIR];
  IdefixArray1D<real> z  = data->x[KDIR];

    if(dir==IDIR) {
        hydro->boundary->BoundaryFor("UserDefBoundary", dir, side,
          KOKKOS_LAMBDA (int k, int j, int i) {
              Vc(RHO,k,j,i) = DensityEqm(x(i), z(k));
              Vc(PRS,k,j,i) = PressureEqm(x(i), z(k));
              Vc(VX1,k,j,i) = ZERO_F;
              Vc(VX2,k,j,i) = VyEqm(x(i), z(k));
              Vc(VX3,k,j,i) = ZERO_F;
          });
    }
}

void Cooling(Hydro *hydro, const real t, const real dtin) {
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;
  IdefixArray4D<real> Uc = hydro->Uc;
  IdefixArray1D<real> x  = data->x[IDIR];
  IdefixArray1D<real> z  = data->x[KDIR];

  COSParams& params = COSParams::getInstance();

  real tcoolLoc   = params.tcool;
  real gammaLoc   = params.gammaIdeal;

  real dt         = dtin;

  idefix_for("Cooling",
    0, data->np_tot[KDIR],
    0, data->np_tot[JDIR],
    0, data->np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                real Pinit      = PressureEqm(x(i), z(k));
                real rhoinit    = DensityEqm(x(i), z(k));
                real Ptarget    = Vc(RHO,k,j,i)*Pinit/rhoinit;
                Uc(ENG,k,j,i)  += -dt*(Vc(PRS,k,j,i)-Ptarget)/(tcoolLoc*(gammaLoc-1.0));
});
}

void BodyForce(DataBlock &data, const real t, IdefixArray4D<real> &force) {
  idfx::pushRegion("BodyForce");
  IdefixArray1D<real> x = data.x[IDIR];
  IdefixArray1D<real> z = data.x[KDIR];

  COSParams& params = COSParams::getInstance();

  // GPUS cannot capture static variables
  real omegaLoc     = params.omega;
  real shearLoc     = params.shear;

  idefix_for("BodyForce",
              data.beg[KDIR] , data.end[KDIR],
              data.beg[JDIR] , data.end[JDIR],
              data.beg[IDIR] , data.end[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {

                force(IDIR,k,j,i) = -2.0*omegaLoc*shearLoc*x(i);
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

  COSParams& params = COSParams::getInstance();

  params.gammaIdeal = data.hydro->eos->GetGamma();
  params.omega      = input.Get<real>("Hydro","rotation",0)*OMEGA;
  params.shear      = input.Get<real>("Hydro","shearingBox",0)*OMEGA;

  params.etahat     = input.Get<real>("Setup","etahat",0);
  params.NsqInf     = input.Get<real>("Setup","NsqInf",0)*OMEGA*OMEGA;
  params.Nsq0       = input.Get<real>("Setup","Nsq0",0)*OMEGA*OMEGA;
  params.NsqWidth   = input.Get<real>("Setup","NsqWidth",0);
 
  params.tcool      = input.Get<real>("Setup","tcool",0)/OMEGA;
  params.Pe         = input.Get<real>("Setup","Pe",0);

  params.pertamp    = input.Get<real>("Setup","pertamp",0);

  params.Re         = input.Get<real>("Setup","Re",0);

  // Add our userstep to the timeintegrator
  data.gravity->EnrollBodyForce(BodyForce);
  data.hydro->EnrollUserSourceTerm(&Cooling);
  // data.hydro->thermalDiffusion->EnrollThermalDiffusivity(&MyThermalDiffusivity);

  //Enroll customzied boundary conditions
  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);

  //Enroll constant kinematic viscosity
  //data.hydro->viscosity->EnrollViscousDiffusivity(&ConstantKinematicViscosity);

  analysis = new Analysis(input, grid, data, output,std::string("analysis.dat"));
  output.EnrollAnalysis(&AnalysisFunction);
  // Reset analysis if required
  if(!input.restartRequested) {
    analysis->ResetAnalysis();
  }
}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
    // Create a host copy
    DataBlockHost d(data);
    real x, z;
    COSParams& params = COSParams::getInstance();

    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {
                x=d.x[IDIR](i);
                z=d.x[KDIR](k);

                d.Vc(RHO,k,j,i) = DensityEqm(x, z);
                d.Vc(PRS,k,j,i) = PressureEqm(x, z);
                d.Vc(VX1,k,j,i) = ZERO_F;
                // d.Vc(VX1,k,j,i)+= (params.pertamp*CS)*(2.0*idfx::randm()-1.0);
                d.Vc(VX2,k,j,i) = VyEqm(x, z);
                d.Vc(VX2,k,j,i)+= (params.pertamp*CS)*(2.0*idfx::randm()-1.0);
                d.Vc(VX3,k,j,i) = ZERO_F;
            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
}

// Analyse data to produce an output

void MakeAnalysis(DataBlock & data) {
}

