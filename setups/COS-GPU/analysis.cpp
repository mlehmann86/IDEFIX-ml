#include "analysis.hpp"
#include "idefix.hpp"
#include "fluid.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

Analysis::Analysis(Input &input, Grid &grid, DataBlock &data, Output &output, std::string filename) {
      this->d = new DataBlockHost(data);
      this->grid = &grid;
      this->filename = filename;
      this->shear = data.hydro->sbS;
      this->precision = 10;
}

/* **************************************************************** */
double Analysis::Average(const int nfields, int fields[])
    /*
     * compute the weighted average: int dphi dz rho *infield/int dphi dz rho
     *
     **************************************************************** */
{
  real outfield = 0;

  for(int k = d->beg[KDIR]; k < d->end[KDIR] ; k++) {
    for(int j = d->beg[JDIR]; j < d->end[JDIR] ; j++) {
      for(int i = d->beg[IDIR]; i < d->end[IDIR] ; i++) {
        real q=1.0;
        for(int n=0 ; n < nfields ; n++) {
          // Substract Keplerian flow if vphi
          if(fields[n]==VX2) {
            q = q*(d->Vc(fields[n],k,j,i)-VyEqm(d->x[IDIR](i), d->x[KDIR](k), qshear, etahat, NsqInf, Nsq0, NsqWidth, gammaIdeal));
          }
          else{
            q = q*d->Vc(fields[n],k,j,i);
          }
        }
        outfield += q;
      }
    }
  }

    // Reduce
#ifdef WITH_MPI
  real reducedValue;
  MPI_Reduce(&outfield, &reducedValue, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  outfield = reducedValue;
#endif

  outfield = outfield / ((double) grid->np_int[IDIR] * grid->np_int[JDIR] * grid->np_int[KDIR]);


  return outfield;
}


/* **************************************************************** */
double Analysis::MaxdV()
    /*
     * find the maximum of sqrt(vx^2 + (vy-vy0)^2 + vz^2)
     * ON HOST
     **************************************************************** */
{
  real vx2, vy2, vz2, dV;
  real dVmax = 0;
  for(int k = d->beg[KDIR]; k < d->end[KDIR] ; k++) {
    for(int j = d->beg[JDIR]; j < d->end[JDIR] ; j++) {
      for(int i = d->beg[IDIR]; i < d->end[IDIR] ; i++) {
        vx2 = d->Vc(VX1,k,j,i)*d->Vc(VX1,k,j,i);
        vy2 = d->Vc(VX2,k,j,i) - VyEqm(d->x[IDIR](i), d->x[KDIR](k), qshear, etahat, NsqInf, Nsq0, NsqWidth, gammaIdeal);
        vy2*= vy2;
        vz2 = d->Vc(VX3,k,j,i)*d->Vc(VX3,k,j,i);
        dV  = sqrt(vx2 + vy2 + vz2);
        dVmax = std::fmax(dV, dVmax);
      }
    }
  }

    // Reduce
#ifdef WITH_MPI
  real dVmaxGlob;
  MPI_Reduce(&dVmax, &dVmaxGlob, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  dVmax = dVmaxGlob;
#endif

  return dVmax;
}

/* **************************************************************** */
double Analysis::AverageAMF()
    /*
     * compute the domain-averaged turbulent angular momentum flux
     *
     **************************************************************** */
{
  real outfield = 0;
  real amf = 0.0;
  for(int k = d->beg[KDIR]; k < d->end[KDIR] ; k++) {
    for(int j = d->beg[JDIR]; j < d->end[JDIR] ; j++) {
      for(int i = d->beg[IDIR]; i < d->end[IDIR] ; i++) {
        amf = d->Vc(VX1,k,j,i)*(d->Vc(VX2,k,j,i)-VyEqm(d->x[IDIR](i), d->x[KDIR](k), qshear, etahat, NsqInf, Nsq0, NsqWidth, gammaIdeal));
        outfield += amf;
      }
    }
  }

    // Reduce
#ifdef WITH_MPI
  real reducedValue;
  MPI_Reduce(&outfield, &reducedValue, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  outfield = reducedValue;
#endif

  outfield = outfield / ((double) grid->np_int[IDIR] * grid->np_int[JDIR] * grid->np_int[KDIR]);

  return outfield;
}


/* **************************************************************** */
void Analysis::WriteField(double data) {
/*
 * Write a global profile to a file
 *
 *
 **************************************************************** */
  if(idfx::prank==0) {
    int col_width = precision + 10;
    this->file << std::scientific << std::setw(col_width) << data;
  }
  return ;
}


void Analysis::ResetAnalysis() {
  GridHost gh(*this->grid);
  gh.SyncFromDevice();
  int col_width = precision + 10;
  if(idfx::prank==0) {
    file.open(filename, std::ios::trunc);
    file << std::setw(col_width) << "t";
    file << std::setw(col_width) << "dVmax";
    file << std::setw(col_width) << "<vx2>";
    file << std::setw(col_width) << "<dvy2>";
    file << std::setw(col_width) << "<vz2>";
    file << std::setw(col_width) << "<amf>";
    // file << std::setw(col_width) << "vy";
    // file << std::setw(col_width) << "vz";
    file << std::endl;
    file.close();
  }
}

void Analysis::PerformAnalysis(DataBlock &data) {
  idfx::pushRegion("Analysis::PerformAnalysis");
  d->SyncFromDevice();
  int fields[3];
  if(idfx::prank==0) {
    file.open(filename, std::ios::app);
    file.precision(precision);
  }

  WriteField(data.t);

  WriteField(MaxdV());

  fields[0] = VX1;
  fields[1] = VX1;
  WriteField(Average(2, fields));

  fields[0] = VX2;
  fields[1] = VX2;
  WriteField(Average(2, fields));

  fields[0] = VX3;
  fields[1] = VX3;
  WriteField(Average(2, fields));

  WriteField(AverageAMF());

  if(idfx::prank==0) {
    file << std::endl;
    file.close();
  }
  idfx::popRegion();
}
