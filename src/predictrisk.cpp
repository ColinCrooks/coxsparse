#include <Rcpp.h>
#include <omp.h>
#include <RcppParallel.h>
#include <RcppInt64>
#include "utils.h"

using namespace Rcpp;
//' predictrisk
//'
//' @description
//'
//' The purpose of this implementation is for fitting a Cox model
//' to data when coxph from the survival package fails due to
//' not enough memory to hold the model and data matrices. The
//' focus is therefore on being memory efficient, which is a
//' slower algorithm than in coxph, but parallelisation is
//' possible to offset this. In this situation compiling the
//' code for the native computer setup would be preferable
//' to providing a standard package binary for multiple systems.
//' The Makevars file therefore contains the options for this.
//'
//' @details
//' A function using the same data structure to calculate individual level linear predictors
//' and survival at the observed times in using the fitted model coefficients and baseline hazards
//' 
//' This function recalculates the individual time level linear predictors,
//' and the survival probability for each person's time point if that covariate
//' level was unchanged throughout the follow up.
//'
//' If time varying covariates are included then the linear predictor
//' for each time point would need to be combined with the incremental 
//' change in cumulative baseline hazard to calculate the cumulative risk.
//'
//' The total number of observations*covariates is allowed to exceed the 
//' maximum integer size in R, so the indexing into covariates
//' needs to use integer64 vectors as defined in the bit64 package,
//' and uses the functions kindly provided by Dirk Eddelbuettel for
//' conversion to C++ vectors (https://github.com/eddelbuettel/RcppInt64).
//' If number of observations and/or ID also exceed the maximum integer size in R
//' then the other vectors will also need changing to integer64 vectors. But
//' this has not currently done to save memory where possible.
//'
//' The data structure is a deconstructed sparse matrix.
//'
//' This uses the same implementation of a Cox proportional hazards model
//' as cox_reg_sparse_parallel
//' OpenMP is used to parallelise the updating of cumulative
//' values and rcppParallel objects are used to make R objects
//' threadsafe.
//'
//' @param beta_in A double vector of starting values for the coefficients
//' of length nvar.
//' @param obs_in An integer vector referencing for each covariate value (sorting as coval) the
//' corresponding unique patient time in the time and outcome vectors. Of the
//' same length as coval. The maximum value is the length of timein and timeout.
//' @param coval_in A double vector of each covariate value sorted first by order 
//' of the covariates then by time then by patient and to be included in model.
//' Of the same longth as obs_in. 
//' \code{coval_in[i] ~ timein_in[obs_in[i]]}, \code{timeout_in[obs_in[i]]}, \code{Outcomes_in[obs_in[i]]},  
//' @param frailty_in A double vector of frailty estimates for each idsorted by id.
//' @param  timein_in An integer vector of the start time for each unique patient 
//' time row, so would be the time that a patient's corresponding
//' covariate value starts. Of the same length as timeout, and outcomes. 
//' Sorted by time out, time in, and patient id
//' @param timeout_in An integer vector of the end time for each unique patient
//' time row, so would be the time that a patient's corresponding outcome
//' occurs. Only used to find maximum time out for survival prediction.
//' Of the same length as timein, timeout and outcomes. Sorted by time out, time in, and patient id
//' @param covstart_in An integer64 (from package bit64) vector of the start row for each covariate in coval 
//' @param covend_in An integer64 (from package bit64) vector of the end row for each covariate in coval
//' @param idn_in An integer vector mapping unique patient IDs sorted by ID to the 
//' corresponding row in observations sorted by time out, time in, and patient id
//' For id = i the corresponding rows in time_in, timeout_in and Outcomes_in 
//' are the rows listed between \code{idn_in[idstart_in[i]]:idn_in[idend_in[i]]} 
//' @param idstart_in An integer vector of the start row for each unique patient ID in idn_in
//' @param idend_in An integer vector of the end row for each unique patient ID in idn_in
//' @param threadn Number of threads to be used - caution as will crash if specify more
//' threads than available memory for copying data for each thread.
//' @return Numeric List with linear predictor and predicted survival.
//'
//' @export
// [[Rcpp::export]]
List predictrisk(
     DoubleVector beta_in,
     IntegerVector obs_in,
     DoubleVector coval_in,
     DoubleVector frailty_in,
     IntegerVector timein_in ,
     IntegerVector timeout_in ,
     NumericVector covstart_in,
     NumericVector covend_in,
     IntegerVector idn_in,
     IntegerVector idstart_in,
     IntegerVector idend_in,
     DoubleVector cumhaz_in,
     int threadn
 )
 {
   Rcpp::Rcout.precision(10);
   omp_set_dynamic(0);
  // Explicitly disable dynamic teams
   omp_set_num_threads(threadn);

     // Vectors from R index begin from 1. Need to convert to 0 index for C++and comparisons

    // Unique times but don't accumulate for time 0 as no events

R_xlen_t maxobs = max(obs_in);
R_xlen_t nvar = covstart_in.length();
R_xlen_t maxid = idstart_in.length(); // Number of unique patients
R_xlen_t ntimes = max(timeout_in);
bool recurrent = maxid > 0;

Rcpp::DoubleVector  zbeta_r(maxobs);
Rcpp::DoubleVector  surv_r(maxobs);

RcppParallel::RVector<double> zbeta(zbeta_r);
RcppParallel::RVector<double> surv(surv_r);

/* Wrap all R objects to make thread safe for read and writing  */
RcppParallel::RVector<double> beta(beta_in);
RcppParallel::RVector<double> frailty(frailty_in);
RcppParallel::RVector<double> cumhaz(cumhaz_in);

RcppParallel::RVector<int> timein(timein_in);
RcppParallel::RVector<int> timeout(timeout_in);

RcppParallel::RVector<double> coval(coval_in);   
RcppParallel::RVector<int> obs(obs_in); 

std::vector<int64_t> covstart = Rcpp::fromInteger64(covstart_in); 
std::vector<int64_t> covend = Rcpp::fromInteger64(covend_in); 

RcppParallel::RVector<int> idn(idn_in);
RcppParallel::RVector<int> idstart(idstart_in);
RcppParallel::RVector<int> idend(idend_in);

double frailty_sum = 0.0;
double frailty_mean = 0.0;
for(R_xlen_t  rowid = 0; rowid < maxid; rowid ++)  frailty_sum += exp(frailty[rowid]);


double min_cumhaz = std::numeric_limits<double>::infinity();

for (R_xlen_t t = 0; t < ntimes; t++) min_cumhaz = (cumhaz[t] > 1e-6 && (cumhaz[t] < min_cumhaz)) ? cumhaz[t] : min_cumhaz;

frailty_mean = safelog(frailty_sum / maxid);

/* Cumulative sums that do not depend on the specific covariate update*/
  for (R_xlen_t  i = 0; i < nvar; i++) // 
  { /* per observation time calculations */
      
      double beta_local =  beta[i] ;
      #pragma omp parallel  default(none)  shared(i, covstart, covend, maxobs, coval, beta_local,  obs, zbeta) 
      {
        #pragma omp for
        for (R_xlen_t  covi = covstart[i] - 1; covi < covend[i]; covi++)
        {
          R_xlen_t  rowobs =  obs[covi] - 1 ;
          double covali =  coval[covi] ;
          zbeta[rowobs] += beta_local * covali ; // each covariate occurs once in an observation time
        }
      }
  }

if (recurrent == 1)
{
  //  Rcout << "summing zbeta with frailty, ";  
  #pragma omp parallel for  default(none) shared(idstart, idend, idn , zbeta, maxid, frailty, frailty_mean) //reduction(+:zbeta[:maxobs])
  for (R_xlen_t  i = 0; i < maxid; i++) // +
    { /* per observation time calculations */
        for (R_xlen_t  idi = idstart[i] - 1; idi < idend[i] ; idi++) // iter over current covariates
      {
        zbeta[idn[idi] - 1] += frailty[i] - frailty_mean ; // should be one frailty per person / observation
      }
    }
}

/* Check zbeta Okay and calculate cumulative sums that do not depend on the specific covariate update*/
#pragma omp parallel  default(none) shared( zbeta, maxobs, timein,surv, cumhaz, min_cumhaz)
{
  #pragma omp for
  for (R_xlen_t  rowobs = 0; rowobs < maxobs; rowobs++)
  {
    R_xlen_t  time_index_entry = timein[rowobs] - 1; // std vectors use unsigned can be negative though for time 0
    
    double tempch = (cumhaz[time_index_entry] == 0) ? min_cumhaz : cumhaz[time_index_entry];
    
    double zbeta_temp = zbeta[rowobs] >22 ? 22 : zbeta[rowobs];
    zbeta_temp = zbeta_temp < -200 ? -200 : zbeta_temp;
    zbeta[rowobs] = zbeta_temp;
    surv[rowobs] = pow(exp(-tempch),exp(zbeta_temp));
  }  
}

return List::create(_["xb"] = zbeta,
                    _["Survival"] = surv);
}
