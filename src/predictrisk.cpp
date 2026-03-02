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
//' and survival at the observed event times using the centred fitted model coefficients 
//' and baseline hazards with efton weighting for ties.
//' 
//' This function recalculates the individual time level linear predictors centred on the mean 
//' across all observed times, and the survival probability for each person's time point 
//' if that covariate level was unchanged throughout the follow up which might be meaningless;
//' if time varying covariates are included then the linear predictor
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
//' @param weights_in A double vector of weights to be applied to each unique
//' patient time point when the model was fitted. 
//' Of the same length as timein, timeout and outcomes. 
//' Sorted by time out, time in, and patient id. 
//' @param timeout_in An integer vector of the end time for each unique patient
//' time row, so would be the time that a patient's corresponding outcome
//' occurs. Only used to find maximum time out for survival prediction.
//' @param Outcomes_in An integer vector of 0 (censored) or 1 (outcome) for the 
//' corresponding unique patient time. Of the same length as timein, timeout and 
//' weights. Sorted by time out, time in, and patient id 
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
//' @return Numeric List with centred linear predictor, 
//' centred baseline hazard increments(zero if no events),
//' cumulative hazard, 
//' and predicted survival.
//'
//' @export
// [[Rcpp::export]]
List predictrisk(
     DoubleVector beta_in,
     IntegerVector obs_in,
     DoubleVector coval_in,
     DoubleVector frailty_in,
     IntegerVector timein_in ,
     DoubleVector  weights_in,
     IntegerVector timeout_in ,
     IntegerVector Outcomes_in ,
     NumericVector covstart_in,
     NumericVector covend_in,
     IntegerVector idn_in,
     IntegerVector idstart_in,
     IntegerVector idend_in,
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
   Rcpp::DoubleVector  basehaz_r(ntimes);
   Rcpp::DoubleVector  cumhaz_r(ntimes);

   RcppParallel::RVector<double> zbeta(zbeta_r);
   RcppParallel::RVector<double> surv(surv_r);
   
   /* Wrap all R objects to make thread safe for read and writing  */
   RcppParallel::RVector<double> beta(beta_in);
   RcppParallel::RVector<double> frailty(frailty_in);
   RcppParallel::RVector<double> cumhaz(cumhaz_r);
   RcppParallel::RVector<double> basehaz(basehaz_r);
   
   RcppParallel::RVector<int> timein(timein_in);
   RcppParallel::RVector<int> timeout(timeout_in);
   
   RcppParallel::RVector<double> coval(coval_in);   
   RcppParallel::RVector<int> obs(obs_in); 
   
   RcppParallel::RVector<int> Outcomes(Outcomes_in);
   
   std::vector<int64_t> covstart = Rcpp::fromInteger64(covstart_in); 
   std::vector<int64_t> covend = Rcpp::fromInteger64(covend_in); 
   
   RcppParallel::RVector<int> idn(idn_in);
   RcppParallel::RVector<int> idstart(idstart_in);
   RcppParallel::RVector<int> idend(idend_in);
   double* OutcomeCounts = new double[ntimes]() ; 
   double* wt_totals = new double[ntimes]() ;
   
   RcppParallel::RVector<double> weights(weights_in);
   
   double* denom = new double[ntimes](); // default zero initialisation
   double* efron_wt = new double[ntimes]();

   /* Weights can mean than some events are not counted, so need to recount event totals */  
   for (R_xlen_t  rowobs = 0; rowobs < maxobs ; rowobs++) // iter over current covariates
   {
     R_xlen_t  time_index_exit = timeout[rowobs] - 1;
     if ( Outcomes[rowobs]*weights[rowobs] > 0 )
       OutcomeCounts[time_index_exit] = OutcomeCounts[time_index_exit] + 1;
   }
   
   int64_t eventtimesN = 0;
   for (R_xlen_t  t = 0; t < ntimes ; t++) // iter over times
     if( OutcomeCounts[t] > 0)  eventtimesN = eventtimesN + 1;
     
   double* OutcomeTotals = new double[eventtimesN]();
   int64_t* OutcomeTotalTimes = new int64_t[eventtimesN]();
     
     R_xlen_t i = 0;
     for (R_xlen_t  t = 0; t < ntimes ; t++) // iter over times
     {
       if (OutcomeCounts[t] > 0) {
         OutcomeTotals[i] = OutcomeCounts[t];
         OutcomeTotalTimes[i] = t + 1;
         i++;
       }
       wt_totals[t] = 0.0;
     }
     
     delete[] OutcomeCounts;
     
#pragma omp parallel default(none) reduction(+:wt_totals[:ntimes]) shared(timeout,  weights,  Outcomes ,ntimes, maxobs)
{  
#pragma omp for
  for (R_xlen_t  rowobs = 0; rowobs < maxobs ; rowobs++) // iter over current covariates
  {
    R_xlen_t  time_index_exit = timeout[rowobs] - 1;
    if (Outcomes[rowobs]*weights[rowobs] > 0 )  wt_totals[time_index_exit] += weights[rowobs];
  }
}
//Rcout << "averaging deaths at each time point, ";
//for(R_xlen_t  r = eventtimesN -1 ; r >=0 ; r--) wt_average[OutcomeTotalTimes[r] - 1] = (OutcomeTotals[r]>0 ? wt_totals[OutcomeTotalTimes[r] - 1]/static_cast<double>(OutcomeTotals[r]) : 0.0);


double frailty_sum = 0.0;
double frailty_mean = 0.0;
for(R_xlen_t  rowid = 0; rowid < maxid; rowid ++)  frailty_sum += exp(frailty[rowid]);


double min_cumhaz = std::numeric_limits<double>::infinity();

for (R_xlen_t t = 0; t < ntimes; t++) min_cumhaz = (cumhaz[t] > 1e-6 && (cumhaz[t] < min_cumhaz)) ? cumhaz[t] : min_cumhaz;

frailty_mean = safelog(frailty_sum / maxid);
double zbeta_mean = 0.0;

/* Cumulative sums that do not depend on the specific covariate update*/
for (R_xlen_t  i = 0; i < nvar; i++) // 
{ /* per observation time calculations */

double beta_local =  beta[i] ;
#pragma omp parallel  default(none) shared(i, covstart, covend, maxobs, coval, beta_local,  obs, zbeta) 
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

#pragma omp parallel for  default(none) reduction(+:zbeta_mean) shared( zbeta, maxobs)
for (R_xlen_t  rowobs = 0; rowobs < maxobs; rowobs++) 
{
  zbeta_mean += zbeta[rowobs];
}

zbeta_mean /= maxobs;

#pragma omp parallel for  default(none) shared( zbeta, maxobs, zbeta_mean)
for (R_xlen_t  rowobs = 0; rowobs < maxobs; rowobs++) 
{
  zbeta[rowobs] -= zbeta_mean;
}


#pragma omp parallel  default(none) reduction(+:denom[:ntimes], efron_wt[:ntimes])  shared(timein, timeout, zbeta, weights,  Outcomes , ntimes,maxobs)
{
#pragma omp for
  for (R_xlen_t  rowobs = 0; rowobs < maxobs; rowobs++)
  {
    R_xlen_t  time_index_entry = timein[rowobs] - 1;  // Time starts at zero but no events at this time to calculate sums. Lowest value of -1 but not used to reference into any vectors
    R_xlen_t  time_index_exit = timeout[rowobs] - 1;
    
    double zbeta_temp = zbeta[rowobs] >22 ? 22 : zbeta[rowobs];
    zbeta_temp = zbeta_temp < -200 ? -200 : zbeta_temp;
    double risk = exp(zbeta_temp) * weights[rowobs];
    zbeta[rowobs] = zbeta_temp;
    //cumumlative sums for all patients
    for (R_xlen_t  r = time_index_exit;  r > time_index_entry ; r--)
      denom[r] += risk;
    if (Outcomes[rowobs]*weights[rowobs] > 0 )
    {
      /*cumumlative sums for event patients */
      efron_wt[time_index_exit] += risk;
    }
    
  }  
}



R_xlen_t  timesN =  eventtimesN -1;
for (R_xlen_t  ir = 0; ir < ntimes; ir++)
  basehaz[ir] = 0.0;
#pragma omp parallel for default(none) shared(timesN,wt_totals, OutcomeTotals, OutcomeTotalTimes, denom,efron_wt, basehaz)
for (R_xlen_t  r =  timesN - 1; r >= 0; r--)
{
  double basehaz_private = 0.0;
  R_xlen_t  time = OutcomeTotalTimes[r] - 1;
 // basehaz_private += (static_cast<double>(OutcomeTotals[r]))/denom[time];
  for (R_xlen_t  k = 0; k < OutcomeTotals[r]; k++)
  {
    double temp = 1/(denom[time] - (efron_wt[time] * (double)k
                                      / (double)OutcomeTotals[r] ));
    basehaz_private += temp/(double)OutcomeTotals[r] ; /* sum(denom) adjusted for tied deaths*/
  }
#pragma omp atomic write
  basehaz[time] = basehaz_private*wt_totals[time]; // should be thread safe as time unique per thread
}

Rcout << " Calculating cumulative baseline hazard..." ;

cumhaz[0] = basehaz[0] ;
for (R_xlen_t  t = 1; t < ntimes; t++)
    cumhaz[t] = cumhaz[t-1] +  basehaz[t];


/* Check zbeta Okay and calculate cumulative sums that do not depend on the specific covariate update*/
#pragma omp parallel  default(none) shared( zbeta, maxobs, timein,surv, cumhaz, min_cumhaz)
{
#pragma omp for
  for (R_xlen_t  rowobs = 0; rowobs < maxobs; rowobs++)
  {
    R_xlen_t  time_index_entry = timein[rowobs] - 1; // std vectors use unsigned can be negative though for time 0
    
    double tempch = (cumhaz[time_index_entry] == 0) ? min_cumhaz : cumhaz[time_index_entry];
    surv[rowobs] = exp(-tempch * exp(zbeta[rowobs]));
  }  
}


delete[] denom;
delete[] efron_wt;
delete[] wt_totals;  

return List::create(_["xb_centred"] = zbeta,
                    _["Survival"] = surv,
                    _["BaseHaz_centred"] = basehaz,
                    _["CumHaz_centred"] = cumhaz);
 }
 