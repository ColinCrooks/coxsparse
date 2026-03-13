#include <Rcpp.h>
#include <omp.h>
#include <RcppParallel.h>
#include <RcppInt64>
#include "utils.h"
#include <algorithm>

using namespace Rcpp;
//' cox_reg_sparse_parallel_TV
//' 
//' @description
//' Implementation of a Cox proportional hazards model using 
//' a sparse data structure. The model is fitted with cyclical 
//' coordinate descent (after Mittal et al (2013).
//' OpenMP is used to parallelise the updating of cumulative 
//' values and rcppParallel objects are used to make R objects
//' threadsafe.
//' 
//' @details
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
//' The total number of observations is allowed to exceed the 
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
//' A function using the same data structure to calculate profile
//' confidence intervals with a crude search pattern is provided.
//' 
//' @param modeldata A list in R of vectors within which to retun the model output.
//' Needs to follow this naming convention of the lists: 
//' * Beta - double vector of length of covariates to be filled with fitted coefficients.
//' * Frailty - double vector of length of frailty terms (e.g. number of unique patients) 
//' to bw filled with the fitted frailty terms on linear predictor scale 
//'   (w in xb + Zw). Exponentiate for the relative scale. No centring applied.
//' * basehaz - double vector of length of max(timeout) for baseline hazard values 
//' for each unique observed time. calculated with the fitted coefficients and Efron weights.
//' * cumhaz - double vector of length of max(timeout) for cumulative hazard values calculated 
//' from the baseline hazard values.
//' * ModelSummary - double vector of length 8 to contain individual values:
//' ** loglik - log likelihood of the model without gamma penalty
//' ** lik_correction - gamma correction for the loglikelihood for including frailty terms
//' ** loglik + lik_correction - the total log likelihood of the model
//' ** theta - the value of theta used in the model
//' ** outer_iter - the number of outer iterations performed
//' ** final convergence of inner loop for covariates abs(1-newlk/loglik)
//' ** final convergence of outer loop for theta
//' ** frailty_mean - the mean of the frailty terms so they can be centred log(mean(exp(frailty)))
//' @param obs_in An integer vector referencing for each covariate value (sorting as coval) the
//' corresponding unique patient time in the time and outcome vectors. Of the
//' same length as coval. The maximum value is the length of timein and timeout.
//' @param coval_in A double vector of each covariate value sorted first by order 
//' of the covariates then by patient then by time and to be included in model.
//' Of the same longth as obs_in. 
//' \code{coval_in[i] ~ timein_in[obs_in[i]]}, \code{timeout_in[obs_in[i]]}, 
//' \code{Outcomes_in[obs_in[i]]},  
//' @param weights_in A double vector of weights to be applied to each unique
//' patient time point. Of the same length as timein, timeout and outcomes. 
//' Sorted by patient id, time out, and time in. 
//' @param  timein_in An integer vector of the start time for each unique patient 
//' time row, so would be the time that a patient's corresponding
//' covariate value starts. Of the same length as weights, timeout, and outcomes. 
//' Sorted by patient id, time out, and time in.
//' @param timeout_in An integer vector of the end time for each unique patient
//' time row, so would be the time that a patient's corresponding outcome
//' occurs. Of the same length as weights, timein, timeout and outcomes. Sorted 
//' by patient id, time out, and time in.
//' @param Outcomes_in An integer vector of 0 (censored) or 1 (outcome) for the 
//' corresponding unique patient time. Of the same length as timein, timeout and 
//' weights. Sorted by patient id, time out, and time in. 
//' @param covstart_in An integer64 (from package bit64) vector of the start row 
//' for each covariate in coval Uses 
//' @param covend_in An integer64 (from package bit64) vector of the end row for 
//' each covariate in coval
//' @param id_in An integer vector with unique patient IDs sorted as the 
//' corresponding rows in time_in, timeout_in and Outcomes_in 
//' @param idn_in An integer vector mapping unique patient IDs sorted by ID to the 
//' corresponding row in observations sorted by patient id, time out, and time in.
//' For id = i the corresponding rows in time_in, timeout_in and Outcomes_in 
//' are the rows listed between \code{idn_in[idstart_in[i]]:idn_in[idend_in[i]]} 
//' @param idstart_in An integer vector of the start row for each unique patient ID 
//' in idn_in
//' @param idend_in An integer vector of the end row for each unique patient ID 
//' in idn_in
//' @param bspl_in A matrix of doubles with containing the spline basis 
//' (ncol = number of betas used, nrow = ntimes at which splines calculated (lookback))
//' First row should be for time zero with weight 1 on 1st beta
//' @param tvbeta_spl_in An integer64 vector for indexing which covariates are to be 
//' converted to time varying with the spline basis
//' @param lambda Penalty weight to include for ridge regression: -log(sqrt(lambda)) * nvar
//' @param theta_in An input starting value for theta or can be set to zero.
//' @param MSTEP_MAX_ITER Maximum number of iterations
//' @param MAX_EPS Threshold for maximum step change in liklihood for convergence. 
//' @param threadn Number of threads to be used - caution as will crash if specify more 
//' threads than available memory for copying data for each thread.
//' @return Void: see the model data input list for the output.
//' @export
// [[Rcpp::export]]
void cox_reg_sparse_parallel_TV( List modeldata, 
                              IntegerVector obs_in,
                              DoubleVector  coval_in,
                              DoubleVector  weights_in,
                              IntegerVector timein_in ,
                              IntegerVector timeout_in ,
                              IntegerVector Outcomes_in ,
                              NumericVector covstart_in,
                              NumericVector covend_in,
                              IntegerVector id_in,
                              IntegerVector idn_in,
                              IntegerVector idstart_in,
                              IntegerVector idend_in,
                              NumericMatrix bspl_in,
                              IntegerVector tvbeta_spl_in,
                              double lambda,
                              double theta_in ,
                              int MSTEP_MAX_ITER,
                              double MAX_EPS,
                              long unsigned int threadn){
   omp_set_dynamic(0);     // Explicitly disable dynamic teams
   omp_set_num_threads(threadn); // Use 8 threads for all consecutive parallel regions

   Rcpp::Rcout.precision(10);
   
   //TODO for each rowobs exposure find other rows that are post exposure in window and update with splines and spline betas
   
   //TODO for timein to timeout  minus timeout exposure from denom and add in time adjusted exposure with splines and spline betas on log scale
   // maybe need denom 2 without time adjusted exposure so don't need to remember last adjustment
   // Then shrink data to remove 
   
   //Weights need to be calculated for every time event or exposure changes or event (ie every timepoint in study for every person )
   
   
   // R objects size uses signed integer Maximum signed integer 2^31 on 32 bit or  R_xlen_t 2^48 on 64 bit so use R_xlen_t as indices. cpp uses size_T
   
   
   // Vectors from R index begin from 1. Need to convert to 0 index for C++ and comparisons
   //Mittal, S., Madigan, D., Burd, R. S., & Suchard, M. a. (2013). High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis. Biostatistics (Oxford, England), 1–15. doi:10.1093/biostatistics/kxt043

   int lookback = bspl_in.nrow();
   int nspl = bspl_in.ncol();

   RcppParallel::RVector<int> tvbeta_spl(tvbeta_spl_in);
   

   R_xlen_t  ntimes = Rcpp::max(timeout_in);  // Unique times but don't accumulate for time 0 as no events
   R_xlen_t  maxobs = Rcpp::max(obs_in);
   R_xlen_t  nvar = covstart_in.length();// +
   R_xlen_t  maxid = idstart_in.length(); // Number of unique patients
   maxid = 0; // TODO disable whilst developing tv
   bool recurrent = maxid > 0; 
   int nvartv = 0;
   for (int nv = 0; nv < nvar; nv++ ) nvartv += tvbeta_spl[nv];
   int nbeta_spl = nvartv*nspl;
   
   R_xlen_t  nallvar = nvar + maxid + nbeta_spl - nvartv; // -1 to avoid double counting tv beta in nvar and nbeta_spl
   
   
   
   double newlk = 0.0;
   double loglik = 0.0;
   double frailty_sum = 0.0;
   double lik_correction = 0.0;
   double d2_sum_private = 0.0;
   double newlk_private = 0.0;
   
   double theta = theta_in;
   double nu = theta_in == 0 ? 0 : 1.0/theta_in;
   double frailty_mean = 0;
   double frailty_penalty = 0.0;
   

   //   Rcout << "Allocating vectors, ";
   int64_t * frailty_group_events = new int64_t [maxid](); // Count of events for each patient (for gamma penalty weight)   
   double* theta_history = new double[MSTEP_MAX_ITER]();
   double* thetalkl_history = new double[MSTEP_MAX_ITER]{-std::numeric_limits<double>::infinity()};
   double* denom = new double[ntimes](); // default zero initialisation
   double* efron_wt = new double[ntimes]();
   double* denom_private = new double[ntimes]();
   double* efron_wt_private = new double[ntimes]();
   double* zbeta = new double[maxobs]();

   double* derivMatrix = new double[ntimes*4]();
   double* step = new double[nallvar]{1.0};
   double* gdiagbeta = new double[nvar + nbeta_spl - nvartv]();
   double* gdiagfrail = new double[maxid]();

   double* OutcomeCounts = new double[ntimes]();
   double* Wts_allEventtimes = new double[ntimes](); 

   /* Wrap all R objects to make thread safe for read and writing  */
   
 //  Rcout << "pointing to R vectors in list, ";
   Rcpp::DoubleVector beta_in = modeldata["Beta"];
   Rcpp::DoubleVector frailty_in = modeldata["Frailty"];
   Rcpp::DoubleVector basehaz_in = modeldata["basehaz"];
   Rcpp::DoubleVector cumhaz_in = modeldata["cumhaz"];
   Rcpp::DoubleVector ModelSummary_in =modeldata["ModelSummary"];

 //  Rcout << "wrapping R vectors, ";

   RcppParallel::RVector<double> weights(weights_in);
   
   RcppParallel::RVector<int> Outcomes(Outcomes_in);
 //  RcppParallel::RVector<int> OutcomeTotals(OutcomeTotals_in);
 //  RcppParallel::RVector<int> OutcomeTotalTimes(OutcomeTotalTimes_in);
   
   RcppParallel::RVector<int> timein(timein_in);
   RcppParallel::RVector<int> timeout(timeout_in);
   
   RcppParallel::RVector<int> obs(obs_in);    
   RcppParallel::RVector<double> coval(coval_in);   
   
   std::vector<int64_t> covstart = Rcpp::fromInteger64(covstart_in); 
   std::vector<int64_t> covend =Rcpp::fromInteger64(covend_in); 
   

   
   RcppParallel::RVector<int>  id(id_in);
   RcppParallel::RVector<int>  idn(idn_in);
   RcppParallel::RVector<int>  idstart(idstart_in);
   RcppParallel::RVector<int>  idend(idend_in);
   
   //vectors for returning calculated results to avoid copying into list for return.
   //To be zeroed before use
//   Rcout << "wrapping R vectors from list, ";
    RcppParallel::RVector<double> basehaz(basehaz_in);
    RcppParallel::RVector<double> cumhaz(cumhaz_in);
    RcppParallel::RVector<double> beta(beta_in);
    RcppParallel::RVector<double> frailty(frailty_in);
    RcppParallel::RVector<double> ModelSummary(ModelSummary_in);
    RcppParallel::RMatrix<double> bspl(bspl_in);

    // for(int64_t itime_diff =0;itime_diff  <lookback;itime_diff ++) 
    // {
    //   for(int64_t bs = 0; bs < nbeta_spl; bs ++)
    //        Rcout << bspl(itime_diff,bs) << ", ";
    //   Rcout << std::endl;
    // }
    // 

  // Objects for TV variables  
  struct ExposureRecord {
    R_xlen_t time;
    double value;
    };
  std::vector<std::vector<ExposureRecord>> tv_data(nvartv);

  Rcout << "MSTEP_MAX_ITER : " << MSTEP_MAX_ITER << " maxobs : " << maxobs <<" ntimes : " << ntimes << " maxid :  " << maxid << " nvar : " << nvar << "\n tv variables : " <<nvartv  << " betas for tv variables : " << nbeta_spl << " nallvar : " << nallvar  << std::endl;    
  int iter_theta = 0;
  double inner_EPS = 1e-5;
  int done = 0;

/* Weights can mean than some events are not counted, so need to recount event totals */  
#pragma omp parallel default(none) reduction(+:OutcomeCounts[:ntimes],Wts_allEventtimes[:ntimes]) shared(timeout,  weights,  Outcomes ,ntimes, maxobs)
{  
#pragma omp for
  for (R_xlen_t  rowobs = 0; rowobs < maxobs ; rowobs++) // iter over current covariates
  {
    R_xlen_t  time_index_exit = timeout[rowobs] - 1;
    if ( Outcomes[rowobs]*weights[rowobs] > 0 ) 
    {
      OutcomeCounts[time_index_exit] += 1;
      Wts_allEventtimes[time_index_exit] += weights[rowobs];
    }
  }
}

// Reduce outcome total counts to a vector at just the outcome times
int64_t eventtimesN = 0;
for (R_xlen_t  t = 0; t < ntimes ; t++) // iter over times
{
  if( OutcomeCounts[t] > 0)  eventtimesN = eventtimesN + 1;
}
double* OutcomeTotals = new double[eventtimesN]();
int64_t* OutcomeTotalTimes = new int64_t[eventtimesN]();
double* wt_average = new double[eventtimesN]();
double* wt_totals = new double[eventtimesN]();

R_xlen_t i = 0;
for (R_xlen_t  t = 0; t < ntimes ; t++) // iter over times
{
  if (OutcomeCounts[t] > 0) {
    OutcomeTotals[i] = OutcomeCounts[t];
    wt_totals[i] = Wts_allEventtimes[t];
    OutcomeTotalTimes[i] = t + 1;
    i++;
  }
  
}

delete[] OutcomeCounts;
delete[] Wts_allEventtimes;
//for (R_xlen_t r = 0; r < eventtimesN; r++) Rcout << OutcomeTotals[r] <<", "<< OutcomeTotalTimes[r] <<std::endl;
Rcout << "averaging deaths at each time point, " << std::endl;
for(R_xlen_t  r = eventtimesN -1 ; r >=0 ; r--) wt_average[r] = (OutcomeTotals[r]>0 ? wt_totals[r]/static_cast<double>(OutcomeTotals[r]) : 0.0);

i = 0;
R_xlen_t bs = 0;
R_xlen_t lastID = -1;
int covi_spl_i = 0;

double gdiag_private = 0.0;

#pragma omp parallel for default(none) reduction(+:gdiag_private) private(tv_data) shared(tvbeta_spl,covi_spl_i, nbeta_spl,nvar,nvartv,lastID,nspl, gdiagbeta, covstart, covend, coval, zbeta,beta,  weights,  Outcomes, obs, i, bs, bspl, lookback, id,timein,timeout) 
for (R_xlen_t  ibeta = 0; ibeta < nvar + nbeta_spl - nvartv; ibeta++) { /* per observation time calculations */

  // std::vector<double> idexposure;
  // std::vector<R_xlen_t> idtime_index;
    
  for (int64_t  covi = covstart[i] - 1; covi < covend[i] ; covi++) {// iter over current covariates
   
    R_xlen_t  rowobs = obs[covi] - 1 ;
    R_xlen_t  time_index = timein[rowobs] - 1;
    R_xlen_t  time_index_exit = timeout[rowobs] - 1;
    R_xlen_t currentID = id[rowobs] - 1;
    double covali =  coval[covi] ;
    
    if (tvbeta_spl[i] == 0) {
#pragma omp atomic 
      zbeta[rowobs] += beta[ibeta] * covali ; // each covariate occurs once in an observation time
      if(Outcomes[rowobs] * weights[rowobs] > 0) 
        gdiag_private +=  covali * weights[rowobs];// don't need to calculate across time if not time varying
      } else {
    
        if(lastID != currentID) tv_data[covi_spl_i].clear(); 
    // {
    //   idexposure.clear();
    //   idtime_index.clear();
    //   }
    
        ExposureRecord exposure_record = {time_index, covali};
        tv_data[covi_spl_i].push_back(exposure_record);
        //if (tv_data[covi_spl_i].size() != 0) {
        
        tv_data[covi_spl_i].erase(std::remove_if(tv_data[covi_spl_i].begin(), tv_data[covi_spl_i].end(), 
          [time_index, lookback](const ExposureRecord& record) {if(time_index - record.time >= lookback ) return true; else return false; }), 
          tv_data[covi_spl_i].end());
          // std::erase_if(tv_data, [r, lookback](const ExposureRecord& record) {
          //     return r - record.time >= lookback; 
          // typedef std::vector<double>::iterator id_cov_iter;
          // typedef std::vector<R_xlen_t>::iterator id_time_iter;
            // for(std::pair<id_time_iter,id_cov_iter> i(idtime_index_multi[covi_spl_i].begin(),idexposure_multi[covi_spl_i].begin()); *i.first <= r && i.first != idtime_index_multi[covi_spl_i].end();) {
        for (auto& record : tv_data[covi_spl_i]) {

      // idexposure.push_back(covali);
      // idtime_index.push_back(time_index);

      // typedef std::vector<double>::iterator id_cov_iter;
      // typedef std::vector<R_xlen_t>::iterator id_time_iter;
      // for(std::pair<id_time_iter,id_cov_iter> i(idtime_index.begin(),idexposure.begin());i.first != idtime_index.end();) {
       
          R_xlen_t itime_diff = time_index - record.time;
        


        // if ( itime_diff >= lookback ) {
        //   i.first = idtime_index.erase(i.first);
        //   i.second = idexposure.erase(i.second);
        // }

#pragma omp atomic
          zbeta[rowobs] += record.value * beta[ibeta] * bspl(itime_diff,bs);
        
          if (Outcomes[rowobs] * weights[rowobs] > 0 ) {
            gdiag_private +=  record.value * bspl(time_index_exit - record.time,bs) * weights[rowobs];
            }
          }
        }
    }
  
  gdiagbeta[ibeta] = gdiag_private;
  
  if (tvbeta_spl[i] == 1 && bs < nspl) {
    bs ++;
    } else if (tvbeta_spl[i] == 1 && bs >= nspl) {
      bs = 0;
      i++;
      covi_spl_i++;
    }
  if (tvbeta_spl[i] == 0) {
    bs = 0;
    i++;
    }
//  Rcout << "i " <<i << " beta " << ibeta << " spline " << bs << " update "<< gdiagbeta[ibeta] << ", "<< std::endl;
  }
//for (R_xlen_t  ibeta = 0; ibeta < nvar + nbeta_spl - nvartv; ibeta++) Rcout << " ," << gdiagbeta[ibeta] ;

   if (recurrent == 1)
   {
#pragma omp parallel for default(none)  shared(gdiagfrail, frailty_group_events, idstart, idend, idn,  nvar, maxid, weights,  Outcomes, obs) 
     for(R_xlen_t  i = 0; i < maxid; i++)
     { /* per observation time calculations */
   
       for (R_xlen_t  idi = idstart[i] - 1; idi < idend[i] ; idi++) // iter over current covariates
       {
         R_xlen_t  rowobs = idn[idi] - 1;  // R_xlen_t is signed long, size_t is unsigned. R long vectors use signed so all numbers should be below R_xlen_t 2^48
         if (Outcomes[rowobs]*weights[rowobs] > 0 ) 
         {
           gdiagfrail[i] += weights[rowobs]; // frailty not in derivative of beta
           frailty_group_events[i] += 1; // i is unique ID at this point so can write directly
         }
       }
     }
   }


/* Set up */
newlk = 0.0;
if (lambda !=0) newlk = -(log(sqrt(lambda)) * nvar);


loglik = 0.0;
frailty_penalty = 0.0;
d2_sum_private = 0.0;
for (R_xlen_t  ivar = 0; ivar < nallvar; ivar++) step[ivar] = 1.0; 



if (recurrent == 1)
{
//  Rcout << "summing zbeta with frailty, ";  
#pragma omp parallel for  default(none) shared(idstart, idend, idn , zbeta, maxid, frailty) //reduction(+:zbeta[:maxobs])
  for (R_xlen_t  i = 0; i < maxid; i++) // +
  { /* per observation time calculations */
    for (R_xlen_t  idi = idstart[i] - 1; idi < idend[i] ; idi++) // iter over current covariates
    {
      zbeta[idn[idi] - 1] += frailty[i] ; // should be one frailty per person / observation
    }
  }
}

Rcout << "accumulating denominator, ";
/* Check zbeta Okay and calculate cumulative sums that do not depend on the specific covariate update*/
newlk_private = 0.0;

R_xlen_t* covi_spl = new R_xlen_t[nvartv]{0} ;
i = 0;
for (R_xlen_t  ibeta = 0; ibeta < nvar; ibeta++) { /*need to adjust each timepoint R for each tv beta in lookback period */ 
  if (tvbeta_spl[ibeta] == 1) {
    covi_spl[i]  = covstart[ibeta] - 1;
//    Rcout << " covi " << covi_spl[i] << " i " << i;
    i++;
    }
  }



// std::vector<std::vector<double>> idexposure_multi(nvartv);
// std::vector<std::vector<R_xlen_t>> idtime_index_multi(nvartv);
for (size_t ntv = 0; ntv < tv_data.size(); ntv++) {
  //    Rcout << "clearing" <<std::endl;
      tv_data[ntv].clear();
      // idexposure_multi[ntv].clear();
      // idtime_index_multi[ntv].clear();
      }

lastID = 0;
//#pragma omp parallel  default(none) reduction(+:newlk_private, denom[:ntimes], efron_wt[:ntimes]) private(i,covi_spl,idexposure_multi,idtime_index_multi,lastID)  shared(obs,bspl,tvbeta_spl,nbeta_spl, nspl, nvar,nvartv,timein, timeout,idstart,idend,id,idn, zbeta, weights,  Outcomes , ntimes,maxobs,lookback, covstart, covend, beta, coval)
//#pragma omp for
for (R_xlen_t  rowobs = 0; rowobs < maxobs; rowobs++) { /* unique per patient/time within each beta /covariate measurement */
  
  R_xlen_t currentID = id[rowobs] - 1;
  R_xlen_t time_index_entry = timein[rowobs] - 1;  // Time starts at zero but no events at this time to calculate sums. Lowest value of -1 but not used to reference into any vectors
  R_xlen_t time_index_exit = timeout[rowobs] - 1;

  double zbeta_temp = zbeta[rowobs] >22 ? 22 : zbeta[rowobs];
  zbeta_temp = zbeta_temp < -200 ? -200 : zbeta_temp;
  zbeta[rowobs] = zbeta_temp;
  
  if (Outcomes[rowobs]*weights[rowobs] > 0 ) {
    /*cumumlative sums for event patients */
    newlk_private += zbeta[rowobs] * weights[rowobs];
    efron_wt[time_index_exit] += exp(zbeta[rowobs]);
    }
  

  
    double* zbeta_tv_exposure = new double[time_index_exit - time_index_entry + 1]{zbeta_temp}; 
    if(lastID != currentID) {
      for (size_t ntv = 0; ntv < tv_data.size(); ntv++) {
    //    Rcout << "clearing" <<std::endl;
        tv_data[ntv].clear();
        // idexposure_multi[ntv].clear();
        // idtime_index_multi[ntv].clear();
        }
      }
    
    
    i = 0;
    covi_spl_i = 0;
    bs = 0;
    for (R_xlen_t  ibeta = 0; ibeta < nvar + nbeta_spl - nvartv; ibeta++) {/*need to adjust each timepoint R for each tv beta in lookback period */ 
     /* Find the time varying covariates indices */
  //   Rcout << " ibeta " << ibeta ;
      if(tvbeta_spl[i] == 1) {
   //     Rcout << " bs " << bs << std::endl;
     //   Rcout << " covi " << covi_spl[covi_spl_i]<< " covend[i] " << covend[i] <<  " iter obs " << obs[covi_spl[covi_spl_i]] -1 << " rowobs "<<  rowobs <<std::endl;
      //  Rcout << " covi_spl_i " << covi_spl_i << " id " << id[covi_spl[covi_spl_i]] - 1  - 1 << " currentID" <<currentID<< std::endl;
        
        int64_t covi = covi_spl[covi_spl_i];
        for (; covi < covend[i] && obs[covi] - 1 <= rowobs;  ) { // For this tv beta find prior exposure at this itime if exists
     //     Rcout << " covi " << covi << " covi_spl_i " << covi_spl_i << " id " << id[obs[covi] - 1]  - 1 << " currentID" <<currentID<< std::endl;
          ExposureRecord exposure_record = {timein[obs[covi] - 1] - 1, coval[covi]};
          tv_data[covi_spl_i].push_back(exposure_record);
          // idexposure_multi[covi_spl_i].push_back(coval[covi]);
          // idtime_index_multi[covi_spl_i].push_back(timein[obs[covi] - 1] - 1);
          ++covi; // only increment if this covi has been used and pushed to vectors
          }
        covi_spl[covi_spl_i] = covi;
   //   Rcout << " Number of obs to review " <<  idexposure[covi_spl_i].size() <<std::endl;
   //   Rcout << " covi " << covi << " covi_spl_i " << covi_spl_i << " covi " << covi_spl[covi_spl_i]<< std::endl;
     
     /*If no time varying observations for this covariate then nothing to do*/
        if (tv_data[covi_spl_i].size() != 0) {
               
          // typedef std::vector<double>::iterator id_cov_iter;
          // typedef std::vector<R_xlen_t>::iterator id_time_iter;
        
    // for(std::pair<id_time_iter,id_cov_iter> i(idtime_index[covi_spl_i].begin(),idexposure[covi_spl_i].begin());*i.first <= time_index_entry;) { 
    // Rcout << " idexposure " << *i.second << " idtime-index " << *i.first << std::endl;
    //   ++i.first;
    //   ++i.second;
    // }
    
          for (R_xlen_t  r = time_index_entry ;  r <= time_index_exit ; r++) {
            tv_data[covi_spl_i].erase(std::remove_if(tv_data[covi_spl_i].begin(), tv_data[covi_spl_i].end(), 
            [r, lookback](const ExposureRecord& record) {if(r - record.time >= lookback ) return true; else return false; }), 
            tv_data[covi_spl_i].end());
          // std::erase_if(tv_data, [r, lookback](const ExposureRecord& record) {
          //     return r - record.time >= lookback; 
          // });
          // typedef std::vector<double>::iterator id_cov_iter;
          // typedef std::vector<R_xlen_t>::iterator id_time_iter;
            // for(std::pair<id_time_iter,id_cov_iter> i(idtime_index_multi[covi_spl_i].begin(),idexposure_multi[covi_spl_i].begin()); *i.first <= r && i.first != idtime_index_multi[covi_spl_i].end();) {
            for (auto& record : tv_data[covi_spl_i]) {
              R_xlen_t itime_diff = r - record.time;
              R_xlen_t indextime_diff = time_index_entry - record.time;
              //if ( itime_diff >= lookback ) { /*observation before look back period so drop*/
                //record = tv_data[covi_spl_i].erase(record);
                // i.first = idtime_index_multi[covi_spl_i].erase(i.first); // advances to next element 
                // i.second = idexposure_multi[covi_spl_i].erase(i.second); // - so continue to next loop as check for limits in for loop control
                //} else
                 if (itime_diff >= 0 && itime_diff < lookback) { 
      //          Rcout << " itime_diff " << itime_diff << " indextime_diff " << indextime_diff << std::endl;
                  //zbeta_tv_exposure[itime_diff] += *i.second * beta[ibeta] * (bspl(itime_diff,bs) - bspl(indextime_diff,bs));
                  zbeta_tv_exposure[itime_diff] += record.value * beta[ibeta] * (bspl(itime_diff,bs) - bspl(indextime_diff,bs));
          //      Rcout << " idexposure " << *i.second << " idtime-index " << *i.first << std::endl;
                  //++record;
                  }
              }
            }
          }
          bs++; 
        }
      if (tvbeta_spl[i] == 0) {
        bs = 0; // Reset spline beta counter
        i++; // Move to next covariate
        } else if (bs > nspl) {
          covi_spl_i++;
          i++;
          bs =0;
          }
    } /* Finished adjusting zbeta_tv_exposure for time varying covariates. If none then will have constant zbeta values*/
    /* sum zbeta for this exposure period into the denominator */
    for (R_xlen_t  r = time_index_exit ;  r >= time_index_entry ; r--) { // Need to adjust each time point for all betas at all points in lookback - expensive if don't want to keep temporary array for time period
   //   Rcout << " r " << r << " time entry " << time_index_entry << " dff " << r-time_index_entry << " update " << std::endl;
  //    Rcout << zbeta_tv_exposure[r-time_index_entry] <<std::endl;
      double zbeta_temp = zbeta_tv_exposure[r-time_index_entry] >22 ? 22 : zbeta_tv_exposure[r-time_index_entry];
      zbeta_temp = zbeta_temp < -200 ? -200 : zbeta_temp;
      double risk = exp(zbeta_temp) * weights[rowobs];
      denom[r] += risk;
      }
    lastID = currentID;
    delete[] zbeta_tv_exposure;
    
}
delete[] covi_spl;


newlk += newlk_private;

Rcout << "Beginning main loop" <<std::endl;

/* Main outer loop for theta and covariate updates */
int outer_iter = 0;
for (outer_iter = 0; outer_iter < MSTEP_MAX_ITER && done == 0; outer_iter++) 
{
  int iter = 0;
  for (iter = 0; iter <= MSTEP_MAX_ITER; iter++)
  {
    i = 0;
    bs = 0;
    covi_spl_i = 0;
    for (R_xlen_t  ibeta = 0; ibeta < nvar + nbeta_spl - nvartv; ibeta++) { 
		std::vector<double> idexposure;
		std::vector<R_xlen_t> idtime_index;
	  
      double gdiag =  -gdiagbeta[ibeta];
      double hdiag = 0.0;
  //    Rcout << "accumulating derivatives, ";        
      for (R_xlen_t  ir = 0; ir < (ntimes*4); ir++) derivMatrix[ir] = 0.0;
      lastID = 0;
      tv_data[covi_spl_i].clear();
//#pragma omp parallel default(none) reduction(+:derivMatrix[:ntimes*4]) private(tv_data) shared(tvbeta_spl, beta, covstart, covend, coval, weights, Outcomes, ntimes, obs, timein, timeout, zbeta,i,ibeta,bs, nvar, idstart, idend, id,idn, bspl,lookback) //covn, 
{
//#pragma omp for
      for (int64_t  covi = covstart[i] - 1; covi < covend[i]; covi++) { 
        R_xlen_t  rowobs = (obs[covi] - 1) ;
        R_xlen_t  time_index_entry = timein[rowobs] - 1; // std vectors use unsigned can be negative though for time 0
        R_xlen_t  time_index_exit = timeout[rowobs] - 1; // std vectors use unsigned
        R_xlen_t currentID = id[rowobs] - 1;         
        
        double covali_tv = coval[covi] ;
        double zbeta_temp = zbeta[rowobs];

        if(lastID != currentID) {
            //    Rcout << "clearing" <<std::endl;
            tv_data[covi_spl_i].clear();
          //  idexposure.clear();
           // idtime_index.clear();
          }
        
        if(tvbeta_spl[i] == 0) {
          double derivFirst = exp(zbeta[rowobs]) * weights[rowobs] * covali_tv;
          double derivSecond = derivFirst * covali_tv;
          for (R_xlen_t  r = time_index_exit ;  r > time_index_entry ; r--) { // keep int for calculations of indices then cast
            derivMatrix[r] += derivFirst;
            derivMatrix[ntimes + r] += derivSecond;
            }
          if (Outcomes[rowobs]*weights[rowobs] > 0) {
            derivMatrix[(2*ntimes) + time_index_exit] += derivFirst ;
            derivMatrix[(3*ntimes) + time_index_exit] += derivSecond ;
            }
          } else {
            ExposureRecord exposure_record = {time_index_entry, covali_tv};
            tv_data[covi_spl_i].push_back(exposure_record);

            //idexposure.push_back(coval[covi]);
            //idtime_index.push_back(timein[obs[covi] - 1] - 1);
            
            if (tv_data[covi_spl_i].size() != 0) {
              
              // typedef std::vector<double>::iterator id_cov_iter;
              // typedef std::vector<R_xlen_t>::iterator id_time_iter;
  
              for (R_xlen_t  r = time_index_entry ;  r <= time_index_exit ; r++) {
                // typedef std::vector<double>::iterator id_cov_iter;
                // typedef std::vector<R_xlen_t>::iterator id_time_iter;
                tv_data[covi_spl_i].erase(std::remove_if(tv_data[covi_spl_i].begin(), tv_data[covi_spl_i].end(), 
                  [r, lookback](const ExposureRecord& record) {if(r - record.time >= lookback ) return true; else return false; }), 
                  tv_data[covi_spl_i].end());
                // for(std::pair<id_time_iter,id_cov_iter> i(idtime_index.begin(),idexposure.begin()); *i.first <= r && i.first != idtime_index.end();) {
                  
                //   R_xlen_t itime_diff = r - *i.first;
                //   R_xlen_t indextime_diff = time_index_entry - *i.first;
                //   if ( itime_diff >= lookback ) { /*observation before look back period so drop*/
                //     i.first = idtime_index.erase(i.first); // advances to next element 
                //     i.second = idexposure.erase(i.second); // - so continue to next loop as check for limits in for loop control
                //     } else if (itime_diff >= 0) { 
                for (auto& record : tv_data[covi_spl_i]) {
                  R_xlen_t itime_diff = r - record.time;
                  R_xlen_t indextime_diff = time_index_entry - record.time;
              //if ( itime_diff >= lookback ) { /*observation before look back period so drop*/
                //record = tv_data[covi_spl_i].erase(record);
                // i.first = idtime_index_multi[covi_spl_i].erase(i.first); // advances to next element 
                // i.second = idexposure_multi[covi_spl_i].erase(i.second); // - so continue to next loop as check for limits in for loop control
                //} else
                  if (itime_diff >= 0 && itime_diff < lookback) { 
                      double temp_covali_tv =  record.value*(bspl(itime_diff,bs));
                      double temp_zbeta_tv =  zbeta_temp + (record.value*beta[ibeta]*(bspl(itime_diff,bs) - bspl(indextime_diff,bs)));   
                      double derivFirst_tv = exp(temp_zbeta_tv) * weights[rowobs] * temp_covali_tv ;
                      double derivSecond_tv = derivFirst_tv * temp_covali_tv;
                      derivMatrix[r] += derivFirst_tv;
                      derivMatrix[ntimes + r] += derivSecond_tv;
                      if (r == time_index_exit && Outcomes[rowobs]*weights[rowobs] > 0) {
                        derivMatrix[(2*ntimes) + time_index_exit] += derivFirst_tv ;
                        derivMatrix[(3*ntimes) + time_index_exit] += derivSecond_tv ;
                        }
                  }
                }
              }
            }
      } /* End of covariate accumulation of derivatives */
}

      R_xlen_t  exittimesN = eventtimesN -1;
//      Rcout << "calculating derivatives, ";
      for(R_xlen_t  r = exittimesN ; r >=0 ; r--) {
        R_xlen_t   time = OutcomeTotalTimes[r] - 1;
        
        for (R_xlen_t  k = 0; k < OutcomeTotals[r]; k++) {
          double temp = (double)k
          / (double)OutcomeTotals[r];
          double d2 = denom[time] - (temp * efron_wt[time]); /* sum(denom) adjusted for tied deaths*/

          double temp2 = (derivMatrix[time] - (temp * derivMatrix[(2*ntimes) + time])) / d2;
          gdiag += wt_average[r]*temp2; 
          hdiag += wt_average[r]*(((derivMatrix[ntimes + time] - (temp * derivMatrix[(3*ntimes) + time])) / d2) -
            (temp2 * temp2)) ;
          }
        //  if (  OutcomeTotals[r] >0 && r > exittimesN-1000) Rcout << "Time " << time << " gdiag " << gdiag<< " hdiag " <<
        //    hdiag << " wt " << wt_average[r] << " totals " << OutcomeTotals[r] <<
        //       " derivs " << derivMatrix[time]  << " " << derivMatrix[time + ntimes] << " " << derivMatrix[time + 2*ntimes] << " " << derivMatrix[time + 3*ntimes]  <<
        //         " denom " << denom[time]  << "\n";
        } 

      double dif = 0; 

        /* Update */
//     Rcout << "calculating increment, ";
      if (lambda !=0) {
        dif = (gdiag + (beta[ibeta] / lambda)) / (hdiag + (1.0 / lambda));
        } else {
          dif = (gdiag ) / (hdiag);
          }

      if (fabs(dif) > step[ibeta]) dif = (dif > 0.0) ? step[ibeta] : -step[ibeta];

      step[ibeta] = ((2.0 * fabs(dif)) > (step[ibeta] / 2.0)) ? 2.0 * fabs(dif) : (step[ibeta] / 2.0);//Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245

      if (lambda !=0) {
        newlk += (beta[ibeta] * beta[ibeta]) / (2.0 * lambda);
        }
      
      beta[ibeta] -= dif;

      if (lambda !=0) {
        newlk -= (beta[ibeta] * beta[ibeta]) / (2.0 * lambda);// Include iteration penalty for this covariate
        }

      Rcout << ibeta <<  ", ";

/* Update cumulative sums dependent on denominator and zbeta so need to accumulate updates then apply them*/
      newlk_private = 0.0;
      for (R_xlen_t  ir = 0 ; ir < ntimes; ir++) {
        denom_private[ir] = 0.0;
        efron_wt_private[ir] = 0.0;
        }
		
		idexposure.clear();
		idtime_index.clear();

//#pragma omp parallel  default(none) reduction(+:newlk_private, denom_private[:ntimes], efron_wt_private[:ntimes]) private(idtime_index,idexposure)  shared(denom,tvbeta_spl,efron_wt,zbeta,covstart, covend, coval, weights, Outcomes, ntimes, obs, timein, timeout, dif, i,ibeta,bs, nvar, idstart, idend,id, idn,bspl,lookback, beta)///*,  denom, efron_wt, newlk*/) covn, 
{
//#pragma omp for
		idexposure.clear();
		idtime_index.clear();
        for (int64_t  covi = covstart[i] - 1; covi < covend[i]; covi++) {
          R_xlen_t  rowobs =  (obs[covi] - 1) ;
          
    //      double riskold = exp(zbeta[rowobs] ); 

          R_xlen_t  time_index_entry = timein[rowobs] - 1; // std vectors use unsigned can be negative though for time 0
          R_xlen_t  time_index_exit = timeout[rowobs] - 1; // std vectors use unsigned
          
          
          if(tvbeta_spl[i] == 0) {
            double riskold = exp(zbeta[rowobs] ); 
            double covali =  coval[covi] ;
            
            double xbdif = dif * covali;
            double zbeta_updated = zbeta[rowobs] - xbdif;
            
            zbeta_updated =  zbeta_updated >22 ? 22 : zbeta_updated;
            zbeta_updated =  zbeta_updated < -200 ? -200 :  zbeta_updated;
            zbeta[rowobs] = zbeta_updated; // Each covariate only once per patient per time so can update directly
            
            double riskdiff = (exp(zbeta_updated ) - riskold) * weights[rowobs];
            for (R_xlen_t  r = time_index_exit;  r > time_index_entry ; r--)
              denom_private[r] += riskdiff; 
            
            if (Outcomes[rowobs]*weights[rowobs] > 0 ) {
              newlk_private += xbdif * weights[rowobs];
              efron_wt_private[time_index_exit] += riskdiff;
              }
            } else {
              
              idexposure.push_back(coval[covi]);
              idtime_index.push_back(timein[obs[covi] - 1] - 1);
              
              if (idexposure.size() != 0) {
                typedef std::vector<double>::iterator id_cov_iter;
                typedef std::vector<R_xlen_t>::iterator id_time_iter;
                
                for (R_xlen_t  r = time_index_entry ;  r <= time_index_exit ; r++) {
                  // typedef std::vector<double>::iterator id_cov_iter;
                  // typedef std::vector<R_xlen_t>::iterator id_time_iter;
                  for(std::pair<id_time_iter,id_cov_iter> i(idtime_index.begin(),idexposure.begin()); *i.first <= r && i.first != idtime_index.end();) {
                    
                    R_xlen_t itime_diff = r - *i.first;
                    R_xlen_t indextime_diff = time_index_entry - *i.first;
                    
                    if ( itime_diff >= lookback ) { /*observation before look back period so drop*/
                      i.first = idtime_index.erase(i.first); // advances to next element 
                      i.second = idexposure.erase(i.second); // - so continue to next loop as check for limits in for loop control
                      } else if (itime_diff >= 0) { 
                        
                        double temp_covali_tv =  *i.second*(bspl(itime_diff,bs));
                        double temp_zbeta_Old_tv =  *i.second*(beta[ibeta]+ dif)*(bspl(itime_diff,bs) - bspl(indextime_diff,bs));
                        temp_zbeta_Old_tv =  temp_zbeta_Old_tv >22 ? 22 : temp_zbeta_Old_tv;
                        temp_zbeta_Old_tv =  temp_zbeta_Old_tv < -200 ? -200 :  temp_zbeta_Old_tv;
                        
                        double temp_zbeta_tv =  *i.second*beta[ibeta]*(bspl(itime_diff,bs) - bspl(indextime_diff,bs));
                        temp_zbeta_tv =  temp_zbeta_tv >22 ? 22 : temp_zbeta_tv;
                        temp_zbeta_tv =  temp_zbeta_tv < -200 ? -200 :  temp_zbeta_tv;
                        
                        denom_private[r] += (exp(temp_zbeta_tv) - exp(temp_zbeta_Old_tv)) * weights[rowobs]; 
                        
                        if (r == time_index_exit &&  Outcomes[rowobs]*weights[rowobs] > 0 ) {
                          newlk_private += dif * temp_covali_tv * weights[rowobs];
                          efron_wt_private[time_index_exit] += (exp(temp_zbeta_tv) - exp(temp_zbeta_Old_tv)) * weights[rowobs];
                          }
                        
                        ++i.first;
                        ++i.second;
                        } else {
                          ++i.first;
                          ++i.second;                  
                          }
                    }
                  }
                }
              }  
          }
} // End of update of covariate sums
      for (R_xlen_t  r = ntimes - 1; r >= 0; r--) {
        efron_wt[r] += efron_wt_private[r];
        denom[r] += denom_private[r];
        }
      newlk -= newlk_private; // min  beta updated = beta - diff

      if (tvbeta_spl[i] == 1) {
        if (bs < nspl - 1) {
          bs++;
          } else {
            i++;
            bs =0;
            covi_spl_i++;
            } 
        }
      if (tvbeta_spl[i] == 0) {
        bs = 0;
        i++;
        } 
    } /* end of inner loop of coefficient updates for beta */
    
    
    for (R_xlen_t  ir = 0 ; ir < ntimes; ir++)
    {
      denom_private[ir] = 0.0;
      efron_wt_private[ir] = 0.0;
    }
    if (recurrent == 1)
    {
      newlk_private = 0.0;
#pragma omp parallel for default(none)  reduction(+:newlk_private,denom_private[:ntimes],efron_wt_private[:ntimes]) shared(eventtimesN, maxid, gdiagfrail, idn, idstart, idend, newlk, OutcomeTotalTimes,OutcomeTotals, denom, efron_wt, wt_average, theta, frailty, frailty_mean, nu, step, derivMatrix,  coval, weights, Outcomes, ntimes, obs, timein, timeout, zbeta, nvar)
      for (R_xlen_t  i = 0; i < maxid ; i++) 
      {
        double gdiag =  -gdiagfrail[i];
        double hdiag = 0.0;
        
        std::vector<double> derivMatrix_private(ntimes*2,0.0);
        
        for (R_xlen_t  idi = idstart[i] - 1; idi < idend[i] ; idi++)
        { // iter over current covariates
          R_xlen_t  rowobs = idn[idi] - 1; // R_xlen_t is signed long, size_t is unsigned. R vectors use signed
          
          R_xlen_t  time_index_entry = timein[rowobs] - 1; // std vectors use unsigned can be negative though for time 0
          R_xlen_t  time_index_exit = timeout[rowobs] - 1; // std vectors use unsigned
          
          double risk = exp(zbeta[rowobs]) * weights[rowobs];
          
          for (R_xlen_t  r = time_index_exit ;  r > time_index_entry ; r--) derivMatrix_private[r] += risk; // keep int for calculations of indices then cast
          
          if (Outcomes[rowobs]*weights[rowobs] > 0) derivMatrix_private[ntimes +  time_index_exit] += risk ;
          
        }
        
        R_xlen_t  exittimesN = eventtimesN -1;
        
        for(R_xlen_t  r = exittimesN ; r >=0 ; r--)
        {
          
          R_xlen_t   time = OutcomeTotalTimes[r] - 1;
          
          for (R_xlen_t  k = 0; k < OutcomeTotals[r]; k++)
          {
            double temp = (double)k
            / (double)OutcomeTotals[r];
            
            double d2 = denom[time] - (temp * efron_wt[time]); /* sum(denom) adjusted for tied deaths*/
            double temp2 = (derivMatrix_private[time] - (temp * derivMatrix_private[ntimes + time])) / d2;

            gdiag += wt_average[r]*temp2; 
            hdiag += wt_average[r]*(temp2 * (1 - temp2)) ; 
          }
        } 
        
        double dif = 0;
        
        /* Update */
        
        
        if ( (theta == 0) || (std::isinf(gdiag)) || (std::isnan(gdiag)) || (std::isinf(hdiag)) || (std::isnan(hdiag)) )
        {
          dif =0;
          frailty[i] = 0;
        } else {
          
          dif = (gdiag + (exp(frailty[i] - frailty_mean) - 1.0) * nu ) /    //-g(d,w)' here is +nu(1-exp(frailty), but gdiag here is negative so use negative form of penalty
            (hdiag + (exp(frailty[i] - frailty_mean) * nu ));  // /*- frailty_mean*/again  -g(d,w)' here is -nu(exp(frailty)), but hdiag here is negative so use negative form of penalty
          
          
          if (fabs(dif) > step[i + nvar]) {
            dif = (dif > 0.0) ? step[i + nvar] : -step[i + nvar];
          }
          
          step[i + nvar] = ((2.0 * fabs(dif)) > (step[i + nvar] / 2.0)) ? 2.0 * fabs(dif) : (step[i + nvar] / 2.0);//Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
          frailty[i] -= dif;
        }
        
        /* Update cumulative sums dependent on denominator and zbeta so need to accumulate updates then apply them*/
        
        for (R_xlen_t  idi = idstart[i] - 1; idi < idend[i] ; idi++)
        {
          R_xlen_t rowobs = (idn[idi] - 1); // R_xlen_t is signed long, size_t is unsigned. R vectors use signed so use int throughout
          
          double riskold = exp(zbeta[rowobs] );
          
          double zbeta_updated = zbeta[rowobs] - dif;
          zbeta_updated =  zbeta_updated >22 ? 22 : zbeta_updated;
          zbeta_updated =  zbeta_updated < -200 ? -200 :  zbeta_updated;
          
          zbeta[rowobs] -= dif; // Each covariate only once per patient per time so can update directly
          
          double riskdiff = (exp(zbeta_updated ) - riskold) * weights[rowobs];
          
          R_xlen_t  time_index_entry =  timein[rowobs] - 1;
          R_xlen_t  time_index_exit =  timeout[rowobs] - 1;
          
          for (R_xlen_t  r = time_index_exit;  r > time_index_entry ; r--)
            denom_private[r] += riskdiff; // need to update exp(xb1 + xb2 + ) + exp(x2b1 + x2b2 +)
          
          if (Outcomes[rowobs]*weights[rowobs] > 0 )
          {
            newlk_private += dif * weights[rowobs];
            efron_wt_private[time_index_exit] += riskdiff;
          }
        }
      } /* next frailty term */
      for (R_xlen_t  r = ntimes - 1; r >= 0; r--)
      {
        efron_wt[r] += efron_wt_private[r];
        denom[r] += denom_private[r];
      }
      newlk -= newlk_private;
    } /* end of frailty updates */
        
    newlk += d2_sum_private;
    d2_sum_private =0;
    
    for(R_xlen_t  r = eventtimesN -1 ; r >=0 ; r--)
    {
      R_xlen_t   time = OutcomeTotalTimes[r] - 1;
      for (R_xlen_t  k = 0; k < OutcomeTotals[r]; k++)
      {
        double temp = (double)k
        / (double)OutcomeTotals[r];
        double d2 = denom[time] - (temp * efron_wt[time]); /* sum(denom) adjusted for tied deaths*/
        
        d2_sum_private += wt_average[r]*safelog(d2); // track this sum to remove from newlk at start of next iteration
        newlk -= wt_average[r]*safelog(d2); 
      }
    }
    
    /* centre exponentiated frailty estimates so penalty is minimised*/
    if (recurrent == 1 ) 
    {
      
      frailty_sum = 0.0;
      for(R_xlen_t  rowid = 0; rowid < maxid; rowid ++)  frailty_sum += exp(frailty[rowid]);
      
      frailty_mean = safelog(frailty_sum / maxid);
      
      newlk  -= frailty_penalty;
      frailty_penalty = 0.0;
      
      for (R_xlen_t  rowid = 0; rowid < maxid ; rowid ++ ) frailty_penalty += (frailty[rowid] - frailty_mean)*nu;
      
      newlk  += frailty_penalty;
    }
    
    
    Rcpp::Rcout << "\r" <<std::flush;
    Rcpp::Rcout << "\r Iter:  " << iter << " Cox likelihood : " << newlk << "   last likelihood :" << loglik << " theta " << theta << "\n";
    /* Check for convergence */
    
    if ((iter > 1) &&  (fabs(1.0 - (newlk / loglik))) <= MAX_EPS) break;
    
    Rcout << " convergence " << 1.0 - ((newlk ) / (loglik )) << "\n";
    loglik = newlk;
    
    Rcout << "Beta : " ;
    for (R_xlen_t  ibeta = 0; ibeta < nvar + nbeta_spl - nvartv; ibeta ++) Rcout << beta[ibeta] << " ";
    Rcout << '\n';
  } /* return for another iteration */
    
    
    if ( recurrent == 1 && iter > 0 && done == 0) 
    {
      
      lik_correction = 0.0;
      Rcout << "*** updating theta ***\n";
      
      if (theta != 0 && nu != 0) 
      {
        for (R_xlen_t  rowid = 0; rowid < maxid; rowid++) {
          
          if(frailty_group_events[rowid] == 0) continue;
          double temp = nu > 1e7 ? frailty_group_events[rowid]*frailty_group_events[rowid]/nu :  frailty_group_events[rowid] + nu*safelog(nu/(nu+(frailty_group_events[rowid])));
          lik_correction += temp +
            std::lgamma(frailty_group_events[rowid] + nu) - 
            std::lgamma(nu) -
            frailty_group_events[rowid]*safelog(nu + frailty_group_events[rowid]); // loglikelihood correction for frailty
        }
      }
      
      /* Update frailty */
      
      double theta_lower_bound = 0;
      double theta_upper_bound = 1;
      
      
      thetalkl_history[iter_theta] = newlk + lik_correction;
      theta_history[iter_theta] = theta;  
      
      for (R_xlen_t  ir = 0; ir <= iter_theta; ir ++) 
      {
        double theta_convergence = (ir < 1) ? 0 : fabs(1.0 - (thetalkl_history[ir] / thetalkl_history[ir-1]));
        Rcout << " Outer iteration : " << ir << " theta : " << theta_history[ir] << " Corrected likelihood : " << thetalkl_history[ir] <<
            " Convergence for theta : " << 
            theta_convergence << std::endl;
      }
      
      if(iter_theta == 0) {
        theta = 1;
        done = 0;
      } else if(iter_theta == 1) 
      {
        
        theta = (thetalkl_history[1] < (thetalkl_history[0] + 1)) ?  (theta_history[0] + theta_history[1]) / 2 : 2 * theta_history[1];
        done = 0;
        
      } else if(iter_theta >= 2) 
      {
        
        done =  (fabs(1.0 - (thetalkl_history[iter_theta]/thetalkl_history[iter_theta-1])) < inner_EPS);
        
        R_xlen_t  best_idx = 0;
        double max_theta = -std::numeric_limits<double>::infinity();
        double min_theta = std::numeric_limits<double>::infinity();
        double max_likl = -std::numeric_limits<double>::infinity();
        
        double max_theta_likl = -std::numeric_limits<double>::infinity();
        double min_theta_likl = std::numeric_limits<double>::infinity();
        
        for(R_xlen_t  it = 0; it <= iter_theta; it++) 
        {
          if (thetalkl_history[it] > max_likl) 
          {
            
            max_likl = thetalkl_history[it];
            best_idx = it;
            
          }
          if (theta_history[it] < min_theta) 
          {
            
            min_theta_likl = thetalkl_history[it];
            min_theta = theta_history[it];
          }
          if (theta_history[it] > max_theta) 
          {
            
            max_theta_likl = thetalkl_history[it];
            max_theta = theta_history[it];
            
          }
        }
        
        double best_thetas[3] = {safesqrt(min_theta),
                                 safesqrt(theta_history[best_idx]),
                                 safesqrt(max_theta)};
        
        double best_likls[3] = {min_theta_likl,
                                thetalkl_history[best_idx],
                                                max_theta_likl};
        
        // Need to find the theta's either side of the best guess theta
        for(R_xlen_t  it = 0; it <= iter_theta; it++) 
        {
          
          if (safesqrt(theta_history[it]) > best_thetas[0] && safesqrt(theta_history[it]) < best_thetas[1]) 
          {
            
            best_thetas[0] = safesqrt(theta_history[it]);
            best_likls[0] = thetalkl_history[it];
            
          }
          if (safesqrt(theta_history[it]) > best_thetas[1] && safesqrt(theta_history[it]) < best_thetas[2]) 
          {
            
            best_thetas[2] = safesqrt(theta_history[it]);
            best_likls[2] = thetalkl_history[it];
            
          }
        }
        
        for (R_xlen_t  r = 0 ; r < 3; r++) 
        {
          Rcout << "  theta " << r << " " << pow(best_thetas[r],2) << " lkl " << best_likls[r] << std::endl;
        }
        
        if (iter_theta == best_idx && max_theta == theta_history[iter_theta])
        {
          theta = 2 * max_theta;
          Rcout << " best lkl is current " << max_likl << " best theta is current " << theta << " done " << done << std::endl;
        } else 
        {
          
          if (theta_history[best_idx] == min_theta) 
          {
            
            theta = safesqrt(min_theta) - 3*(safesqrt( best_thetas[2]) - safesqrt(min_theta));  // safesqrt returns zero for zero, same behaviour as r sqrt
            
            if (theta < theta_lower_bound) 
            {
              
              /* Repeat search for minimum that is above lower bound */
              double min_theta = theta_upper_bound;
              
              for(R_xlen_t  it = 0; it <= iter_theta; it++) if(theta_history[it] > theta_lower_bound && theta_history[it] < min_theta) min_theta = theta_history[it];
              
              theta = safesqrt(theta_lower_bound) + (safesqrt(min_theta)-safesqrt(theta_lower_bound))/10;
              
            }
            theta = pow(theta,2);
            Rcout << " best lkl is " << max_likl << " best theta is smallest " << theta << " done " << done << std::endl;
            
          } else if (theta_history[best_idx] == max_theta ) 
          {
            
            theta =  safesqrt(max_theta) + 3*(safesqrt(max_theta) - safesqrt( best_thetas[0]));
            
            if (theta > theta_upper_bound) 
            {
              /* Repeat search for maximum that is below upper bound */
              double max_theta = theta_lower_bound;
              
              for(R_xlen_t  it = 0; it <= iter_theta; it++) if (theta_history[it] < theta_upper_bound && theta_history[it] > max_theta) max_theta = theta_history[it];
              
              theta = safesqrt(theta_upper_bound) + (safesqrt(max_theta)-safesqrt(theta_upper_bound))/10;
              
            }
            theta = pow(theta,2);
            Rcout << " best lkl is " << max_likl << " best theta is largest " << theta << " done " << done << std::endl;
            
          } else 
          {
            
            /* Brent search update */
            
            double temp1 = (pow(best_thetas[1] - best_thetas[0],2) *
            (best_likls[1] - best_likls[2])) -
            (pow(best_thetas[1] - best_thetas[2],2) *
            (best_likls[1] - best_likls[0]));
            
            double temp2 = ((best_thetas[1] - best_thetas[0]) *
                            (best_likls[1] - best_likls[2])) -
                            ((best_thetas[1] - best_thetas[2]) *
                            (best_likls[1] - best_likls[0]));
            
            
            theta = best_thetas[1] - (0.5*(temp1/temp2));
            
            Rcout << " theta is updated " << max_likl << " best theta is new " << pow(theta,2) << " done " << done << std::endl;
            
            if ((theta < best_thetas[0]) ||
                (theta > best_thetas[2]) ||
                ((iter_theta > 3) &&
                (theta - safesqrt(theta_history[iter_theta])) >
                0.5 *
                abs(safesqrt(theta_history[iter_theta - 1]) - safesqrt(theta_history[iter_theta - 2]))
                ))
            {
              if ((best_thetas[1] - best_thetas[0]) >
                    (best_thetas[2] - best_thetas[1])) 
              {
                
                theta = best_thetas[1] - (.38*(best_thetas[1] - best_thetas[0]));
                
              } else 
              {
                
                theta = best_thetas[1] + (.32 * (best_thetas[2] - best_thetas[1]));
                
              }
              
              Rcout << " theta is bouncing " << max_likl << " best theta is new " << pow(theta,2) << " done " << done << std::endl;
            }
            
            theta = pow(theta,2);
            
          }
        }
    
      }
      
      Rcout << std::endl << " theta: " << theta << " correction " << lik_correction << " newlk " << newlk  << std::endl << std::endl;
      
      
      iter_theta++;
      
      nu = 1 / theta;
      
    } 
    if (recurrent == 0) done = 1;
    /* end of theta loop */
    
}


Rcout << std::endl << "Final betas " <<  std::endl;
for (R_xlen_t  ibeta = 0; ibeta < nvar + nbeta_spl - nvartv; ibeta ++) Rcout << beta[ibeta] << " ";

Rcout << '\n';
Rcout << "Log likelihood : "  << loglik + lik_correction << std::endl;
Rcout << "Theta : "  << theta << std::endl;
Rcout << "Outer iterations  : " << outer_iter << '\n';
Rcout << "Mean Frailty log(mean(exponentiated)) : "  << frailty_mean << std::endl;

/* baseline hazard whilst zbeta in memory */
// need to have cumulative baseline hazard
Rcout << " Calculating baseline hazard..." ;


if (recurrent == 1)
{
#pragma omp parallel for  default(none) shared(idstart, idend, idn , zbeta, maxid, frailty_mean) //reduction(+:zbeta[:maxobs])
  for (R_xlen_t  i = 0; i < maxid; i++) // +
  { /* per observation time calculations */
    for (R_xlen_t  idi = idstart[i] - 1; idi < idend[i] ; idi++) // iter over current covariates
    {
      zbeta[idn[idi] - 1] -= frailty_mean ; // should be one frailty per person / observation
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
  basehaz[time] = basehaz_private*wt_totals[r]; // should be thread safe as time unique per thread
}

Rcout << " Calculating cumulative baseline hazard..." ;

cumhaz[0] = basehaz[0] ;
for (R_xlen_t  t = 1; t < ntimes; t++)
   cumhaz[t] = cumhaz[t-1] +  basehaz[t];


ModelSummary[0] = loglik;
ModelSummary[1] = lik_correction;
ModelSummary[2] = loglik + lik_correction;
ModelSummary[3] = theta;
ModelSummary[4] = outer_iter;
ModelSummary[5] = fabs(1.0 - (newlk / loglik));
ModelSummary[6] = iter_theta < 2 ? fabs(1.0 - (newlk / loglik)) : fabs(1.0 - (thetalkl_history[iter_theta-1]/thetalkl_history[iter_theta-2]));
ModelSummary[7] = frailty_mean;

Rcout << " Model Statistics : LogLikelihood , Gamma likelihood correction , corrected likelihood , theta , outer iterations , inner convergence , outer convergence , frailty mean\n" ;

for(R_xlen_t  i = 0; i < 8; i++) Rcout << ModelSummary[i] << ", ";
Rcout <<  std::endl;


Rcout << " Freeing arrays..." ;

delete[] denom;
delete[] efron_wt;
delete[] denom_private;
delete[] efron_wt_private;
delete[] wt_average;
delete[] wt_totals;  
delete[] derivMatrix;
delete[] frailty_group_events ;
delete[] theta_history;
delete[] thetalkl_history;
delete[] zbeta ;
delete[] step ;
delete[] gdiagbeta ;
delete[] gdiagfrail;
delete[] OutcomeTotals;
delete[] OutcomeTotalTimes;

Rcout << " Freed\n " ;

 }
 
 
