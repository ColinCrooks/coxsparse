#include <Rcpp.h>
#include <omp.h>
#include <RcppParallel.h>
#include <RcppInt64>
#include "utils.h"

using namespace Rcpp;
//' profile_ci
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
//' A function using the same data structure to calculate profile
//' confidence intervals with a crude search pattern is provided.
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
//' coval_in[i] ~ timein_in[obs_in[i]], timeout_in[obs_in[i]], Outcomes_in[obs_in[i]],  
//' @param weights_in A double vector of weights to be applied to each unique
//' patient time point. Of the same length as timein, timeout and outcomes. 
//' Sorted by time out, time in, and patient id. 
//' @param frailty_in A double vector of frailty estimates for each idsorted by id.
//' @param  timein_in An integer vector of the start time for each unique patient 
//' time row, so would be the time that a patient's corresponding
//' covariate value starts. Of the same length as weights, timeout, and outcomes. 
//' Sorted by time out, time in, and patient id
//' @param timeout_in An integer vector of the end time for each unique patient
//' time row, so would be the time that a patient's corresponding outcome
//' occurs. Of the same length as weights, timein, timeout and outcomes. Sorted by time out, time in, and patient id
//' @param Outcomes_in An integer vector of 0 (censored) or 1 (outcome) for the 
//' corresponding unique patient time. Of the same length as timein, timeout and 
//' weights. Sorted by time out, time in, and patient id 
//' @param OutcomeTotals_in An integer vector of the total number of outcomes that
//' occur at each unique time point. Length is the number of unique times in cohort. Sorted by time
//' @param OutcomeTotalTimes_in An integer vector of each unique time point that
//' outcome events are observed in the cohort. Same length as OutcomeTotals. Sorted by time
//' @param covstart_in An integer64 (from package bit64) vector of the start row for each covariate in coval 
//' @param covend_in An integer64 (from package bit64) vector of the end row for each covariate in coval
//' @param idn_in An integer vector mapping unique patient IDs sorted by ID to the 
//' corresponding row in observations sorted by time out, time in, and patient id
//' For id = i the corresponding rows in time_in, timeout_in and Outcomes_in 
//' are the rows listed between idn_in[idstart_in[i]]:idn_in[idend_in[i]] 
//' @param idstart_in An integer vector of the start row for each unique patient ID in idn_in
//' @param idend_in An integer vector of the end row for each unique patient ID in idn_in
//' @param lambda Penalty weight to include for ridge regression:-log(sqrt(lambda)) * nvar
//' @param theta_in The value of theta to use in the model. This is the frailty parameter.
//' @param MSTEP_MAX_ITER Maximum number of iterations
//' @param decimals Precision required for confidence intervals defined by
//' number of decimal places.
//' @param confint_width e.g. for 95% confidence interval confint_width = 0.95.
//' @param threadn Number of threads to be used - caution as will crash if specify more
//' threads than available memory for copying data for each thread.
//' @return Numeric matrix with nvar rows and lower and upper confidence intervals in 2 columns.
//'
//' @export
// [[Rcpp::export]]
 NumericMatrix profile_ci(
     DoubleVector beta_in,
     IntegerVector obs_in,
     DoubleVector coval_in,
     DoubleVector weights_in,
     DoubleVector frailty_in,
     IntegerVector timein_in ,
     IntegerVector timeout_in ,
     IntegerVector Outcomes_in ,
     IntegerVector OutcomeTotals_in ,
     IntegerVector OutcomeTotalTimes_in,
     NumericVector covstart_in,
     NumericVector covend_in,
     IntegerVector idn_in,
     IntegerVector idstart_in,
     IntegerVector idend_in,
     double lambda, 
     double theta_in,
     int MSTEP_MAX_ITER,
     int decimals,
     double confint_width,
     int threadn
 )
 {
   Rcpp::Rcout.precision(10);
   omp_set_dynamic(0);
  // Explicitly disable dynamic teams
   omp_set_num_threads(threadn);

     // Vectors from R index begin from 1. Need to convert to 0 index for C++and comparisons
   
   R_xlen_t ntimes = max(timeout_in);
     
    // Unique times but don't accumulate for time 0 as no events
     
   R_xlen_t maxobs = max(obs_in);
   R_xlen_t nvar = covstart_in.length();
   R_xlen_t maxid = idstart_in.length(); // Number of unique patients
   bool recurrent = maxid > 0;
   double dif = 0.0;
   double * denom  = new double [ntimes]();
   double * efron_wt  = new double [ntimes]();
   double * wt_average  = new double [ntimes]();  // Average weight of people with event at each time point.
   double * denom_update  = new double [ntimes]();
   double * efron_wt_update  = new double [ntimes]();
   double * zbeta = new double [maxobs]();
   
   double frailty_penalty = 0.0;
   int64_t * frailty_group_events = new int64_t [maxid](); // Count of events for each patient (for gamma penalty weight)
   
   /* Wrap all R objects to make thread safe for read and writing  */
   RcppParallel::RVector<double> beta(beta_in);
   RcppParallel::RVector<double> frailty(frailty_in);

   RcppParallel::RVector<double> weights(weights_in);
   
   RcppParallel::RVector<int> Outcomes(Outcomes_in);
   RcppParallel::RVector<int> OutcomeTotals(OutcomeTotals_in);
   RcppParallel::RVector<int> OutcomeTotalTimes(OutcomeTotalTimes_in);

   RcppParallel::RVector<int> timein(timein_in);
   RcppParallel::RVector<int> timeout(timeout_in);

   RcppParallel::RVector<double> coval(coval_in);   
   RcppParallel::RVector<int> obs(obs_in); 
   
   std::vector<int64_t> covstart = Rcpp::fromInteger64(covstart_in); 
   std::vector<int64_t> covend = Rcpp::fromInteger64(covend_in); 
   
   RcppParallel::RVector<int> idn(idn_in);
   RcppParallel::RVector<int> idstart(idstart_in);
   RcppParallel::RVector<int> idend(idend_in);
   
   double step = 1/pow(10,decimals);
   double  d2sum = 0.0;
   
   double newlk = 0.0;
   if (lambda !=0) newlk = -(log(sqrt(lambda)) * nvar);

   
   if (recurrent == 1)
   {
#pragma omp parallel for default(none)  shared(frailty_group_events, idstart, idend, idn,  nvar, maxid, weights,  Outcomes, obs) 
     for(R_xlen_t  i = 0; i < maxid; i++)
     { /* per observation time calculations */
   
       for (R_xlen_t  idi = idstart[i] - 1; idi < idend[i] ; idi++) // iter over current covariates
       {
         R_xlen_t  rowobs = idn[idi] - 1;  // R_xlen_t is signed long, size_t is unsigned. R long vectors use signed so all numbers should be below R_xlen_t 2^48
         if (Outcomes[rowobs] > 0 ) 
         {
           frailty_group_events[i] += 1; // i is unique ID at this point so can write directly
         }
       }
     }
   }
   
   
   if ((recurrent == 1) && (theta_in != 0))
   {
     double nu = 1 / theta_in;
     double frailty_sum = 0.0;
     for(R_xlen_t rowid = 0; rowid < maxid; rowid ++)  frailty_sum += exp(frailty[rowid]);

     double frailty_mean = safelog(frailty_sum / maxid);
     frailty_penalty = 0.0;

     for (R_xlen_t ir = 0; ir < maxid ; ir ++ ) frailty_penalty += (frailty[ir] - frailty_mean)*nu;

     newlk  += frailty_penalty;
     double lik_correction = 0.0;

     if ((theta_in != 0) && (nu != 0)) {
       for (R_xlen_t rowid = 0; rowid < maxid; rowid++) {

         if(frailty_group_events[rowid] == 0) continue;
         double temp = nu > 1e7 ? frailty_group_events[rowid]*frailty_group_events[rowid]/nu :  frailty_group_events[rowid] + nu*safelog(nu/(nu+(frailty_group_events[rowid])));
         lik_correction += temp +
           std::lgamma(frailty_group_events[rowid] + nu) -
           std::lgamma(nu) -
           frailty_group_events[rowid]*safelog(nu + frailty_group_events[rowid]); // loglikelihood correction for frailty
       }
     }

     newlk += lik_correction;
   }

   
#pragma omp parallel default(none) reduction(+:wt_average[:ntimes]) shared(timeout,  weights,  Outcomes ,ntimes, maxobs)
{  
#pragma omp for
  for (R_xlen_t  rowobs = 0; rowobs < maxobs ; rowobs++) // iter over current covariates
  {
    R_xlen_t  time_index_exit = timeout[rowobs] - 1;
    if (Outcomes[rowobs] > 0 )  wt_average[time_index_exit] += weights[rowobs];
  }
}

  for(R_xlen_t r = OutcomeTotalTimes.size() -1 ; r >=0 ; r--) wt_average[OutcomeTotalTimes[r] - 1] = (OutcomeTotals[r]>0 ? wt_average[OutcomeTotalTimes[r] - 1]/static_cast<double>(OutcomeTotals[r]) : 0.0);

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
#pragma omp parallel for  default(none) shared(idstart, idend, idn , zbeta, maxid, frailty) //reduction(+:zbeta[:maxobs])
     for (R_xlen_t  i = 0; i < maxid; i++) // +
     { /* per observation time calculations */
       for (R_xlen_t  idi = idstart[i] - 1; idi < idend[i] ; idi++) // iter over current covariates
       {
         zbeta[idn[idi] - 1] += frailty[i] ; // should be one frailty per person / observation
       }
     }
   }
   
   /* Check zbeta Okay and calculate cumulative sums that do not depend on the specific covariate update*/
 double  newlk_private = 0.0;
#pragma omp parallel  default(none) reduction(+:newlk_private, denom[:ntimes], efron_wt[:ntimes])  shared(timein, timeout, zbeta, weights,  Outcomes , ntimes,maxobs)
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
    
    if (Outcomes[rowobs] > 0 )
    {
      /*cumumlative sums for event patients */
      newlk_private += zbeta_temp * weights[rowobs];
      efron_wt[time_index_exit] += risk;
      
    }
  }  
}
newlk += newlk_private;



for(R_xlen_t r = OutcomeTotalTimes.size() - 1 ; r >=0 ; r--)
{
  for (R_xlen_t k = 0; k < OutcomeTotals[r]; k++)
  {
    R_xlen_t time = OutcomeTotalTimes[r] - 1;
    double temp = (double)k
    / (double)OutcomeTotals[r];
    double d2 = denom[time] - (temp * efron_wt[time]); /* sum(denom) adjusted for tied deaths*/
      newlk -= wt_average[time]*safelog(d2);
      d2sum += wt_average[time]*safelog(d2);
  }
}

//Rcout << " d2sum " << d2sum << " newlk " << newlk << std::endl;

NumericMatrix confinterval(nvar, 2);
double threshold = R::qchisq(confint_width, 1,true, false );

Rcout << "Coefficient" << "\t" << "Lower "<< confint_width*100<<"% CI" <<  "\t" << "Upper "<< confint_width*100<<"% CI" << "\n";
for (R_xlen_t i = 0; i < nvar; i++)
{

  // lower beta
  confinterval(i,0) = beta[i];
  confinterval(i,1) = beta[i];
  
  double updatelk = newlk;
  double stepi = 1;
  
  while(stepi >=step)
  {
    int iter = 0;
    while((iter == 0) | (std::isfinite(updatelk) && (abs((-2)*(updatelk -newlk)) < threshold ) && (iter < MSTEP_MAX_ITER)))
    {

      confinterval(i,0) = confinterval(i,0) - stepi;
      dif = beta[i] - confinterval(i,0); // replicating beta -= dif - > confinterval = beta - dif -> dif = beta - confinterval
      updatelk = newlk + d2sum;
      
      for (R_xlen_t ir = 0 ; ir < ntimes; ir++)
      {
        denom_update[ir] = denom[ir];
        efron_wt_update[ir] = efron_wt[ir];
      }
      
      
      double updatelk_private = 0.0;
#pragma omp parallel  default(none) reduction(+:updatelk_private)  shared(denom_update,efron_wt_update,zbeta,covstart, covend, coval, weights, Outcomes, ntimes, obs, timein, timeout, dif, i, nvar)///*,  denom, efron_wt, newlk*/)
{
  double *denom_private = new double [ntimes];
  double *efron_wt_private= new double [ntimes];
  
  for (R_xlen_t ir = 0 ; ir < ntimes; ir++)
  {
    denom_private[ir] = 0.0;
    efron_wt_private[ir] = 0.0;
  }
  
#pragma omp for
  for (R_xlen_t covi = covstart[i] - 1; covi < covend[i]; covi++)
  {
    R_xlen_t rowobs =  (obs[covi] - 1) ;
    
    double riskold = exp(zbeta[rowobs] ); // + frailty_old_temp
    double covali =  coval[covi] ;
    double xbdif = dif * covali ; // don't update zbeta when only patient level frailty terms updated!
    
    double zbeta_updated = zbeta[(rowobs)] - xbdif;
    zbeta_updated =  zbeta_updated >22 ? 22 : zbeta_updated;
    zbeta_updated =  zbeta_updated < -200 ? -200 :  zbeta_updated;
    
    double riskdiff = (exp(zbeta_updated ) - riskold) * weights[rowobs]; //+ frailty[rowid] 
    
    R_xlen_t time_index_entry =  timein[rowobs] - 1;
    R_xlen_t time_index_exit =  timeout[rowobs] - 1;
    
    for (R_xlen_t r = time_index_exit; r > time_index_entry ; r--)
      denom_private[(r)] += riskdiff; // need to update exp(xb1 + xb2 + ) + exp(x2b1 + x2b2 +)
    
    if (Outcomes[rowobs] > 0 )
    {
      updatelk_private += xbdif * weights[rowobs];
      efron_wt_private[(time_index_exit)] += riskdiff;
    }
  }
  for (R_xlen_t r = ntimes - 1; r >= 0; r--)
  {
#pragma omp atomic
    efron_wt_update[(r)] += efron_wt_private[(r)];
#pragma omp atomic
    denom_update[(r)] += denom_private[(r)];
  }
  delete[] efron_wt_private;
  delete[] denom_private;
}

updatelk -= updatelk_private;

      for(R_xlen_t r = OutcomeTotalTimes.size() -1 ; r >=0 ; r--)
      {
        R_xlen_t time = OutcomeTotalTimes[r] - 1;
        for (R_xlen_t k = 0; k < OutcomeTotals[r]; k++)
        {
          double temp = (double)k
          / (double)OutcomeTotals[r];
          double d2 = denom_update[time] - (temp * efron_wt_update[time]); /* sum(denom) adjusted for tied deaths*/
            updatelk -= wt_average[time]*safelog(d2); // track this sum to remove nvar from newlk at start of next iteration
        }
      }
      iter++;
      Rcout << '.';
    }
  //Return the confidence interval to the penultimate iteration before the likelhood passed the 95% CI.
    confinterval(i,0) = confinterval(i,0) + stepi;
    stepi /= 10 ;
    Rcout << "<";
    
  }
  Rcout << "|\n";
  stepi = 1;
  while(stepi >=step)
  {
    int iter = 0;
    while((iter == 0) | (std::isfinite(updatelk) && (abs((-2)*(updatelk -newlk)) < threshold)  && (iter < MSTEP_MAX_ITER)))
    {
      confinterval(i,1) = confinterval(i,1) + stepi;
      dif = beta[i] - confinterval(i,1);
      updatelk = newlk + d2sum;
      
      double updatelk_private = 0.0;
      
      for (R_xlen_t ir = 0 ; ir < ntimes; ir++)
      {
        denom_update[ir] = denom[ir];
        efron_wt_update[ir] = efron_wt[ir];
      }
  updatelk_private = 0.0;
#pragma omp parallel  default(none) reduction(+:updatelk_private)  shared(denom_update,efron_wt_update,zbeta,covstart, covend,coval, weights, Outcomes, ntimes, obs, timein, timeout, dif, i, nvar)///*,  denom, efron_wt, newlk*/)
{
  double *denom_private = new double [ntimes];
  double *efron_wt_private= new double [ntimes];
  
  for (R_xlen_t ir = 0 ; ir < ntimes; ir++)
  {
    denom_private[ir] = 0.0;
    efron_wt_private[ir] = 0.0;
  }
  
#pragma omp for
  for (R_xlen_t covi = covstart[i] - 1; covi < covend[i]; covi++)
  {
    R_xlen_t rowobs =  (obs[covi] - 1) ;
    
    double riskold = exp(zbeta[rowobs] ); // + frailty_old_temp
    double covali =  coval[covi] ;
    double xbdif = dif * covali ; // don't update zbeta when only patient level frailty terms updated!
    
    double zbeta_updated = zbeta[rowobs] - xbdif;
    zbeta_updated =  zbeta_updated >22 ? 22 : zbeta_updated;
    zbeta_updated =  zbeta_updated < -200 ? -200 :  zbeta_updated;
    
    double riskdiff = (exp(zbeta_updated ) - riskold) * weights[rowobs]; //+ frailty[rowid] 
    
    R_xlen_t time_index_entry =  timein[rowobs] - 1;
    R_xlen_t time_index_exit =  timeout[rowobs] - 1;
    
    for (R_xlen_t r = time_index_exit; r > time_index_entry ; r--)
      denom_private[(r)] += riskdiff; // need to update exp(xb1 + xb2 + ) + exp(x2b1 + x2b2 +)
    
    if (Outcomes[rowobs] > 0 )
    {
      updatelk_private += xbdif * weights[rowobs];
      efron_wt_private[(time_index_exit)] += riskdiff;
    }
  }
  for (R_xlen_t r = ntimes - 1; r >= 0; r--)
  {
#pragma omp atomic
    efron_wt_update[(r)] += efron_wt_private[(r)];
#pragma omp atomic
    denom_update[(r)] += denom_private[(r)];
  }
  delete[] efron_wt_private;
  delete[] denom_private;
}

updatelk -= updatelk_private;

for(R_xlen_t r = OutcomeTotalTimes.size() -1 ; r >=0 ; r--)
{
  for (R_xlen_t k = 0; k < OutcomeTotals[r]; k++)
  {
    R_xlen_t time = OutcomeTotalTimes[r] - 1;
    double temp = (double)k
    / (double)OutcomeTotals[r];
    double d2 = denom_update[time] - (temp * (efron_wt_update[time])); /* sum(denom) adjusted for tied deaths*/
    updatelk -= wt_average[time]*safelog(d2); // track this sum to remove from newlk at start of next iteration
    
  }
}
iter++;
Rcout << '.';
if (lambda !=0) updatelk += (beta[i] * beta[i]) / (2.0 * lambda);
if (lambda !=0) newlk -= ((beta[i]-dif) * (beta[i]-dif)) / (2.0 * lambda);// Include iteration penalty for this covariate

    }
    
    
    confinterval(i,1) = confinterval(i,1) - stepi;
    stepi /= 10 ;
    Rcout << "<";
  }
  Rcout << '\n' << beta[i] << "\t" << confinterval(i,0) <<  "\t" << confinterval(i,1) << "\n";
  
}

delete[] zbeta;
delete[] denom;
delete[] efron_wt;
delete[] denom_update;
delete[] efron_wt_update;
delete[] frailty_group_events;
delete[] wt_average;

Rcout << "\nHazard ratios" << "\t" << "Lower "<< confint_width*100<<"% CI" <<  "\t" << "Upper "<< confint_width*100<<"% CI" << "\n";


for (R_xlen_t i = 0; i < nvar; i ++ )
{
  Rcout << exp(beta[i]) << "\t(" << exp(confinterval(i,0)) <<  " to " << exp(confinterval(i,1)) << ")\n";
}
return confinterval;
 }
 