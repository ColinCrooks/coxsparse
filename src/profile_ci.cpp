#include <Rcpp.h>
#include "utils.h"
#include <omp.h>
#include <RcppParallel.h>
using namespace Rcpp;
//using namespace RcppParallel;
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
//' @param obs_in An integer vector referencing for each covariate value the
//' corresponding unique patient time in the time and outcome vectors. Of the
//' same length as coval. The maximum value is the length of timein and timeout.
//' @param coval_in A double vector of each covariate value sorted first by
//' time then by patient and by order of the covariates to be included in model.
//' Of the same longth as obs_in.
//' @param weights_in A double vector of weights to be applied to each unique
//' patient time point. Of the same length as timein, timeout and outcomes.
//' @param frailty_in A double vector of the frailty term for each unique patient.
//' @param timein_in An integer vector of the start time for each unique patient
//' time row, so would be the time that a patient's corresponding
//' covariate value starts. Of the same length as weights, timeout, and outcomes.
//' @param timeout_in An integer vector of the end time for each unique patient
//' time row, so would be the time that a patient's corresponding outcome
//' occurs. Of the same length as weights, timein, timeout and outcomes.
//' @param Outcomes_in An integer vector of 0 (censored) or 1 (outcome) for the
//' corresponding unique patient time. Of the same length as timein, timeout and
//' weights
//' @param OutcomeTotals_in An integer vector of the total number of outcomes that
//' occur at each unique time point. Length is the number of unique times in cohort.
//' @param OutcomeTotalTimes_in An integer vector of each unique time point that
//' outcome events are observed in the cohort. Same length as OutcomeTotals.
//' @param covn_in An integer vector mapping covariates sorted by covariate to the 
//' corresponding covariate value row in coval sorted by time and id
//' @param covstart_in An integer vector of the start row for each covariate in covn_in
//' @param covend_in An integer vector of the end row for each covariate in covn_in
//' @param idn_in An integer vector mapping unique patient IDs sorted by ID to the 
//' corresponding row in observations sorted by time and id
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
     IntegerVector covn_in,
     IntegerVector covstart_in,
     IntegerVector covend_in,
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
   
   int ntimes = max(timeout_in);
     
    // Unique times but don't accumulate for time 0 as no events
     
   int maxobs = max(obs_in);
   int nvar = covstart_in.length();
   int maxid = idstart_in.length(); // Number of unique patients
   bool recurrent = maxid > 0;
   double dif = 0.0;
   double * denom  = new double [ntimes];
   double * efron_wt  = new double [ntimes];
   for (int ir = 0 ; ir < ntimes; ir++)
   {
     denom[ir] = 0.0;
     efron_wt[ir] = 0.0;
   }
   
   double * wt_average  = new double [ntimes];  // Average weight of people with event at each time point.
   double * denom_update  = new double [ntimes];
   double * efron_wt_update  = new double [ntimes];
   for (int ir = 0 ; ir < ntimes; ir++)
   {
     denom_update[ir] = 0.0;
     efron_wt_update[ir] = 0.0;
     wt_average[ir] = 0.0;
   }
   
   double * zbeta = new double [maxobs];
   for (int ir = 0; ir < maxobs; ir++) zbeta[ir] = 0.0;

   double frailty_penalty = 0.0;
   int * frailty_group_events = new int [maxid]; // Count of events for each patient (for gamma penalty weight)
   for (int ir = 0; ir < maxid; ir++)  frailty_group_events[ir] = 0;
   
   /* Wrap all R objects to make thread safe for read and writing  */
   RcppParallel::RVector<double> beta(beta_in);
   RcppParallel::RVector<double> coval(coval_in);
   RcppParallel::RVector<double> weights(weights_in);
   RcppParallel::RVector<double> frailty(frailty_in);
   RcppParallel::RVector<int> Outcomes(Outcomes_in);
   RcppParallel::RVector<int> OutcomeTotals(OutcomeTotals_in);
   RcppParallel::RVector<int> OutcomeTotalTimes(OutcomeTotalTimes_in);
   RcppParallel::RVector<int> obs(obs_in);
   RcppParallel::RVector<int> timein(timein_in);
   RcppParallel::RVector<int> timeout(timeout_in);
   RcppParallel::RVector<int>  covn(covn_in);
   RcppParallel::RVector<int>  covstart(covstart_in);
   RcppParallel::RVector<int>  covend(covend_in);
   RcppParallel::RVector<int>  idn(idn_in);
   RcppParallel::RVector<int>  idstart(idstart_in);
   RcppParallel::RVector<int>  idend(idend_in);
   
   double step = 1/pow(10,decimals);
   double  d2sum = 0.0;
   
   double newlk = 0.0;
   if (lambda !=0) newlk = -(log(sqrt(lambda)) * nvar);

   
   if (recurrent ==1)
   {
#pragma omp parallel for default(none) shared( idstart, idend, idn,  frailty_group_events, nvar, maxid, weights,  Outcomes, obs) //re
     for(int i = 0; i < maxid; i++)
     { /* per observation time calculations */
       int group_events = 0;
       for (int idi = idstart[i] - 1; idi < idend[i] ; idi++) // iter over current covariates
       {
         int rowobs = idn[idi] - 1;  // R_xlen_t is signed long, size_t is unsigned. R vectors use signed so all numbers should be below 2,147,483,647
         if (Outcomes[rowobs] > 0 ) {
           group_events ++; // i is unique ID at this point so can write directly
         }
       }
#pragma omp atomic write
       frailty_group_events[i] = group_events;
     }
   }
   
   
   if ((recurrent == 1) && (theta_in != 0))
   {
     double nu = 1 / theta_in;
     double frailty_sum = 0.0;
     for(int rowid = 0; rowid < maxid; rowid ++)  frailty_sum += exp(frailty[rowid]);

     double frailty_mean = safelog(frailty_sum / maxid);
     frailty_penalty = 0.0;

     for (int ir = 0; ir < maxid ; ir ++ ) frailty_penalty += (frailty[ir] - frailty_mean)*nu;

     newlk  += frailty_penalty;
     double lik_correction = 0.0;

     if ((theta_in != 0) && (nu != 0)) {
       for (int rowid = 0; rowid < maxid; rowid++) {

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

   
#pragma omp parallel default(none) shared(wt_average, timeout,  weights,  Outcomes ,ntimes, maxobs)
{  
  double * wt_average_private = new double [ntimes];
  for (int ir = 0 ; ir < ntimes; ir++)    wt_average_private[ir] = 0.0;
  
#pragma omp for
  for (int rowobs = 0; rowobs < maxobs ; rowobs++) // iter over current covariates
  {
    int time_index_exit = timeout[rowobs] - 1;
    if (Outcomes[rowobs] > 0 )  wt_average_private[time_index_exit] += weights[rowobs];
  }
  
  for (int r = 0; r < ntimes ; r++)
  {
#pragma omp atomic
    wt_average[r] += wt_average_private[r];
  }
  
  delete[] wt_average_private;
}

for(int r = OutcomeTotalTimes.size() -1 ; r >=0 ; r--) wt_average[OutcomeTotalTimes[r] - 1] = (OutcomeTotals[r]>0 ? wt_average[OutcomeTotalTimes[r] - 1]/static_cast<double>(OutcomeTotals[r]) : 0.0);


   /* Cumulative sums that do not depend on the specific covariate update*/
   for (int i = 0; i < nvar; i++) // 
   { /* per observation time calculations */
   
   double beta_local =  beta[i] ;
     
#pragma omp parallel  default(none) shared(i, covstart, covend, covn, maxobs, coval, beta_local,  obs, zbeta) //reduction(+:zbeta[:maxobs])
{
  double* zbeta_private = new double [maxobs];
  for (int ir = 0; ir < maxobs; ir++) zbeta_private[ir] = 0.0;
  
#pragma omp for
  for (int covi = covstart[i] - 1; covi < covend[i]; covi++)
  {
    int row = covn[covi] - 1; // R_xlen_t is signed long, size_t is unsigned. R vectors use signed so all numbers should be below 2,147,483,647
    int rowobs =  obs[row] - 1 ;
    double covali =  coval[row] ;
    zbeta_private[rowobs] += beta_local * covali ;
  }
  
  for (int rowobs = 0; rowobs < maxobs ; rowobs++)
  {
#pragma omp atomic
    zbeta[rowobs] += zbeta_private[rowobs];
  }
  delete[] zbeta_private;
}
   }
   
   if (recurrent == 1)
   {
#pragma omp parallel for  default(none) shared(idstart, idend, idn , zbeta, maxid, frailty) //reduction(+:zbeta[:maxobs])
     for (int i = 0; i < maxid; i++) // +
     { /* per observation time calculations */
   
       for (int idi = idstart[i] - 1; idi < idend[i] ; idi++) // iter over current covariates
       {
    #pragma omp atomic
         zbeta[idn[idi] - 1] += frailty[i] ; // should be one frailty per person / observation
       }
     }
   }
   
   /* Check zbeta Okay and calculate cumulative sums that do not depend on the specific covariate update*/
 double  newlk_private = 0.0;
#pragma omp parallel  default(none) reduction(+:newlk_private)  shared(efron_wt,denom,timein, timeout, zbeta, weights,  Outcomes , ntimes,maxobs)
{
  double * denom_private  = new double [ntimes];
  double * efron_wt_private  = new double [ntimes];
  for (int ir = 0 ; ir < ntimes; ir++)
  {
    denom_private[ir] = 0.0;
    efron_wt_private[ir] = 0.0;
  }

#pragma omp for
  for (int rowobs = 0; rowobs < maxobs; rowobs++)
  {
    int time_index_entry = timein[rowobs] - 1;  // Time starts at zero but no events at this time to calculate sums. Lowest value of -1 but not used to reference into any vectors
    int time_index_exit = timeout[rowobs] - 1;
    //      int rowid = id[rowobs] - 1;
    
    double zbeta_temp = zbeta[(rowobs)] >22 ? 22 : zbeta[(rowobs)];
    zbeta_temp = zbeta_temp < -200 ? -200 : zbeta_temp;
    double risk = exp(zbeta_temp ) * weights[rowobs];
    
    //cumumlative sums for all patients
    for (int r = time_index_exit; r > time_index_entry ; r--)
      denom_private[(r)] += risk;
    
    if (Outcomes[rowobs] > 0 )
    {
      /*cumumlative sums for event patients */
      newlk_private += (zbeta_temp) * weights[rowobs];
      efron_wt_private[(time_index_exit)] += risk;
      
    }
#pragma omp atomic write
    zbeta[(rowobs)] = zbeta_temp; // should be threadsafe without atomic as threads by rowobs
  }  
  for (int r = 0; r < ntimes ; r++)
  {
#pragma omp atomic
    efron_wt[(r)] += efron_wt_private[(r)];
#pragma omp atomic
    denom[(r)] += denom_private[(r)];
  }
  delete[] efron_wt_private;
  delete[] denom_private;
}
newlk += newlk_private;



for(int r = OutcomeTotalTimes.size() - 1 ; r >=0 ; r--)
{
  for (int k = 0; k < OutcomeTotals[r]; k++)
  {
    int time = OutcomeTotalTimes[r] - 1;
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
for (int i = 0; i < nvar; i++)
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
      
      for (int ir = 0 ; ir < ntimes; ir++)
      {
        denom_update[ir] = denom[ir];
        efron_wt_update[ir] = efron_wt[ir];
      }
      
      
      double updatelk_private = 0.0;
#pragma omp parallel  default(none) reduction(+:updatelk_private)  shared(denom_update,efron_wt_update,zbeta,covstart, covend,covn, coval, weights, Outcomes, ntimes, obs, timein, timeout, dif, i, nvar)///*,  denom, efron_wt, newlk*/)
{
  double *denom_private = new double [ntimes];
  double *efron_wt_private= new double [ntimes];
  
  for (int ir = 0 ; ir < ntimes; ir++)
  {
    denom_private[ir] = 0.0;
    efron_wt_private[ir] = 0.0;
  }
  
#pragma omp for
  for (int covi = covstart[i] - 1; covi < covend[i]; covi++)
  {
    int row = (covn[covi] - 1); // R_xlen_t is signed long, size_t is unsigned. R vectors use signed so use int throughout
    int rowobs =  (obs[row] - 1) ;
    
    double riskold = exp(zbeta[rowobs] ); // + frailty_old_temp
    double covali =  coval[row] ;
    double xbdif = dif * covali ; // don't update zbeta when only patient level frailty terms updated!
    
    double zbeta_updated = zbeta[(rowobs)] - xbdif;
    zbeta_updated =  zbeta_updated >22 ? 22 : zbeta_updated;
    zbeta_updated =  zbeta_updated < -200 ? -200 :  zbeta_updated;
    
    double riskdiff = (exp(zbeta_updated ) - riskold) * weights[rowobs]; //+ frailty[rowid] 
    
    int time_index_entry =  timein[rowobs] - 1;
    int time_index_exit =  timeout[rowobs] - 1;
    
    for (int r = time_index_exit; r > time_index_entry ; r--)
      denom_private[(r)] += riskdiff; // need to update exp(xb1 + xb2 + ) + exp(x2b1 + x2b2 +)
    
    if (Outcomes[rowobs] > 0 )
    {
      updatelk_private += xbdif * weights[rowobs];
      efron_wt_private[(time_index_exit)] += riskdiff;
    }
  }
  for (int r = ntimes - 1; r >= 0; r--)
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

      for(int r = OutcomeTotalTimes.size() -1 ; r >=0 ; r--)
      {
        int time = OutcomeTotalTimes[r] - 1;
        for (int k = 0; k < OutcomeTotals[r]; k++)
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
      
      for (int ir = 0 ; ir < ntimes; ir++)
      {
        denom_update[ir] = denom[ir];
        efron_wt_update[ir] = efron_wt[ir];
      }
  updatelk_private = 0.0;
#pragma omp parallel  default(none) reduction(+:updatelk_private)  shared(denom_update,efron_wt_update,zbeta,covstart, covend,covn, coval, weights, Outcomes, ntimes, obs, timein, timeout, dif, i, nvar)///*,  denom, efron_wt, newlk*/)
{
  double *denom_private = new double [ntimes];
  double *efron_wt_private= new double [ntimes];
  
  for (int ir = 0 ; ir < ntimes; ir++)
  {
    denom_private[ir] = 0.0;
    efron_wt_private[ir] = 0.0;
  }
  
#pragma omp for
  for (int covi = covstart[i] - 1; covi < covend[i]; covi++)
  {
    int row = (covn[covi] - 1); // R_xlen_t is signed long, size_t is unsigned. R vectors use signed so use int throughout
    int rowobs =  (obs[row] - 1) ;
    
    double riskold = exp(zbeta[rowobs] ); // + frailty_old_temp
    double covali =  coval[row] ;
    double xbdif = dif * covali ; // don't update zbeta when only patient level frailty terms updated!
    
    double zbeta_updated = zbeta[(rowobs)] - xbdif;
    zbeta_updated =  zbeta_updated >22 ? 22 : zbeta_updated;
    zbeta_updated =  zbeta_updated < -200 ? -200 :  zbeta_updated;
    
    double riskdiff = (exp(zbeta_updated ) - riskold) * weights[rowobs]; //+ frailty[rowid] 
    
    int time_index_entry =  timein[rowobs] - 1;
    int time_index_exit =  timeout[rowobs] - 1;
    
    for (int r = time_index_exit; r > time_index_entry ; r--)
      denom_private[(r)] += riskdiff; // need to update exp(xb1 + xb2 + ) + exp(x2b1 + x2b2 +)
    
    if (Outcomes[rowobs] > 0 )
    {
      updatelk_private += xbdif * weights[rowobs];
      efron_wt_private[(time_index_exit)] += riskdiff;
    }
  }
  for (int r = ntimes - 1; r >= 0; r--)
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

for(int r = OutcomeTotalTimes.size() -1 ; r >=0 ; r--)
{
  for (int k = 0; k < OutcomeTotals[r]; k++)
  {
    int time = OutcomeTotalTimes[r] - 1;
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


for (int i = 0; i < nvar; i ++ )
{
  Rcout << exp(beta[i]) << "\t(" << exp(confinterval(i,0)) <<  " to " << exp(confinterval(i,1)) << ")\n";
}
return confinterval;
 }
 