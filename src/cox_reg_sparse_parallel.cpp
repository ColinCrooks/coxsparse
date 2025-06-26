#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <Rcpp.h>
#include <omp.h>
#include <RcppParallel.h>
#include "utils.h"
using namespace Rcpp;
//using namespace RcppParallel;
//' cox_reg_sparse_parallel
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
 //' * BaseHazardEntry - double vector of max(obs) for baseline hazard at each individual observation.
 //' * cumhazEntry - double vector of max(obs) for cumulative hazard at each individual observation. 
 //' * cumhaz1year - double vector of max(obs) for cumulative hazard one year after each individual observation. 
 //' * Risk - double vector of max(obs) for exp(xb) at each individual observation.
 //' * ModelSummary - double vector of length 8 to contain individual values:
 //' ** loglik - log likelihood of the model without gamma penalty
 //' ** lik_correction - gamma correction for the loglikelihood for including frailty terms
 //' ** loglik + lik_correction - the total log likelihood of the model
 //' ** theta - the value of theta used in the model
 //' ** outer_iter - the number of outer iterations performed
 //' ** final convergence of inner loop for covariates abs(1-newlk/loglik)
 //' ** final convergence of outer loop for theta
 //' ** frailty_mean - the mean of the frailty terms so they can be centred log(mean(exp(frailty)))
 //' @param obs_in An integer vector referencing for each covariate value the
 //' corresponding unique patient time in the time and outcome vectors. Of the
 //' same length as coval. The maximum value is the length of timein and timeout.
 //' @param coval_in A double vector of each covariate value sorted first by 
 //' time then by patient and by order of the covariates to be included in model.
 //' Of the same longth as obs_in.
 //' @param weights_in A double vector of weights to be applied to each unique
 //' patient time point. Of the same length as timein, timeout and outcomes. 
 //' @param  timein_in An integer vector of the start time for each unique patient 
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
 //' @param lambda Penalty weight to include for ridge regression: -log(sqrt(lambda)) * nvar
 //' @param theta_in An input starting value for theta or can be set to zero.
 //' @param MSTEP_MAX_ITER Maximum number of iterations
 //' @param MAX_EPS Threshold for maximum step change in liklihood for convergence. 
 //' @param threadn Number of threads to be used - caution as will crash if specify more 
 //' threads than available memory for copying data for each thread.
 //' @return Void: see the model data input list for the output.
 //' @export
 // [[Rcpp::export]]
Rcpp::List cox_reg_sparse_parallel(  
                              IntegerVector obs_in,
                              DoubleVector  coval_in,
                              DoubleVector  weights_in,
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
                              double theta_in ,
                              int MSTEP_MAX_ITER,
                              double MAX_EPS,
                              long unsigned int threadn) {
   omp_set_dynamic(0);     // Explicitly disable dynamic teams
   omp_set_num_threads(threadn); // Use 8 threads for all consecutive parallel regions
   
   Rcpp::Rcout.precision(10);
   
   
   // R objects size uses signed integer Maximum signed integer 2^31 - 1 ~ 2*10^9 = 2,147,483,647 so overflow unlikely using int as indices. But cpp uses size_T
   
   
   // Vectors from R index begin from 1. Need to convert to 0 index for C++ and comparisons
   //Mittal, S., Madigan, D., Burd, R. S., & Suchard, M. a. (2013). High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis. Biostatistics (Oxford, England), 1–15. doi:10.1093/biostatistics/kxt043
   
   int ntimes = max(timeout_in);  // Unique times but don't accumulate for time 0 as no events
   int maxobs = max(obs_in);
   int nvar = covstart_in.length();
   int maxid = idstart_in.length(); // Number of unique patients
   bool recurrent = maxid > 0;
   int nallvar = nvar + maxid;
   
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
   Rcout << "MSTEP_MAX_ITER : " << MSTEP_MAX_ITER << " maxobs : " << maxobs <<" ntimes : " << ntimes << " maxid :  " << maxid << " nvar : " << nvar << " nallvar : " << nallvar << std::endl;
   Rcout << "Allocating vectors, ";
   std::vector<int> frailty_group_events(maxid,0); // Count of events for each patient (for gamma penalty weight)   
   std::vector<double> theta_history(MSTEP_MAX_ITER,0.0);
   std::vector<double> thetalkl_history(MSTEP_MAX_ITER,-std::numeric_limits<double>::infinity());
  // std::vector<double> denom(ntimes,0.0);      // sum of risk of all patients at each time point
  // std::vector<double> efron_wt(ntimes,0.0);  // Sum of risk of patients with events at each time point
  double* denom = new double[ntimes]{0.0};
  double* efron_wt = new double[ntimes]{0.0};
  double* denom_private = new double[ntimes]{0.0};
  double* efron_wt_private = new double[ntimes]{0.0};
  double* wt_average = new double[ntimes]{0.0};
//   std::vector<double> wt_average(ntimes,0.0);  // Average weight of people with event at each time point.
   std::vector<double> zbeta(maxobs,0.0);
   double* derivMatrix = new double[ntimes*4]{0.0};
   std::vector<double> step(nallvar,1.0);
   std::vector<double> gdiagbeta(nvar,0.0);
   std::vector<double> gdiagfrail(maxid,0.0);
   Rcpp::DoubleVector frailty_r(maxid);
   Rcpp::DoubleVector beta_r(nvar);
   Rcpp::DoubleVector basehaz_r(ntimes);
   Rcpp::DoubleVector cumhaz_r(ntimes);
   Rcpp::DoubleVector ModelSummary_r(8);

   Rcout << "setting to zero, ";

   /* Wrap all R objects to make thread safe for read and writing  */
   
   Rcout << "pointing to R vectors in list, ";
   //Rcpp::DoubleVector beta_in = modeldata["Beta"];
   //Rcpp::DoubleVector frailty_in = modeldata["Frailty"];
   //Rcpp::DoubleVector basehaz_in = modeldata["basehaz"];
   //Rcpp::DoubleVector cumhaz_in = modeldata["cumhaz"];
   //DoubleVector BaseHazardEntry_in = modeldata["BaseHazardEntry"];
   //DoubleVector cumhazEntry_in = modeldata["cumhazEntry"];
   //DoubleVector cumhaz1year_in = modeldata["cumhaz1year"];
   //DoubleVector Risk_in = modeldata["Risk"];    */
   //Rcpp::DoubleVector ModelSummary_in =modeldata["ModelSummary"];

   Rcout << "wrapping R vectors, ";
   RcppParallel::RVector<double> coval(coval_in);
   RcppParallel::RVector<double> weights(weights_in);
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
   
   //vectors for returning calculated results to avoid copying into list for return.
   //To be zeroed before use
   Rcout << "wrapping R vectors from list, ";
   //RcppParallel::RVector<double> cumhazEntry(cumhazEntry_in);
   //RcppParallel::RVector<double> BaseHazardEntry(BaseHazardEntry_in);
   //RcppParallel::RVector<double> cumhaz1year(cumhaz1year_in);
   //RcppParallel::RVector<double> Risk(Risk_in);
  // Rcpp::DoubleVector basehaz(ntimes);
  // Rcpp::DoubleVector cumhaz(ntimes);
  // Rcpp::DoubleVector beta(nvar);
   RcppParallel::RVector<double> basehaz(basehaz_r);
   RcppParallel::RVector<double> cumhaz(cumhaz_r);
   RcppParallel::RVector<double> beta(beta_r);
   RcppParallel::RVector<double> frailty(frailty_r);
   RcppParallel::RVector<double> ModelSummary(ModelSummary_r);

   int iter_theta = 0;
   double inner_EPS = 1e-5;
   int done = 0;
   
   Rcout << "counting numerators, ";
   for (int i = 0; i < nvar; i++)
   { /* per observation time calculations */
   
   double gdiag_private = 0.0;
     
#pragma omp parallel for default(none) reduction(+:gdiag_private) shared(gdiagbeta, covstart, covend, covn, coval,  weights,  Outcomes, obs, i) //reduction(+:zbeta[:maxobs])
     for (int covi = covstart[i] - 1; covi < covend[i] ; covi++) // iter over current covariates
     {
       int row = covn[covi] - 1;//covrows[covi] - 1;  // R_xlen_t is signed long, size_t is unsigned. R vectors use signed so all numbers should be below 2,147,483,647
       int rowobs = obs[row] - 1 ;
       if (Outcomes[rowobs] > 0 ) {
         gdiag_private +=  coval[row] * weights[rowobs];//(i < nvar ? coval[row] : 1.0) * weights[rowobs]; // frailty not in derivative of beta
       }
     }
     
     gdiagbeta[i] = gdiag_private;
   }
   
   Rcout << "adding in frailty to numerators (no atomic), ";
   if (recurrent == 1)
   {
   // int*  frailty_group_events_data = frailty_group_events.data();
   // double* gdiagfrail_data = gdiagfrail.data(); 
#pragma omp parallel for default(none)  shared(gdiagfrail, frailty_group_events, idstart, idend, idn,  nvar, maxid, weights,  Outcomes, obs) //re frailty_group_events, reduction(+:gdiagfrail_data[:maxid], frailty_group_events_data[:maxid])
     for(int i = 0; i < maxid; i++)
     { /* per observation time calculations */
   
  // double gdiag_private = 0.0;
  //     int group_events = 0;
       for (int idi = idstart[i] - 1; idi < idend[i] ; idi++) // iter over current covariates
       {
         int rowobs = idn[idi] - 1;  // R_xlen_t is signed long, size_t is unsigned. R vectors use signed so all numbers should be below 2,147,483,647
         if (Outcomes[rowobs] > 0 ) 
         {
           
           gdiagfrail[i] += weights[rowobs]; // frailty not in derivative of beta
           
           frailty_group_events[i] += 1; // i is unique ID at this point so can write directly
         }
       }
// #pragma omp atomic write
//        frailty_group_events[i] = group_events;
// #pragma omp atomic write
//        gdiagvar[i + nvar] = gdiag_private;
     }
   }
   
   Rcout << "summing deaths at each time point, ";
   //double* wt_average_data = wt_average.data();
#pragma omp parallel default(none) reduction(+:wt_average[:ntimes]) shared(timeout,  weights,  Outcomes ,ntimes, maxobs)
{  
  // std::vector<double> wt_average_private(ntimes);
  // wt_average_private = {0};
#pragma omp for
  for (int rowobs = 0; rowobs < maxobs ; rowobs++) // iter over current covariates
  {
    int time_index_exit = timeout[rowobs] - 1;
    if (Outcomes[rowobs] > 0 )  wt_average[time_index_exit] += weights[rowobs];
  }
  
//   for (int r = 0; r < ntimes ; r++)
//   {
// #pragma omp atomic
//     wt_average[r] += wt_average_private[r];
//   }
  
}
Rcout << "averaging deaths at each time point, ";
for(int r = OutcomeTotalTimes.size() -1 ; r >=0 ; r--) wt_average[OutcomeTotalTimes[r] - 1] = (OutcomeTotals[r]>0 ? wt_average[OutcomeTotalTimes[r] - 1]/static_cast<double>(OutcomeTotals[r]) : 0.0);

/* Set up */
newlk = 0.0;
if (lambda !=0) newlk = -(log(sqrt(lambda)) * nvar);


loglik = 0.0;
frailty_penalty = 0.0;
d2_sum_private = 0.0;
for (int ivar = 0; ivar < nallvar; ivar++) step[ivar] = 1.0; 

// for (int ir = 0 ; ir < ntimes; ir++)
// {
//   denom[ir] = 0.0;
//   efron_wt[ir] = 0.0;
// }

// for (int ir = 0; ir < maxobs; ir++) zbeta[ir] = 0.0;
// for (int it = 0; it < ntimes*4; it++) derivMatrix[it] = 0.0;
//  for (int ivar = 0; ivar < maxid; ivar++) frailty[ivar] = 0.0;
//  for (int ivar = 0; ivar < nvar; ivar++  ) beta[ivar] = 0.0;

Rcout << "summing zbeta with covariates, ";
for (int i = 0; i < nvar; i++) // 
{ /* per observation time calculations */

  double beta_local =  beta[i] ;
//  std::vector<double> zbeta_private(maxobs);
//  zbeta_private = {0.0};
//  double* zbeta_private_data = zbeta_private.data();
#pragma omp parallel  default(none)  shared(i, covstart, covend, covn, maxobs, coval, beta_local,  obs, zbeta) //reduction(+:zbeta[:maxobs]) reduction(+:zbeta_private_data[:maxobs])
{
#pragma omp for
  for (int covi = covstart[i] - 1; covi < covend[i]; covi++)
  {
    int row = covn[covi] - 1; // R_xlen_t is signed long, size_t is unsigned. R vectors use signed so all numbers should be below 2,147,483,647
    int rowobs =  obs[row] - 1 ;
    double covali =  coval[row] ;
  //  zbeta_private_data[rowobs] += beta_local * covali ;
    zbeta[rowobs] += beta_local * covali ; // each covariate occurs once in an observation time
  }
}
// for (int rowobs = 0; rowobs < maxobs ; rowobs++)
// {
// //#pragma omp atomic
//   zbeta[rowobs] += zbeta_private[rowobs];
// }
}


if (recurrent == 1)
{
  Rcout << "summing zbeta with frailty, ";  
#pragma omp parallel for  default(none) shared(idstart, idend, idn , zbeta, maxid, frailty) //reduction(+:zbeta[:maxobs])
  for (int i = 0; i < maxid; i++) // +
  { /* per observation time calculations */

    for (int idi = idstart[i] - 1; idi < idend[i] ; idi++) // iter over current covariates
    {
    //#pragma omp atomic
      zbeta[idn[idi] - 1] += frailty[i] ; // should be one frailty per person / observation
    }
  }
}

Rcout << "accumulating denominator, ";
/* Check zbeta Okay and calculate cumulative sums that do not depend on the specific covariate update*/
newlk_private = 0.0;

#pragma omp parallel  default(none) reduction(+:newlk_private, denom[:ntimes], efron_wt[:ntimes])  shared(timein, timeout, zbeta, weights,  Outcomes , ntimes,maxobs)
{
#pragma omp for
  for (int rowobs = 0; rowobs < maxobs; rowobs++)
  {
    int time_index_entry = timein[rowobs] - 1;  // Time starts at zero but no events at this time to calculate sums. Lowest value of -1 but not used to reference into any vectors
    int time_index_exit = timeout[rowobs] - 1;
    
    double zbeta_temp = zbeta[rowobs] >22 ? 22 : zbeta[rowobs];
    zbeta_temp = zbeta_temp < -200 ? -200 : zbeta_temp;
    double risk = exp(zbeta_temp ) * weights[rowobs];
    zbeta[rowobs] = zbeta_temp;
    //cumumlative sums for all patients
    for (int r = time_index_exit; r > time_index_entry ; r--)
      denom[r] += risk;
    
    if (Outcomes[rowobs] > 0 )
    {
      /*cumumlative sums for event patients */
      newlk_private += (zbeta_temp) * weights[rowobs];
      efron_wt[time_index_exit] += risk;
      
    }
// #pragma omp atomic write
//     zbeta[(rowobs)] = zbeta_temp; // should be threadsafe without atomic as threads by rowobs
  }  
//   for (int r = 0; r < ntimes ; r++)
//   {
// #pragma omp atomic
//     efron_wt[(r)] += efron_wt_private[(r)];
// #pragma omp atomic
//     denom[(r)] += denom_private[(r)];
//   }
}

newlk += newlk_private;

Rcout << "main outer loop, ";
/* Main outer loop for theta and covariate updates */
int outer_iter = 0;
for (outer_iter = 0; outer_iter < MSTEP_MAX_ITER && done == 0; outer_iter++) 
{
  int iter = 0;
  for (iter = 0; iter <= MSTEP_MAX_ITER; iter++)
  {
    //  if (iter != 0 || outer_iter == 0)  // Don't update beta coefficients on first iteration if only change is theta/nu for gamma penalty
    {
      for (int i = 0; i < nvar; i++)
      { 
        
        double gdiag =  -gdiagbeta[i];
        double hdiag = 0.0;
        Rcout << "accumulating derivatives, ";        
        for (int ir = 0; ir < (ntimes*4); ir++) derivMatrix[ir] = 0.0;
        //double* derivMatrix_data = derivMatrix.data();
#pragma omp parallel default(none) reduction(+:derivMatrix[:ntimes*4]) shared( covstart, covend, covn, coval, weights, Outcomes, ntimes, obs, timein, timeout, zbeta,i, nvar)
{
  // std::vector<double> derivMatrix_private(ntimes*4);
  // derivMatrix_private = {0.0};

#pragma omp for
        for (int covi = covstart[i] - 1; covi < covend[i]; covi++)
        {
          int row = covn[covi] - 1; // R_xlen_t is signed long, size_t is unsigned. R vectors use signed
          int rowobs = (obs[row] - 1) ;
          
          int time_index_entry = timein[rowobs] - 1; // std vectors use unsigned can be negative though for time 0
          int time_index_exit = timeout[rowobs] - 1; // std vectors use unsigned
          
          double risk = exp(zbeta[rowobs]) * weights[rowobs];
          double covali = coval[row] ;
          double derivFirst = risk * covali;
          double derivSecond = derivFirst * covali;
          for (int r = time_index_exit ; r >time_index_entry ; r--) // keep int for calculations of indices then cast
          {
            derivMatrix[r] += derivFirst;
            derivMatrix[ntimes + r] += derivSecond;
          }
          if (Outcomes[rowobs] > 0)
          {
            
            derivMatrix[(2*ntimes) + time_index_exit] += derivFirst ;
            derivMatrix[(3*ntimes) + time_index_exit] += derivSecond ;
            
          }
        }
  
//   for (int r = 0; r < (ntimes*4); r++)
//   {
// #pragma omp atomic
//     derivMatrix[(r)] += derivMatrix_private[(r)];
//   }
}

        int exittimesN = OutcomeTotalTimes.size() -1;
        Rcout << "calculating derivatives, ";
        for(int r = exittimesN ; r >=0 ; r--)
        {
          int  time = OutcomeTotalTimes[r] - 1;
          
          for (int k = 0; k < OutcomeTotals[r]; k++)
          {
            double temp = (double)k
            / (double)OutcomeTotals[r];
            double d2 = denom[time] - (temp * efron_wt[time]); /* sum(denom) adjusted for tied deaths*/

            double temp2 = (derivMatrix[(time)] - (temp * derivMatrix[((2*ntimes) + time)])) / d2;
            gdiag += wt_average[time]*temp2; 
            hdiag += wt_average[time]*(((derivMatrix[ntimes + time] - (temp * derivMatrix[(3*ntimes) + time])) / d2) -
              (temp2 * temp2)) ;
          }
        } 

        double dif = 0; 

          /* Update */
        Rcout << "calculating increment, ";
        if (lambda !=0) {
          dif = (gdiag + (beta[i] / lambda)) / (hdiag + (1.0 / lambda));
        } else {
          dif = (gdiag ) / (hdiag);
        }

        if (fabs(dif) > step[i]) dif = (dif > 0.0) ? step[i] : -step[i];

        step[i] = ((2.0 * fabs(dif)) > (step[i] / 2.0)) ? 2.0 * fabs(dif) : (step[i] / 2.0);//Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245

        if (lambda !=0) {
          newlk += (beta[i] * beta[i]) / (2.0 * lambda);
        }

        beta[i] -= dif;

        if (lambda !=0) {
          newlk -= (beta[i] * beta[i]) / (2.0 * lambda);// Include iteration penalty for this covariate
        }

        Rcout << "updating zbeta and denominator, ";

/* Update cumulative sums dependent on denominator and zbeta so need to accumulate updates then apply them*/
        newlk_private = 0.0;
//std::vector<double> denom_private(ntimes);
//std::vector<double> efron_wt_private(ntimes);
//std::vector<double> zbeta_private(maxobs);
//denom_private = {0};
//efron_wt_private = {0};
// zbeta_private = {0.0};
// double* zbeta_private_data = zbeta_private.data();
for (int ir = 0 ; ir < ntimes; ir++)
{
  denom_private[ir] = 0.0;
  efron_wt_private[ir] = 0.0;
}
#pragma omp parallel  default(none) reduction(+:newlk_private, denom_private[:ntimes], efron_wt_private[:ntimes])  shared(denom,efron_wt,zbeta,covstart, covend,covn, coval, weights, Outcomes, ntimes, obs, timein, timeout, dif, i, nvar)///*,  denom, efron_wt, newlk*/)
{
#pragma omp for
          for (int covi = covstart[i] - 1; covi < covend[i]; covi++)
          {
            int row = (covn[covi] - 1); // R_xlen_t is signed long, size_t is unsigned. R vectors use signed so use int throughout
            int rowobs =  (obs[row] - 1) ;
            
            double riskold = exp(zbeta[rowobs] ); 
            double covali =  coval[row] ;
            
            double xbdif = dif * covali ; // don't update zbeta when only patient level frailty terms updated!
            
            double zbeta_updated = zbeta[(rowobs)] - xbdif;
            zbeta_updated =  zbeta_updated >22 ? 22 : zbeta_updated;
            zbeta_updated =  zbeta_updated < -200 ? -200 :  zbeta_updated;
        //    zbeta_private_data[(rowobs)] = zbeta_updated -  zbeta[(rowobs)] ; // Update zbeta for this covariate
        //#pragma omp atomic write
            zbeta[rowobs] = zbeta_updated; // Each covariate only once per patient per time so can update directly
            
            double riskdiff = (exp(zbeta_updated ) - riskold) * weights[rowobs]; 
            
            int time_index_entry =  timein[rowobs] - 1;
            int time_index_exit =  timeout[rowobs] - 1;
            
            for (int r = time_index_exit; r > time_index_entry ; r--)
              denom_private[(r)] += riskdiff; 
            
            if (Outcomes[rowobs] > 0 )
            {
              newlk_private += xbdif * weights[rowobs];
              efron_wt_private[time_index_exit] += riskdiff;
            }
          }
}
// for (int rowobs = 0; rowobs < maxobs ; rowobs++)
//   zbeta[rowobs] += zbeta_private[rowobs]; // Update zbeta for all covariates

          for (int r = ntimes - 1; r >= 0; r--)
          {
          //#pragma omp atomic
            efron_wt[r] += efron_wt_private[r];
          //#pragma omp atomic
            denom[r] += denom_private[r];
          }
          newlk -= newlk_private; // min  beta updated = beta - diff
      } /* end of coefficient updates for beta */
    }
    
    // std::vector<double> denom_private(ntimes,0.0);
    // std::vector<double> efron_wt_private(ntimes,0.0);

    // double* denom_private_data = denom_private.data();
    // double* efron_wt_private_data = efron_wt_private.data();
    for (int ir = 0 ; ir < ntimes; ir++)
    {
      denom_private[ir] = 0.0;
      efron_wt_private[ir] = 0.0;
    }
    if (recurrent == 1)
    {
      newlk_private = 0.0;
#pragma omp parallel for default(none)  reduction(+:newlk_private,denom_private[:ntimes],efron_wt_private[:ntimes]) shared(maxid, gdiagfrail, idn, idstart, idend, newlk, OutcomeTotalTimes,OutcomeTotals, denom, efron_wt, wt_average, theta, frailty, frailty_mean, nu, step, derivMatrix,  coval, weights, Outcomes, ntimes, obs, timein, timeout, zbeta, nvar)
      for (int i = 0; i < maxid ; i++) 
      {
        double gdiag =  -gdiagfrail[i];
        double hdiag = 0.0;
        
        std::vector<double> derivMatrix_private(ntimes*2,0.0);
        
        for (int idi = idstart[i] - 1; idi < idend[i] ; idi++)
        { // iter over current covariates
          int rowobs = idn[idi] - 1; // R_xlen_t is signed long, size_t is unsigned. R vectors use signed
          
          int time_index_entry = timein[rowobs] - 1; // std vectors use unsigned can be negative though for time 0
          int time_index_exit = timeout[rowobs] - 1; // std vectors use unsigned
          
          double risk = exp(zbeta[rowobs]) * weights[rowobs];
          
          for (int r = time_index_exit ; r >time_index_entry ; r--) derivMatrix_private[r] += risk; // keep int for calculations of indices then cast
          
          if (Outcomes[rowobs] > 0) derivMatrix_private[ntimes +  time_index_exit] += risk ;
          
        }
        
        int exittimesN = OutcomeTotalTimes.size() -1;
        
        for(int r = exittimesN ; r >=0 ; r--)
        {
          
          int  time = OutcomeTotalTimes[r] - 1;
          
          for (int k = 0; k < OutcomeTotals[r]; k++)
          {
            double temp = (double)k
            / (double)OutcomeTotals[r];
            
            double d2 = denom[time] - (temp * efron_wt[time]); /* sum(denom) adjusted for tied deaths*/
            double temp2 = (derivMatrix_private[time] - (temp * derivMatrix_private[ntimes + time])) / d2;

            gdiag += wt_average[time]*temp2; 
            hdiag += wt_average[time]*(temp2 * (1 - temp2)) ; 
          }
        } 
        
        double dif = 0;
        
        /* Update */
        
        
        if ( (theta == 0) || (std::isinf(gdiag)) || (std::isnan(gdiag)) || (std::isinf(hdiag)) || (std::isnan(hdiag)) )
        {
          dif =0;
//#pragma omp atomic write
          frailty[i] = 0;
        } else {
          
          dif = (gdiag + (exp(frailty[i] - frailty_mean) - 1.0) * nu ) /    //-g(d,w)' here is +nu(1-exp(frailty), but gdiag here is negative so use negative form of penalty
            (hdiag + (exp(frailty[i] - frailty_mean) * nu ));  // /*- frailty_mean*/again  -g(d,w)' here is -nu(exp(frailty)), but hdiag here is negative so use negative form of penalty
          
          
          if (fabs(dif) > step[i + nvar]) {
            dif = (dif > 0.0) ? step[i + nvar] : -step[i + nvar];
          }
          
//#pragma omp atomic write
          step[i + nvar] = ((2.0 * fabs(dif)) > (step[i + nvar] / 2.0)) ? 2.0 * fabs(dif) : (step[i + nvar] / 2.0);//Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
//#pragma omp atomic
          frailty[i] -= dif;
        }
        
        /* Update cumulative sums dependent on denominator and zbeta so need to accumulate updates then apply them*/
        
        // std::vector<double> denom_private(ntimes);
        // std::vector<double> efron_wt_private(ntimes);
        // denom_private = {0};
        // efron_wt_private = {0};

        for (int idi = idstart[i] - 1; idi < idend[i] ; idi++)
        {
          int rowobs = (idn[idi] - 1); // R_xlen_t is signed long, size_t is unsigned. R vectors use signed so use int throughout
          
          double riskold = exp(zbeta[rowobs] );
          
          double zbeta_updated = zbeta[rowobs] - dif;
          zbeta_updated =  zbeta_updated >22 ? 22 : zbeta_updated;
          zbeta_updated =  zbeta_updated < -200 ? -200 :  zbeta_updated;
          
//#pragma omp atomic 
          zbeta[rowobs] -= dif; // Each covariate only once per patient per time so can update directly
          
          double riskdiff = (exp(zbeta_updated ) - riskold) * weights[rowobs];
          
          int time_index_entry =  timein[rowobs] - 1;
          int time_index_exit =  timeout[rowobs] - 1;
          
          for (int r = time_index_exit; r > time_index_entry ; r--)
            denom_private[r] += riskdiff; // need to update exp(xb1 + xb2 + ) + exp(x2b1 + x2b2 +)
          
          if (Outcomes[rowobs] > 0 )
          {
            newlk_private += dif * weights[rowobs];
            efron_wt_private[time_index_exit] += riskdiff;
          }
        }

        
        
      } /* next frailty term */
      for (int r = ntimes - 1; r >= 0; r--)
      {
//#pragma omp atomic
        efron_wt[r] += efron_wt_private[r];
//#pragma omp atomic
        denom[r] += denom_private[r];
      }
      newlk -= newlk_private;
    } /* end of frailty updates */
        
    newlk += d2_sum_private;
    d2_sum_private =0;
    
    for(int r = OutcomeTotalTimes.size() -1 ; r >=0 ; r--)
    {
      int  time = OutcomeTotalTimes[r] - 1;
      for (int k = 0; k < OutcomeTotals[r]; k++)
      {
        double temp = (double)k
        / (double)OutcomeTotals[r];
        double d2 = denom[time] - (temp * efron_wt[time]); /* sum(denom) adjusted for tied deaths*/
        
        d2_sum_private += wt_average[time]*safelog(d2); // track this sum to remove from newlk at start of next iteration
        newlk -= wt_average[time]*safelog(d2); 
      }
    }
    
    /* centre exponentiated frailty estimates so penalty is minimised*/
    if (recurrent == 1 ) 
    {
      
      frailty_sum = 0.0;
      for(int rowid = 0; rowid < maxid; rowid ++)  frailty_sum += exp(frailty[rowid]);
      
      frailty_mean = safelog(frailty_sum / maxid);
      
      newlk  -= frailty_penalty;
      frailty_penalty = 0.0;
      
      for (int rowid = 0; rowid < maxid ; rowid ++ ) frailty_penalty += (frailty[rowid] - frailty_mean)*nu;
      
      newlk  += frailty_penalty;
    }
    
    
    
    Rcpp::Rcout << " Iter:  " << iter << " Cox likelihood : " << newlk << "   last likelihood :" << loglik << " theta " << theta << "\n";
    /* Check for convergence */
    
    if ((iter > 0) &&  (fabs(1.0 - (newlk / loglik))) <= MAX_EPS) break;
    
    Rcout << " done " << done << " frailty_penalty "<< frailty_penalty << " nu "<< nu << " convergence " << 1.0 - ((newlk ) / (loglik )) << "\n";
    loglik = newlk;
    
    Rcout << "Beta : " ;
    for (int i = 0; i < nvar; i ++) Rcout << beta[i] << " ";
    Rcout << '\n';
  } /* return for another iteration */
    
    
    if ( recurrent == 1 && iter > 0 && done == 0) 
    {
      
      lik_correction = 0.0;
      Rcout << "*** updating theta ***\n";
      
      if (theta != 0 && nu != 0) 
      {
        for (int rowid = 0; rowid < maxid; rowid++) {
          
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
      
      for (int ir = 0; ir <= iter_theta; ir ++) 
      {
        double theta_convergence = (ir < 1) ? 0 : fabs(1.0 - (thetalkl_history[ir] / thetalkl_history[ir-1]));
        Rcout << " iter_theta " << ir << " theta " << theta_history[ir] << " lkl " << thetalkl_history[ir] <<
          " done " << done << "convergence theta " << 
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
        
        int best_idx = 0;
        double max_theta = -std::numeric_limits<double>::infinity();
        double min_theta = std::numeric_limits<double>::infinity();
        double max_likl = -std::numeric_limits<double>::infinity();
        
        double max_theta_likl = -std::numeric_limits<double>::infinity();
        double min_theta_likl = std::numeric_limits<double>::infinity();
        
        for(int it = 0; it <= iter_theta; it++) 
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
        for(int it = 0; it <= iter_theta; it++) 
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
        
        for (int r = 0 ; r < 3; r++) 
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
              
              for(int it = 0; it <= iter_theta; it++) if(theta_history[it] > theta_lower_bound && theta_history[it] < min_theta) min_theta = theta_history[it];
              
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
              
              for(int it = 0; it <= iter_theta; it++) if (theta_history[it] < theta_upper_bound && theta_history[it] > max_theta) max_theta = theta_history[it];
              
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
for (int i = 0; i < nvar; i ++) Rcout << beta[i] << " ";
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
  for (int i = 0; i < maxid; i++) // +
  { /* per observation time calculations */
for (int idi = idstart[i] - 1; idi < idend[i] ; idi++) // iter over current covariates
{
#pragma omp atomic
  zbeta[idn[idi] - 1] -= frailty_mean ; // should be one frailty per person / observation
}
  }
}


int timesN =  OutcomeTotals.size() -1;
for (int ir = 0; ir < ntimes; ir++)
  basehaz[ir] = 0.0;
//#pragma omp parallel for default(none) shared(timesN,wt_average, OutcomeTotals, OutcomeTotalTimes, denom, efron_wt, basehaz)
for (int r =  timesN - 1; r >= 0; r--)
{
  double basehaz_private = 0.0;
  int time = OutcomeTotalTimes[r] - 1;
  for (int k = 0; k < OutcomeTotals[r]; k++)
  {
    double temp = (double)k
    / (double)OutcomeTotals[r];
    basehaz_private += wt_average[time]/(denom[time] - (temp * efron_wt[time])); /* sum(denom) adjusted for tied deaths*/
  }
  if (std::isnan(basehaz_private) || basehaz_private < 1e-100)  basehaz_private = 1e-100; //log(basehaz) required so a minimum measureable hazard is required to avoobs NaN errors.
  
#pragma omp atomic write
  basehaz[time] = basehaz_private; // should be thread safe as time unique per thread
}
Rcout << " done" <<std::endl;

Rcout << " Calculating cumulative baseline hazard..." ;

/* Carry forward last value of basehazard */
double last_value = 0.0;

cumhaz[0] = basehaz[0] ;
for (int t = 0; t < ntimes; t++)
{
  if (t>0) cumhaz[t] = cumhaz[t-1] +  basehaz[t];
  if (basehaz[t] == 0.0)
  {
    basehaz[t] = last_value;
  } else 
  {
    last_value = basehaz[t];
  }
}
Rcout << " done" <<std::endl;

 Rcout << " Freeing arrays..." ;

delete[] denom;
delete[] efron_wt;
delete[] denom_private;
delete[] efron_wt_private;
delete[] wt_average;  
delete[] derivMatrix;
/*
#pragma omp parallel for  default(none)  shared(ntimes,frailty_mean, cumhaz1year,cumhazEntry, cumhaz, BaseHazardEntry, Risk, basehaz, timein, zbeta, weights, maxobs)
for (int rowobs = 0; rowobs < maxobs ; rowobs++)
{
  int time_index_entry = timein[rowobs] - 1;  // Time starts at zero but no events at this time to calculate sums. Lowest value of -1 but not used to reference into any vectors
  int time_one_year = time_index_entry + 365;
  if (time_one_year >= ntimes) time_one_year = ntimes -1;
  BaseHazardEntry[rowobs] = basehaz[time_index_entry];
  cumhazEntry[rowobs] = cumhaz[time_index_entry];
  cumhaz1year[rowobs] = cumhaz[time_one_year];
  Risk[rowobs] = exp(zbeta[rowobs]);
}
Rcout << " done" <<std::endl;
 */

//for (int i = 0; i < maxid; i++) frailty_return[i] = frailty[i];

ModelSummary[0] = loglik;
ModelSummary[1] = lik_correction;
ModelSummary[2] = loglik + lik_correction;
ModelSummary[3] = theta;
ModelSummary[4] = outer_iter;
ModelSummary[5] = fabs(1.0 - (newlk / loglik));
ModelSummary[6] = iter_theta < 2 ? fabs(1.0 - (newlk / loglik)) : fabs(1.0 - (thetalkl_history[iter_theta-1]/thetalkl_history[iter_theta-2]));
ModelSummary[7] = frailty_mean;

Rcout << " Returning : " ;
for(int i = 0; i < 8; i++) Rcout << ModelSummary[i] << ", ";

Rcout <<  std::endl;

Rcpp::List returnList = List::create(Named("Beta") = beta_r ,
                                  _["Frailty"] = frailty_r, 
                                    _["basehaz"] = basehaz_r,
                                  _["cumhaz"] = cumhaz_r,
                                  _["ModelSummary"] = ModelSummary_r);

Rcout << "Return list created" << std::endl;
return(returnList);
Rcout << " Returned\n " ;
 }
 
 
