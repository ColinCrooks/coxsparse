#include <Rcpp.h>
#include <omp.h>
#include <RcppParallel.h>
#include "utils.h"
using namespace Rcpp;
using namespace RcppParallel;
//' cox_reg_sparse_parallel
//' 
//' Description
//' Implementation of a Cox proportional hazards model using 
//' a sparse data structure. The model is fitted with cyclical 
//' coordinate descent (after Mittal et al (2013).
//' OpenMP is used to parallelise the updating of cumulative 
//' values and rcppParallel objects are used to make R objects
//' threadsafe.
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
//' The data structure is a deconstructed sparse matrix.
//' 
//' A function using the same data structure to calculate profile
//' confidence intervals with a crude search pattern is provided.
//' 
//' @param covrowlist_in A list in R of integer vectors of length nvar. 
//' Each entry in the list corresponds to a covariate.
//' Each vector contains the indices for the rows in the coval list 
//' that correspond to that vector. Maximum value of any indices corresponds
//' therefore to the length of coval.
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
//' @param nvar Number of covariates
//' @param lambda Penalty weight to include for ridge regression: -log(sqrt(lambda)) * nvar
//' @param MSTEP_MAX_ITER Maximum number of iterations
//' @param MAX_EPS Threshold for maximum step change in liklihood for convergence. 
//' @param threadn Number of threads to be used - caution as will crash if specify more 
//' threads than available memory for copying data for each thread.
//' @return A list of:
//' * Beta Fitted coefficients
//' * BaseHaz Baseline hazard values for each unique observed time
//'   calculated with the fitted coefficients and Efron weights.
//' * CumHaz Cumulative values from the baseline hazard values.
//' * BaseHazardEntry Baseline hazard expanded and sorted to correspond to each 
//'   patient time in the original data provided in timein.
//' * CumHazAtEntry Cumulative hazard values expanded and sorted to correspond to each 
//'   patient time in the original data provided in timein.
//' * CumHazOneYear Cumulative hazard values expanded and sorted to correspond to one
//'   full year each after each patient time in the original data provided in timein.
//' * Risk The hazard (exp(xb)) for each  patient time in the original data 
//'   provided in timein.
//' 
//' @export
// [[Rcpp::export]]
List cox_reg_sparse_parallel(List covrowlist_in,
                              DoubleVector beta_in,
                              IntegerVector id_in,
                              int recurrent,
                              IntegerVector obs_in,
                              DoubleVector coval_in,
                              DoubleVector weights_in,
                              IntegerVector timein_in ,
                              IntegerVector timeout_in ,
                              IntegerVector Outcomes_in ,
                              IntegerVector OutcomeTotals_in ,
                              IntegerVector OutcomeTotalTimes_in,
                              int nvar,
                              double lambda,
                              double theta_in ,
                              int MSTEP_MAX_ITER,
                              double MAX_EPS,
                              long unsigned int threadn) {
  omp_set_dynamic(0);     // Explicitly disable dynamic teams
  omp_set_num_threads(threadn); // Use 8 threads for all consecutive parallel regions

  Rcpp::Rcout.precision(10);

  List covrowlist(covrowlist_in);
// R objects size uses signed integer Maximum signed integer 2^31 - 1 ~ 2*10^9 = 2,147,483,647 so overflow unlikely using int as indices. But cpp uses size_T

  //  RVector<double> zbeta_internal(zbeta); // if want to bring in zbeta from the user
  // Vectors from R index begin from 1. Need to convert to 0 index for C++ and comparisons
  //Mittal, S., Madigan, D., Burd, R. S., & Suchard, M. a. (2013). High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis. Biostatistics (Oxford, England), 1–15. doi:10.1093/biostatistics/kxt043

  int ntimes = max(timeout_in);  // Unique times but don't accumulate for time 0 as no events
  int maxobs = max(obs_in);
  int maxid = max(id_in);

  double  loglik = 0.0;
  double frailty_sum = 0.0;
  double lik_correction = 0.0;
  
  double theta = theta_in;
  double nu = theta_in == 0 ? 0 : 1.0/theta_in;
  double frailty_mean = 0;
//  double old_lik_correction = 0.0;
  

  DoubleVector frl(maxid);
  RVector<double> frailty(frl);
  double frailty_penalty_old = 0.0;
  double frailty_penalty = 0.0;
  int * frailty_group_events = new int [maxid]; // Count of events for each patient (for gamma penalty weight)
//  double * outcomes_wt = new double [maxid]; // sum of weights of events for each patient for first derivative for frailty
//  double * frailty_old = new double [maxid];
  for (int ir = 0; ir < maxid; ir++) {
    frailty[ir] = 0.0;
 //   frailty_old[ir] = 0.0;
//    outcomes_wt[ir] = 0.0;
    frailty_group_events[ir] = 0;
  }

  double * denom  = new double [ntimes];      // sum of risk of all patients at each time point
  double * efron_wt  = new double [ntimes];  // Sum of risk of patients with events at each time point
  double * wt_average  = new double [ntimes];  // Average weight of people with event at each time point.
  
  for (int ir = 0 ; ir < ntimes; ir++)
  {
    denom[ir] = 0.0;
    efron_wt[ir] = 0.0;
    wt_average[ir] = 0.0;
  }

  double * zbeta = new double [maxobs];
  for (int ir = 0; ir < maxobs; ir++) zbeta[ir] = 0.0;

  double * d2sum = new double [nvar + maxid];
  for (int ivar = 0; ivar < (nvar + maxid); ivar++) d2sum[ivar] = 0.0;

  double * derivMatrix = new double [ntimes*4];
  for (int ir = 0; ir < ntimes*4; ir++) derivMatrix[ir] = 0.0;

  /* Wrap all R objects to make thread safe for read and writing  */
  RVector<double> beta(beta_in);
  RVector<double> coval(coval_in);
  RVector<double> weights(weights_in);
  RVector<int> Outcomes(Outcomes_in);
  RVector<int> OutcomeTotals(OutcomeTotals_in);
  RVector<int> OutcomeTotalTimes(OutcomeTotalTimes_in);
  RVector<int> obs(obs_in);
  RVector<int> id(id_in);
  RVector<int> timein(timein_in);
  RVector<int> timeout(timeout_in);

  double * step = new double [nvar + maxid];
  for (int ivar = 0; ivar < nvar + maxid; ivar++) {
    step[ivar] = 1.0;
  }
  
  double * gdiagvar = new double [nvar + maxid];
  for (int ivar = 0; ivar < nvar + maxid; ivar++) gdiagvar[ivar] = 0.0;
  
   int iter_theta = 0;
   double inner_EPS = 0.0001;
   int done = 0;
   std::vector<double> theta_history(MSTEP_MAX_ITER);
   theta_history = {0.0};
   std::vector<double> thetalkl_history(MSTEP_MAX_ITER);
   thetalkl_history = {-std::numeric_limits<double>::infinity()};

   double** coef_history = new double* [MSTEP_MAX_ITER];
   for (int it = 0; it < MSTEP_MAX_ITER; it++)
   {
     coef_history[it] = new double [nvar];
     for (int j = 0; j < nvar; j++)
       coef_history[it][j] = 0.0;
   }
   double** frailty_history = new double* [MSTEP_MAX_ITER];
   for (int it = 0; it < MSTEP_MAX_ITER; it++)
   {
     frailty_history[it] = new double [maxid];
     for (int j = 0; j < maxid; j++)
       frailty_history[it][j] = 0.0;
   }  
 // int Outcome_ntimes = OutcomeTotalTimes.size();
  
  DoubleVector bh(ntimes);
  RVector<double> basehaz(bh);
  for (int ir = 0; ir < ntimes; ir++)
    basehaz[ir] = 0.0;
  
  DoubleVector ch(ntimes);
  RVector<double> cumhaz(ch);

  
  // Weight and count of events summed per person
  for(int rowobs = 0; rowobs < maxobs; rowobs++) {
    int rowid = id[rowobs] - 1;
    if(Outcomes[rowobs] > 0) { 
 //     outcomes_wt[rowid] += weights[rowobs];
      if (weights[rowobs] > 0) frailty_group_events[rowid] += Outcomes[rowobs];
    }
  }
  
  
for (int theta_iter = 0; theta_iter < MSTEP_MAX_ITER && done == 0; theta_iter++) 
{
    double newlk = 0.0;
    loglik = 0.0;
    frailty_sum = 0.0;
    if (lambda !=0) newlk = -(log(sqrt(lambda)) * nvar);
    
    for (int ivar = 0; ivar < nvar + maxid; ivar++) 
    {
      gdiagvar[ivar] = 0.0; 
 //     beta[ivar] = 0.0;
    }
    
    for (int ivar = 0; ivar < nvar + maxid; ivar++) 
    {
      step[ivar] = 1.0; 
      d2sum[ivar] = 0.0;
    }
    
 //   for (int ir = 0; ir < maxid; ir++) {
  //    frailty[ir] = 0.0;
  //    frailty_old[ir] = 0.0;
   // }

    for (int ir = 0 ; ir < ntimes; ir++)
    {
      denom[ir] = 0.0;
      efron_wt[ir] = 0.0;
      wt_average[ir] = 0.0;
    }
    
    for (int ir = 0; ir < maxobs; ir++) zbeta[ir] = 0.0;
    for (int ir = 0; ir < ntimes*4; ir++) derivMatrix[ir] = 0.0;
    
    for (int i = 0; i < nvar + maxid; i++)
    { /* per observation time calculations */
    IntegerVector covrows_in = covrowlist[i];
      RVector<int> covrows(covrows_in);
      int rowN = covrows.size();
  
      double gdiag_private = 0.0;
      double beta_local = i < nvar ? beta[i] : frailty[i];
  #pragma omp parallel  default(none) reduction(+:gdiag_private) shared(rowN, i, maxobs, covrows, coval, beta_local, weights, frailty, Outcomes, obs, id, zbeta) //reduction(+:zbeta[:maxobs])
  {
    double* zbeta_private = new double [maxobs];
    for (int ir = 0; ir < maxobs; ir++) zbeta_private[ir] = 0.0;
  
    int size = omp_get_num_threads(); // get total number of processes
    int rank = omp_get_thread_num(); // get rank of current ( range 0 -> (num_threads - 1) )
  
    for (int covi = rank * rowN / size; covi < (rank + 1) * rowN / size ; covi++)
    {
      int row = covrows[covi] - 1; // R_xlen_t is signed long, size_t is unsigned. R vectors use signed so all numbers should be below 2,147,483,647
      int rowobs = (i < nvar) ? (obs[row] - 1) : row ;
      //    int rowid = id[rowobs] - 1;
      double covali = i < nvar ? coval[row] : 1.0;
      zbeta_private[rowobs] += beta_local * covali ;
      if (Outcomes[rowobs] > 0 ) gdiag_private += covali * weights[rowobs]; // frailty not in derivative of beta
    }
    for (int rowobs = 0; rowobs < maxobs ; rowobs++)
    {
  #pragma omp atomic
      zbeta[rowobs] += zbeta_private[rowobs];
    }
    delete[] zbeta_private;
  }
  //    Rcout << "zbeta " <<  zbeta[0]  <<  " newlk " << newlk << " gdiag " << gdiag_private<<"\n";
  gdiagvar[i] = gdiag_private;
    }
  
    /* Cumulative sums that do not depend on the specific covariate update*/
    double newlk_private = 0.0;
  #pragma omp parallel  default(none) reduction(+:newlk_private)  shared(efron_wt,wt_average,denom,timein, timeout, zbeta, weights, frailty, Outcomes ,id, ntimes,maxobs)
  {
    double * denom_private  = new double [ntimes];
    double * efron_wt_private  = new double [ntimes];
    double * wt_average_private = new double [ntimes];
    for (int ir = 0 ; ir < ntimes; ir++)
    {
      denom_private[ir] = 0.0;
      efron_wt_private[ir] = 0.0;
      wt_average_private[ir] = 0.0;
    }
    int size = omp_get_num_threads(); // get total number of processes
    int rank = omp_get_thread_num(); // get rank of current ( range 0 -> (num_threads - 1) )
  
    for (int rowobs = (rank * maxobs / size); rowobs < ((rank + 1) * maxobs / size) ; rowobs++)
    {
      int time_index_entry = timein[rowobs] - 1;  // Time starts at zero but no events at this time to calculate sums. Lowest value of -1 but not used to reference into any vectors
      int time_index_exit = timeout[rowobs] - 1;
      int rowid = id[rowobs] - 1;
      
      double zbeta_temp = zbeta[(rowobs)] >22 ? 22 : zbeta[(rowobs)];
      zbeta_temp = zbeta_temp < -200 ? -200 : zbeta_temp;
      double risk = exp(zbeta_temp ) * weights[rowobs]; //+ frailty[rowid] 
  
      //cumumlative sums for all patients
      for (int r = time_index_exit; r > time_index_entry ; r--)
        denom_private[(r)] += risk;
  
      if (Outcomes[rowobs] > 0 )
      {
        /*cumumlative sums for event patients */
        newlk_private += (zbeta_temp) * weights[rowobs];
        efron_wt_private[(time_index_exit)] += risk;
        wt_average_private[time_index_exit] += weights[rowobs];
      }
  #pragma omp atomic write
      zbeta[(rowobs)] = zbeta_temp; // should be threadsafe without atomic as threads by rowobs
    }
    for (int r = 0; r < ntimes ; r++)
    {
  #pragma omp atomic
      efron_wt[(r)] += efron_wt_private[(r)];
  #pragma omp atomic
      wt_average[r] += wt_average_private[r];
  #pragma omp atomic
      denom[(r)] += denom_private[(r)];
    }
    delete[] efron_wt_private;
    delete[] denom_private;
    delete[] wt_average_private;
  }
  newlk += newlk_private;
  
  for(int r = OutcomeTotalTimes.size() -1 ; r >=0 ; r--) wt_average[OutcomeTotalTimes[r] - 1] = (OutcomeTotals[r]>0 ? wt_average[OutcomeTotalTimes[r] - 1]/static_cast<double>(OutcomeTotals[r]) : 0.0);
  
  /* Vectors for holding intermediate values in inference loop */
  int lastvar = nvar + maxid -1;

  for (int iter = 0; iter <= MSTEP_MAX_ITER; iter++)
  {
    int rowN = 0;
    for (int i = 0; i < (nvar + maxid); i++)
    { 
   //   if (update_theta == 0 && i >= nvar) continue;
      double gdiag = 0.0;
      
      IntegerVector covrows_in = covrowlist[i];
      RVector<int> covrows(covrows_in);
      rowN = covrows.size();
      
      newlk += d2sum[lastvar];   // newlk is sum across all event times delta*risk -log(denom) - newlk contains sum across all variables, but updated with each beta update
  
      d2sum[i] = 0.0;
   //   if (i < nvar) {
      gdiag = -gdiagvar[i];
    //  } else {
      /* for frailty groups need estimates per frailty group */
    //    gdiag = -outcomes_wt[i - nvar];
    //  }
  
      double hdiag = 0.0;
      
      for (int ir = 0; ir < (ntimes*4); ir++) derivMatrix[ir] = 0.0;
  
  #pragma omp parallel default(none) shared(rowN, derivMatrix, covrows, coval, weights, frailty, frailty_mean, Outcomes, ntimes, obs, id, timein, timeout, zbeta,i, nvar)
  {
      double * derivMatrix_private= new double [ntimes*4];
      for (int ir = 0; ir < (ntimes*4); ir++) derivMatrix_private[ir] = 0.0;
      
      int size = omp_get_num_threads(); // get total number of processes
      int rank = omp_get_thread_num(); // get rank of current ( range 0 -> (num_threads - 1) )
    
      for (int covi = (rank * rowN / size); covi < (rank + 1) * rowN / size; covi++)
      {
        int row = covrows[covi] - 1; // R_xlen_t is signed long, size_t is unsigned. R vectors use signed
        int rowobs = (i < nvar) ? (obs[row] - 1) : row ;
        int rowid = id[rowobs] - 1;
        
        int time_index_entry = timein[rowobs] - 1; // std vectors use unsigned can be negative though for time 0
        int time_index_exit = timeout[rowobs] - 1; // std vectors use unsigned
    
        double risk = exp(zbeta[(rowobs)] ) * weights[rowobs]; //+ frailty[rowid]
        double covali = (i < nvar) ? coval[row]: 1;
        double derivFirst = risk * covali;
        double derivSecond = derivFirst * covali;
        for (int r = time_index_exit ; r >time_index_entry ; r--) // keep int for calculations of indices then cast
        {
          derivMatrix_private[(r)] += derivFirst;
          derivMatrix_private[(ntimes + r)] += derivSecond;
        }
        if (Outcomes[rowobs] > 0)
        {
       
          derivMatrix_private[((2*ntimes) + time_index_exit)] += derivFirst ;
          derivMatrix_private[((3*ntimes) + time_index_exit)] += derivSecond ;
       
        }
      }
    
      for (int r = 0; r < (ntimes*4); r++)
      {
    #pragma omp atomic
        derivMatrix[(r)] += derivMatrix_private[(r)];
      }
      
      delete[] derivMatrix_private;
  }
  
    int   exittimesN = OutcomeTotalTimes.size() -1;
  
    for(int r = exittimesN ; r >=0 ; r--)
    {
  
      int  time = OutcomeTotalTimes[r] - 1;
      int   eventsum = OutcomeTotals[r]; 
      double   mean_wt = wt_average[time];
  
      for (int k = 0; k < eventsum; k++)
      {
        double temp = (double)k
        / (double)eventsum;
        double d2 = denom[time] - (temp * efron_wt[time]); /* sum(denom) adjusted for tied deaths*/
        d2sum[i] += mean_wt*safelog(d2); // track this sum to remove from newlk at start of next iteration
        newlk -= mean_wt*safelog(d2);
        
        double temp2 = (derivMatrix[(time)] - (temp * derivMatrix[((2*ntimes) + time)])) / d2;
        gdiag += mean_wt*temp2;
        hdiag += mean_wt*(((derivMatrix[(ntimes + time)] - (temp * derivMatrix[((3*ntimes) + time)])) / d2) -
                  (temp2 * temp2)) ;// if covariates were 1 this reduces to temp2 - temp2*temp2 or (temp2 * (1 - temp2));
      }
    } 
    
    /* Check if newlk changes are positive or negative and do halving here */
    double dif = 0; 
  
  /* Update */
  
    if (i < nvar) { 
 //     oldbeta[i] = beta[i]; // track for halving
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
      
   //   Rcout << " old beta2[i] " << oldbeta2[i] << " old beta[i] " << oldbeta[i] <<" beta[i] "<<beta[i] << std::endl;
      
    } else if ((recurrent == 1 ) && (theta != 0)) {
      
  //    frailty_old[i-nvar] = frailty[i-nvar];
   
       
       if ( (theta == 0) || (std::isinf(gdiag)) || (std::isnan(gdiag)) || (std::isinf(hdiag)) || (std::isnan(hdiag)) ) {
             dif =0;
             frailty[i - nvar] = 0;
       } else {
  
         dif = (gdiag + ((exp(frailty[i - nvar] ) - 1.0) * nu )) /    //-g(d,w)' here is +nu(1-exp(frailty), but gdiag here is negative so use negative form of penalty
           (hdiag + (exp(frailty[i - nvar] ) * nu ) );  // /*- frailty_mean*/again  -g(d,w)' here is -nu(exp(frailty)), but hdiag here is negative so use negative form of penalty
       
  //Rcout << " gdiag " << gdiag << " hdiag " << hdiag << " frailty "  << frailty[i-nvar] << " frailty mean " << frailty_mean << " 1o penalty " << (exp(frailty[i-nvar]- frailty_mean)-1)*nu << " 2o penalty " << (exp(frailty[i-nvar]- frailty_mean))*nu << " diff " << (gdiag + (exp(frailty[i-nvar]- frailty_mean)-1)*nu) /  (hdiag + (exp(frailty[i-nvar]- frailty_mean))*nu) <<  std::endl;   //penalty substracted, but gdiag here is negative
       
       if (fabs(dif) > step[i]) {
         dif = (dif > 0.0) ? step[i] : -step[i];
       }
  //     Rcout << " i " << i - nvar << " old frail2[i - nvar] " << frailty_old2[i - nvar] << " old frail[i - nvar] " << frailty_old[i - nvar] <<" frail[i - nvar] "<<frailty[i - nvar] << std::endl;
       
       step[i] = ((2.0 * fabs(dif)) > (step[i] / 2.0)) ? 2.0 * fabs(dif) : (step[i] / 2.0);//Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
       frailty[i - nvar] -= dif;
       }
        
    }
  
  
  //  if ( i == 1 )   Rcout << "After diff update newlk " << newlk << " loglik " << loglik << " delta_newlk " << delta_newlk[i] << std::endl;
    /* Update cumulative sums dependent on denominator and zbeta so need to accumulate updates then apply them*/
    double newlk_private = 0.0;
  //#pragma omp parallel  default(none) reduction(+:newlk_private)  shared(rowN, denom,efron_wt,zbeta,covrows, coval, weights, frailty, frailty_old, Outcomes, ntimes, obs, id, timein, timeout,maxobs, dif, i, nvar)///*,  denom, efron_wt, newlk*/)
  {
    double *denom_private = new double [ntimes];
    double *efron_wt_private= new double [ntimes];
    
    for (int ir = 0 ; ir < ntimes; ir++)
    {
      denom_private[ir] = 0.0;
      efron_wt_private[ir] = 0.0;
    }
    
    int size = omp_get_num_threads(); // get total number of processes
    int rank = omp_get_thread_num(); // get rank of current ( range 0 -> (num_threads - 1) )
    for (int covi = (rank * rowN / size); covi < ((rank + 1) * rowN / size) ; covi++)
    {
      int row = (covrows[(covi)] - 1); // R_xlen_t is signed long, size_t is unsigned. R vectors use signed so use int throughout
      int rowobs = (i < nvar) ? (obs[row] - 1) : row ;
      int rowid = id[rowobs] -1;
      
     // double frailty_old_temp = (i < nvar) ?  frailty[rowid] : frailty_old[rowid]; // don't keep updating frailty when it isn't being updated!
      
      double riskold = exp(zbeta[rowobs]  ); //+ frailty_old_temp
      double covali = (i < nvar) ? coval[row] : 1.0;
      double xbdif =   dif * covali ; //(i < nvar) ?: 0.0; // don't update zbeta when only patient level frailty terms updated!
      
      double zbeta_updated = zbeta[(rowobs)] - xbdif;
      zbeta_updated =  zbeta_updated >22 ? 22 : zbeta_updated;
      zbeta_updated =  zbeta_updated < -200 ? -200 :  zbeta_updated;
      
 //     if ( i < nvar) {
  #pragma omp atomic write
      zbeta[(rowobs)] = zbeta_updated; // Each covariate only once per patient per time so can update directly
  //    }
      
      double riskdiff = (exp(zbeta_updated ) - riskold) * weights[rowobs]; // + frailty[rowid] 
      
      int time_index_entry =  timein[rowobs] - 1;
      int time_index_exit =  timeout[rowobs] - 1;
      
      for (int r = time_index_exit; r > time_index_entry ; r--)
        denom_private[(r)] += riskdiff; // need to update exp(xb1 + xb2 + ) + exp(x2b1 + x2b2 +)
      
      if (Outcomes[rowobs] > 0 )
      {
   //     if ( i == 1 ) Rcout << xbdif * weights[rowobs] <<std::endl;
        
        newlk_private += xbdif * weights[rowobs];
        efron_wt_private[(time_index_exit)] += riskdiff;
      }
    }
    for (int r = ntimes - 1; r >= 0; r--)
    {
  #pragma omp atomic
      efron_wt[(r)] += efron_wt_private[(r)];
  #pragma omp atomic
      denom[(r)] += denom_private[(r)];
    }
    delete[] efron_wt_private;
    delete[] denom_private;
  }
  
    newlk -= newlk_private; // min  beta updated = beta - diff
  
  
   // if ( i == 1 )   Rcout << "After zbeta update newlk " << newlk << " loglik " << loglik << " delta_newlk " << delta_newlk[i] << std::endl;
    //if (i < nvar) 
    lastvar = i;
  
    
    }
    /* centre exponentiated frailty estimates so penalty is minimised*/
   if (recurrent == 1 ) 
   {
     
    frailty_sum = 0.0;
    for(int rowid = 0; rowid < maxid; rowid ++)  frailty_sum += exp(frailty[rowid]);
    
    frailty_mean = safelog(frailty_sum / maxid);
    frailty_penalty = 0.0;
    
    for (int ir = 0; ir < maxid ; ir ++ ) {
      frailty[ir] -= frailty_mean;
      frailty_penalty += frailty[ir] * nu ;
    }
   } 

    Rcpp::Rcout << " Iter:  " << iter << " Cox likelihood : " << newlk << "   last likelihood :" << loglik << " theta " << theta << "\n";
    /* Check for convergence and previous halving */
    
    if (fabs(1.0 - ((newlk  + frailty_penalty) / (loglik + frailty_penalty_old))) <= MAX_EPS) break;
  //   
     Rcout << " done " << done << " frailty_penalty "<< frailty_penalty << " nu "<< nu << " convergence " << 1.0 - ((newlk + frailty_penalty) / (loglik + frailty_penalty_old)) << "\n";
     loglik = newlk;
     frailty_penalty_old = frailty_penalty;
     
  // 
  // 
    Rcout << "Beta : " ;
    for (int i = 0; i < nvar; i ++) Rcout << beta[i] << " ";
    Rcout << '\n';
  } /* return for another iteration */
  
  
  if ( recurrent == 1 && done == 0 ) {
      
      lik_correction = 0.0;
      Rcout << "*** updating theta ***\n";
      
      if (theta != 0 && nu != 0) {
        for (int rowid = 0; rowid < maxid; rowid++) {
          
          if(frailty_group_events[rowid] == 0) continue;
          double temp = nu > 1e7 ? frailty_group_events[rowid]*frailty_group_events[rowid]/nu :  frailty_group_events[rowid] + nu*safelog(nu/(nu+(frailty_group_events[rowid])));
          lik_correction += temp +
            std::lgamma(frailty_group_events[rowid] + nu) - std::lgamma(nu) -
            frailty_group_events[rowid]*safelog(nu + frailty_group_events[rowid]); // loglikelihood correction for frailty
          }
      }
      
      /* Update frailty */
      
      double theta_lower_bound = 0;
      double theta_upper_bound = 1;
      
      
      thetalkl_history[iter_theta] = newlk + frailty_penalty + lik_correction;
      theta_history[iter_theta] = theta;  // coxph sqrt then square output
      for (int ivar = 0; ivar < nvar; ivar ++ ) {
        coef_history[iter_theta][ivar] = beta[ivar];
      }
      for (int rowid = 0; rowid < maxid; rowid++) {
        frailty_history[iter_theta][rowid] = frailty[rowid];
      }
      
      for (int ir = 0; ir <= iter_theta; ir ++) {
        Rcout << " iter_theta " << ir << " theta " << theta_history[ir] << " lkl " << thetalkl_history[ir] <<
          " done " << fabs(1 - thetalkl_history[ir] / thetalkl_history[ir - 1]) << std::endl;
      }
      
      if(iter_theta == 0) {
        theta = 1;
        done = 0;
      } else if(iter_theta == 1) {
        
        theta = (thetalkl_history[1] < (thetalkl_history[0] + 1)) ?  (theta_history[0] + theta_history[1]) / 2 : 2 * theta_history[1];
        done = 0;
        
      } else if(iter_theta >= 2) {
        
        done =  (fabs(1.0 - thetalkl_history[iter_theta]/thetalkl_history[iter_theta-1]) < inner_EPS);
        
        int best_idx = 0;
        double max_theta = -std::numeric_limits<double>::infinity();
        double last_max_theta = -std::numeric_limits<double>::infinity();
        double min_theta = std::numeric_limits<double>::infinity();
        double last_min_theta = std::numeric_limits<double>::infinity();
        
        double max_likl = -std::numeric_limits<double>::infinity();
        
        double max_theta_likl = -std::numeric_limits<double>::infinity();
        //      double last_max_theta_likl = -std::numeric_limits<double>::infinity();
        
        double min_theta_likl = std::numeric_limits<double>::infinity();
        //       double last_min_theta_likl = std::numeric_limits<double>::infinity();
        
        for(int it = 0; it <= iter_theta; it++) {
          if (thetalkl_history[it] > max_likl) {
            
            max_likl = thetalkl_history[it];
            best_idx = it;
          }
          if (theta_history[it] < min_theta) {
            
            last_min_theta = min_theta ;
            //          last_min_theta_likl = min_theta_likl;
            min_theta_likl = thetalkl_history[it];
            min_theta = theta_history[it];
            
          }
          if (theta_history[it] > max_theta) {
            
            last_max_theta = max_theta;
            //        last_max_theta_likl = max_theta_likl;
            max_theta_likl = thetalkl_history[it];
            max_theta = theta_history[it];
            
          }
        }
        
        if (iter_theta == best_idx && max_theta == theta_history[iter_theta])
        {
          theta = 2 * max_theta;
          Rcout << " best lkl is current " << max_likl << " best theta is current " << theta << " done " << done << std::endl;
        } else {
          
          if (theta_history[best_idx] == min_theta) {
            
            theta = min_theta - 3*(last_min_theta - min_theta);
            
            if (theta < theta_lower_bound) {
              
              double min_theta =theta_upper_bound;
              
              for(int it = 0; it <= iter_theta; it++) {
                
                if (theta_history[it] > theta_lower_bound && theta_history[it] < min_theta) {
                  
                  min_theta = theta_history[it];
                  
                }
              }
              
              theta = theta_lower_bound + (min_theta-theta_lower_bound)/10;
              
            }
            Rcout << " best lkl is " << max_likl << " best theta is smallest " << theta << " done " << done << std::endl;
            
          } else if (theta_history[best_idx] == max_theta ) {
            
            theta =  max_theta + 3*(max_theta - last_max_theta);
            
            if (theta > theta_upper_bound) {

              double max_theta = theta_lower_bound;

              for(int it = 0; it <= iter_theta; it++) {

                if (theta_history[it] < theta_upper_bound && theta_history[it] > max_theta) {

                  max_theta = theta_history[it];

                }
              }

              theta = theta_upper_bound + (max_theta-theta_upper_bound)/10;

            }
            Rcout << " best lkl is " << max_likl << " best theta is largest " << theta << " done " << done << std::endl;
            
          } else {
            
            double best_thetas[3] = {sqrt(min_theta),
                                     sqrt(theta_history[best_idx]),
                                     sqrt(max_theta)};
            
            double best_likls[3] = {min_theta_likl,
                                    thetalkl_history[best_idx],
                                                    max_theta_likl};
            
            // Need to find the theta's either side of the best guess theta
            for(int it = 0; it <= iter_theta; it++) {
              
              if (theta_history[it] > best_thetas[0] && theta_history[it] < best_thetas[1]) {
                
                best_thetas[0] = theta_history[it];
                best_likls[0] = thetalkl_history[it];
                
              }
              if (theta_history[it] > best_thetas[1] && theta_history[it] < best_thetas[2]) {
                
                best_thetas[2] = theta_history[it];
                best_likls[2] = thetalkl_history[it];
                
              }
            }
            
            for (int r = 0 ; r < 3; r++) {
              Rcout << " sqrt theta " << r << " " << best_thetas[r] << " lkl " << best_likls[r] << std::endl;
            }
            
            double temp1 = (pow(best_thetas[1] - best_thetas[0],2) *
                            (best_likls[1] - best_likls[2])) -
                            (pow(best_thetas[1] - best_thetas[2],2) *
                            (best_likls[1] - best_likls[0]));
            
            double temp2 = ((best_thetas[1] - best_thetas[0]) *
                            (best_likls[1] - best_likls[2])) -
                            ((best_thetas[1] - best_thetas[2]) *
                            (best_likls[1] - best_likls[0]));
            
            //           Rcout << " temp1 " << temp1 << " temp2 " << temp2 << std::endl;
            
            theta = best_thetas[1] - .5*temp1/temp2;
            
            Rcout << " theta is updated " << max_likl << " best theta is new " << pow(theta,2) << " done " << done << std::endl;
            
            if ((theta < best_thetas[0]) ||
                (theta > best_thetas[2]) ||
                ((iter_theta > 3) &&
                (theta - sqrt(theta_history[iter_theta])) >
                0.5 *
                abs(sqrt(theta_history[iter_theta - 1]) - sqrt(theta_history[iter_theta - 2]))
                )
            )
            {
              if ((best_thetas[1] - best_thetas[0]) >
                    (best_thetas[2] - best_thetas[1])) {
                
                theta = best_thetas[1] - .38*(best_thetas[1] - best_thetas[0]);
                
              } else {
                
                theta = best_thetas[1] + .32 * (best_thetas[2] - best_thetas[1]);
                
              }
              
              Rcout << " theta is bouncing " << max_likl << " best theta is new " << pow(theta,2) << " done " << done << std::endl;
            }
            
            theta = pow(theta,2);
            
          }
        }
      }
      if (theta == theta_history[iter_theta]) {
        done = 1;
      }
      Rcout << std::endl << " theta: " << theta << " correction " << lik_correction << " newlk " << newlk  << std::endl << std::endl;
      
      
      iter_theta++;
      
      //  if (theta >= 0.5) theta = theta/2;  //Fudge to stop runaway
      nu = 1 / theta;
      
    if (done == 0 && iter_theta > 0)
     {
       int closest_last_theta_idx = 0;
       double closest_last_theta_dif = std::numeric_limits<double>::infinity();
       for(int it = 0; it <= iter_theta; it++) {
         double temp_dif = abs(theta - theta_history[it]);
         if (temp_dif < closest_last_theta_dif) {
           closest_last_theta_dif = temp_dif;
           closest_last_theta_idx = it;
         }
       }
       for (int ivar = 0; ivar < nvar; ivar++) {
         double updatebeta = 0;
         updatebeta = coef_history[closest_last_theta_idx][ivar];
         beta[ivar] = updatebeta;
       }


       /* update cummulative sums and newlk to closest last estimates */

       for (int rowid = 0; rowid < maxid; rowid++) {

      //   frailty_old[rowid] = frailty[rowid];
         frailty[rowid] = frailty_history[closest_last_theta_idx][rowid];
       }
     }
    
    } else {
      break;
    }
}

  for (int i = 0; i<nvar; i++) beta_in[i] = beta[i];
/* baseline hazard whilst zbeta in memory */
// need to have cumulative baseline hazard

int timesN =  OutcomeTotals.size() -1;

#pragma omp parallel default(none) shared(timesN, OutcomeTotals, OutcomeTotalTimes, denom, efron_wt, basehaz)
{
  int size = omp_get_num_threads(); // get total number of processes
  int rank = omp_get_thread_num(); // get rank of current ( range 0 -> (num_threads - 1) )
  
  for (int r =  timesN * (rank + 1)/ size; r >= timesN * rank/ size ; r--)
  {
    double basehaz_private = 0.0;
    int time = OutcomeTotalTimes[r] - 1;
    for (int k = 0; k < OutcomeTotals[r]; k++)
    {
      double temp = (double)k
      / (double)OutcomeTotals[r];
      basehaz_private += 1.0/(denom[time] - (temp * efron_wt[time])); /* sum(denom) adjusted for tied deaths*/
    }
    if (std::isnan(basehaz_private) || basehaz_private < 1e-100)
    {
      basehaz_private = 1e-100; //log(basehaz) required so a minimum measureable hazard is required to avoobs NaN errors.
    }
#pragma omp atomic write
    basehaz[time] = basehaz_private; // should be thread safe as time unique per thread
  }
}

/* Carry forward last value of basehazard */
double last_value = 0.0;

cumhaz[0] = basehaz[0] ;
for (int t = 0; t < ntimes; t++)
{
  if (t>0) cumhaz[t] = cumhaz[t-1] +  basehaz[t];
  if (basehaz[t] == 0.0)
  {
    basehaz[t] = last_value;
  } else {
    last_value = basehaz[t];
  }
}

DoubleVector chentry(maxobs);
RVector<double> cumhazEntry(chentry);
DoubleVector bhentry(maxobs);
RVector<double> BaseHazardEntry(bhentry);
DoubleVector ch1yr(maxobs);
RVector<double> cumhaz1year(ch1yr);
DoubleVector rsk(maxobs);
RVector<double> Risk(rsk);

#pragma omp parallel  default(none)  shared(ntimes,cumhaz1year,cumhazEntry, cumhaz, BaseHazardEntry, Risk, basehaz, timein, zbeta, weights, maxobs)
{
  int size = omp_get_num_threads(); // get total number of processes
  int rank = omp_get_thread_num(); // get rank of current ( range 0 -> (num_threads - 1) )

  for (int rowobs = (rank * maxobs / size); rowobs < ((rank + 1) * maxobs / size) ; rowobs++)
  {
    int time_index_entry = timein[rowobs] - 1;  // Time starts at zero but no events at this time to calculate sums. Lowest value of -1 but not used to reference into any vectors
    int time_one_year = time_index_entry + 365;
    if (time_one_year >= ntimes) time_one_year = ntimes -1;
    BaseHazardEntry[rowobs] = basehaz[time_index_entry];
    cumhazEntry[rowobs] = cumhaz[time_index_entry];
    cumhaz1year[rowobs] = cumhaz[time_one_year];
    Risk[rowobs] = exp(zbeta[rowobs]);
  }
}

delete[] step;
delete[] gdiagvar;
delete[] denom;
delete[] efron_wt;
delete[] zbeta;
delete[] derivMatrix;
delete[] d2sum;
//delete[] frailty_old;
//delete[] outcomes_wt;
delete[] wt_average;
delete[] frailty_group_events;
delete[] coef_history;
delete[] frailty_history;

return List::create(_["Loglik"] = loglik + lik_correction,
                    _["Beta"] = beta,
                    _["BaseHaz"] = basehaz,
                    _["CumHaz"] = cumhaz,
                    _["BaseHazardAtEntry"] = BaseHazardEntry,
                    _["CumHazAtEntry"] = cumhazEntry,
                    _["CumHazOneYear"] = cumhaz1year,
                    _["Risk"] = Risk,
                    _["Frailty"] = frailty);

}

// 
// 
// List update_theta(int maxid,
//              int iter_theta,
//              double theta,
//              double nu,
//              double newlk,
//              double inner_EPS,
//              double * thetalkl_history,
//              double * theta_history,
//             // double ** coef_history,
//             // double ** frailty_history,
//              int * frailty_group_events,
//              int * frailty,
//              int * beta,
// //             int * coval_in,
// //             int * weights_in,
// //             int * Outcomes_in,
// //             int * OutcomeTotals_in,
// //             int * OutcomeTotalTimes_in,
// //             int * obs_in,
// //             int * id_in, 
//              double frailty_penalty, 
//              double lambda, 
//              int done) 
// {
//   
//   
//   double lik_correction = 0.0;
//   Rcout << "*** updating theta ***\n";
//   
//   if (theta != 0 && nu != 0) {
//     for (int rowid = 0; rowid < maxid; rowid++) {
//       
//       if(frailty_group_events[rowid] == 0) continue;
//       double temp = nu > 1e7 ? frailty_group_events[rowid]*frailty_group_events[rowid]/nu :  frailty_group_events[rowid] + nu*safelog(nu/(nu+(frailty_group_events[rowid])));
//       lik_correction += temp +
//         std::lgamma(frailty_group_events[rowid] + nu) - std::lgamma(nu) -
//         frailty_group_events[rowid]*safelog(nu + frailty_group_events[rowid]); // loglikelihood correction for frailty
//     }
//   }
//   
//   /* Update frailty */
//   
//   double theta_lower_bound = 0.0000001;
//   double theta_upper_bound = 1;
//   
//   
//   thetalkl_history[iter_theta] = newlk + frailty_penalty + lik_correction;
//   theta_history[iter_theta] = theta;  // coxph sqrt then square output
// 
//   for (int ir = 0; ir <= iter_theta; ir ++) {
//     Rcout << " iter_theta " << ir << " theta " << theta_history[ir] << " lkl " << thetalkl_history[ir] <<
//       " done " << fabs(1 - thetalkl_history[ir] / thetalkl_history[ir - 1]) << std::endl;
//   }
//   
//   if(iter_theta == 0) {
//     theta = 1;
//     done = 0;
//   } else if(iter_theta == 1) {
//     
//     theta = (thetalkl_history[1] < (thetalkl_history[0] + 1)) ?  (theta_history[0] + theta_history[1]) / 2 : 2 * theta_history[1];
//     done = 0;
//     
//   } else if(iter_theta >= 2) {
//     
//     done =  (fabs(1.0 - thetalkl_history[iter_theta]/thetalkl_history[iter_theta-1]) < inner_EPS);
//     
//     int best_idx = 0;
//     double max_theta = -std::numeric_limits<double>::infinity();
//     double last_max_theta = -std::numeric_limits<double>::infinity();
//     double min_theta = std::numeric_limits<double>::infinity();
//     double last_min_theta = std::numeric_limits<double>::infinity();
//     
//     double max_likl = -std::numeric_limits<double>::infinity();
//     
//     double max_theta_likl = -std::numeric_limits<double>::infinity();
//     //      double last_max_theta_likl = -std::numeric_limits<double>::infinity();
//     
//     double min_theta_likl = std::numeric_limits<double>::infinity();
//     //       double last_min_theta_likl = std::numeric_limits<double>::infinity();
//     
//     for(int it = 0; it <= iter_theta; it++) {
//       if (thetalkl_history[it] > max_likl) {
//         
//         max_likl = thetalkl_history[it];
//         best_idx = it;
//       }
//       if (theta_history[it] < min_theta) {
//         
//         last_min_theta = min_theta ;
//         //          last_min_theta_likl = min_theta_likl;
//         min_theta_likl = thetalkl_history[it];
//         min_theta = theta_history[it];
//         
//       }
//       if (theta_history[it] > max_theta) {
//         
//         last_max_theta = max_theta;
//         //        last_max_theta_likl = max_theta_likl;
//         max_theta_likl = thetalkl_history[it];
//         max_theta = theta_history[it];
//         
//       }
//     }
//     
//     if (iter_theta == best_idx && max_theta == theta_history[iter_theta])
//     {
//       
//       theta = 2 * max_theta;
//       Rcout << " best lkl is current " << max_likl << " best theta is current " << theta << " done " << done << std::endl;
//       
//     } else {
//       
//       if (theta_history[best_idx] == min_theta) {
//         
//         theta = min_theta - 3*(last_min_theta - min_theta);
//         
//         if (theta < theta_lower_bound) {
//           
//           double min_theta =theta_upper_bound;
//           
//           for(int it = 0; it <= iter_theta; it++) {
//             
//             if (theta_history[it] > theta_lower_bound && theta_history[it] < min_theta) {
//               
//               min_theta = theta_history[it];
//               
//             }
//           }
//           
//           theta = theta_lower_bound + (min_theta-theta_lower_bound)/10;
//           
//         }
//         Rcout << " best lkl is " << max_likl << " best theta is smallest " << theta << " done " << done << std::endl;
//         
//       } else if (theta_history[best_idx] == max_theta ) {
//         
//         theta =  max_theta + 3*(max_theta - last_max_theta);
//         
//         if (theta > theta_upper_bound) {
//           
//           double max_theta = theta_lower_bound;
//           
//           for(int it = 0; it <= iter_theta; it++) {
//             
//             if (theta_history[it] < theta_upper_bound && theta_history[it] > max_theta) {
//               
//               max_theta = theta_history[it];
//               
//             }
//           }
//           
//           theta = theta_upper_bound + (max_theta-theta_upper_bound)/10;
//           
//         }
//         Rcout << " best lkl is " << max_likl << " best theta is largest " << theta << " done " << done << std::endl;
//         
//       } else {
//         
//         double best_thetas[3] = {sqrt(min_theta),
//                                  sqrt(theta_history[best_idx]),
//                                  sqrt(max_theta)};
//         
//         double best_likls[3] = {min_theta_likl,
//                                 thetalkl_history[best_idx],
//                                                 max_theta_likl};
//         
//         // Need to find the theta's either side of the best guess theta
//         for(int it = 0; it <= iter_theta; it++) {
//           
//           if (theta_history[it] > best_thetas[0] && theta_history[it] < best_thetas[1]) {
//             
//             best_thetas[0] = theta_history[it];
//             best_likls[0] = thetalkl_history[it];
//             
//           }
//           if (theta_history[it] > best_thetas[1] && theta_history[it] < best_thetas[2]) {
//             
//             best_thetas[2] = theta_history[it];
//             best_likls[2] = thetalkl_history[it];
//             
//           }
//         }
//         
//         for (int r = 0 ; r < 3; r++) {
//           Rcout << " theta " << r << " " << best_thetas[r] << " lkl " << best_likls[r] << std::endl;
//         }
//         
//         double temp1 = (pow(best_thetas[1] - best_thetas[0],2) *
//                         (best_likls[1] - best_likls[2])) -
//                         (pow(best_thetas[1] - best_thetas[2],2) *
//                         (best_likls[1] - best_likls[0]));
//         
//         double temp2 = ((best_thetas[1] - best_thetas[0]) *
//                         (best_likls[1] - best_likls[2])) -
//                         ((best_thetas[1] - best_thetas[2]) *
//                         (best_likls[1] - best_likls[0]));
//         
//         //           Rcout << " temp1 " << temp1 << " temp2 " << temp2 << std::endl;
//         
//         theta = best_thetas[1] - .5*temp1/temp2;
//         
//         Rcout << " theta is updated " << max_likl << " best theta is new " << pow(theta,2) << " done " << done << std::endl;
//         
//         if ((theta < best_thetas[0]) ||
//             (theta > best_thetas[2]) ||
//             ((iter_theta > 3) &&
//             (theta - sqrt(theta_history[iter_theta])) >
//             0.5 *
//             abs(sqrt(theta_history[iter_theta - 1]) - sqrt(theta_history[iter_theta - 2]))
//             )
//         )
//         {
//           if ((best_thetas[1] - best_thetas[0]) >
//                 (best_thetas[2] - best_thetas[1])) {
//             
//             theta = best_thetas[1] - .38*(best_thetas[1] - best_thetas[0]);
//             
//           } else {
//             
//             theta = best_thetas[1] + .32 * (best_thetas[2] - best_thetas[1]);
//             
//           }
//           
//           Rcout << " theta is bouncing " << max_likl << " best theta is new " << pow(theta,2) << " done " << done << std::endl;
//         }
//         
//         theta = pow(theta,2);
//         
//       }
//     }
//   }
//   if (theta == theta_history[iter_theta]) {
//     done = 1;
//   }
//   Rcout << " theta: " << theta << " correction " << lik_correction << " newlk " << newlk ;
//   
//   
//   iter_theta++;
//   
//   //  if (theta >= 0.5) theta = theta/2;  //Fudge to stop runaway
// 
//   nu = 1 / theta;  
//   return List::create(_["theta"] = theta,
//                       _["nu"] = nu,
//                       _["done"] = done,
//                       _["iter_theta"] = iter_theta);
// }
