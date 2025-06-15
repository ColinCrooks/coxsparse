//#include <Rcpp.h>
#include <omp.h>
#include <RcppParallel.h>
#include "utils.h"
//using namespace Rcpp;
using namespace RcppParallel;
//' cox_reg_sparse_hess
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
//' @param cov_in An integer vector mapping covariates to the 
//' corresponding covariate value row in coval sorted by time and id
//' @param id_in An integer vector mapping unique patient IDs to the 
//' corresponding row in observations sorted by time and id
//' @param lambda Penalty weight to include for ridge regression: -log(sqrt(lambda)) * nvar
//' @param theta_in An input starting value for theta or can be set to zero.
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
//' * Frailty The frailty value for each unique ID group on linear predictor scale 
//'   (w in xb + Zw). Exponentiate for the relative scale. No centring applied.
//' @export
// [[Rcpp::export]]
void cox_reg_sparse_hess(IntegerVector obs_in,
                              DoubleVector  coval_in,
                              DoubleVector  weights_in,
                              IntegerVector timein_in ,
                              IntegerVector timeout_in ,
                              IntegerVector Outcomes_in ,
                              IntegerVector OutcomeTotals_in ,
                              IntegerVector OutcomeTotalTimes_in,
                              IntegerVector cov_in,
                              IntegerVector id_in,
                              double lambda,
                              double theta_in ,
                              int MSTEP_MAX_ITER,
                              double MAX_EPS,
                              long unsigned int threadn) {
  omp_set_dynamic(0);     // Explicitly disable dynamic teams
  omp_set_num_threads(threadn); // Use 8 threads for all consecutive parallel regions

  Rcpp::Rcout.precision(10);

 // List covrowlist(covrowlist_in);
// R objects size uses signed integer Maximum signed integer 2^31 - 1 ~ 2*10^9 = 2,147,483,647 so overflow unlikely using int as indices. But cpp uses size_T

  //  RVector<double> zbeta_internal(zbeta); // if want to bring in zbeta from the user
  // Vectors from R index begin from 1. Need to convert to 0 index for C++ and comparisons
  //Mittal, S., Madigan, D., Burd, R. S., & Suchard, M. a. (2013). High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis. Biostatistics (Oxford, England), 1–15. doi:10.1093/biostatistics/kxt043

  int ntimes = max(timeout_in);  // Unique times but don't accumulate for time 0 as no events
  int maxobs = max(obs_in);
  int nvar = max(cov_in);
  int maxid =  max(id_in); // Number of unique patients
  bool recurrent = maxid > 0;
  
  double newlk = 0.0;
  double loglik = 0.0;
  double frailty_sum = 0.0;
  double lik_correction = 0.0;
  double d2_sum_private = 0.0;
  double newlk_private = 0.0;
  
  double theta = theta_in;
  double nu = theta_in == 0 ? 0 : 1.0/theta_in;
  double frailty_mean = 0;

  DoubleVector frl(maxid);
  RVector<double> frailty(frl);

  double frailty_penalty = 0.0;
  int * frailty_group_events = new int [maxid]; // Count of events for each patient (for gamma penalty weight)
  for (int ir = 0; ir < maxid; ir++) {
    frailty[ir] = 0.0;
    frailty_group_events[ir] = 0;
  }

  std::vector<double> theta_history(MSTEP_MAX_ITER);
  theta_history = {0.0};
  std::vector<double> thetalkl_history(MSTEP_MAX_ITER);
  thetalkl_history = {-std::numeric_limits<double>::infinity()};
  
  double  denom  = 0.0;      // sum of risk of all patients at each time point
  double  efron_wt  = 0.0;      // sum of risk of all patients at each time point
// double * denom  = new double [ntimes];      // sum of risk of all patients at each time point
//  double * efron_wt  = new double [ntimes];  // Sum of risk of patients with events at each time point
  double * wt_average  = new double [ntimes];  // Average weight of people with event at each time point.
  
  for (int ir = 0 ; ir < ntimes; ir++)
  {
  //  denom[ir] = 0.0;
  //  efron_wt[ir] = 0.0;
    wt_average[ir] = 0.0;
  }

  double * zbeta = new double [maxobs];
  for (int ir = 0; ir < maxobs; ir++) zbeta[ir] = 0.0;

  double * fdiag = new double [maxid];
  for (int ir = 0; ir < maxid; ir++) fdiag[ir] = 0.0; // diagonal of the frailty matrix
  
  double * tmean = new double [nvar + maxid];
  double * u = new double [nvar + maxid];
  double * a = new double [nvar + maxid];
  double * a2 = new double [nvar + maxid];
  double ** cmat = new double* [nvar];
  double ** cmat2 = new double* [nvar];
  double ** jmat = new double* [nvar];
  
  for (int ivar = 0; ivar < nvar + maxid; ivar++) 
  { 
    tmean[ivar] = 0.0;
    a[ivar] = 0.0;
    a2[ivar] = 0.0;
    u[ivar] = 0.0;
  }
  
  for (int ivar = 0; ivar < nvar; ivar++) 
  {
    cmat[ivar] = new double [nvar + maxid];
    cmat2[ivar] = new double [nvar + maxid];
    jmat[ivar] = new double [nvar + maxid];
    for (int jvar = 0; jvar < nvar + maxid; jvar++) 
    {
      cmat[ivar][jvar] = 0.0;
      cmat2[ivar][jvar] = 0.0;
      jmat[ivar][jvar] = 0.0; // Hessian matrix
    }
  }


  
  /* Wrap all R objects to make thread safe for read and writing  */
  DoubleVector beta_in(nvar);
  beta_in.fill(0.0);
  RVector<double> beta(beta_in);
  RVector<double> coval(coval_in);
  RVector<double> weights(weights_in);
  RVector<int> Outcomes(Outcomes_in);
  RVector<int> OutcomeTotals(OutcomeTotals_in);
  RVector<int> OutcomeTotalTimes(OutcomeTotalTimes_in);
  RVector<int> obs(obs_in);
  RVector<int> timein(timein_in);
  RVector<int> timeout(timeout_in);
  RVector<int>  cov(cov_in);
  RVector<int>  id(id_in);
  double * step = new double [nvar + maxid];
  for (int ivar = 0; ivar < nvar + maxid; ivar++) {
    step[ivar] = 1.0;
  }
  
  double * gdiagvar = new double [nvar + maxid];
  for (int ivar = 0; ivar < nvar + maxid; ivar++) gdiagvar[ivar] = 0.0;
  
  int iter_theta = 0;
  double inner_EPS = 1e-5;
  int done = 0;

  for (int obsi = 0; obsi < maxobs; obsi++)
    zbeta[obsi] += frailty[id[obsi] - 1]; // initial frailty values
  
  for (int covi = cov.length(); covi >= 0; covi--)
    zbeta[obs[covi] - 1] += coval[covi] * beta[cov[covi] - 1]; // cov[covi] is the covariate index, obs[covi] is the observation time index
  
  int covp = cov.length() - 1;
  int ndead = 0;
  
  for (int covi = cov.length() - 1; covi >= 0; covi--)
  {
    if (covi == (int)cov.length() - 1 || 
        id[covi] != id[covi + 1]) covp = cov[covi] - 1; // start of covariate index
    
    cmat[covi][id[covi] - 1] += coval[covi]*exp(zbeta[obs[covi] - 1]);  
    
    for (int covi2 = covp; covi2 >= covi; covi2--)
      cmat[covi][covi2 + maxid] += coval[covi]*coval[covi2]*exp(zbeta[obs[covi] - 1]);
    
    if (Outcomes[obs[covi] - 1] > 0 )   
    {
      u[covi + maxid] += weights[id[covi] - 1] * coval[covi]; // u is the cumulative sum of weights for each patient ID
      a2[covi + maxid] += exp(zbeta[obs[covi] - 1]) * coval[covi]; // a2 is the cumulative sum of risk for each patient ID
      cmat2[covi][id[covi] - 1] += coval[covi]*exp(zbeta[obs[covi] - 1]);  
      
      for (int covi2 = covp; covi2 >= covi; covi2--)
        cmat2[covi][covi2 + maxid] += coval[covi]*coval[covi2]*exp(zbeta[obs[covi] - 1]);
    }
    
    
    if (covi == 0 || 
        id[covi] != id[covi - 1] || 
        timeout[covi] !=  timeout[covi - 1]) 
    {
      denom += exp(zbeta[obs[covi] - 1]);
      a[id[covi] - 1] += exp(zbeta[obs[covi] - 1]);
      
      if (Outcomes[obs[covi] - 1] > 0 )   
      {
        efron_wt += exp(zbeta[obs[covi] - 1]); 
        newlk += weights[id[covi - 1]] * exp(zbeta[obs[covi] - 1]); // weights[id[covi - 1]] is the weight for the patient at this time point
        
        u[id[covi] - 1] += weights[id[covi - 1]];
        a2[id[covi] - 1] += exp(zbeta[obs[covi] - 1]);
      }
    }
    
    if (ndead > 0 && (covi == 0 || timeout[covi] !=  timeout[covi - 1])) 
    {
      for (int k=0; k<ndead; k++) {
        double temp = (double)k / ndead;
        double d2= denom - temp*efron_wt;
        newlk -= wt_average[timeout[covi] - 1] *safelog(d2);
        
        for (int ivar = 0; ivar < nvar + maxid; ivar++) {  /* by row of full matrix */
          double temp2 = (a[ivar] - temp*a2[ivar])/d2;
          tmean[ivar] = temp2;
          u[ivar] -= wt_average[timeout[covi] - 1] *temp2;
          if (ivar < maxid) fdiag[ivar] += temp2 * (1-temp2);
          else {
            int ii = ivar-maxid;     /*actual row in c/j storage space */
            for (int j=0; j<=ivar; j++) 
              jmat[ii][j] +=  wt_average[timeout[covi] - 1] *
                ((cmat[ii][j] - temp*cmat2[ii][j]) /d2 - temp2*tmean[j]);
          }
        }
      }
      efron_wt =0;
      for (int i=0; i<nvar + maxid; i++) {
        a2[i]=0;
        for (int j=0; j<nvar; j++)  cmat2[j][i]=0;
      }
      ndead = 0;
    }
  }
  
  frailty_sum = 0.0;
  for(int rowid = 0; rowid < maxid; rowid ++)  frailty_sum += exp(frailty[rowid]);
  
  frailty_mean = safelog(frailty_sum / maxid);
  
  for (int ivar = 0; ivar < maxid; ivar ++) 
  {
    u[ivar] += (exp(frailty[ivar] - frailty_mean) - 1.0) * nu; 
    fdiag[ivar] += exp(frailty[ivar] - frailty_mean) * nu; 
  }
  
  

// return Rcpp::List::create(_["Loglik"] = loglik + lik_correction,
//                     _["Beta"] = beta,
//                  //   _["BaseHaz"] = basehaz,
//                 //    _["CumHaz"] = cumhaz,
//                 //    _["BaseHazardAtEntry"] = BaseHazardEntry,
//                 //    _["CumHazAtEntry"] = cumhazEntry,
//                 //    _["CumHazOneYear"] = cumhaz1year,
//                 //    _["Risk"] = Risk,
//                     _["Frailty"] = frailty,
//                     _["Theta"] = theta);

}
