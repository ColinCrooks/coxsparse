#include <Rcpp.h>
#include "utils.h"
#include <omp.h>
#include <RcppParallel.h>
using namespace Rcpp;
using namespace RcppParallel;

//' profile_ci
//' 
//' Description
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
//' @param covrowlist_in A list in R of integer vectors of length nvar. 
//' Each entry in the list corresponds to a covariate.
//' Each vector contains the indices for the rows in the coval list 
//' that correspond to that vector. Maximum value of any indices corresponds
//' therefore to the length of coval.
//' @param beta_in A double vector of starting values for the coefficients 
//' of length nvar.
//' @param id_in An integer vector referencing for each covariate value the
//' corresponding unique patient time in the time and outcome vectors. Of the
//' same length as coval. The maximum value is the length of timein and timeout.
//' @param coval_in A double vector of each covariate value sorted first by 
//' time then by patient and by order of the covariates to be included in model.
//' Of the same longth as id_in.
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
//' @param lambda Penalty weight to include for ridge regression: -log(sqrt(lambda)) * nvar
//' @param nvar Number of covariates
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
    List covrowlist_in,
    DoubleVector beta_in,
    IntegerVector id_in,
    DoubleVector coval_in,
    DoubleVector weights_in,
    IntegerVector timein_in ,
    IntegerVector timeout_in ,
    IntegerVector Outcomes_in ,
    IntegerVector OutcomeTotals_in ,
    IntegerVector OutcomeTotalTimes_in,
    double lambda,
    int nvar,
    int MSTEP_MAX_ITER,
    int decimals,
    double confint_width,
    int threadn)
{
  Rcpp::Rcout.precision(10);
  List covrowlist(covrowlist_in);
  omp_set_dynamic(0);     // Explicitly disable dynamic teams
  omp_set_num_threads(threadn); // Use 8 threads for all consecutive parallel regions

  // Vectors from R index begin from 1. Need to convert to 0 index for C++and comparisons

  int ntimes = max(timeout_in);
  // Unique times but don't accumulate for time 0 as no events

  int maxid = max(id_in);

  double dif = 0.0;
  double * denom  = new double [ntimes];
  double * efron_wt  = new double [ntimes];
  for (int ir = 0 ; ir < ntimes; ir++)
  {
    denom[ir] = 0.0;
    efron_wt[ir] = 0.0;
  }

  double * zbeta = new double [maxid];
  for (int ir = 0; ir < maxid; ir++) zbeta[ir] = 0.0;

  /* Wrap all R objects to make thread safe for read and writing  */
  RVector<double> beta(beta_in);
  RVector<double> coval(coval_in);
  RVector<double> weights(weights_in);
  RVector<int> Outcomes(Outcomes_in);
  RVector<int> OutcomeTotals(OutcomeTotals_in);
  RVector<int> OutcomeTotalTimes(OutcomeTotalTimes_in);
  RVector<int> id(id_in);
  RVector<int> timein(timein_in);
  RVector<int> timeout(timeout_in);

  double step = 1/pow(10,decimals);
  double  d2sum = 0.0;

  double newlk = 0.0;
  if (lambda !=0) newlk = -(log(sqrt(lambda)) * nvar);


  for (int i = 0; i < nvar; i++)
  { /* per observation time calculations */
  IntegerVector covrows_in = covrowlist[i];
    RVector<int> covrows(covrows_in);
    double beta_local = beta[i];
#pragma omp parallel  default(none) shared(zbeta, maxid, covrows, coval, beta_local, weights, Outcomes, id)
{
  double *zbeta_private = new double [maxid];
  for (int ir = 0; ir < maxid; ir++) zbeta_private[static_cast<size_t>(ir)] = 0.0;

  int size = omp_get_num_threads(); // get total number of processes
  int rank = omp_get_thread_num(); // get rank of current ( range 0 -> (num_threads - 1) )
  int rowN = covrows.size();

  for (size_t covi = static_cast<size_t>(rank * rowN / size); covi < static_cast<size_t>((rank + 1) * rowN / size) ; covi++)
  {
    R_xlen_t row = static_cast<R_xlen_t>(covrows[covi] - 1); // R_xlen_t is signed long, size_t is unsigned. R vectors use signed
    R_xlen_t rowid = static_cast<R_xlen_t>(id[row] - 1);

    double covali = coval[row];
    zbeta_private[static_cast<size_t>(rowid)] += beta_local * covali;
  }
  for (int rowid = 0; rowid < maxid ; rowid++)
  {
#pragma omp atomic
    zbeta[rowid] += zbeta_private[rowid];
  }
  delete[] zbeta_private;
}
  }

  /* Cumulative sums that do not depend on the specific covariate update*/
  double newlk_private = 0.0;
#pragma omp parallel  default(none) reduction(+:newlk_private) shared(timein, timeout, zbeta, weights,  Outcomes,denom,efron_wt, ntimes,maxid)
{
  double * denom_private  = new double [ntimes];
  double * efron_wt_private  = new double [ntimes];
  for (int ir = 0 ; ir < ntimes; ir++)
  {
    denom_private[ir] = 0.0;
    efron_wt_private[ir] = 0.0;
  }
  int size = omp_get_num_threads(); // get total number of processes
  int rank = omp_get_thread_num(); // get rank of current ( range 0 -> (num_threads - 1) )
  int rowN = maxid;

  for (R_xlen_t rowid = static_cast<R_xlen_t>(rank * rowN / size); rowid < static_cast<R_xlen_t>((rank + 1) * rowN / size) ; rowid++)
  {
    int time_index_entry = timein[rowid] - 1;  // Time starts at zero but no events at this time to calculate sums. Lowest value of -1 but not used to reference into any vectors
    int time_index_exit = timeout[rowid] - 1;

    double zbeta_temp = zbeta[(rowid)] >22 ? 22 : zbeta[(rowid)];
    zbeta_temp = zbeta_temp < -200 ? -200 : zbeta_temp;
    double risk = exp(zbeta_temp) * weights[rowid];

    //cumumlative sums for all patients
    for (int r = time_index_exit; r > time_index_entry ; r--)
      denom_private[static_cast<size_t>(r)] += risk;

    if (Outcomes[rowid] > 0 )
    {
      /*cumumlative sums for event patients */
      newlk_private += zbeta_temp * weights[rowid];
      efron_wt_private[static_cast<size_t>(time_index_exit)] += risk;
    }
#pragma omp atomic write
    zbeta[(rowid)] = zbeta_temp; // should be threadsafe without atomic as threads by rowid
  }
  for (int r = 0; r < ntimes ; r++)
  {
#pragma omp atomic
    efron_wt[static_cast<size_t>(r)] += efron_wt_private[static_cast<size_t>(r)];
#pragma omp atomic
    denom[static_cast<size_t>(r)] += denom_private[static_cast<size_t>(r)];
  }
  delete[] efron_wt_private;
  delete[] denom_private;
}


newlk += newlk_private;


for(int r = OutcomeTotalTimes.size() -1 ; r >=0 ; r--)
{
  for (int k = 0; k < OutcomeTotals[r]; k++)
  {
    int time = OutcomeTotalTimes[r] - 1;
    double temp = (double)k
    / (double)OutcomeTotals[r];
    double d2 = denom[time] - (temp * efron_wt[time]); /* sum(denom) adjusted for tied deaths*/
      newlk -= safelog(d2);
      d2sum += safelog(d2);
  }
}

NumericMatrix confinterval(nvar, 2);
double threshold = R::qchisq(confint_width, 1,true, false )/2;

Rcout << "Coefficient" << "\t" << "Lower "<< confint_width*100<<"% CI" <<  "\t" << "Upper "<< confint_width*100<<"% CI" << "\n";
for (int i = 0; i < nvar; i++)
{
  IntegerVector covrows_in = covrowlist[i];
  RVector<int> covrows(covrows_in);


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
      dif = confinterval(i,0) - beta[i];
      updatelk = newlk + d2sum;

      double updatelk_private = 0.0;

#pragma omp parallel  default(none) reduction(+:updatelk_private) shared(covrows, coval, weights, Outcomes, ntimes, id, timein, timeout,maxid, zbeta, dif,  denom, efron_wt)
{
  int size = omp_get_num_threads(); // get total number of processes
  int rank = omp_get_thread_num(); // get rank of current ( range 0 -> (num_threads - 1) )
  int rowN = covrows.size();

  double *denom_private = new double [ntimes];
  double *efron_wt_private= new double [ntimes];
  double *zbeta_private = new double [maxid];
  for (int ir = 0; ir < maxid; ir++)
    zbeta_private[ir] = 0.0;
  for (int ir = 0 ; ir < ntimes; ir++)
  {
    denom_private[ir] = 0.0;
    efron_wt_private[ir] = 0.0;
  }

  for (size_t covi = static_cast<size_t>(rank * rowN / size); covi < static_cast<size_t>(((rank + 1) * rowN / size)) ; covi++)
  {
    R_xlen_t row = static_cast<R_xlen_t>(covrows[static_cast<R_xlen_t>(covi)] - 1); // R_xlen_t is signed long, size_t is unsigned. R vectors use signed
    R_xlen_t rowid = static_cast<R_xlen_t>(id[row] - 1);

    double riskold = exp(zbeta[static_cast<size_t>(rowid)]);

    double xbdif = dif * coval[row];
    zbeta_private[static_cast<size_t>(rowid)] += xbdif;

    double zbeta_updated = zbeta[static_cast<size_t>(rowid)] - xbdif;
    zbeta_updated =  zbeta_updated >22 ? 22 : zbeta_updated;
    zbeta_updated =  zbeta_updated < -200 ? -200 :  zbeta_updated;

    double riskdiff = (exp(zbeta_updated) - riskold) * weights[rowid];

    int time_index_entry =  timein[rowid] - 1;
    int time_index_exit =  timeout[rowid] - 1;

    for (int r = time_index_exit; r > time_index_entry ; r--)
      denom_private[static_cast<size_t>(r)] += riskdiff; //

    if (Outcomes[rowid] > 0 )
    {
      updatelk_private += xbdif * weights[rowid];
      efron_wt_private[static_cast<size_t>(time_index_exit)] += riskdiff;
    }
  }
  for (int rowid = 0; rowid < maxid ; rowid++)
  {
#pragma omp atomic
    zbeta[static_cast<size_t>(rowid)] -= zbeta_private[static_cast<size_t>(rowid)];
    zbeta[static_cast<size_t>(rowid)] =  zbeta[static_cast<size_t>(rowid)] >22 ? 22 : zbeta[static_cast<size_t>(rowid)];
    zbeta[static_cast<size_t>(rowid)] =  zbeta[static_cast<size_t>(rowid)] < -200 ? -200 :  zbeta[static_cast<size_t>(rowid)];
  }
  for (int r = 0; r < ntimes ; r++)
  {
#pragma omp atomic
    efron_wt[static_cast<size_t>(r)] += efron_wt_private[static_cast<size_t>(r)];
#pragma omp atomic
    denom[static_cast<size_t>(r)] += denom_private[static_cast<size_t>(r)];
  }
  delete[] efron_wt_private;
  delete[] denom_private;
  delete[] zbeta_private;
}
updatelk -= updatelk_private;
for(int r = OutcomeTotalTimes.size() -1 ; r >=0 ; r--)
{
  for (int k = 0; k < OutcomeTotals[r]; k++)
  {
    int time = OutcomeTotalTimes[r] - 1;
    double temp = (double)k
    / (double)OutcomeTotals[r];
    double d2 = denom[time] + denom[time] - (temp * (efron_wt[time] + efron_wt[time])); /* sum(denom) adjusted for tied deaths*/
      updatelk -= safelog(d2); // track this sum to remove nvar from newlk at start of next iteration

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
      dif = confinterval(i,1) - beta[i];
      updatelk = newlk + d2sum;

      double updatelk_private = 0.0;

#pragma omp parallel  default(none) reduction(+:updatelk_private) shared(covrows, coval, weights, Outcomes, ntimes, id, timein, timeout,maxid, zbeta, dif,  denom, efron_wt)
{
  double *denom_private = new double [ntimes];
  double *efron_wt_private= new double [ntimes];
  double *zbeta_private = new double [maxid];
  for (int ir = 0; ir < maxid; ir++)
    zbeta_private[ir] = 0.0;
  for (int ir = 0 ; ir < ntimes; ir++)
  {
    denom_private[ir] = 0.0;
    efron_wt_private[ir] = 0.0;
  }
  int size = omp_get_num_threads(); // get total number of processes
  int rank = omp_get_thread_num(); // get rank of current ( range 0 -> (num_threads - 1) )
  int rowN = covrows.size();
  for (size_t covi = static_cast<size_t>(rank * rowN / size); covi < static_cast<size_t>(((rank + 1) * rowN / size)) ; covi++)
  {
    R_xlen_t row = static_cast<R_xlen_t>(covrows[static_cast<R_xlen_t>(covi)] - 1); // R_xlen_t is signed long, size_t is unsigned. R vectors use signed
    R_xlen_t rowid = static_cast<R_xlen_t>(id[row] - 1);

    double riskold = exp(zbeta[static_cast<size_t>(rowid)]);

    double xbdif = dif * coval[row];
    zbeta_private[static_cast<size_t>(rowid)] += xbdif;

    double zbeta_updated = zbeta[static_cast<size_t>(rowid)] - xbdif;
    zbeta_updated =  zbeta_updated >22 ? 22 : zbeta_updated;
    zbeta_updated =  zbeta_updated < -200 ? -200 :  zbeta_updated;

    double riskdiff = (exp(zbeta_updated) - riskold) * weights[rowid];

    int time_index_entry =  timein[rowid] - 1;
    int time_index_exit =  timeout[rowid] - 1;

    for (int r = time_index_exit; r > time_index_entry ; r--)
      denom_private[static_cast<size_t>(r)] += riskdiff; //

    if (Outcomes[rowid] > 0 )
    {
      updatelk_private += xbdif * weights[rowid];
      efron_wt_private[static_cast<size_t>(time_index_exit)] += riskdiff;
    }
  }
  for (int rowid = 0; rowid < maxid ; rowid++)
  {
#pragma omp atomic
    zbeta[static_cast<size_t>(rowid)] -= zbeta_private[static_cast<size_t>(rowid)];
    zbeta[static_cast<size_t>(rowid)] =  zbeta[static_cast<size_t>(rowid)] >22 ? 22 : zbeta[static_cast<size_t>(rowid)];
    zbeta[static_cast<size_t>(rowid)] =  zbeta[static_cast<size_t>(rowid)] < -200 ? -200 :  zbeta[static_cast<size_t>(rowid)];
  }
  for (int r = 0; r < ntimes ; r++)
  {
#pragma omp atomic
    efron_wt[static_cast<size_t>(r)] += efron_wt_private[static_cast<size_t>(r)];
#pragma omp atomic
    denom[static_cast<size_t>(r)] += denom_private[static_cast<size_t>(r)];
  }
  delete[] efron_wt_private;
  delete[] denom_private;
  delete[] zbeta_private;
}
updatelk -= updatelk_private;

for(int r = OutcomeTotalTimes.size() -1 ; r >=0 ; r--)
{
  for (int k = 0; k < OutcomeTotals[r]; k++)
  {
    int time = OutcomeTotalTimes[r] - 1;
    double temp = (double)k
    / (double)OutcomeTotals[r];
    double d2 = denom[time] - (temp * (efron_wt[time])); /* sum(denom) adjusted for tied deaths*/
    updatelk -= safelog(d2); // track this sum to remove from newlk at start of next iteration

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


Rcout << "\nHazard ratios" << "\t" << "Lower "<< confint_width*100<<"% CI" <<  "\t" << "Upper "<< confint_width*100<<"% CI" << "\n";


for (int i = 0; i < nvar; i ++ )
{
  Rcout << exp(beta[i]) << "\t(" << exp(confinterval(i,0)) <<  " to " << exp(confinterval(i,1)) << ")\n";
}
return confinterval;
}
