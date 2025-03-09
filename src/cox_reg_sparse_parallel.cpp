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
                              DoubleVector coval_in,
                              DoubleVector weights_in,
                              IntegerVector timein_in ,
                              IntegerVector timeout_in ,
                              IntegerVector Outcomes_in ,
                              IntegerVector OutcomeTotals_in ,
                              IntegerVector OutcomeTotalTimes_in,
                              int nvar,
                              double lambda,
                              int MSTEP_MAX_ITER,
                              double MAX_EPS,
                              int threadn) {
  omp_set_dynamic(0);     // Explicitly disable dynamic teams
  omp_set_num_threads(threadn); // Use 8 threads for all consecutive parallel regions

  Rcpp::Rcout.precision(10);

  List covrowlist(covrowlist_in);
// R objects size uses signed integer Maximum signed integer 2^31 - 1 ~ 2*10^9 = 2,147,483,647 so overflow unlikely using int as indices. But cpp uses size_T

  //  RVector<double> zbeta_internal(zbeta); // if want to bring in zbeta from the user
  // Vectors from R index begin from 1. Need to convert to 0 index for C++ and comparisons
  //Mittal, S., Madigan, D., Burd, R. S., & Suchard, M. a. (2013). High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis. Biostatistics (Oxford, England), 1–15. doi:10.1093/biostatistics/kxt043



  int ntimes = max(timeout_in);  // Unique times but don't accumulate for time 0 as no events
  int maxid = max(id_in);

  double * denom  = new double [ntimes];
  double * efron_wt  = new double [ntimes];
  for (int ir = 0 ; ir < ntimes; ir++)
  {
    denom[ir] = 0.0;
    efron_wt[ir] = 0.0;
  }

  double * zbeta = new double [maxid];
  for (int ir = 0; ir < maxid; ir++) zbeta[ir] = 0.0;

  double * d2sum = new double [nvar];
  for (int ir = 0; ir < nvar; ir++) d2sum[ir] = 0.0;

  double * derivMatrix = new double [ntimes*4];
  for (int ir = 0; ir < ntimes*4; ir++) derivMatrix[ir] = 0.0;

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

  double * step = new double [nvar];
  double * gdiagvar = new double [nvar];
  for (int ir = 0; ir < nvar; ir++)
  {
    step[ir] = 1.0;
    gdiagvar[ir] = 0.0;
  }

  double newlk = 0.0;
  if (lambda !=0) newlk = -(log(sqrt(lambda)) * nvar);

  for (int i = 0; i < nvar; i++)
  { /* per observation time calculations */
  IntegerVector covrows_in = covrowlist[i];
    RVector<int> covrows(covrows_in);
    int rowN = covrows.size();

    double gdiag_private = 0.0;
    double beta_local = beta[i];
#pragma omp parallel  default(none) reduction(+:gdiag_private) shared(rowN, maxid, covrows, coval, beta_local, weights, Outcomes, id, zbeta) //reduction(+:zbeta[:maxid])
{
  double* zbeta_private = new double [maxid];
  for (int ir = 0; ir < maxid; ir++) zbeta_private[ir] = 0.0;

  int size = omp_get_num_threads(); // get total number of processes
  int rank = omp_get_thread_num(); // get rank of current ( range 0 -> (num_threads - 1) )

  for (int covi = rank * rowN / size; covi < (rank + 1) * rowN / size ; covi++)
  {
    int row = covrows[covi] - 1; // R_xlen_t is signed long, size_t is unsigned. R vectors use signed so all numbers should be below 2,147,483,647
    int rowid = id[row] - 1;
    double covali = coval[row];
    zbeta_private[rowid] += beta_local * covali;
    if (Outcomes[rowid] > 0 ) gdiag_private += covali * weights[rowid];
  }
  for (int rowid = 0; rowid < maxid ; rowid++)
  {
#pragma omp atomic
    zbeta[rowid] += zbeta_private[rowid];
  }
  delete[] zbeta_private;
}
//    Rcout << "zbeta " <<  zbeta[0]  <<  " newlk " << newlk << " gdiag " << gdiag_private<<"\n";
gdiagvar[i] = gdiag_private;
  }

  /* Cumulative sums that do not depend on the specific covariate update*/
  double newlk_private = 0.0;
#pragma omp parallel  default(none) reduction(+:newlk_private)  shared(efron_wt,denom,timein, timeout, zbeta, weights,  Outcomes,/* denom,efron_wt,*/ ntimes,maxid)
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

  for (int rowid = (rank * maxid / size); rowid < ((rank + 1) * maxid / size) ; rowid++)
  {
    int time_index_entry = timein[rowid] - 1;  // Time starts at zero but no events at this time to calculate sums. Lowest value of -1 but not used to reference into any vectors
    int time_index_exit = timeout[rowid] - 1;

    double zbeta_temp = zbeta[(rowid)] >22 ? 22 : zbeta[(rowid)];
    zbeta_temp = zbeta_temp < -200 ? -200 : zbeta_temp;
    double risk = exp(zbeta_temp) * weights[rowid];

    //cumumlative sums for all patients
    for (int r = time_index_exit; r > time_index_entry ; r--)
      denom_private[(r)] += risk;

    if (Outcomes[rowid] > 0 )
    {
      /*cumumlative sums for event patients */
      newlk_private += zbeta_temp * weights[rowid];
      efron_wt_private[(time_index_exit)] += risk;
    }
#pragma omp atomic write
    zbeta[(rowid)] = zbeta_temp; // should be threadsafe without atomic as threads by rowid
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
//Rcout << "zbeta " <<  zbeta[0]  << " denom " << denom[ntimes - 1] << " newlk " << newlk << " efron " << efron_wt[0] <<"\n";

/* Vectors for holding intermediate values in inference loop */
int lastvar = nvar-1;
double  loglik = 0.0;

for (int iter = 1; iter <= MSTEP_MAX_ITER; iter++)
{
  for (int i = 0; i < nvar; i++)
  {
    newlk += d2sum[lastvar];   // newlk is sum across all event times delta*risk -log(denom) - newlk contains sum across all variables, but updated with each beta update
    d2sum[i] = 0.0;
    double gdiag = -gdiagvar[i];
    double hdiag = 0.0;
    for (int ir = 0; ir < (ntimes*4); ir++) derivMatrix[ir] = 0.0;

    IntegerVector covrows_in = covrowlist[i];
    RVector<int> covrows(covrows_in);
    int rowN = covrows.size();

#pragma omp parallel default(none) shared(rowN, derivMatrix, covrows, coval, weights, Outcomes, ntimes, id, timein, timeout, zbeta)
{
  double * derivMatrix_private= new double [ntimes*4];
  for (int ir = 0; ir < (ntimes*4); ir++) derivMatrix_private[ir] = 0.0;

  int size = omp_get_num_threads(); // get total number of processes
  int rank = omp_get_thread_num(); // get rank of current ( range 0 -> (num_threads - 1) )

  for (int covi = (rank * rowN / size); covi < (rank + 1) * rowN / size; covi++)
  {
    int row = covrows[covi] - 1; // R_xlen_t is signed long, size_t is unsigned. R vectors use signed
    int rowid = id[row] - 1;

    int time_index_entry = timein[rowid] - 1; // std vectors use unsigned can be negative though for time 0
    int time_index_exit = timeout[rowid] - 1; // std vectors use unsigned

    double risk = exp(zbeta[(rowid)]) * weights[rowid];
    double covali = coval[row];
    double derivFirst = risk * covali;
    double derivSecond = derivFirst * covali;
    for (int r = time_index_exit ; r >time_index_entry ; r--) // keep int for calculations of indices then cast
    {
      derivMatrix_private[(r)] += derivFirst;
      derivMatrix_private[(ntimes + r)] += derivSecond;
    }
    if (Outcomes[rowid] > 0)
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

for(int r = OutcomeTotalTimes.size() -1 ; r >=0 ; r--)
{
  int time = OutcomeTotalTimes[r] - 1;
  for (int k = 0; k < OutcomeTotals[r]; k++)
  {
    double temp = (double)k
    / (double)OutcomeTotals[r];
    double d2 = denom[time] - (temp * efron_wt[time]); /* sum(denom) adjusted for tied deaths*/
d2sum[i] += safelog(d2); // track this sum to remove from newlk at start of next iteration
newlk -= safelog(d2);
double temp2 = (derivMatrix[(time)] - (temp * derivMatrix[((2*ntimes) + time)])) / d2;
gdiag += temp2;
hdiag += ((derivMatrix[(ntimes + time)] - (temp * derivMatrix[((3*ntimes) + time)])) / d2) -
  (temp2 * temp2);
  }
}

/* Update */
double dif = 0;
if (lambda !=0) {
  dif = (gdiag + (beta[i] / lambda)) / (hdiag + (1.0 / lambda));
} else {
  dif = (gdiag ) / (hdiag);
}

if (fabs(dif) > step[i]) dif = (dif > 0.0) ? step[i] : -step[i];

step[i] = ((2.0 * fabs(dif)) > (step[i] / 2.0)) ? 2.0 * fabs(dif) : (step[i] / 2.0);//Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245

if (lambda !=0) newlk += (beta[i] * beta[i]) / (2.0 * lambda);

beta[i] -= dif;

if (lambda !=0) newlk -= (beta[i] * beta[i]) / (2.0 * lambda);// Include iteration penalty for this covariate

double newlk_private = 0.0;

/* Update cumulative sums dependent on denominator and zbeta so need to accumulate updates then apply them*/
#pragma omp parallel  default(none) reduction(+:newlk_private)  shared(rowN, denom,efron_wt,zbeta,covrows, coval, weights, Outcomes, ntimes, id, timein, timeout,maxid, dif)///*,  denom, efron_wt, newlk*/)
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
    int rowid = (id[row] - 1);

    double riskold = exp(zbeta[(rowid)]);

    double xbdif = dif * coval[row];

    double zbeta_updated = zbeta[(rowid)] - xbdif;
    zbeta_updated =  zbeta_updated >22 ? 22 : zbeta_updated;
    zbeta_updated =  zbeta_updated < -200 ? -200 :  zbeta_updated;
#pragma omp atomic write
    zbeta[(rowid)] = zbeta_updated; // Each covariate only once per patient per time so can update directly
    double riskdiff = (exp(zbeta_updated) - riskold) * weights[rowid];

    int time_index_entry =  timein[rowid] - 1;
    int time_index_exit =  timeout[rowid] - 1;

    for (int r = time_index_exit; r > time_index_entry ; r--)
      denom_private[(r)] += riskdiff; // need to update exp(xb1 + xb2 + ) + exp(x2b1 + x2b2 +)

    if (Outcomes[rowid] > 0 )
    {
      newlk_private += xbdif * weights[rowid];
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
newlk -= newlk_private;

lastvar = i;
  }


  if (fabs(1.0 - (newlk / loglik)) <= MAX_EPS) break;
  loglik = newlk;

  Rcpp::Rcout << " Iter:  " << iter << " Cox likelihood : " << loglik << "\n";
  for (int i = 0; i<nvar; i++) beta_in[i] = beta[i];

  Rcout << "Beta : " ;
  for (int i = 0; i < nvar; i ++) Rcout << beta[i] << " ";
  Rcout << '\n';
} /* return for another iteration */


/* baseline hazard whilst zbeta in memory */
DoubleVector bh(ntimes);
RVector<double> basehaz(bh);
for (int ir = 0; ir < ntimes; ir++)
  basehaz[ir] = 0.0;

DoubleVector ch(ntimes);
RVector<double> cumhaz(ch);

DoubleVector chentry(maxid);
RVector<double> cumhazEntry(chentry);

DoubleVector bhentry(maxid);
RVector<double> BaseHazardEntry(bhentry);
DoubleVector ch1yr(maxid);
RVector<double> cumhaz1year(ch1yr);
DoubleVector rsk(maxid);
RVector<double> Risk(rsk);


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
      basehaz_private = 1e-100; //log(basehaz) required so a minimum measureable hazard is required to avoid NaN errors.
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


#pragma omp parallel  default(none)  shared(ntimes,cumhaz1year,cumhazEntry, cumhaz, BaseHazardEntry, Risk, basehaz, timein, zbeta, weights, maxid)
{
  int size = omp_get_num_threads(); // get total number of processes
  int rank = omp_get_thread_num(); // get rank of current ( range 0 -> (num_threads - 1) )

  for (int rowid = (rank * maxid / size); rowid < ((rank + 1) * maxid / size) ; rowid++)
  {
    int time_index_entry = timein[rowid] - 1;  // Time starts at zero but no events at this time to calculate sums. Lowest value of -1 but not used to reference into any vectors
    int time_one_year = time_index_entry + 365;
    if (time_one_year >= ntimes) time_one_year = ntimes -1;
    BaseHazardEntry[rowid] = basehaz[time_index_entry];
    cumhazEntry[rowid] = cumhaz[time_index_entry];
    cumhaz1year[rowid] = cumhaz[time_one_year];
    Risk[rowid] = exp(zbeta[rowid]);
  }
}

delete[] step;
delete[] gdiagvar;
delete[] denom;
delete[] efron_wt;
delete[] zbeta;
delete[] derivMatrix;
delete[] d2sum;

return List::create(_["Beta"] = beta,
                    _["BaseHaz"] = basehaz,
                    _["CumHaz"] = cumhaz,
                    _["BaseHazardAtEntry"] = BaseHazardEntry,
                    _["CumHazAtEntry"] = cumhazEntry,
                    _["CumHazOneYear"] = cumhaz1year,
                    _["Risk"] = Risk);

}

