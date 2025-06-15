#include <Rcpp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <map>
#include <limits> // Required for numeric_limits
#include <iomanip> // For printing output

using namespace Rcpp;


// [[Rcpp::export]]
double safelog (double x)
{
  return(x < 1e-200 ? 1e-200 : log(x));
}



// [[Rcpp::export]]
double safesqrt (double x)
{
  return(x <= 0 ? 0 : sqrt(x));
}


// --- Mathematical Helper Functions ---

// Log-gamma function (approximation or use library like Boost.Math if available)
// Using Lanczos approximation (simplified version)
// double log_gamma_approx(double x) {
//   if (x <= 0.0) return std::numeric_limits<double>::infinity(); // Or handle error
//   static const double coeffs[] = {
//     76.18009172947146, -86.50532032941677, 24.01409824083091,
//     -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5
//   };
//   double y = x;
//   double tmp = x + 5.5;
//   tmp -= (x + 0.5) * log(tmp);
//   double ser = 1.000000000190015;
//   for (int j = 0; j < 6; ++j) {
//     ser += coeffs[j] / ++y;
//   }
//   return -tmp + log(2.5066282746310005 * ser / x);
// }

// Digamma function (approximation - often derived from log_gamma)
// psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + ... for large x
// More robust approximations exist, or use numerical differentiation of lgamma
// double digamma_approx(double x) {
//   if (x <= 0.0) return -std::numeric_limits<double>::infinity(); // Or handle error
//   if (x < 1e-5) return -0.5772156649; // Euler-Mascheroni constant for small x limit (approx)
//   
//   // Use asymptotic expansion for x >= 8
//   if (x >= 8.0) {
//     double invX = 1.0 / x;
//     double invX2 = invX * invX;
//     // Formula: log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
//     return std::log(x) - 0.5 * invX - invX2 / 12.0 + invX2 * invX2 / 120.0 - invX2 * invX2 * invX2 / 252.0;
//   }
//   
//   // Use recurrence relation psi(x+1) = psi(x) + 1/x for smaller x
//   // Increase x until it's >= 8
//   double result = 0.0;
//   while (x < 8.0) {
//     result -= 1.0 / x;
//     x += 1.0;
//   }
//   double invX = 1.0 / x;
//   double invX2 = invX * invX;
//   result += std::log(x) - 0.5 * invX - invX2 / 12.0 + invX2 * invX2 / 120.0 - invX2 * invX2 * invX2 / 252.0;
//   return result;
//   
// }

// double gamma_theta_lkl(double nu, double * outcomes_wt, double* group_lambda, int maxid) {
//   double lkl = 0;
//   
//   // for(int ir = 0; ir < maxid; ir ++) lkl +=(nu*safelog(nu))  - std::lgamma(nu) +  
//   //       digamma_approx(nu + outcomes_wt[ir]) -
//   //       safelog(nu + group_lambda[ir]) +
//   //       (nu + outcomes_wt[ir])/(nu + group_lambda[ir] + 1e-100) ;
//   for(int ir = 0; ir < maxid; ir ++) {
//   if (outcomes_wt[ir] == 0 && group_lambda[ir] == 0.0) continue;
//   lkl += log_gamma_approx(outcomes_wt[ir] + nu) - log_gamma_approx(nu);
//   lkl -= (outcomes_wt[ir] + nu) * safelog(std::max(group_lambda[ir], 1e-100) + nu);
//   lkl += nu * safelog(nu);
//   }
  //     
  //     safelog(std::tgamma(nu + outcomes_wt[ir])/std::tgamma(nu))  + 
  //      outcomes_wt[ir] ;
  
  // if (1/nu <= 1e-8) return -std::numeric_limits<double>::infinity(); // Avoid theta=0 or negative
  // 
  // for(int ir = 0; ir < maxid; ir ++) {
  // 
  //   if (outcomes_wt[ir] == 0 && group_lambda[ir] == 0.0) continue; // Group contributed nothing
  //   
  //   // Prevent log(0) issues if lambda_k is tiny but positive
  //   double lambda_plus_inv_theta = std::max(group_lambda[ir], 1e-100) + nu;
  //   
  //   lkl += log_gamma_approx(outcomes_wt[ir] + nu) - log_gamma_approx(nu);
  //   lkl -= (outcomes_wt[ir] + nu) * std::log(lambda_plus_inv_theta);
  //   lkl += nu * std::log(nu);
  // }
//   return lkl;
// 
// }

// Function to calculate the profile log-likelihood for theta given beta and group summaries
// This is the function we want to MAXIMIZE w.r.t. theta
// double profile_loglik_theta(double theta,
//                             const std::map<int, int>& group_event_counts, // d_k
//                             const std::map<int, double>& group_lambda_hat) // Lambda_k
// {
//   if (theta <= 1e-8) return -std::numeric_limits<double>::infinity(); // Avoid theta=0 or negative
//   
//   double loglik = 0.0;
//   double inv_theta = 1.0 / theta;
//   
//   for (const auto& pair : group_event_counts) {
//     int group_k = pair.first;
//     int d_k = pair.second;
//     double lambda_k = group_lambda_hat.count(group_k) ? group_lambda_hat.at(group_k) : 0.0;
//     
//     if (d_k == 0 && lambda_k == 0.0) continue; // Group contributed nothing
//     
//     // Prevent log(0) issues if lambda_k is tiny but positive
//     double lambda_plus_inv_theta = std::max(lambda_k, 1e-100) + inv_theta;
//     
//     loglik += log_gamma_approx(d_k + inv_theta) - log_gamma_approx(inv_theta);
//     loglik -= (d_k + inv_theta) * std::log(lambda_plus_inv_theta);
//     loglik += inv_theta * std::log(inv_theta);
//   }
//   return loglik;
// }


// --- Simple Brent's Method for 1D Optimization ---
// (A basic implementation, robust versions exist in libraries)
// Finds the maximum of func within [ax, bx]
// double brent_maximize(double ax, double bx,
//                       double * outcomes_wt,  double* group_lambda, int maxid,
//                       double tol = 1e-7, int max_iter = 100)
// {
//   const double CGOLD = 0.3819660;
//   const double ZEPS = std::numeric_limits<double>::epsilon() * 1.0e-3;
//   double a, b, d = 0.0, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v, w, x, xm;
//   int iter;
//   
//   a = ax;
//   b = bx;
//   x = w = v = b; // Initialize x, w, v to one endpoint
//   fw = fv = fx = gamma_theta_lkl(1/x,  outcomes_wt, group_lambda,  maxid);
//   
//   for (iter = 0; iter < max_iter; ++iter) {
//     xm = 0.5 * (a + b);
//     tol2 = 2.0 * (tol1 = tol * std::abs(x) + ZEPS);
//     if (std::abs(x - xm) <= (tol2 - 0.5 * (b - a))) {
//       return x; // Converged
//     }
//     if (std::abs(d) > tol1) { // Try parabolic fit
//       r = (x - w) * (fx - fv);
//       q = (x - v) * (fx - fw);
//       p = (x - v) * q - (x - w) * r;
//       q = 2.0 * (q - r);
//       if (q > 0.0) p = -p;
//       q = std::abs(q);
//       etemp = d;
//       d = p; // Accept interpolation? Check bounds and previous step size
//       if (std::abs(p) >= std::abs(0.5 * q * etemp) || p <= q * (a - x) || p >= q * (b - x))
//         d = CGOLD * (x >= xm ? a - x : b - x); // Golden section step
//     } else {
//       d = CGOLD * (x >= xm ? a - x : b - x); // Golden section step
//     }
//     u = (std::abs(d) >= tol1) ? x + d : x + (d > 0.0 ? tol1 : -tol1);
//     fu = gamma_theta_lkl(1/u,  outcomes_wt, group_lambda, maxid);
//     
//     // Update a, b, v, w, x based on fu and fx
//     if (fu > fx) {
//       if (u >= x) a = x; else b = x;
//       v = w; w = x; x = u;
//       fv = fw; fw = fx; fx = fu;
//     } else {
//       if (u < x) a = u; else b = u;
//       if (fu > fw || w == x) {
//         v = w; w = u;
//         fv = fw; fw = fu;
//       } else if (fu > fv || v == x || v == w) {
//         v = u;
//         fv = fu;
//       }
//     }
//   }
//   // std::cerr << "Warning: Brent optimization reached max iterations." << std::endl;
//   return x; // Return best estimate
// }