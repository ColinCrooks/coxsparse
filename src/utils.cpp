#include <Rcpp.h>


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
