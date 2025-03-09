#include <Rcpp.h>

using namespace Rcpp;


// [[Rcpp::export]]
double safelog (double x)
{
  return(x < 1e-200 ? 1e-200 : log(x));
}

