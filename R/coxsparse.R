#' @name coxsparse
#' @description
#' Implementation of a Cox proportional hazards model using 
#' a sparse data structure. The model is fitted with cyclical 
#' coordinate descent (after Mittal et al (2013).
#' OpenMP is used to parallelise the updating of cumulative 
#' values and rcppParallel objects are used to make R objects
#' threadsafe.
#' @details
#' The purpose of this implementation is for fitting a Cox model
#' to data when coxph from the survival package fails due to
#' not enough memory to hold the model and data matrices. The
#' focus is therefore on being memory efficient, which is a 
#' slower algorithm than in coxph, but parallelisation is 
#' possible to offset this. In this situation compiling the 
#' code for the native computer setup would be preferable
#' to providing a standard package binary for multiple systems.
#' The Makevars file therefore contains the options for this.
#' 
#' The data structure is a deconstructed sparse matrix.
#' 
#' A function using the same data structure to calculate profile
#' confidence intervals with a crude search pattern is provided.
#' 
#' @keywords internal 
"_PACKAGE"
#' @author Colin Crooks <colin,crooks@nottingham.ac.uk>
#' @import Rcpp RcppParallel
#' @importFrom Rcpp evalCpp
#' @importFrom RcppParallel RcppParallelLibs
#' @importFrom survival coxph basehaz
#' @importFrom data.table ':='
#' @useDynLib coxsparse
#' @name coxsparse
NULL  