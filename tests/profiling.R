library(syrup)
library(future)
library(tidymodels)
library(rlang)

local_options(parallelly.fork.enable = TRUE)
plan(multicore, workers = =10)

set.seed(1)
dat <- sim_regression(100)