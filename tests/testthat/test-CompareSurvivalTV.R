pkgbuild::compiler_flags(debug = T)
pkgbuild::compile_dll(debug = T)
library(bit64)
library(coxsparse)
library(RcppParallel)
library(testthat)
devtools::load_all()

######################TV######################################################
covlist <- c('age', 'sex', 'expAntiTNF')
idvars <- c('id', 'tstart', 'fuptime', 'anyevent')

data("dataExample")


data("miniCESAME")
setDT(miniCESAME)
tvidvars <- c('id', 'fuptime', 'tstart', 'event')
data.table::setkeyv(miniCESAME, tvidvars)
miniCESAME[, rowobs := .I]
miniCESAME[, anyevent := event > 0]
miniCESAME[,sex := as.numeric(sex == 'M')]

for (c in covlist) {
  miniCESAME[, (c) := as.numeric(as.character(get(c)))]
}


coxph(
  Surv(tstart, fuptime, anyevent != 0) ~ age +
    sex +
#expThiop +
#expMetho +
    expAntiTNF,
  id = id,
  data = miniCESAME
)


############ covariates #########################


OutcomesTotals <- miniCESAME[
  anyevent == 1,
  sum(anyevent, na.rm = T),
  by = fuptime
]

data.table::setnames(OutcomesTotals, 'V1', 'anyevent')
Outcomes <- as.integer(miniCESAME[['anyevent']])
OutcomesTotalUnique <- OutcomesTotals$anyevent
OutcomesTotalTimes <- OutcomesTotals$fuptime

timein <- as.integer(miniCESAME[['tstart']])
timeout <- as.integer(miniCESAME[['fuptime']])
id <- as.integer(miniCESAME[['id']])


idtable <- data.table::data.table(
  'idn' = 1:length(id),
  id,
  key = c('id', 'idn')
)
idtable[, idlength := .N, id]
idend <- as.integer(idtable[c(T, id[-1] != id[-.N]), cumsum(idlength)])
idstart <- as.integer(c(1, idend[-length(idend)] + 1))
idn <- as.integer(idtable[['idn']])
rm(idtable)

#################################################################
data.table::setkeyv(miniCESAME, tvidvars)

covstart <- as.integer64(1)
covbuild <- data.table("rowobs" = integer(), coval = double())
i <- 1
for (x in covlist) {
  if (i == 1) {
    covstart <- as.integer64(1)
  } else {
    covstart <- c(covstart, as.integer64(covend[length(covend)] + 1))
  }
  miniCESAME[is.finite(x), (x) := as.double(x)]
  covbuild <- rbind(
    covbuild,
    miniCESAME[
      is.finite(get(x)) &
        get(x) != 0,
      c('rowobs', x),
      keyby = .(rowobs),
      with = F
    ],
    use.names = FALSE
  )
  if (i == 1) {
    covend <- as.integer64(nrow(covbuild))
  } else {
    covend <- c(covend, as.integer64(nrow(covbuild)))
  }
  i <- i + 1
}
################################################################

obs <- covbuild$rowobs
coval <- as.numeric(covbuild$coval)

##########################

i <- as.integer(covstart[1:3])
cbind(id[obs[i]], timein[obs[i]], timeout[obs[i]], Outcomes[obs[i]], coval[i])

i <- as.integer(covend[1:3])
cbind(id[obs[i]], timein[obs[i]], timeout[obs[i]], Outcomes[obs[i]], coval[i])


i <- 1:100
cbind(id[obs[i]], timein[obs[i]], timeout[obs[i]], Outcomes[obs[i]], coval[i])

j <- 1
checkcovids <- c()
checkobs <- c()
for (i in idn[idstart[j]]:idn[idend[j]]) {
  tempcheck <- which(obs == i)
  checkobs <- c(checkobs, obs[tempcheck])
  checkcovids <- c(checkcovids, tempcheck)
}

cbind(
  timein[idn[idstart[i]]:idn[idend[i]]],
  timeout[idn[idstart[i]]:idn[idend[i]]],
  Outcomes[idn[idstart[i]]:idn[idend[i]]],
  dcast(data.table(checkobs, checkcovids), checkobs ~ .)
)



########### Splines ###########################
wnd <- 50
n.knots <- 1
knots <- unique(seq(1, wnd, wnd / n.knots))
measure.points <- c(0:(2 * wnd))

bspl <- splines::bs(
  knots = unique(c(knots)),
  x = measure.points,
  intercept = T
)

#####################################

miniCESAME[(expAntiTNF == 1),exposurestart := tstart]

miniCESAME.long <- miniCESAME[miniCESAME[,.(start = seq(tstart, fuptime-1)), by = rowobs],, on = 'rowobs']
miniCESAME.long[, stop := shift(start,n =1L,type ='lead'), rowobs ]
miniCESAME.long[is.na(stop), stop := fuptime]
miniCESAME.long[,event := anyevent & fuptime == stop]

miniCESAME.long[,exposure := exposurestart == start]
miniCESAME.long[is.na(exposure),exposure := 0]

#######################
cumulative.range <- Rcpp::cppFunction(
  "DoubleVector cumulativerange(IntegerVector duration, DoubleVector dosedays, IntegerVector id, const int upperlimit, const int lowerlimit) {
  int n = duration.size();
  DoubleVector result(n,0.0);
  for (int i = n - 1; i >= 0; i--) {
  int runofdays = 0;
  int ii = i;
  while((runofdays < upperlimit) & (id[i] == id[ii])) {
    if (runofdays >= lowerlimit) 
      result[i] += dosedays[ii];
    runofdays += duration[ii];
    ii--;
  }
//  result[i] = (result[i] > 183.0 ? 183.0 : result[i]);
//  result[i] /= 183.0;
  }
  return result;
}"
)
miniCESAME.long[, duration := stop - start]
miniCESAME.long[,
  cumulativedays := cumsum((duration) * expAntiTNF) - ((duration) * expAntiTNF),
  by = id
]


miniCESAME.long[,
  prevduration := shift(duration, n = 1L, type = 'lag', fill = 0),
  by = id
]
miniCESAME.long[, prevdays := c(0, diff(cumulativedays)), by = id]


data.table::setkeyv(miniCESAME.long, idvars)
exposures <-
  as.matrix(cbind(
    miniCESAME.long[, expAntiTNF],
    miniCESAME.long[,
      lapply(1:length(measure.points), function(x) {
        cumulative.range(
          duration = prevduration,
          dosedays = prevdays,
          id = id,
          upperlimit = measure.points[x],
          lowerlimit = measure.points[x - 1]
        )
      })
    ]
  ))
expThiop.splines <- miniCESAME.long[,
  .(rowobs,start, as.data.table(tcrossprod(exposures, t(bspl))))
]
rm(exposures)

miniCESAME.long <- merge(expThiop.splines, miniCESAME.long, by = c('rowobs','start'), all.x = T)




lambda = 0.0
##########################
CoxRegListParallelbeta <- list()
weights <- rep(1, length(Outcomes))
tvbeta_spl <- c(0, 0, 0)

CoxRegListParallelbeta[['Beta']] <- vector('double', length(covstart)) 
CoxRegListParallelbeta[['Frailty']] <- vector('double', length(idstart))
CoxRegListParallelbeta[['basehaz']] <- vector('double', max(timeout))
CoxRegListParallelbeta[['cumhaz']] <- vector('double', max(timeout))
CoxRegListParallelbeta[['ModelSummary']] <- vector('double', 8)

Sys.time()
system.time(cox_reg_sparse_parallel_TV(
  modeldata = CoxRegListParallelbeta,
  obs_in = obs,
  Outcomes_in = Outcomes,
  timein_in = timein,
  timeout_in = timeout,
  weights_in = weights,
  coval_in = coval,
  covstart_in = covstart[1:3],
  covend_in = covend[1:3],
  id_in = id,
  idn_in = vector('integer', 0L),
  idstart_in = vector('integer', 0L),
  idend_in = vector('integer', 0L),
  tvbeta_spl_in = tvbeta_spl,
  bspl_in = bspl,
  lambda = lambda,
  theta_in = 0,
  MSTEP_MAX_ITER = 10,
  MAX_EPS = 1e-10,
  threadn = 32
))

coxuntv <- coxph(
  Surv(tstart, fuptime, anyevent ) ~ age +
    sex +
    expAntiTNF,
  id = id,
  data = miniCESAME
)

coxuntv

names(CoxRegListParallelbeta$Beta) <- covlist
expect_equal(coef(coxuntv), CoxRegListParallelbeta$Beta, tolerance = 1e-1)

rbind((coef(coxuntv)), CoxRegListParallelbeta$Beta)

################# TV ####################

miniCESAME.long[event == 1,lapply(.SD,sum),.SDcols = paste0('X',1:5)]

coxspline <-coxph(
  Surv(start, stop, event) ~ age + sex + X1 + X2 + X3 + X4 + X5,
  id = id,
  data = miniCESAME.long
)
coxspline

lambda = 0.1
weights <- rep(1, length(Outcomes))

tvbeta_spl <- c(0, 0, 1)
CoxRegListParallelbeta_tv <- list()
CoxRegListParallelbeta_tv[['Beta']] <- vector(
  'double',
  length(covstart[1:3]) + (sum(tvbeta_spl) * ncol(bspl)) - sum(tvbeta_spl)
)
CoxRegListParallelbeta_tv[['Frailty']] <- vector('double', length(idstart))
CoxRegListParallelbeta_tv[['basehaz']] <- vector('double', max(timeout))
CoxRegListParallelbeta_tv[['cumhaz']] <- vector('double', max(timeout))
CoxRegListParallelbeta_tv[['ModelSummary']] <- vector('double', 8)

Sys.time()

system.time(cox_reg_sparse_parallel_TV(
  modeldata = CoxRegListParallelbeta_tv,
  obs_in = obs,
  Outcomes_in = Outcomes,
  timein_in = timein,
  timeout_in = timeout,
  weights_in = weights,
  coval_in = coval,
  covstart_in = covstart[1:3],
  covend_in = covend[1:3],
  id_in = id,
  idn_in = vector('integer', 0L),
  idstart_in = vector('integer', 0L),
  idend_in = vector('integer', 0L),
  tvbeta_spl_in =  tvbeta_spl,
  bspl_in = bspl,
  lambda = lambda,
  theta_in = 0,
  MSTEP_MAX_ITER = 20,
  MAX_EPS = 1e-10,
  threadn = 32
))
