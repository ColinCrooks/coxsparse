pkgbuild::compiler_flags(debug = F)
pkgbuild::compile_dll(debug = F)

library(coxsparse)
library(RcppParallel)
library(testthat)
idvars <- c('id','start','stop','event')
covlist <- c('age','year','surgery','transplant')
reshapevars <- c(idvars,covlist)
heart <- survival::heart
heart.dt <- data.table::setDT(data.table::copy(heart))
heart.dt <- rbind(heart.dt,data.table::data.table("start" = c(3,8,10),
                    "stop"  = c(5,9,18),
                    "event" = c(0,1,0),
                    "age"   = c(0,0,0),
                    "year"  = c(0,0,0),
                    "surgery" = c(0,0,0),
                    "transplant" = c(0,0,0),
                    "id" = 104:106))

heart.dt <- data.table::rbindlist(lapply(1:10,function(i) cbind(heart.dt, heart.dt$id+((i-1) * max(heart.dt$id)))))
heart.dt[,id := V2]

set.seed(235739801)
heart.dt[, event := runif(nrow(heart.dt)) > 0.7 | event]

for (c in covlist) {
  heart.dt[,(c) := as.numeric(as.character(get(c)))]

}


tvidvars <- c('stop','start','id','event')
data.table::setkeyv(heart.dt, tvidvars)
heart.dt[,rowobs := .I]

OutcomesTotals <- heart.dt[event == 1,sum(event,na.rm = T), by = stop]

data.table::setnames(OutcomesTotals,'V1','TotalDeaths')
Outcomes <- as.integer(heart.dt[['event']])
OutcomesTotalUnique <-  OutcomesTotals$TotalDeaths  
OutcomesTotalTimes <- OutcomesTotals$stop 

timein <- as.integer(heart.dt[['start']])
timeout <- as.integer(heart.dt[['stop']])
id <- as.integer(heart.dt[['id']])


idtable <- data.table::data.table('idn' = 1:length(id),id, key = c('id','idn'))
idtable[,idlength := .N,id]
idend <- as.integer(idtable[c(T,id[-1]!=id[-.N]),cumsum(idlength)])
idstart <- as.integer(c(1,idend[-length(idend)]+1))
idn <- as.integer(idtable[['idn']])
rm(idtable,id)

#################################################################
data.table::setkeyv(heart.dt, tvidvars)

covstart <- 1
covbuild <- data.table("rowobs" = integer(), coval = double())
i <- 1
for (x in covlist)
{
  if(i == 1) {
    covstart <- 1 
  } else {
    covstart <-  c(covstart,covend[length(covend)] + 1)
  }
  heart.dt[is.finite(x),(x) := as.double(x)]
  covbuild <- rbind(covbuild,
                    heart.dt[is.finite(get(x)) &
                                   get(x) != 0,
                                 c('rowobs',x),
                                 keyby = .(rowobs),
                                 with = F],
                    use.names=FALSE)
  if(i == 1) {
    covend <- nrow(covbuild)
    } else {
     covend <- c(covend,nrow(covbuild))
    }
  i <- i + 1
}
################################################################

obs <- covbuild$rowobs
coval <- covbuild$coval


nvar <- length(covstart)
weights <- rep(1,length(Outcomes))

Sys.time()
system.time(coxunreg <- survival::coxph(Surv(start, stop, event) ~ age + year + surgery + transplant ,weights = weights, data = heart.dt))
coxunreg_bh <- survival::basehaz(coxunreg, centered = F)
Sys.time() #20s
coef(coxunreg)
coxunreg$loglik


setThreadOptions(numThreads = 32)

lambda = 0.0

 CoxRegListParallelbeta <- list()

 CoxRegListParallelbeta[['Beta']] <- vector('double',length(covstart))
 CoxRegListParallelbeta[['Frailty']] <- vector('double',length(idstart))
 CoxRegListParallelbeta[['basehaz']] <- vector('double',max(timeout))
 CoxRegListParallelbeta[['cumhaz']] <- vector('double',max(timeout))
 CoxRegListParallelbeta[['ModelSummary']] <- vector('double',8)

Sys.time()
system.time(coxsparse::cox_reg_sparse_parallel(modeldata = CoxRegListParallelbeta,
                                                   obs_in = obs,                                                   
                                                    Outcomes_in = Outcomes,
                                                    OutcomeTotals_in = OutcomesTotalUnique,
                                                    OutcomeTotalTimes_in = OutcomesTotalTimes,
                                                    timein_in = timein,
                                                    timeout_in = timeout,
                                                    weights_in = weights,
                                                    coval_in = coval,
                                                  covstart_in = covstart,
                                                  covend_in = covend,
                                                  idn_in = vector('integer',0L),
                                                  idstart_in = vector('integer',0L),
                                                  idend_in = vector('integer',0L),
                                                    lambda = lambda,
                                                    theta_in = 0,
                                                    MSTEP_MAX_ITER = 1,
                                                    MAX_EPS = 1e-10,
                                                    threadn = 32))
Sys.time() # 17 s

names(CoxRegListParallelbeta$Beta) <- c('age', 'year', 'surgery', 'transplant')
expect_equal(coef(coxunreg), CoxRegListParallelbeta$Beta, tolerance =  1e-1)



rbind(coef(coxunreg), CoxRegListParallelbeta$Beta)

################### Frailty ###########
library(coxsparse)
library(RcppParallel)
library(testthat)
library(survival)
idvars <- c('id','start','stop','event')
covlist <- c('rx','sex')
reshapevars <- c(idvars,covlist)
rats <- survival::rats
rats.dt <- data.table::setDT(data.table::copy(rats))
rats.dt[,start := 0]
data.table::setnames(rats.dt, 'time', 'stop')
data.table::setnames(rats.dt, 'litter', 'id')
data.table::setnames(rats.dt, 'status', 'event')


rats.dt <- rbind(rats.dt[,.(id,rx,stop,event, sex, start)],data.table::data.table("id" = (max(rats.dt$id)+1):(max(rats.dt$id)+6),
                                                                               "rx"   = c(0,0,0),
                                                                               "stop"  = c(5,9,18),
                                                                               "event" = c(1,1,1),
                                                                               "sex"  = c('m','m','m'),
                                                                               "start" = c(3,8,10)
                                                  ))

rats.dt <- data.table::rbindlist(lapply(1:100,function(i) cbind(rats.dt, rats.dt$id+((i-1) * max(rats.dt$id)))))
rats.dt[,id := V2]

for (c in covlist) {
  rats.dt[,(c) := as.numeric(as.factor(get(c))) - 1]
  
}

idtimevar <- c('stop', 'start', 'id','event')

data.table::setkeyv(rats.dt, idtimevar)
rats.dt[,rowobs := .I]
####################################################################

covstart <- 1
covbuild <- data.table("rowobs" = integer(), coval = double())
i <- 1
for (x in covlist)
{
  if(i == 1) {
    covstart <- 1 
  } else {
    covstart <-  c(covstart,covend[length(covend)] + 1)
  }
  rats.dt[is.finite(x),(x) := as.double(x)]
  covbuild <- rbind(covbuild,
                    rats.dt[is.finite(get(x)) &
                               get(x) != 0,
                             c('rowobs',x),
                             keyby = .(rowobs),
                             with = F],
                    use.names=FALSE)
  if(i == 1) {
    covend <- nrow(covbuild)
  } else {
    covend <- c(covend,nrow(covbuild))
  }
  i <- i + 1
}

obs <- covbuild$rowobs
coval <- covbuild$coval
##################################################################

data.table::setkeyv(rats.dt, idtimevar)

OutcomesTotals <- rats.dt[event == 1,sum(event,na.rm = T) , keyby = stop]
data.table::setnames(OutcomesTotals,'V1','TotalDeaths')

data.table::setkeyv(rats.dt, idtimevar)
rats.dt[, OutcomeTotals := OutcomesTotals[rats.dt,.(TotalDeaths) ,on = 'stop', mult = 'first']]

rats.dt[, OutcomeTotals := OutcomeTotals*(seq_len(.N) == 1),by = stop]  ## Counting time backwards in accumulator loops so need to count at patient with lowest ID per time
rats.dt[is.na(OutcomeTotals)  , OutcomeTotals := 0]

Outcomes <- as.integer(rats.dt[['event']])
OutcomesTotalUnique <- OutcomesTotals[['TotalDeaths']]
OutcomesTotalTimes <- OutcomesTotals[['stop']]

timein <- as.integer(rats.dt[['start']])
timeout <- as.integer(rats.dt[['stop']])
id <- as.integer(rats.dt[['id']])

idtable <- data.table::data.table('idn' = 1:length(id),id, key = c('id','idn'))
idtable[,idlength := .N,id]
idend <- as.integer(idtable[c(T,id[-1]!=id[-.N]),cumsum(idlength)])
idstart <- as.integer(c(1,idend[-length(idend)]+1))
idn <- as.integer(idtable[['idn']])
rm(idtable,id)

set.seed(235739801)
weights <- runif(length(Outcomes))

setThreadOptions(numThreads = 32)

lambda = 0
CoxRegListParallelbetanoFrailty <- list()

CoxRegListParallelbetanoFrailty[['Beta']] <- vector('double',length(covstart))
CoxRegListParallelbetanoFrailty[['Frailty']] <- vector('double',length(idstart))
CoxRegListParallelbetanoFrailty[['basehaz']] <- vector('double',max(timeout))
CoxRegListParallelbetanoFrailty[['cumhaz']] <- vector('double',max(timeout))
CoxRegListParallelbetanoFrailty[['ModelSummary']] <- vector('double',8)

weights <- runif(length(Outcomes))
Sys.time()

system.time(coxnofrail<- survival::coxph(Surv(start, stop, event) ~ rx + sex  ,weights = rep(1,length(weights)),  data = rats.dt))
Sys.time()

system.time(cox_reg_sparse_parallel(modeldata = CoxRegListParallelbetanoFrailty,               
                                                           obs_in = obs,
                                                         coval_in = coval,
                                                         weights_in = rep(1,length(weights)),
                                                         timein_in = timein,
                                                         timeout_in = timeout,
                                                         Outcomes_in = Outcomes,
                                                         OutcomeTotals_in = OutcomesTotalUnique,
                                                         OutcomeTotalTimes_in = OutcomesTotalTimes,
                                                         covstart_in = covstart,
                                                         covend_in = covend,
                                                         idn_in = vector('integer',0L),
                                                         idstart_in = vector('integer',0L),
                                                         idend_in = vector('integer',0L),
                                                         lambda = lambda,
                                                         theta_in = 0,
                                                         MSTEP_MAX_ITER = 10,
                                                         MAX_EPS = 1e-10,
                                                         threadn = 32
                                                          ))
Sys.time()

names(CoxRegListParallelbetanoFrailty$Beta) <- c('rx', 'sex')
 expect_equal(c(coef(coxnofrail)), c(CoxRegListParallelbetanoFrailty$Beta), tolerance = 1e-6)

 coxfrail<- survival::coxph(Surv(start, stop, event) ~ rx + sex + frailty(id, trace =T) ,weights = rep(1,length(weights)),  data = rats.dt,model = T)

setThreadOptions(numThreads = 32)
set.seed(235739801)
lambda =0
 CoxRegListParallelbetaFrailty <- list()

 CoxRegListParallelbetaFrailty[['Beta']] <- vector('double',length(covstart))
 CoxRegListParallelbetaFrailty[['Frailty']] <- vector('double',length(idstart))
 CoxRegListParallelbetaFrailty[['basehaz']] <- vector('double',max(timeout))
 CoxRegListParallelbetaFrailty[['cumhaz']] <- vector('double',max(timeout))
 CoxRegListParallelbetaFrailty[['ModelSummary']] <- vector('double',8)

weights <- runif(length(Outcomes))
Sys.time()
history <- list()

system.time(cox_reg_sparse_parallel(modeldata = CoxRegListParallelbetaFrailty,                
                                                           obs_in = obs,                                                   
                                                           Outcomes_in = Outcomes,
                                                           OutcomeTotals_in = OutcomesTotalUnique,
                                                           OutcomeTotalTimes_in = OutcomesTotalTimes,
                                                           timein_in = timein,
                                                           timeout_in = timeout,
                                                           weights_in = rep(1,length(weights)),
                                                           coval_in = coval,
                                                         covstart_in = covstart,
                                                         covend_in = covend,
                                                         idn_in = idn,
                                                         idstart_in = idstart,
                                                         idend_in = idend,
                                                           lambda = lambda,
                                                         theta_in = 0,
                                                           MSTEP_MAX_ITER = 20,
                                                           MAX_EPS = 1e-9,
                                                           threadn = 32))
 
expect_equal(c(t(CoxRegListParallelbetaFrailty$Frailty)-unlist(CoxRegListParallelbetaFrailty$ModelSummary[8])),
as.vector(coxfrail$frail),tolerance = 10-9)

names(CoxRegListParallelbetaFrailty$Beta) <- c('rx', 'sex')

expect_equal(CoxRegListParallelbetaFrailty$Beta,
             coxfrail$coefficients,tolerance = 10-9)

cbind(CoxRegListParallelbetaFrailty$cumhaz[1:max(timeout) %in% basehaz(coxfrail)$time], basehaz(coxfrail, centered = F)$hazard)
plot(cbind(CoxRegListParallelbetaFrailty$cumhaz[1:max(timeout) %in% basehaz(coxfrail)$time], basehaz(coxfrail, centered = F)$hazard))


rbind(coef(coxnofrail),
      CoxRegListParallelbetanoFrailty$Beta,
      coef(coxfrail),
CoxRegListParallelbetaFrailty$Beta
)

par(mfrow = c(2,2))
hist(coxfrail$frail)
hist(c(t(CoxRegListParallelbetaFrailty$Frailty)))
hist(c(t(CoxRegListParallelbetaFrailty$Frailty)-unlist(CoxRegListParallelbetaFrailty$ModelSummary[8])))

set.seed(235739801)
profileCI <- profile_ci(beta_in = c(CoxRegListParallelbetaFrailty$Beta),
                        obs_in = obs,
                        coval_in = coval,
                         weights_in =  rep(1,length(weights)),
                         frailty_in = c(t(CoxRegListParallelbetaFrailty$Frailty)),
                         timein_in = timein ,
                         timeout_in = timeout ,
                         Outcomes_in = Outcomes ,
                         OutcomeTotals_in = OutcomesTotalUnique ,
                         OutcomeTotalTimes_in = OutcomesTotalTimes,
                        covstart_in = covstart,
                        covend_in = covend,
                        idn_in = idn,
                        idstart_in = idstart,
                        idend_in = idend,
                         lambda = lambda,
                         theta_in = c(CoxRegListParallelbetaFrailty$ModelSummary[8]),
                         MSTEP_MAX_ITER = 100000,
                         decimals = 10,
                         confint_width = 0.95,
                         threadn = 32)
cbind(CoxRegListParallelbetaFrailty$Beta,profileCI )
rbind(c(confint(coxfrail,type='profile')) , c(profileCI))

expect_equal(c(confint(coxfrail,type='profile')),c(profileCI), tolerance = 0.2)

###################################################################################

lambda =0.1
system.time(coxfrailpenal <- survival::coxph(Surv(start, stop, event) ~ ridge(rx,sex, theta = 1/lambda, scale = F) + frailty(id, trace =T) ,weights = rep(1,length(weights)),  data = rats.dt,model = T))

# user  system elapsed 
# 0.03    0.00    0.03 

set.seed(235739801)

 CoxRegListParallelbetaFrailtypenal <- list()

 CoxRegListParallelbetaFrailtypenal[['Beta']] <- vector('double',length(covstart))
 CoxRegListParallelbetaFrailtypenal[['Frailty']] <- vector('double',length(idstart))
 CoxRegListParallelbetaFrailtypenal[['basehaz']] <- vector('double',max(timeout))
 CoxRegListParallelbetaFrailtypenal[['cumhaz']] <- vector('double',max(timeout))
 CoxRegListParallelbetaFrailtypenal[['ModelSummary']] <- vector('double',8)

weights <- runif(length(Outcomes))
Sys.time()
history <- list()

# using vector, parallel for and private vectors to reduce
# user  system elapsed 
# 0.81    0.15    0.23 

system.time(
 cox_reg_sparse_parallel(modeldata = CoxRegListParallelbetaFrailtypenal,
                        obs_in = obs,                                                   
                        Outcomes_in = Outcomes,
                        OutcomeTotals_in = OutcomesTotalUnique,
                        OutcomeTotalTimes_in = OutcomesTotalTimes,
                        timein_in = timein,
                        timeout_in = timeout,
                        weights_in = rep(1,length(weights)),
                        coval_in = coval,
                        covstart_in = covstart,
                        covend_in = covend,
                        idn_in = idn,
                        idstart_in = idstart,
                        idend_in = idend,
                        lambda = lambda,
                        theta_in = 0,
                        MSTEP_MAX_ITER = 20,
                        MAX_EPS = 1e-9,
                        threadn = 32)
  )


rbind(coef(coxnofrail),
      CoxRegListParallelbetanoFrailty$Beta,
      coef(coxfrail),
      CoxRegListParallelbetaFrailty$Beta,
      coef(coxfrailpenal),
      CoxRegListParallelbetaFrailtypenal$Beta
)
names(CoxRegListParallelbetaFrailtypenal$Beta) <- c('ridge(rx)', 'ridge(sex)')
expect_equal(CoxRegListParallelbetaFrailtypenal$Beta,
             coxfrailpenal$coefficients,tolerance = 10-9)
expect_equal(c(t(CoxRegListParallelbetaFrailtypenal$Frailty))-unlist(CoxRegListParallelbetaFrailtypenal$ModelSummary[8]),
             coxfrailpenal$frail,tolerance = 10-9)
