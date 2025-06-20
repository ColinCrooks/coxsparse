library(coxsparse)
library(RcppParallel)
library(testthat)
idvars <- c('id','obs','start','stop','event')
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

heart.dt <- data.table::rbindlist(lapply(1:1000,function(i) cbind(heart.dt, heart.dt$id+((i-1) * max(heart.dt$id)))))
heart.dt[,id := V2]
heart.dt[,obs := .I]

set.seed(235739801)
heart.dt[, event := runif(nrow(heart.dt)) > 0.7 | event]

for (c in covlist) {
  heart.dt[,(c) := as.numeric(as.character(get(c)))]

}

for (c in  c('age','year','surgery','transplant')) {
  heart.dt[,(c) := scale(get(c))]

}

idtimevar <- c('stop', 'start', 'obs','event')
data.table::setkeyv(heart.dt, idvars)


heart.tv.build <-  data.table::melt(heart.dt[,mget(reshapevars)],
                       id.vars = idvars,
                       measure.vars = covlist,
                       variable.name = 'covariate',
                       value.name = 'value')


data.table::setkeyv(heart.tv.build, idvars)
heart.tv.build[,idx := seq_len(.N), by = .(obs,start)]

heart.tv.build[, keep := cummax(value!=0), by = .(obs,start)]
heart.tv.build[, keep := keep[.N], by = .(obs,start)]
heart.tv.build[, keep := value != 0 | (keep == F & idx == 1)]

heart.tv.build <- heart.tv.build[(keep),]

heart.tv.build[, `:=`(covlength = .N,
                     colid = seq_len(.N)),
              by = .(obs,start)]

heart.tv.build <- data.table::dcast(heart.tv.build,
                       id+obs+start+stop+event+covlength ~ colid,
                       value.var = c('idx', 'value'))
data.table::setkeyv(heart.tv.build, idtimevar)

OutcomesTotals <- heart.tv.build[event == 1,sum(event,na.rm = T) , keyby = stop]

OutcomesTotals[,event := 1]
data.table::setnames(OutcomesTotals,'V1','TotalDeaths')

data.table::setkeyv(heart.tv.build, idtimevar)
heart.tv.build[, OutcomeTotals := OutcomesTotals[heart.tv.build,.(TotalDeaths) ,on = 'stop', mult = 'first']]


heart.tv.build[, OutcomeTotals := OutcomeTotals*(seq_len(.N) == 1),by = stop]  ## Counting time backwards in accumulator loops so need to count at patient with lowest ID per time
heart.tv.build[is.na(OutcomeTotals)  , OutcomeTotals := 0]
data.table::setkeyv(heart.tv.build, idtimevar)
heart.tv.build[,rowobs := .I]
OutcomesTotals <- as.integer(heart.tv.build[['OutcomeTotals']])
Outcomes <- as.integer(heart.tv.build[['event']])
OutcomesTotalUnique <- as.integer(heart.tv.build[OutcomeTotals > 0,OutcomeTotals])
OutcomesTotalTimes <- as.integer(heart.tv.build[OutcomeTotals > 0,stop])

timein <- as.integer(heart.tv.build[['start']])
timeout <- as.integer(heart.tv.build[['stop']])
Covlength <- as.integer(heart.tv.build[['covlength']])
id <- as.integer(heart.tv.build[['id']])

nvar <- length(grep("idx_",names(heart.tv.build)))

cov <- data.table::melt(heart.tv.build[,c('rowobs',paste0('idx_',1:nvar)), with = F],id.vars = 'rowobs', na.rm = T)
data.table::setkey(cov, rowobs, variable)
obs <- as.integer(cov[['rowobs']])
cov <- as.integer(cov[['value']])
coval <- data.table::melt(heart.tv.build[,c('rowobs',paste0('value_',1:nvar)), with = F],id.vars = 'rowobs', na.rm = T)
data.table::setkey(coval, rowobs, variable)
all.equal(obs,coval[['rowobs']])
coval <- as.double(coval[['value']])

data.table::data.table(obs = 1:length(id), id, key = c('id','obs'))

nvar <- max(cov, na.rm = T)



idtable <- data.table::data.table('idn' = 1:length(id),id, key = c('id','idn'))
idtable[,idlength := .N,id]
idend <- as.integer(idtable[c(T,id[-1]!=id[-.N]),cumsum(idlength)])
idstart <- as.integer(c(1,idend[-length(idend)]+1))
idn <- as.integer(idtable[['idn']])
rm(idtable,id)

covtable <- data.table::data.table('covn' = 1:length(cov),cov, key = c('cov','covn'))
covtable[,covlength := .N,cov]
covend <- as.integer(covtable[c(T,cov[-1]!=cov[-.N]),cumsum(covlength)])
covstart <- as.integer(c(1,covend[-length(covend)]+1))
covn <- as.integer(covtable[['covn']])
rm(covtable,cov)


weights <- runif(length(Outcomes))


Sys.time()
system.time(coxunreg <- survival::coxph(Surv(start, stop, event) ~ age + year + surgery + transplant ,weights = weights, data = heart.dt))
coxunreg_bh <- survival::basehaz(coxunreg, centered = F)
Sys.time() #20s
coef(coxunreg)
coxunreg$loglik


setThreadOptions(numThreads = 32)

lambda = 0.00

CoxRegListParallelbeta <- list()

CoxRegListParallelbeta[['Beta']] <- vector('double',length(covstart))
CoxRegListParallelbeta[['Frailty']] <- vector('double',length(idstart))
CoxRegListParallelbeta[['basehaz']] <- vector('double',max(timeout))
CoxRegListParallelbeta[['cumhaz']] <- vector('double',max(timeout))
CoxRegListParallelbeta[['BaseHazardEntry']] <- vector('double',max(obs))
CoxRegListParallelbeta[['cumhazEntry']] <- vector('double',max(obs))
CoxRegListParallelbeta[['cumhaz1year']] <- vector('double',max(obs))
CoxRegListParallelbeta[['Risk']] <- vector('double',max(obs))
CoxRegListParallelbeta[['ModelSummary']] <- vector('double',8)

Sys.time()
system.time( cox_reg_sparse_parallel(modeldata = CoxRegListParallelbeta,#beta_in = CoxRegListParallelbeta[['Beta']] ,
                                                                    #frailty_in = CoxRegListParallelbeta[['Frailty']],
                                                                    #basehaz_in  = CoxRegListParallelbeta[['basehaz']],
                                                                    #cumhaz_in = CoxRegListParallelbeta[['cumhaz']] ,
                                                                    #BaseHazardEntry_in = CoxRegListParallelbeta[['BaseHazardEntry']],
                                                                    #cumhazEntry_in = CoxRegListParallelbeta[['cumhazEntry']],
                                                                    #cumhaz1year_in = CoxRegListParallelbeta[['cumhaz1year']],
                                                                    #Risk_in = CoxRegListParallelbeta[['Risk']] ,
                                                   obs_in = obs,                                                   
                                                    Outcomes_in = Outcomes,
                                                    OutcomeTotals_in = OutcomesTotalUnique,
                                                    OutcomeTotalTimes_in = OutcomesTotalTimes,
                                                    timein_in = timein,
                                                    timeout_in = timeout,
                                                    weights_in = weights,
                                                    coval_in = coval,
                                                  covn_in = covn,
                                                  covstart_in = covstart,
                                                  covend_in = covend,
                                                  idn_in = vector('integer',0L),
                                                  idstart_in = vector('integer',0L),
                                                  idend_in = vector('integer',0L),
                                                    lambda = lambda,
                                                    theta_in = 0,
                                                    MSTEP_MAX_ITER = 100,
                                                    MAX_EPS = 1e-10,
                                                    threadn = 32))
Sys.time() # 17 s

names(CoxRegListParallelbeta$Beta) <- c('age', 'year', 'surgery', 'transplant')
expect_equal(coef(coxunreg), CoxRegListParallelbeta$Beta, tolerance =  1e-1)



#expect_equal(CoxRegListParallelbeta$BaseHaz[c(timeout[-1]!=timeout[-length(timeout)],T)], coxunreg_bh$hazard)


# 
# CoxRegListParallelbetaFrail <- cox_reg_sparse_parallel(beta_in = beta,
#                                                   obs_in = obs,                                                   
#                                                   Outcomes_in = Outcomes,
#                                                   OutcomeTotals_in = OutcomesTotalUnique,
#                                                   OutcomeTotalTimes_in = OutcomesTotalTimes,
#                                                   timein_in = timein,
#                                                   timeout_in = timeout,
#                                                   weights_in = weights,
#                                                   coval_in = coval,
#                                                   covn_in = covn,
#                                                   covstart_in = covstart,
#                                                   covend_in = covend,
#                                                   idn_in = idn,
#                                                   idstart_in = idstart,
#                                                   idend_in = idend,
#                                                   lambda = lambda,
#                                                   theta_in = 0,
#                                                   MSTEP_MAX_ITER = 1000,
#                                                   MAX_EPS = 1e-5,
#                                                   threadn = 32)

rbind(coef(coxunreg), CoxRegListParallelbeta$Beta)
# 
# 
# set.seed(235739801)
# profileCI <- profile_ci(covrowlist_in = covrowlist,
#                          beta_in = CoxRegListParallelbeta$Beta,
#                         obs_in = obs,
#                         coval_in = coval,
#                          weights_in =  rep(1,length(weights)),
#                          frailty_in = CoxRegListParallelbeta$Frailty,
#                          timein_in = timein ,
#                          timeout_in = timeout ,
#                          Outcomes_in = Outcomes ,
#                          OutcomeTotals_in = OutcomesTotalUnique ,
#                          OutcomeTotalTimes_in = OutcomesTotalTimes,
#                          nvar = nvar,
#                          lambda = lambda,
#                          MSTEP_MAX_ITER = 100000,
#                          decimals = 10,
#                          confint_width = 0.95,
#                          threadn = 32)
# cbind(CoxRegListParallelbeta$Beta,profileCI )
# rbind(c(confint(coxunreg,type='profile')) , c(profileCI))
# expect_true(max(c(confint(coxunreg,type='profile')) - c(profileCI)) < 0.003)


################### Frailty ###########
library(coxsparse)
library(RcppParallel)
library(testthat)
library(survival)
idvars <- c('id','obs','start','stop','event')
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

rats.dt <- data.table::rbindlist(lapply(1:1,function(i) cbind(rats.dt, rats.dt$id+((i-1) * max(rats.dt$id)))))
rats.dt[,id := V2]
rats.dt[,obs := .I]


for (c in covlist) {
  rats.dt[,(c) := as.numeric(as.factor(get(c))) - 1]
  
}



idtimevar <- c('stop', 'start', 'obs','event')
data.table::setkeyv(rats.dt, idvars)


rats.tv.build <-  data.table::melt(rats.dt[,mget(reshapevars)],
                                    id.vars = idvars,
                                    measure.vars = covlist,
                                    variable.name = 'covariate',
                                    value.name = 'value')


data.table::setkeyv(rats.tv.build, idvars)
rats.tv.build[,idx := seq_len(.N), by = .(obs,start)]

rats.tv.build[, keep := cummax(value!=0), by = .(obs,start)]
rats.tv.build[, keep := keep[.N], by = .(obs,start)]
rats.tv.build[, keep := value != 0 | (keep == F & idx == 1)]

rats.tv.build <- rats.tv.build[(keep),]

rats.tv.build[, `:=`(covlength = .N,
                      colid = seq_len(.N)),
               by = .(obs,start)]

rats.tv.build <- data.table::dcast(rats.tv.build,
                                    id+obs+start+stop+event+covlength ~ colid,
                                    value.var = c('idx', 'value'))
data.table::setkeyv(rats.tv.build, idtimevar)

OutcomesTotals <- rats.tv.build[event == 1,sum(event,na.rm = T) , keyby = stop]

OutcomesTotals[,event := 1]
data.table::setnames(OutcomesTotals,'V1','TotalDeaths')

data.table::setkeyv(rats.tv.build, idtimevar)
rats.tv.build[, OutcomeTotals := OutcomesTotals[rats.tv.build,.(TotalDeaths) ,on = 'stop', mult = 'first']]


rats.tv.build[, OutcomeTotals := OutcomeTotals*(seq_len(.N) == 1),by = stop]  ## Counting time backwards in accumulator loops so need to count at patient with lowest ID per time
rats.tv.build[is.na(OutcomeTotals)  , OutcomeTotals := 0]
data.table::setkeyv(rats.tv.build, idtimevar)
rats.tv.build[,rowobs := .I]
OutcomesTotals <- as.integer(rats.tv.build[['OutcomeTotals']])
Outcomes <- as.integer(rats.tv.build[['event']])
OutcomesTotalUnique <- as.integer(rats.tv.build[OutcomeTotals > 0,OutcomeTotals])
OutcomesTotalTimes <- as.integer(rats.tv.build[OutcomeTotals > 0,stop])

timein <- as.integer(rats.tv.build[['start']])
timeout <- as.integer(rats.tv.build[['stop']])
#Covlength <- as.integer(rats.tv.build[['covlength']])
id <- as.integer(rats.tv.build[['id']])

nvar <- length(grep("idx_",names(rats.tv.build)))

cov <- data.table::melt(rats.tv.build[,c('rowobs',paste0('idx_',1:nvar)), with = F],id.vars = 'rowobs', na.rm = T)
data.table::setkey(cov, rowobs, variable)
obs <- as.integer(cov[['rowobs']])
cov <- as.integer(cov[['value']])
coval <- data.table::melt(rats.tv.build[,c('rowobs',paste0('value_',1:nvar)), with = F],id.vars = 'rowobs', na.rm = T)
data.table::setkey(coval, rowobs, variable)
all.equal(obs,coval[['rowobs']])
coval <- as.double(coval[['value']])

nvar <- max(cov, na.rm = T)

#covrowlist <-lapply(1:nvar, function(i) which(cov == i))
#frailrowlist <- lapply(1:max(id), function(i) which(id == i))


idtable <- data.table::data.table('idn' = 1:length(id),id, key = c('id','idn'))
idtable[,idlength := .N,id]
idend <- as.integer(idtable[c(T,id[-1]!=id[-.N]),cumsum(idlength)])
idstart <- as.integer(c(1,idend[-length(idend)]+1))
idn <- as.integer(idtable[['idn']])
rm(idtable,id)

covtable <- data.table::data.table('covn' = 1:length(cov),cov, key = c('cov','covn'))
covtable[,covlength := .N,cov]
covend <- as.integer(covtable[c(T,cov[-1]!=cov[-.N]),cumsum(covlength)])
covstart <- as.integer(c(1,covend[-length(covend)]+1))
covn <- as.integer(covtable[['covn']])
rm(covtable,cov)


#frailrowlist <- split(data.table::data.table('idn' = 1:length(id),id, key = c('id','idn')),by = 'id', keep.by = F, drop = T, sorted = T, flatten = T)
#frailrowlist <- lapply(frailrowlist, unlist)

weights <- runif(length(Outcomes))


#coxnofrail<- survival::coxph(Surv(start, stop, event) ~ rx + sex ,weights = rep(1,length(weights)),  data = rats.dt)


setThreadOptions(numThreads = 32)
set.seed(235739801)
lambda = 0
CoxRegListParallelbetanoFrailty <- list()

CoxRegListParallelbetanoFrailty[['Beta']] <- vector('double',length(covstart))
CoxRegListParallelbetanoFrailty[['Frailty']] <- vector('double',length(idstart))
CoxRegListParallelbetanoFrailty[['basehaz']] <- vector('double',max(timeout))
CoxRegListParallelbetanoFrailty[['cumhaz']] <- vector('double',max(timeout))
CoxRegListParallelbetanoFrailty[['BaseHazardEntry']] <- vector('double',max(obs))
CoxRegListParallelbetanoFrailty[['cumhazEntry']] <- vector('double',max(obs))
CoxRegListParallelbetanoFrailty[['cumhaz1year']] <- vector('double',max(obs))
CoxRegListParallelbetanoFrailty[['Risk']] <- vector('double',max(obs))
CoxRegListParallelbetanoFrailty[['ModelSummary']] <- vector('double',8)

weights <- runif(length(Outcomes))
Sys.time()

coxnofrail<- survival::coxph(Surv(start, stop, event) ~ rx + sex  ,weights = rep(1,length(weights)),  data = rats.dt)


cox_reg_sparse_parallel(modeldata = CoxRegListParallelbetanoFrailty, #beta_in = CoxRegListParallelbetanoFrailty[['Beta']] ,
                                                           #frailty_in = CoxRegListParallelbetanoFrailty[['Frailty']],
                                                           #basehaz_in  = CoxRegListParallelbetanoFrailty[['basehaz']],
                                                           #cumhaz_in = CoxRegListParallelbetanoFrailty[['cumhaz']] ,
                                                           #BaseHazardEntry_in = CoxRegListParallelbetanoFrailty[['BaseHazardEntry']],
                                                           #cumhazEntry_in = CoxRegListParallelbetanoFrailty[['cumhazEntry']],
                                                           #cumhaz1year_in = CoxRegListParallelbetanoFrailty[['cumhaz1year']],
                                                           #Risk_in = CoxRegListParallelbetanoFrailty[['Risk']] ,
                                                         obs_in = obs,
                                                         coval_in = coval,
                                                         weights_in = rep(1,length(weights)),
                                                         timein_in = timein,
                                                         timeout_in = timeout,
                                                         Outcomes_in = Outcomes,
                                                         OutcomeTotals_in = OutcomesTotalUnique,
                                                         OutcomeTotalTimes_in = OutcomesTotalTimes,
                                                         covn_in = covn,
                                                         covstart_in = covstart,
                                                         covend_in = covend,
                                                         idn_in = vector('integer',0L),
                                                         idstart_in = vector('integer',0L),
                                                         idend_in = vector('integer',0L),
                                                         lambda = lambda,
                                                         theta_in = 0,
                                                         MSTEP_MAX_ITER = 10,
                                                         MAX_EPS = 1e-10,
                                                         threadn = 32)
  
names(CoxRegListParallelbetanoFrailty$Beta) <- c('rx', 'sex')
 expect_equal(c(coef(coxnofrail)), c(CoxRegListParallelbetanoFrailty$Beta), tolerance = 1e-8)


# 
# 
coxfrail<- survival::coxph(Surv(start, stop, event) ~ rx + sex + frailty(id, trace =T) ,weights = rep(1,length(weights)),  data = rats.dt,model = T)

# rats.dt[, cumhz := data.table::setDT(basehaz(coxfrail,centered = F))[rats.dt[,.(stop)], , on = c('time' = 'stop')]$hazard]
# rats.dt[,wt := ((sum(event))/(sum(cumhz))), by = id]
# rats.dt[,wt := wt -(exp((wt))/exp((wt) - 1))]
# rats.dt[,wt := wt -mean(wt)]
# coxfrail$coxlist1$first/coxfrail$coxlist1$second

#discoxfrail <- discfrail::npdf_cox( Surv( stop, event) ~rx + sex, groups=id, data=rats.dt, K = 1, eps_conv=10^-4)
#https://github.com/therneau/survival/blob/master/src/coxfit5.c

#Frailty(id)


setThreadOptions(numThreads = 32)
set.seed(235739801)
lambda =0
CoxRegListParallelbetaFrailty <- list()

CoxRegListParallelbetaFrailty[['Beta']] <- vector('double',length(covstart))
CoxRegListParallelbetaFrailty[['Frailty']] <- vector('double',length(idstart))
CoxRegListParallelbetaFrailty[['basehaz']] <- vector('double',max(timeout))
CoxRegListParallelbetaFrailty[['cumhaz']] <- vector('double',max(timeout))
CoxRegListParallelbetaFrailty[['BaseHazardEntry']] <- vector('double',max(obs))
CoxRegListParallelbetaFrailty[['cumhazEntry']] <- vector('double',max(obs))
CoxRegListParallelbetaFrailty[['cumhaz1year']] <- vector('double',max(obs))
CoxRegListParallelbetaFrailty[['Risk']] <- vector('double',max(obs))
CoxRegListParallelbetaFrailty[['ModelSummary']] <- vector('double',8)

weights <- runif(length(Outcomes))
Sys.time()
history <- list()
# i <- 1
# while( done == 0) {
cox_reg_sparse_parallel(modeldata = CoxRegListParallelbetaFrailty,
                                                          #beta_in = CoxRegListParallelbetaFrailty[['Beta']] ,
                                                          #frailty_in = CoxRegListParallelbetaFrailty[['Frailty']],
                                                           #basehaz_in  = CoxRegListParallelbetaFrailty[['basehaz']],
                                                           #cumhaz_in = CoxRegListParallelbetaFrailty[['cumhaz']] ,
                                                           #BaseHazardEntry_in = CoxRegListParallelbetaFrailty[['BaseHazardEntry']],
                                                           #cumhazEntry_in = CoxRegListParallelbetaFrailty[['cumhazEntry']],
                                                           #cumhaz1year_in = CoxRegListParallelbetaFrailty[['cumhaz1year']],
                                                           #Risk_in = CoxRegListParallelbetaFrailty[['Risk']] ,
                                                           obs_in = obs,                                                   
                                                           Outcomes_in = Outcomes,
                                                           OutcomeTotals_in = OutcomesTotalUnique,
                                                           OutcomeTotalTimes_in = OutcomesTotalTimes,
                                                           timein_in = timein,
                                                           timeout_in = timeout,
                                                           weights_in = rep(1,length(weights)),
                                                           coval_in = coval,
                                                         covn_in = covn,
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
 

expect_equal(CoxRegListParallelbetaFrailty$Frailty-CoxRegListParallelbetaFrailty$ModelSummary[8],
coxfrail$frail,tolerance = 10-9)

names(CoxRegListParallelbetaFrailty$Beta) <- c('rx', 'sex')

expect_equal(CoxRegListParallelbetaFrailty$Beta,
             coxfrail$coefficients,tolerance = 10-9)

cbind(CoxRegListParallelbetaFrailty$cumhaz[1:max(timeout) %in% basehaz(coxfrail)$time], basehaz(coxfrail, centered = F)$hazard)
plot(CoxRegListParallelbetaFrailty$cumhaz[1:max(timeout) %in% basehaz(coxfrail)$time], basehaz(coxfrail, centered = F)$hazard)


cumsum(OutcomesTotalTimes/
         cumsum(
           vapply(1:length(OutcomesTotalTimes), 
                  FUN.VALUE = 1.0, 
                  function(i) sum(CoxRegListParallelbetaFrailty$Risk[timeout>OutcomesTotalTimes[i] & timein<=OutcomesTotalTimes[i]]))))

  # correction <- attr(survival::Frailty.gamma(CoxRegListParallelbetaFrailty$Frailty),'pfun')(CoxRegListParallelbetaFrailty$Frailty, 1, OutcomesTotals)
   # 
   # history[[i]] <- as.vector(theta , CoxRegListParallelbetaFrailty$loglik + correction) 
   # attr(survival::Frailty.gamma(CoxRegListParallelbetaFrailty$Frailty),'pfun')()
   # i <- i + 1
  
#}
#attr(Frailty.gamma(CoxRegListParallelbetaFrailty$Frailty),'pfun')(CoxRegListParallelbetaFrailty$Frailty, 1, OutcomesTotals) 

rbind(coef(coxnofrail),
      CoxRegListParallelbetanoFrailty$Beta,
      coef(coxfrail),
CoxRegListParallelbetaFrailty$Beta
)

par(mfrow = c(2,2))
hist(coxfrail$frail)
hist(CoxRegListParallelbetaFrailty$Frailty)
hist(CoxRegListParallelbetaFrailty$Frailty - log(mean(exp(CoxRegListParallelbetaFrailty$Frailty))))

rxloglik <- function(beta,max,rats.dt) {
  temp <- survival::coxph(Surv(start, stop, event) ~ offset(beta*rx) + sex + frailty(id, trace =F) ,weights = rep(1,length(weights)),  data = rats.dt)
  loglikratio <- 2*(max - temp$loglik[2])
  list("-2LLR" = loglikratio)
}

#emplik::findUL(fun = rxloglik, MLE = coxfrail$coefficients[1], max = coxfrail$loglik[2], rats.dt = rats.dt)


set.seed(235739801)
profileCI <- profile_ci(beta_in = CoxRegListParallelbetaFrailty$Beta,
                        obs_in = obs,
                        coval_in = coval,
                         weights_in =  rep(1,length(weights)),
                         frailty_in = CoxRegListParallelbetaFrailty$Frailty,
                         timein_in = timein ,
                         timeout_in = timeout ,
                         Outcomes_in = Outcomes ,
                         OutcomeTotals_in = OutcomesTotalUnique ,
                         OutcomeTotalTimes_in = OutcomesTotalTimes,
                        covn_in = covn,
                        covstart_in = covstart,
                        covend_in = covend,
                        idn_in = idn,
                        idstart_in = idstart,
                        idend_in = idend,
                         lambda = lambda,
                         theta_in = CoxRegListParallelbetaFrailty[['ModelSummary']][4],
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

# rats.dt[, cumhz := data.table::setDT(basehaz(coxfrail,centered = F))[rats.dt[,.(stop)], , on = c('time' = 'stop')]$hazard]
# rats.dt[,wt := ((sum(event))/(sum(cumhz))), by = id]
# rats.dt[,wt := wt -(exp((wt))/exp((wt) - 1))]
# rats.dt[,wt := wt -mean(wt)]
# coxfrail$coxlist1$first/coxfrail$coxlist1$second

#discoxfrail <- discfrail::npdf_cox( Surv( stop, event) ~rx + sex, groups=id, data=rats.dt, K = 1, eps_conv=10^-4)
#https://github.com/therneau/survival/blob/master/src/coxfit5.c

#Frailty(id)


set.seed(235739801)

CoxRegListParallelbetaFrailtypenal <- list()

CoxRegListParallelbetaFrailtypenal[['Beta']] <- vector('double',length(covstart))
CoxRegListParallelbetaFrailtypenal[['Frailty']] <- vector('double',length(idstart))
CoxRegListParallelbetaFrailtypenal[['basehaz']] <- vector('double',max(timeout))
CoxRegListParallelbetaFrailtypenal[['cumhaz']] <- vector('double',max(timeout))
CoxRegListParallelbetaFrailtypenal[['BaseHazardEntry']] <- vector('double',max(obs))
CoxRegListParallelbetaFrailtypenal[['cumhazEntry']] <- vector('double',max(obs))
CoxRegListParallelbetaFrailtypenal[['cumhaz1year']] <- vector('double',max(obs))
CoxRegListParallelbetaFrailtypenal[['Risk']] <- vector('double',max(obs))
CoxRegListParallelbetaFrailtypenal[['ModelSummary']] <- vector('double',8)

weights <- runif(length(Outcomes))
Sys.time()
history <- list()
# i <- 1
# while( done == 0) {

# using vector, parallel for and private vectors to reduce
# user  system elapsed 
# 0.81    0.15    0.23 

system.time(cox_reg_sparse_parallel(modeldata = CoxRegListParallelbetaFrailtypenal,
                        #beta_in = CoxRegListParallelbetaFrailty[['Beta']] ,
                        #frailty_in = CoxRegListParallelbetaFrailty[['Frailty']],
                        #basehaz_in  = CoxRegListParallelbetaFrailty[['basehaz']],
                        #cumhaz_in = CoxRegListParallelbetaFrailty[['cumhaz']] ,
                        #BaseHazardEntry_in = CoxRegListParallelbetaFrailty[['BaseHazardEntry']],
                        #cumhazEntry_in = CoxRegListParallelbetaFrailty[['cumhazEntry']],
                        #cumhaz1year_in = CoxRegListParallelbetaFrailty[['cumhaz1year']],
                        #Risk_in = CoxRegListParallelbetaFrailty[['Risk']] ,
                        obs_in = obs,                                                   
                        Outcomes_in = Outcomes,
                        OutcomeTotals_in = OutcomesTotalUnique,
                        OutcomeTotalTimes_in = OutcomesTotalTimes,
                        timein_in = timein,
                        timeout_in = timeout,
                        weights_in = rep(1,length(weights)),
                        coval_in = coval,
                        covn_in = covn,
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
expect_equal(CoxRegListParallelbetaFrailtypenal$Frailty-CoxRegListParallelbetaFrailtypenal$ModelSummary[8],
             coxfrailpenal$frail,tolerance = 10-9)
