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
heart.dt <- heart.dt[,id := NULL]

heart.dt <- data.table::rbindlist(lapply(1:1000,function(i) heart.dt))
heart.dt[,id := .I]


for (c in covlist) {
  heart.dt[,(c) := as.numeric(as.character(get(c)))]

}

for (c in  c('age','year','surgery','transplant')) {
  heart.dt[,(c) := scale(get(c))]

}

idtimevar <- c('stop', 'start', 'id','event')
data.table::setkeyv(heart.dt, idvars)


heart.tv.build <-  data.table::melt(heart.dt[,mget(reshapevars)],
                       id.vars = idvars,
                       measure.vars = covlist,
                       variable.name = 'covariate',
                       value.name = 'value')


data.table::setkeyv(heart.tv.build, idvars)
heart.tv.build[,idx := seq_len(.N), by = .(id,start)]

heart.tv.build[, keep := cummax(value!=0), by = .(id,start)]
heart.tv.build[, keep := keep[.N], by = .(id,start)]
heart.tv.build[, keep := value != 0 | (keep == F & idx == 1)]

heart.tv.build <- heart.tv.build[(keep),]

heart.tv.build[, `:=`(covlength = .N,
                     colid = seq_len(.N)),
              by = .(id,start)]

heart.tv.build <- data.table::dcast(heart.tv.build,
                       id+start+stop+event+covlength ~ colid,
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
heart.tv.build[,rowid := .I]
OutcomesTotals <- as.integer(heart.tv.build[['OutcomeTotals']])
Outcomes <- as.integer(heart.tv.build[['event']])
OutcomesTotalUnique <- as.integer(heart.tv.build[OutcomeTotals > 0,OutcomeTotals])
OutcomesTotalTimes <- as.integer(heart.tv.build[OutcomeTotals > 0,stop])

timein <- as.integer(heart.tv.build[['start']])
timeout <- as.integer(heart.tv.build[['stop']])
Covlength <- as.integer(heart.tv.build[['covlength']])

nvar <- length(grep("idx_",names(heart.tv.build)))

cov <- data.table::melt(heart.tv.build[,c('rowid',paste0('idx_',1:nvar)), with = F],id.vars = 'rowid', na.rm = T)
data.table::setkey(cov, rowid, variable)
id <- as.integer(cov[['rowid']])
cov <- as.integer(cov[['value']])
coval <- data.table::melt(heart.tv.build[,c('rowid',paste0('value_',1:nvar)), with = F],id.vars = 'rowid', na.rm = T)
data.table::setkey(coval, rowid, variable)
all.equal(id,coval[['rowid']])
coval <- as.double(coval[['value']])
covrowlist <- lapply(1:nvar, function(i) which(cov == i))

nvar <- max(cov, na.rm = T)
weights <- runif(length(Outcomes))


Sys.time()
coxunreg <- survival::coxph(Surv(start, stop, event) ~ age + year + surgery + transplant,weights = rep(1,length(weights)), id = id, data = heart.dt)
coxunreg_bh <- survival::basehaz(coxunreg)
Sys.time() #20s
coef(coxunreg)


setThreadOptions(numThreads = 32)
set.seed(235739801)
lambda = 0.00
beta <- vector('double',nvar)
beta <- rep(0,nvar)
weights <- runif(length(Outcomes))
Sys.time()
CoxRegListParallelbeta2 <- cox_reg_sparse_parallel(beta_in = beta,
                                                    id_in = id,
                                                    Outcomes_in = Outcomes,
                                                    OutcomeTotals_in = OutcomesTotalUnique,
                                                    OutcomeTotalTimes_in = OutcomesTotalTimes,
                                                    timein_in = timein,
                                                    timeout_in = timeout,
                                                    weights_in = rep(1,length(weights)),
                                                    coval_in = coval,
                                                    covrowlist_in = covrowlist,
                                                    nvar = nvar,
                                                    lambda = lambda,
                                                    MSTEP_MAX_ITER = 1000,
                                                    MAX_EPS = 1e-10,
                                                    threadn = 32)
Sys.time() # 17 s
expect_setequal(round(as.numeric(coef(coxunreg)),9), round(CoxRegListParallelbeta2$Beta,9))

rbind(coef(coxunreg), CoxRegListParallelbeta2$Beta)

cbind(coxunreg_bh$hazard[coxunreg_bh$time %in% OutcomesTotalTimes],
                  cumsum(CoxRegListParallelbeta2$BaseHaz[OutcomesTotalTimes]))

expect_setequal(round(coxunreg_bh$hazard[coxunreg_bh$time %in% OutcomesTotalTimes],9),
      round(cumsum(CoxRegListParallelbeta2$BaseHaz[OutcomesTotalTimes]),9))

set.seed(235739801)
profileCI <- profile_ci(covrowlist_in = covrowlist,
                         beta_in = CoxRegListParallelbeta2$Beta,
                         id_in = id,
                         coval_in = coval,
                         weights_in =  rep(1,length(weights)),
                         timein_in = timein ,
                         timeout_in = timeout ,
                         Outcomes_in = Outcomes ,
                         OutcomeTotals_in = OutcomesTotalUnique ,
                         OutcomeTotalTimes_in = OutcomesTotalTimes,
                         nvar = nvar,
                         lambda = lambda,
                         MSTEP_MAX_ITER = 100000,
                         decimals = 10,
                         confint_width = 0.95,
                         threadn = 32)
cbind(CoxRegListParallelbeta2$Beta,profileCI )
expect_true(max(c(confint(coxunreg,type='profile')) - c(profileCI)) < 0.01)
