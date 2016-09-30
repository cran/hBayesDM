#' Norm-Training Ultimatum Game
#' 
#' @description 
#' Hierarchical Bayesian Modeling of the Norm-Training Ultimatum Game using the following parameters: "alpha" (envy), "ep" (norm adaptation rate), "tau" (inverse temperature).
#' 
#' \strong{MODEL:}
#' Rescorla-Wagner (delta) Model (Gu et al., 2015, J Neuro)
#' 
#' @param data A .txt file containing the data to be modeled. Data columns should be labelled as follows: "subjID", "offer", and "accept". See \bold{Details} below for more information. 
#' @param niter Number of iterations, including warm-up.
#' @param nwarmup Number of iterations used for warm-up only.
#' @param nchain Number of chains to be run.
#' @param ncore Integer value specifying how many CPUs to run the MCMC sampling on. Defaults to 1. 
#' @param nthin Every \code{i == nthin} sample will be used to generate the posterior distribution. Defaults to 1. A higher number can be used when auto-correlation within the MCMC sampling is high. 
#' @param inits Character value specifying how the initial values should be generated. Options are "fixed" or "random" or your own initial values.
#' @param indPars Character value specifying how to summarize individual parameters. Current options are: "mean", "median", or "mode".
#' @param saveDir Path to directory where .RData file of model output (\code{modelData}) can be saved. Leave blank if not interested.
#' @param email Character value containing email address to send notification of completion. Leave blank if not interested. 
#' @param modelRegressor Exporting model-based regressors? TRUE or FALSE. Currently not available for this model.
#' @param adapt_delta Floating point number representing the target acceptance probability of a new sample in the MCMC chain. Must be between 0 and 1. See \bold{Details} below.
#' @param stepsize Integer value specifying the size of each leapfrog step that the MCMC sampler can take on each new iteration. See \bold{Details} below.
#' @param max_treedepth Integer value specifying how many leapfrog steps that the MCMC sampler can take on each new iteration. See \bold{Details} below. 
#'  
#' @return \code{modelData}  A class \code{"hBayesDM"} object with the following components:
#' \describe{
#'  \item{\code{model}}{Character string with the name of the model (\code{"ug_delta"}).}
#'  \item{\code{allIndPars}}{\code{"data.frame"} containing the summarized parameter 
#'    values (as specified by \code{"indPars"}) for each subject.}
#'  \item{\code{parVals}}{A \code{"list"} where each element contains posterior samples
#'    over different model parameters. }
#'  \item{\code{fit}}{A class \code{"stanfit"} object containing the fitted model.}
#'  \item{\code{rawdata}}{\code{"data.frame"} containing the raw data used to fit the model, as specified by the user.}
#' }
#' 
#' @importFrom rstan stan rstan_options extract
#' @importFrom modeest mlv
#' @importFrom mail sendmail
#' @importFrom stats median qnorm
#' @importFrom utils read.table
#'
#' @details 
#' This section describes some of the function arguments in greater detail.
#' 
#' \strong{data} should be assigned a character value specifying the full path and name of the file, including the file extension 
#' (e.g. ".txt"), that contains the behavioral data of all subjects of interest for the current analysis. 
#' The file should be a text (.txt) file whose rows represent trial-by-trial observations and columns 
#' represent variables. For the Norm-Training Ultimatum Game, there should be three columns of data 
#' with the labels "subjID", "offer", and "accept". It is not necessary for the columns to be in this particular order, 
#' however it is necessary that they be labelled correctly and contain the information below:
#' \describe{
#'  \item{\code{"subjID"}}{A unique identifier for each subject within data-set to be analyzed.}
#'  \item{\code{"offer"}}{An real value representing the offer made within the given trial (e.g., 10, 11, 4, etc..).}
#'  \item{\code{"accept"}}{A 1 or 0 indicating an offer was accepted or not (1 = accepted, 0 = rejected).}
#' } 
#' \strong{*}Note: The data.txt file may contain other columns of data (e.g. "Reaction_Time", "trial", etc.), but only the data with the column
#' names listed above will be used for analysis/modeling. As long as the columns above are present and labelled correctly,
#' there is no need to remove other miscellaneous data columns.    
#'  
#' \strong{nwarmup} is a numerical value that specifies how many MCMC samples should not be stored upon the 
#' beginning of each chain. For those familiar with Bayesian methods, this value is equivalent to a burn-in sample. 
#' Due to the nature of MCMC sampling, initial values (where the sampling chain begins) can have a heavy influence 
#' on the generated posterior distributions. The \code{nwarmup} argument can be set to a high number in order to curb the 
#' effects that initial values have on the resulting posteriors.  
#' 
#' \strong{nchain} is a numerical value that specifies how many chains (i.e. independent sampling sequences) should be
#' used to draw samples from the posterior distribution. Since the posteriors are generated from a sampling 
#' process, it is good practice to run multiple chains to ensure that a representative posterior is attained. When
#' sampling is completed, the multiple chains may be checked for convergence with the \code{plot(myModel, type = "trace")}
#' command. The chains should resemble a "furry caterpillar".
#' 
#' \strong{nthin} is a numerical value that specifies the "skipping" behavior of the MCMC samples being chosen 
#' to generate the posterior distributions. By default, \code{nthin} is equal to 1, hence every sample is used to 
#' generate the posterior. 
#' 
#' \strong{Contol Parameters:} adapt_delta, stepsize, and max_treedepth are advanced options that give the user more control 
#' over Stan's MCMC sampler. The Stan creators recommend that only advanced users change the default values, as alterations
#' can profoundly change the sampler's behavior. Refer to Hoffman & Gelman (2014, Journal of Machine Learning Research) for 
#' more information on the functioning of the sampler control parameters. One can also refer to section 58.2 of the  
#' \href{http://mc-stan.org/documentation/}{Stan User's Manual} for a less technical description of these arguments. 
#' 
#' @export 
#' 
#' @references 
#' Gu, X., Wang, X., Hula, A., Wang, S., Xu, S., Lohrenz, T. M., et al. (2015). Necessary, Yet Dissociable Contributions of the 
#' Insular and Ventromedial Prefrontal Cortices to Norm Adaptation: Computational and Lesion Evidence in Humans. Journal of 
#' Neuroscience, 35(2), 467-473. http://doi.org/10.1523/JNEUROSCI.2906-14.2015
#' 
#' Hoffman, M. D., & Gelman, A. (2014). The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. The 
#' Journal of Machine Learning Research, 15(1), 1593-1623.
#'
#' @seealso 
#' We refer users to our in-depth tutorial for an example of using hBayesDM: \url{https://rpubs.com/CCSL/hBayesDM}
#' 
#' @examples 
#' \dontrun{
#' # Run the model and store results in "output"
#' output <- ul_m1("example", 2000, 1000, 3, 3)
#' 
#' # Plot the posterior distributions of the hyper-parameters
#' plot(output)
#' 
#' # Show the WAIC and LOOIC model fit estimates 
#' printFit(output)
#' }

ug_delta <- function(data          = "choose",
                     niter         = 3000, 
                     nwarmup       = 1000, 
                     nchain        = 1,
                     ncore         = 1, 
                     nthin         = 1,
                     inits         = "random",  
                     indPars       = "mean", 
                     saveDir       = NULL,
                     email         = NULL,
                     modelRegressor= FALSE,
                     adapt_delta   = 0.8,
                     stepsize      = 1,
                     max_treedepth = 10 ) {

  # Path to .stan model file
  if (modelRegressor) { # model regressors (for model-based neuroimaging, etc.)
    stop("** Model-based regressors are not available for this model **\n")
  } else {
    modelPath <- system.file("stan", "ug_delta.stan", package="hBayesDM")
  }
  
  # To see how long computations take
  startTime <- Sys.time()    

  # For using example data
  if (data=="example") {
    data <- system.file("extdata", "ug_exampleData.txt", package = "hBayesDM")
  } else if (data=="choose") {
    data <- file.choose()
  }
  
  # Load data
  if (file.exists(data)) {
    rawdata <- read.table( data, header = T )
  } else {
    stop("** The data file does not exist. Please check it again. **\n  e.g., data = '/MyFolder/SubFolder/dataFile.txt', ... **\n")
  }  
  
  # Individual Subjects
  subjList <- unique(rawdata[,"subjID"])  # list of subjects x blocks
  numSubjs <- length(subjList)  # number of subjects
  
  # Specify the number of parameters and parameters of interest 
  numPars <- 3
  POI     <- c("mu_ep", "mu_tau", "mu_alpha",
               "sigma", 
               "ep", "tau", "alpha", 
               "log_lik") 
  
  modelName <- "ug_delta"
  
  # Information for user
  cat("\nModel name = ", modelName, "\n")
  cat("Data file  = ", data, "\n")
  cat("\nDetails:\n")
  cat(" # of chains                   = ", nchain, "\n")
  cat(" # of cores used               = ", ncore, "\n")
  cat(" # of MCMC samples (per chain) = ", niter, "\n")
  cat(" # of burn-in samples          = ", nwarmup, "\n")
  cat(" # of subjects                 = ", numSubjs, "\n")
  
  ################################################################################
  # THE DATA.  ###################################################################
  ################################################################################
  
  Tsubj <- as.vector( rep( 0, numSubjs ) ) # number of trials for each subject
  
  for ( i in 1:numSubjs )  {
    curSubj  <- subjList[ i ]
    Tsubj[i] <- sum( rawdata$subjID == curSubj )  # Tsubj[N]
  }
  
  maxTrials <- max(Tsubj)
  
  # Information for user continued
  cat(" # of (max) trials per subject = ", maxTrials, "\n\n")

  offer  <- array(0, c(numSubjs, maxTrials) )
  accept <- array(0, c(numSubjs, maxTrials) )
  
  for (i in 1:numSubjs) {
    curSubj      <- subjList[i]
    useTrials    <- Tsubj[i]
    tmp          <- subset(rawdata, rawdata$subjID == curSubj)
    offer[i,  1:useTrials] <- tmp$offer
    accept[i, 1:useTrials] <- tmp$accept
  }
  
  dataList <- list(
    N       = numSubjs,
    T       = maxTrials,
    Tsubj   = Tsubj,
    offer   = offer, 
    accept  = accept
  )
  
  # inits
  if (inits[1] != "random") {
    if (inits[1] == "fixed") {
      inits_fixed <- c(1.0, 0.5, 1.0)
    } else {
      if (length(inits)==numPars) {
        inits_fixed <- inits
      } else {
        stop("Check your inital values!")
      }
    }  
    genInitList <- function() {
      list(
        mu_p     = c( qnorm(inits_fixed[1] ), qnorm( inits_fixed[2]/10), qnorm( inits_fixed[3]/20) ),
        sigma    = c( 1.0, 1.0, 1.0),
        ep_pr    = rep( qnorm(inits_fixed[1]), numSubjs),
        tau_pr   = rep( qnorm(inits_fixed[2]/10), numSubjs),
        alpha_pr = rep( qnorm(inits_fixed[3]/20), numSubjs)
      )
    }
  } else {
    genInitList <- "random"
  }
    
  rstan::rstan_options(auto_write = TRUE)
  if (ncore > 1) {
    numCores <- parallel::detectCores()
    if (numCores < ncore){
      options(mc.cores = numCores)
      warning('Number of cores specified for parallel computing greater than number of locally available cores. Using all locally available cores.')
    }
    else{
      options(mc.cores = ncore)
    }
  }
  else {
    options(mc.cores = 1)
  }
  
  cat("************************************\n")
  cat("** Building a model. Please wait. **\n")
  cat("************************************\n")
  
  # Fit the Stan model
  fit <- rstan::stan(file   = modelPath,
                     data   = dataList, 
                     pars   = POI,
                     warmup = nwarmup,
                     init   = genInitList, 
                     iter   = niter, 
                     chains = nchain,
                     thin   = nthin,
                     control = list(adapt_delta   = adapt_delta, 
                                    max_treedepth = max_treedepth, 
                                    stepsize      = stepsize) )
  
  ## Extract parameters
  parVals <- rstan::extract(fit, permuted=T)
  
  ep    <- parVals$ep
  tau   <- parVals$tau
  alpha <- parVals$alpha
  
  # Individual parameters (e.g., individual posterior means)
  allIndPars <- array(NA, c(numSubjs, numPars))
  allIndPars <- as.data.frame(allIndPars)
  
  for (i in 1:numSubjs) {
    if (indPars=="mean") {
      allIndPars[i, ] <- c( mean(ep[, i]), 
                            mean(tau[, i]), 
                            mean(alpha[, i]) )
    } else if (indPars=="median") {
      allIndPars[i, ] <- c( median(ep[, i]), 
                            median(tau[, i]), 
                            median(alpha[, i]) )
    } else if (indPars=="mode") {
      allIndPars[i, ] <- c( as.numeric(modeest::mlv(ep[, i], method="shorth")[1]),
                            as.numeric(modeest::mlv(tau[, i], method="shorth")[1]),
                            as.numeric(modeest::mlv(alpha[, i], method="shorth")[1]) )
    }
  }
  
  allIndPars           <- cbind(allIndPars, subjList)
  colnames(allIndPars) <- c("ep", 
                            "tau", 
                            "alpha", 
                            "subjID")

  # Wrap up data into a list
  modelData        <- list(modelName, allIndPars, parVals, fit, rawdata)
  names(modelData) <- c("model", "allIndPars", "parVals", "fit", "rawdata")
  class(modelData) <- "hBayesDM"

  # Total time of computations
  endTime  <- Sys.time()
  timeTook <- endTime - startTime
  
  # If saveDir is specified, save modelData as a file. If not, don't save
  # Save each file with its model name and time stamp (date & time (hr & min))
  if (!is.null(saveDir)) {  
    currTime  <- Sys.time()
    currDate  <- Sys.Date()
    currHr    <- substr(currTime, 12, 13)
    currMin   <- substr(currTime, 15, 16)
    timeStamp <- paste0(currDate, "_", currHr, "_", currMin)
    save(modelData, file=file.path(saveDir, paste0(modelName, "_", timeStamp, ".RData"  ) ) )
  }
  
  # Send email to notify user of completion
  if (is.null(email)==F) {
    mail::sendmail(email, paste("model=", modelName, ", fileName = ", data),
             paste("Check ", getwd(), ". It took ", as.character.Date(timeTook), sep="") )
  }
  # Inform user of completion
  cat("\n************************************\n")
  cat("**** Model fitting is complete! ****\n")
  cat("************************************\n")
  
  return(modelData)
}