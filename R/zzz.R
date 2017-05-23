#' @noRd
 
.onAttach <- function(libname, pkgname) {
  ver <- utils::packageVersion("hBayesDM")
  packageStartupMessage("\n\nThis is hBayesDM version ", ver, "\n",
                        "*********************************************************************\n",
                        "If you are a Mac/Linux user, We strongly recommend users\n",
                        "install hBayesDM from GitHub (https://github.com/CCS-Lab/hBayesDM).\n",
                        "The GitHub version is identical to the CRAN version, except that \n",
                        "all models are precompiled in the GitHub version, which saves time\n", 
                        "for compiling Stan models\n",
                        "*********************************************************************",
                        "\n\n")
}
