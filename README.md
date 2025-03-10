<a id="readme-top"></a>

<br />
<div align="center">
  <a href="https://github.com/colincrooks/coxsparse">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Implementation of a Cox proportional hazards model using a sparse data structure</h3>

  <p align="center">
    The purpose of this implementation is for fitting a Cox model to data when coxph from the survival package fails due to not enough memory to hold the model and data matrices. 
    The focus is therefore on being memory efficient, which is a slower algorithm than in coxph, but parallelisation is possible to offset this. 
    In this situation compiling the code for the native computer setup would be preferable to providing a standard package binary for multiple systems. 
    The Makevars file therefore contains the options for this.  The data structure is a deconstructed sparse matrix.  
    A function using the same data structure to calculate profile confidence intervals with a crude search pattern is provided.
    <br />
    <a href="https://github.com/colincrooks/coxsparse"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/colincrooks/coxsparse">View Demo</a>
    &middot;
    <a href="https://github.com/colincrooks/coxsparse/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/colincrooks/coxsparse/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Package

The purpose of this implementation is for fitting a Cox model to data when coxph from the survival package fails due to not enough memory to hold the model and data matrices. 
The focus is therefore on being memory efficient, which is a slower algorithm than in coxph, but parallelisation is possible to offset this. 
In this situation compiling the code for the native computer setup would be preferable to providing a standard package binary for multiple systems. 
The Makevars file therefore contains the options for this.

The data structure is a deconstructed sparse matrix.

A function using the same data structure to calculate profile confidence intervals with a crude search pattern is provided.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This package depends on rcpp, rcppParallel and OpenMP and is intended to be compiled with rtools installed. 
It has a Makevars file in /src/ which contains optimisation flags to compile and tune natively to local system, these can be edited if not suitable.

### Prerequisites

In R can use:
  ```r
  install.packages('installr','Rcpp','RcppParallel')
  library(installr)
  installr::install.Rtools()
  
  ```

### Installation

In R can use:
```r
devtools::install_git(url = "https://github.com/ColinCrooks/coxsparse", ref = 'master')
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

For example see comparison with coxph from the survival package in the tests folder [CompareSurvival](https://github.com/ColinCrooks/coxsparse/blob/master/tests/testthat/test-CompareSurvival.R)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the GPLv3. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Colin Crooks - email@colin.crooks@nottingham.ac.uk.com

Project Link: [https://github.com/colincrooks/coxsparse](https://github.com/colincrooks/coxsparse)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
Mittal, S., Madigan, D., Burd, R. S., & Suchard, M. a. (2013). High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis. Biostatistics, 1–15. doi:10.1093/biostatistics/kxt043
* [Mittal et al](https://academic.oup.com/biostatistics/article/15/2/207/226074)
Therneau T (2024). _A Package for Survival Analysis in R_. R package version 3.8-3,
*[Therneau (2024)](https://CRAN.R-project.org/package=survival).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
