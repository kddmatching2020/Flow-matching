## Propensity Score based Matching via Distribution Learning


There has been tremendous research interests in methods for treatment effect estimation. Randomized field experiments have been considered as an efficient and robust tool for this purpose. However, it is costly and the randomization in some situations is not possible. Finding alternatives with observational data for treatment effect estimation becomes a necessity, including structural modeling, instrumental variables, and machine learning based approaches. Among which propensity score matching has been widely used and theoretically proven to obtain unbiased estimations given the correct model assumption, which usually does not hold because the distribution of data is complicated in practice. To deal with the issue of model mis-specification, in this study we propose a distribution learning framework for propensity score prediction and subsequent treatment effect estimation. Specifically, we implement a normalizing flow by taking a sequence of invertible transformations starting from a simple Gaussian to approximate the true distribution of covariates conditioned on the treatment assignment. Then the propensity score can be obtained via the Bayesian theory. 

To evaluate our algorithm and compare with existing benchmarks, we conduct extensive experiments using synthetic and real data. The results demonstrate that our flow-based estimator can achieve up to approximately 5 times better performance of the ATT estimation over state-of-the-art matching approaches. It also shows the capability of alleviating model mis-specification issue when using propensity score matching with observational data. Additionally, we empirically find that our continuous flows can approximate true distributions with a fast convergence rate. 



### Note that the number of transformations in the flow model (K) is 4, for the purpose of simplicity.
