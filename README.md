# 590PR Final_Project
Fork from here to create your final project repository.

Two things are different than all the previous assignments in 590PR regarding the permissions settings:

1. Please KEEP the "All_Students" team to have Read access.  
2. Whenever you choose to, you are welcome to change your Final Project repository to private or to public.  This will enable you to list it in your resume, website, or other portfolio.

DELETE these lines from TEMPLATE up.

TEMPLATE for your report to fill out:

# Title: Distributional properties of Rasch Standardized Outfit Statistics

## Team Member(s):
Rajat Chadha

# Monte Carlo Simulation Scenario & Purpose:
Rasch Model (Rasch, 1960; 1981) is a probabilistic psychometric item response model used to create interval-level measures from categorical data, such as persons’ answers to items on an exam. Rasch Model is used in education, health professions, and market research.

In measurement, our intent is to use numbers to indicate "more" or "less" of the trait that is presumed to be homogeneous. Using the Rasch Model we can investigate if the data conforms to this homogeneity. Rasch modelling is a unique approach of mathematical modeling based upon a latent trait and accomplishes stochastic (probabilistic) conjoint additivity (conjoint means measurement of persons and items on the same scale and additivity is the equal-interval property of the scale) (Granger, 2008). For more information please visit: https://www.rasch.org/rmt/rmt213d.htm.

The probability of a correct answer by a person on an item is modelled as a function of the difference in the person and item locations on measurement scale.

An important aspect of analyzing data using the Rasch Model is to assess the fit of persons’ responses to individual items to the measurement model. Various types of measurement disturbances can manifest as misfit of the data to the model expected. One statistic that is widely used to assess the fit is the standardized outfit statistic.

This project aims to develop a Python program to assess the distributional properties of the standardized outfit statistics. Studies (Smith, 1991; Smith, Schumacker, & Bush, 1998) have been conducted in the past to study this. 

## Simulation's variables of uncertainty
List and describe your simulation's variables of uncertainty (where you're using pseudo-random number generation). For each such variable, how did you decide the range and probability distribution to use?  Do you think it's a good representation of reality?

Person measures: Person measures were pseudo-randomly selected from a normally-distributed population with mean = 0 logits
and standard deviation = 1 logit. Normal distribution was selected because person measures are usually normally distributed. Moreover, this program can be easily modified for a uniform distribution of person measures.

Item locations: Pseudo-random sample of item locations from a uniformly-distributed population ranging from -2.0 to 2.0 logits. Uniform distribution was selected because it is desirable to have a set of items on an exam that follow uniform distribution to adequately measure most of the persons with similar precision. 

Replication sample: A pseudo-random sample numpy array with the size [n_persons, n_items].
n_persons*n_items samples are drawn from a uniformly distributed population in range 0.0001 to <1.0.
These constraints are because theoretically the probability of a correct response is never equal to 0 or 1. 

## Hypothesis or hypotheses before running the simulation:
With large number of replications, the summary statistics for the standardized outfit statistics are expected to converge to following values: 
95th percentile: 1.645;
5th percentile: -1.645;
Probability of observing a value of 2 or smaller: 97.725;
Probability of observing a value of -2 or smaller: 2.275;
Mean = 0;
SD = 1

## Analytical Summary of your findings: (e.g. Did you adjust the scenario based on previous simulation outcomes?  What are the management decisions one could make from your simulation's output, etc.)
The standardized outfit statistics seem to converge to the hypothesized values. I ran multiple replications and am attaching results from two simulations: one with 45 items, 500 persons, and 10000 replications; and second with 50 items, 1500 persons, and 100000 replications. For my research, I will also review how the distribution of the standardized outfit statistic changes with different item locations. This is beyond the scope of this particular project. 

## Instructions on how to use the program:
Run the program and input number of items (suggested between 10 and 50), number of persons (suggested between 500 to 2000), number of replications (suggestion 100 to 100000). The program generates: 1) a csv file with item locations, 2) a csv file with person locations, 3) a csv files with summary statistics, and 4) histograms with one plot saved in the directory for each item.

## All Sources Used:
Rash, G. (1960). Probabilistic models for some intelligence and attainment tests. Copenhagen: Danish Institute for Educational Research.
Smith, R. M. (1991). The distributional properties of Rasch item fit statistics. Educational and psychological measurement, 51(3), 541-565.
