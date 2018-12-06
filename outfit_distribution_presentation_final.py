import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def generate_prob_table(n_items, n_persons):
    """
    This function generates the expected probability table for each person's response to each item.
    Uses a random sample of student locations from a normally-distributed population with mean = 0 logits
    and standard deviation = 1 logit.
    Uses a random sample of item locations from a uniformly-distributed population
    ranging from -2.0 to 2.0 logits.
    :param items: number of items
    :param persons: number of persons
    :return: numpy array with the probability table
    doctests - min, max, size
    """
    mu, sigma = 0, 1  # mean and standard deviation
    persons = np.random.normal(mu, sigma, n_persons)
    items = np.random.uniform(-2.0, 2.01, n_items)

    print("Items:", np.around(items, decimals=2))
    print("Persons:", np.around(persons, decimals=2))
    for i in range(n_persons):
        numerator = np.exp(-1*(items - persons[i]))
        denomominator = 1 + numerator
        prob_temp = numerator / denomominator
        if i == 0:
            prob_table = prob_temp
        else:
            prob_table = np.vstack((prob_table, prob_temp))
    # print(np.around(prob_table, decimals=2))
    return prob_table

def replication_sample (n_items, n_persons):
    """
    This function generates and returns a pseudo-random sample numpy array with the size [n_persons, n_items].
    n_persons*n_items samples are drawn from a uniformly distributed population in range 0.001 to 0.99999.
    EXPLAIN WHY THIS RANGE

    :param n_items: number of items
    :param n_persons: number of persons
    :return: sample numpy array
    doctests - max, min and size of the array

    """
    sample = np.random.uniform(low=0.0001, high=1.0, size=(n_items,n_persons))
    # print("Sample",sample.shape,"min=",sample.min(),"max=",sample.max(),"\n",np.around(sample, decimals=2))
    # print("Sample to generate response matrix:\n",np.around(sample,decimals=2))
    return sample

def generate_response_matrix(prob_table, sample):
    """
    This function generates a response matrix with 0s and 1s for each item and person interaction.
    In takes in the probability matrix numpy array with probability of correct response for each person and
    the sample numpy array. It then compares the corresponding values in the two arrays.
    If the value of the sample is lower than the corresponding value in the probability matrix, then the response
    value is set to 1. Else, it is set to 0 (only 0 and 1 are possible scores for a person on any item).
    The logic is that the probability matrix defines the probability of correct response by a person to an item
    which is theoretically greater than 0 and less than 1.
    The sample numpy array was created with a uniform distribution with a minimum 0.00001 and a maximum of
    less than 1. Therefore, when we compare the corresponding sample and probability matrix values and assign a
    1 for sample values lower than the corresponding probability matrix value and assign a 0 for equal to or greater
    than the corresponding probability matrix value,
    over an infinite number of random samples the number of times the sample value is lower than the corresponding
    probability matrix value is likely to converge with the probability of correct response.


    :param prob_table: probability matrix numpy array with probability of correct response for each person
    and item interaction
    :param sample: sample numpy array
    :return: response matrix
    """
    response_table = prob_table - sample
    response_table[(response_table > 0)] = 1
    response_table[(response_table <= 0)] = 0
    return response_table

def check_extreme_scores(resp_matrix_sample):
    """
    This function checks if any item has an extreme score (all 0 or all 1).
    It is important to exclude items with extreme scores because the location estimates
    necessary to calculate the probability of a correct response for these score groups is infinite.
    :param resp_matrix_sample: response matrix of shape number of items x number of persons.
    :return: extreme_score_flag: Flag variable with value of 1 if there is any extreme item
    or person score in the sample, else 0.
    doctest - supply with perfect score and return 0
    """
    extreme_score_flag = 0
    for i in range(np.size(resp_matrix_sample, 1)):
        if resp_matrix_sample[:, i].max() == 0 or resp_matrix_sample[:, i].min() == 1:
            extreme_score_flag = 1
            break
    return extreme_score_flag

def calculate_outfit(prob,resp_matrix):
    """
    This function calculates the standardized outfit statistic for each item given
    a probability table and a response matrix
    :param prob: probability table with probability for each item person interaction
    :param resp_matrix: response matrix with 0 and 1 scores for each item person interaction
    :return: std_outfit_items: an array with standardized outfit statistic for each item
    doctest....
    """
    # mean squares
    std_res_sq_numer = np.square(resp_matrix - prob)
    std_res_sq_denom = (np.ones_like(prob) - prob)*prob
    std_res_sq = std_res_sq_numer/std_res_sq_denom
    n_persons = np.size(prob,0)
    item_mean_square = np.sum(std_res_sq, axis=0)/n_persons

    # standard deviation of mean squares
    w_reciprocal = np.ones_like(prob)/std_res_sq_denom
    numer = np.sqrt(np.sum( w_reciprocal,axis=0) - 4*n_persons)
    stddev_item_mean_square = numer/n_persons

    # cube root transformation to approximate t distribution
    term_1 = np.cbrt(item_mean_square)-1
    term_2 = 3/stddev_item_mean_square + stddev_item_mean_square/3
    std_outfit_item = term_1*term_2
    return std_outfit_item


def generate_summary_stats(std_outfit,n_items,n_persons,n_replications):
    """
    This functions takes in a numpy array with all the standardized fit statistics and the number of items.
    Computes the mean, standard deviation, min, max, percentile ranks for critical values of -2 and 2, 5th and 95th percentiles
    Saves the summay statistics in a CSV file with the name summary_stats_n_items_n_persons_n_replications
    :param std_outfit: numpy array with all the standardized outfit statistics
    :param n_items: number of items
    :param n_persons: number of persons
    :param n_replications: number of replications
    :return: summary: Pandas data frame with the summary statistics
    """
    mean_std_outfit = np.mean(std_outfit, axis=0)
    stddev_std_outfit = np.std(std_outfit, axis=0)
    percentile_95 = np.percentile(std_outfit, 95, axis=0)
    percentile_5 = np.percentile(std_outfit, 5, axis=0)
    min_std_outfit = np.min(std_outfit, axis=0)
    max_std_outfit = np.max(std_outfit, axis=0)
    prank_pos_2 = np.array([])
    prank_neg_2 = np.array([])
    for i in range(np.size(std_outfit, 1)):
        prank_pos_2 = np.append(prank_pos_2, stats.percentileofscore(std_outfit[:, i], 2))
        prank_neg_2 = np.append(prank_neg_2, stats.percentileofscore(std_outfit[:, i], -2))
    # print(prank_pos_2)
    # print(prank_neg_2)
    summary_stats = np.vstack((mean_std_outfit,stddev_std_outfit,min_std_outfit,max_std_outfit,prank_pos_2,prank_neg_2,percentile_95,percentile_5))
    # print(np.around(summary_stats, decimals=2))
    item_ref = []
    for i in range(n_items):
        item_ref.append(i+1)
    summary_dataframe = pd.DataFrame({'Item': item_ref,
                            'Mean':summary_stats[0,:],
                            'SD':summary_stats[1,:],
                            'Min':summary_stats[2,:],
                            'Max':summary_stats[3,:],
                            'Percentile rank of 2.0':summary_stats[4,:],
                            'Percentile rank of -2.0': summary_stats[5,:],
                            'Percentile (95th)': summary_stats[6,:],
                            'Percentile (5th)': summary_stats[7,:]})

    filename = 'Summary_stats' + '_' + str(n_items) + '_' + str(n_persons) + '_' + str(n_replications) + '.csv'
    summary_dataframe.to_csv(filename)
    print(filename, " generated.")
    # return summary_dataframe

def generate_plots(std_outfit):
    """
    This function generates a histogram for standardized outfit statistic for each item.
    :param std_outfit: numpy array with the standardized outfit statistics
    :return: none
    """

    for i in range(np.size(std_outfit, 1)):
        plt.hist(std_outfit[:, i],bins='auto')
        plot_title = 'Item' + str(i+1) + ' standardized outfit statistics'
        plt.title(plot_title)
        plt.show()

if __name__ == '__main__':

    while True:
        try:
            n_items = int(input("Please enter the number of items:"))
            n_persons = int(input("Please enter the number of persons:"))
            n_replications = int(input("Please enter the number of replications(>1):"))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue

        if n_replications < 2:
            print("Sorry, please enter number of replications greater than 1.")
            continue
        else:
            break


    prob_of_corr_response = generate_prob_table(n_items,n_persons)
    # print("Prob of correct response",prob_of_corr_response.shape,"min=",prob_of_corr_response.min(),"max=",prob_of_corr_response.max(),"\n",np.around(prob_of_corr_response, decimals=2))
    std_outfit_stats = np.array([])
    for replication in range(n_replications):
        prob_sample = replication_sample(n_persons, n_items)
        sample_resp_matrix = generate_response_matrix(prob_of_corr_response,prob_sample)
        while check_extreme_scores(sample_resp_matrix) == 1:
            # print("Sample response matrix (not useful):\n", sample_resp_matrix)
            prob_sample = replication_sample(n_persons, n_items)
            sample_resp_matrix = generate_response_matrix(prob_of_corr_response, prob_sample)
            if check_extreme_scores(sample_resp_matrix) == 0:
                break
        if std_outfit_stats.size == 0:
            std_outfit_stats = calculate_outfit(prob_of_corr_response,sample_resp_matrix)
        else:
            std_outfit_stats = np.vstack((std_outfit_stats, calculate_outfit(prob_of_corr_response,sample_resp_matrix)))
        # print("Sample response matrix (useful):\n", sample_resp_matrix)
    generate_summary_stats(std_outfit_stats,n_items,n_persons,n_replications)
    generate_plots(std_outfit_stats)
