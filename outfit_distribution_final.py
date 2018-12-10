import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def generate_prob_table(n_items, n_persons, n_replications):
    """
    This function generates the expected probability table for each person's response to each item.
    Uses a pseudo-random sample of student locations from a normally-distributed population with mean = 0 logits
    and standard deviation = 1 logit.
    Uses a random sample of item locations from a uniformly-distributed population
    ranging from -2.0 to 2.0 logits.
    :param n_items: number of items
    :param n_persons: number of persons
    :param n_replications: number of replications (to save the item and person locations with appropriate name)
    :return: numpy array with the probability table

    >>> test = generate_prob_table(2,3,10) # to check the shape of the returned array
    >>> test.shape
    (3, 2)
    >>> np.min(test)>0 # 0 is invalid because probability of a correct response is theoretically never 0
    True
    >>> np.max(test)>=1 # 1 or greater is invalid because probability of a correct response is theoretically  never 1
    False
    """
    mu, sigma = 0, 1  # mean and standard deviation
    persons = np.random.normal(mu, sigma, n_persons)
    items = np.random.uniform(-2.0, 2.01, n_items)

    # save person and item locations as csv
    items_filename = "item_locations_" + str(n_items) + "_"+ str(n_persons) +"_"+ str(n_replications)+".csv"
    # print("Item locations have been stored in ", items_filename)
    persons_filename = "person_measures_" + str(n_items) +"_"+ str(n_persons) + "_"+str(n_replications)+".csv"
    # print("Person measures have been stored in ", persons_filename)
    np.savetxt(items_filename, items)
    np.savetxt(persons_filename, persons)

    # Iterate over persons to create an np array with probabilities
    for i in range(n_persons):
        numerator = np.exp(-1*(items - persons[i]))
        denomominator = 1 + numerator
        prob_temp = numerator / denomominator
        if i == 0: # for first person to initialize the array
            prob_table = prob_temp
        else: # add the array to existing
            prob_table = np.vstack((prob_table, prob_temp))
    return prob_table

def replication_sample (n_items, n_persons):
    """
    This function generates and returns a pseudo-random sample numpy array with the size [n_persons, n_items].
    n_persons*n_items samples are drawn from a uniformly distributed population in range 0.0001 to <1.0.
    These constraints are because theoretically the probability of a correct response is never equal to 0 or 1
    :param n_items: number of items
    :param n_persons: number of persons
    :return: sample numpy array
    >>> np.min(replication_sample(20,30))>0 # 0 is invalid because probability of a correct response is theoretically never 0
    True
    >>> np.max(replication_sample(20,30))>=1 # 1 or greater is invalid because probability of a correct response is theoretically  never 1
    False
    """
    sample = np.random.uniform(low=0.0001, high=1.0, size=(n_items,n_persons))
    return sample

def generate_response_matrix(prob_table, sample):
    """
    This function generates a response matrix with 0s and 1s for each item and person combination.
    It takes in the probability matrix numpy array with probability of correct response for each person and
    the replication sample numpy array. It then compares the corresponding values in the two arrays.
    If the value of the replication sample element is lower than the corresponding element value in the probability matrix,
    then the response value is set to 1. Else, it is set to 0 (only 0 and 1 are possible scores for a person on any item).
    The logic is that the probability matrix defines the probability of correct response by a person to an item
    which is theoretically greater than 0 and less than 1.
    The replication sample numpy array was created with a uniform distribution with a minimum 0.0001 and a maximum of
    less than 1. Therefore, when we compare the corresponding sample and probability matrix values and assign a
    1 for sample values lower than the corresponding probability matrix value and assign a 0 for equal to or greater
    than the corresponding probability matrix value, over an infinite number of random samples,
    the number of times the sample value is lower than the corresponding probability matrix value is likely to converge
    with the probability of correct response.
    For example, if the probability of correct response is 0.47 in the probability matrix and the corresponding elemtent
    For example, if the probability of correct response is 0.47 in the probability matrix and the corresponding elemtent
    value in the replication sample is 0.12, the corresponding value in the response matrix will be 1.
    Prob of correct response = 0.45
    Corresponding value in replication sample = 0.12
    Corresponding value in the response matrix = 1 because 0.12 is less than 0.45
    The replication sample values are drawn randomly from a uniform distribution between 0.0001 and less than 1.
    So, over infinite replications, the value will be less than 0.45 about 45% of the times - which is the probability
    of correct response. As a result, the over infinite replications, we expect that the response matrix score will be
    1 about 45% of the times.

    :param prob_table: probability matrix numpy array with probability of correct response for each person
    and item combination
    :param sample: sample numpy array
    :return: response matrix
    >>> prob_table = generate_prob_table(2,3,5)
    >>> sample = replication_sample(3,2)
    >>> resp_matrix = generate_response_matrix(prob_table, sample)
    >>> print(resp_matrix.shape)
    (3, 2)
    """
    response_table = prob_table - sample
    response_table[(response_table > 0)] = 1
    response_table[(response_table <= 0)] = 0
    return response_table

def check_extreme_scores(resp_matrix_sample):
    """
    This function checks if any item has an extreme score (all 0 or all 1).
    It is important to exclude items with extreme scores because the location estimates
    necessary to calculate the probability of a correct response for these score groups is infinite and hence not useful.
    :param resp_matrix_sample: response matrix of shape number of items x number of persons.
    :return: extreme_score_flag: Flag variable with value of 1 if there is any extreme item
    or person score in the sample, else 0.
    doctest - supply with perfect score and return 0
    >>> test_1=np.ones([4,5])
    >>> check_extreme_scores(test_1)
    1
    >>> test_2=np.zeros([4,5])
    >>> check_extreme_scores(test_2)
    1
    >>> test_3 = np.arange(0,24,2).reshape((4,3))
    >>> check_extreme_scores(test_3)
    0
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
    Saves the summary statistics in a CSV file with the name summary_stats_n_items_n_persons_n_replications.
    :param std_outfit: numpy array with all the standardized outfit statistics
    :param n_items: number of items
    :param n_persons: number of persons
    :param n_replications: number of replications
    :return: none
    """
    # Summary statistics
    mean_std_outfit = np.mean(std_outfit, axis=0)
    stddev_std_outfit = np.std(std_outfit, axis=0)
    percentile_95 = np.percentile(std_outfit, 95, axis=0)
    percentile_5 = np.percentile(std_outfit, 5, axis=0)
    min_std_outfit = np.min(std_outfit, axis=0)
    max_std_outfit = np.max(std_outfit, axis=0)
    prank_pos_2 = np.array([]) # percentile rank of +2 - probability of observing a value less than or equal to +2
    prank_neg_2 = np.array([]) # percentile rank of -2 - probability of observing a value less than or equal to -2

    for i in range(np.size(std_outfit, 1)):
        prank_pos_2 = np.append(prank_pos_2, stats.percentileofscore(std_outfit[:, i], 2))
        prank_neg_2 = np.append(prank_neg_2, stats.percentileofscore(std_outfit[:, i], -2))
    summary_stats = np.vstack((mean_std_outfit,stddev_std_outfit,min_std_outfit,max_std_outfit,prank_pos_2,prank_neg_2,percentile_95,percentile_5))

    item_ref = []
    for i in range(n_items):
        item_ref.append(i+1)

    # Save summary statistics to a pandas dataframe
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

def generate_plots(std_outfit, n_items, n_persons):
    """
    This function generates a histogram for standardized outfit statistic for each item.
    :param std_outfit: numpy array with the standardized outfit statistics
    :param n_items: number of items to save in the file name
    :param n_personss: number of persons to save in the file name
    :return: none
    """

    for i in range(np.size(std_outfit, 1)):
        plt.hist(std_outfit[:, i],bins='auto')
        plot_title = 'Item' + str(i+1) + ' standardized outfit statistics'
        plot_filename = 'Item_' + str(i+1) + '_' + str(n_items) + '_' + str(n_persons) + ".png"
        plt.title(plot_title)
        plt.savefig(plot_filename)
        plt.show()
    print("Standardized outfit statistics plot saved.")


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

    prob_of_corr_response = generate_prob_table(n_items,n_persons, n_replications)
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

    generate_summary_stats(std_outfit_stats,n_items,n_persons,n_replications)
    generate_plots(std_outfit_stats, n_items,n_persons)
