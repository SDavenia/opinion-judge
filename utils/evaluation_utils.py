from itertools import combinations
import math 

def get_den_normalization(n:int, range_size:float):

    return math.floor(pow(n,2)/4*range_size)


def calculate_rs(scores: list[list[float]],
                 range: tuple[float, float] = (0.0,1.0)) -> float:

    n = len(scores)
    tot_sum = 0.0
    size_range = range[1] - range[0]
    for scores_query in scores:
        combined_score_pairs = combinations(scores_query, 2)
        abs_diff = [abs(s-s2) for (s, s2) in combined_score_pairs]
        sum_abs_diff = sum(abs_diff)
        den_normalization = get_den_normalization(len(scores_query),size_range) #this normalization is inside in case a number of different trials is conducted for different queries
        tot_sum += (sum_abs_diff/den_normalization)
        
    rs = 1-((1/n) * tot_sum)
    return rs

def calculate_pc(scores: list[list[tuple[float, float]]],
                 range: tuple[float, float] = (0.0,1.0)) -> float:

    n = len(scores)
    tot_sum = 0.0
    size_range = range[1] - range[0]
    for scores_couple in scores:
        diff_couples = [abs(s-s2) for (s, s2) in scores_couple]
        sum_diff_couples = sum(diff_couples)
        sum_diff_couples = sum_diff_couples/size_range
        sum_diff_couples = sum_diff_couples/len(scores_couple) #this normalization is inside in case a number of different trials is conducted for different queries
        tot_sum += sum_diff_couples

    pc = 1 -((1/n) * tot_sum)
    return pc


