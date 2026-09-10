from itertools import combinations
import math 
import pandas as pd

def get_den_normalization(n:int, range_size:float):

    return math.floor(pow(n,2)/4*range_size)

def calculate_rs(scores: list[list[float]], # [[esempio1_prompt01_1, esempio1_prompt01_2, ...], ]
                 range: tuple[float, float] = (0.0,1.0)) -> float:

    tot_sum = 0.0
    n = 0
    size_range = range[1] - range[0]
    for scores_query in scores:
        if len(scores_query) < 2:
            # No repeated judgments to compare for this query (den_normalization
            # would be 0, since there are no pairs to take an abs-diff over) --
            # skip it rather than divide by zero; it carries no stability signal.
            continue
        combined_score_pairs = combinations(scores_query, 2)
        abs_diff = [abs(s-s2) for (s, s2) in combined_score_pairs]
        sum_abs_diff = sum(abs_diff)
        den_normalization = get_den_normalization(len(scores_query),size_range) #this normalization is inside in case a number of different trials is conducted for different queries
        tot_sum += (sum_abs_diff/den_normalization)
        n += 1

    if n == 0:
        return float("nan")
    rs = 1-((1/n) * tot_sum)
    return rs

def calculate_pc(scores: list[list[tuple[float, float]]], # [[(esempio1_prompt01_1_1to2, esempio1_prompt01_1_2to1), (esempio1_prompt01_2_1to2, esempio1_prompt01_2_2to1) , ...], ]
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

def _get_grouped_scores_rs(df_scoring: pd.DataFrame,
                        score_column: str) -> list:
    result = (
        df_scoring.groupby(["id_1", "id_2"])
        .agg(lambda x: x.tolist())
        [score_column]
        .tolist()
    )
    return result

def calculate_rs_from_df(df_scoring: pd.DataFrame,
                         range: tuple[float, float] = (0.0,1.0)) -> float:

    sc_1_to_2 = _get_grouped_scores_rs(df_scoring, "parsed_score_1to2")
    sc_2_to_1 = _get_grouped_scores_rs(df_scoring, "parsed_score_2to1")
    all_scores = sc_1_to_2 + sc_2_to_1
    return calculate_rs(all_scores, range=range)

def _get_grouped_scores_pc(df_scoring: pd.DataFrame) -> list:
    result = (df_scoring.groupby(["id_1", "id_2"])
       .agg(lambda x: x.tolist())
       .apply(lambda row: list(zip(row["parsed_score_1to2"], row["parsed_score_2to1"])), axis=1)
       .tolist()
    )
    return result
    #domanda che mi sto facendo, ma in pc ci importa farlo raggruppando? non ha senso prendere tutto assieme?

def calculate_pc_from_df(df_scoring: pd.DataFrame,
                         range: tuple[float, float] = (0.0,1.0)) -> float:

    scores_pc = _get_grouped_scores_pc(df_scoring)
    return calculate_pc(scores_pc, range=range)