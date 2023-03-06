# %%
import itertools
from typing import List, Tuple, Dict


# %%
# Create a function that returns with a subset of a list.
# The subset's starting and ending indexes should be set as input parameters (the list as well).
# return type: list
# function name must be: subset
# input parameters: input_list,start_index,end_index
# %%
def subset(input_list: List, start_index: int, end_index: int) -> List:
    subset_of_list: List
    subset_of_list = input_list[start_index:end_index]
    return subset_of_list


# %%
# Create a function that returns every nth element of a list.
# return type: list
# function name must be: every_nth
# input parameters: input_list,step_size
# %%
def every_nth(input_list: List, step_size: int) -> List:
    list_of_nth: List
    list_of_nth = input_list[::step_size]
    return list_of_nth


# %%
# Create a function that can decide whether a list contains unique values or not
# return type: bool
# function name must be: unique
# input parameters: input_list
# %%
def unique(input_list: List) -> bool:
    only_unique: bool
    only_unique = len(set(input_list)) == len(input_list)
    return only_unique


# %%
# Create a function that can flatten a nested list ([[..],[..],..])
# return type: list
# function name must be: flatten
# input parameters: input_list
# %%
def flatten(input_list: List) -> List:
    flatted_list: List
    flatted_list = list(itertools.chain(*input_list))
    return flatted_list


# %%
# Create a function that concatenates n lists
# return type: list
# function name must be: merge_lists
# input parameters: *args

# %%
def merge_lists(*args: List) -> List:
    merged_list: List
    merged_list = list(itertools.chain(*args))
    return merged_list


# %%
# Create a function that can reverse a list of tuples
# example [(1,2),...] => [(2,1),...]
# return type: list
# function name must be: reverse_tuples
# input parameters: input_list
# %%
def reverse_tuples(input_list: List[Tuple]) -> List[Tuple]:
    reversed_tuples_in_list: List[Tuple]
    reversed_tuples_in_list = [tuples[::-1] for tuples in input_list]
    return reversed_tuples_in_list


# %%
# Create a function that removes duplicates from a list
# return type: list
# function name must be: remove_tuplicates
# input parameters: input_list
# %%
def remove_duplicates(input_list: List) -> List:
    uniques_only: List
    uniques_only = list(set(input_list))
    return uniques_only


# %%
# Create a function that transposes a nested list (matrix)
# return type: list
# function name must be: transpose
# input parameters: input_list
# %%
def transpose(input_list: List) -> List:
    transposed: List
    transposed = list(map(list, zip(*input_list)))
    return transposed


# %%
# Create a function that can split a nested list into chunks
# chunk size is given by parameter
# return type: list
# function name must be: split_into_chunks
# input parameters: input_list,chunk_size
# %%
def split_into_chunks(input_list: List, chunk_size: int) -> List[List]:
    nested_list: List[List]
    nested_list = [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
    return nested_list

# %%
# Create a function that can merge n dictionaries
# return type: dictionary
# function name must be: merge_dicts
# input parameters: *dict
# %%
def merge_dicts(*dict: Dict) -> Dict:
    merged_dict: Dict = {}
    for one_dict in dict:
        merged_dict.update(one_dict)
    return merged_dict


# %%
# Create a function that receives a list of integers and sort them by parity
# and returns with a dictionary like this: {"even":[...],"odd":[...]}
# return type: dict
# function name must be: by_parity
# input parameters: input_list
# %%
def by_parity(input_list: List) -> Dict:
    output_dict: Dict
    even_list: List = [element for element in input_list if element % 2 == 0]
    odd_list: List = [element for element in input_list if element % 2 != 0]
    output_dict = {"even": even_list, "odd": odd_list}
    return output_dict


# %%
# Create a function that receives a dictionary like this: {"some_key":[1,2,3,4],"another_key":[1,2,3,4],....}
# and return a dictionary like this : {"some_key":mean_of_values,"another_key":mean_of_values,....}
# in short calculates the mean of the values key wise
# return type: dict
# function name must be: mean_key_value
# input parameters: input_dict
# %%
def mean_key_value(input_dict: Dict) -> Dict:
    output_dict: Dict
    output_dict = {keys: sum(list_of_values) / len(list_of_values) for (keys, list_of_values) in input_dict.items()}
    return output_dict
# %%
# If all the functions are created convert this notebook into a .py file and push to your repo
