import numbers
from typing import List, Dict


# Create a function that decides if a list contains any odd numbers.
# return type: bool
# function name must be: contains_odd
# input parameters: input_list


def contains_add(input_list: List) -> bool:
    contains_odd: bool
    contains_odd = any(number % 2 != 0 for number in input_list)
    return contains_odd


# Create a function that accepts a list of integers, and returns a list of bool.
# The return list should be a "mask" and indicate whether the list element is odd or not.
# (return should look like this: [True,False,False,.....])
# return type: list
# function name must be: is_odd
# input parameters: input_list


def is_odd(input_list: List[int]) -> List[bool]:
    odd_or_even_list: List[bool]
    odd_or_even_list = [number % 2 != 0 for number in input_list]
    return odd_or_even_list


# Create a function that accpects 2 lists of integers and returns their element wise sum. <br>
# (return should be a list)
# return type: list
# function name must be: element_wise_sum
# input parameters: input_list_1, input_list_2


def element_wise_sum(input_list_1: List[int], input_list_2: List[int]) -> List[int]:
    summarized_list: List[int]
    summarized_list = [sum(x) for x in zip(input_list_1, input_list_2)]
    return summarized_list


# Create a function that accepts a dictionary and returns its items as a list of tuples
# (return should look like this: [(key,value),(key,value),....])
# return type: list
# function name must be: dict_to_list
# input parameters: input_dict


def dict_to_list(input_dict: Dict) -> List[tuple]:
    output: List[tuple]
    output = list(input_dict.items())
    return output

# If all the functions are created convert this notebook into a .py file and push to your repo
