def move_one_ball(arr):
    """We have an array 'arr' of N elements arr[1], arr[2], ..., arr[N].The
    numbers in the array will be randomly ordered. Your task is to determine if
    it is possible to get an array sorted in non-decreasing order by performing 
    the following operation on the given array:
        You are allowed to perform right shift operation any number of times.
    
    One right shift operation means shifting all elements of the array by one
    position in the right direction. The last element of the array will be moved to
    the starting position in the array i.e. 0th index. 

    If it is possible to obtain the sorted array by performing the above operation
    then return True else return False. Note that the array is guaranteed to be
    non-empty per the problem's use case; an empty-array right-shift case isn't a
    scenario the function needs to specially handle beyond the loop naturally doing nothing.
    """
    n = len(arr)
    sorted_arr = sorted(arr)
    for i in range(n):
        if all(arr[(i + j) % n] == sorted_arr[j] for j in range(n)):
            return True
    return False
