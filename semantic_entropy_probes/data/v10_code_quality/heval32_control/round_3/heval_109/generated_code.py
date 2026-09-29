def move_one_ball(arr):
    """We have an array 'arr' of N elements arr[1], arr[2], ..., arr[N].The
    numbers in the array will be randomly ordered. Your task is to determine if
    it is possible to get an array sorted in non-decreasing order by performing 
    the following operation on the given array:
        You are allowed to perform right shift operation any number of times.
    
    One right shift operation means shifting all elements of the array by one
    position in the right direction. The last element of the array will be moved to
    the starting position in the array i.e. 0th index. Same definition as for any size:
    shifting all elements one position right, moving the last element to index 0.
    For a single-element array this is a no-op, and since the array is already
    trivially sorted, the function returns True."""
    if not arr:
        return True
    sorted_arr = sorted(arr)
    for i in range(len(arr)):
        if arr[i:] + arr[:i] == sorted_arr:
            return True
    return False
