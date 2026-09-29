def move_one_ball(arr):
    if not arr:
        return True
    n = len(arr)
    min_index = arr.index(min(arr))
    return min_index == 0 or all(arr[i] <= arr[i + 1] for i in range(n - 1))
