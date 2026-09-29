def move_one_ball(arr):
    if not arr:
        return True
    sorted_arr = sorted(arr)
    n = len(arr)
    for i in range(n):
        if all(arr[(i + j) % n] == sorted_arr[j] for j in range(n)):
            return True
    return False
