def move_one_ball(arr):
    if len(arr) <= 1:
        return True
    sorted_arr = sorted(arr)
    for i in range(len(arr)):
        if all(arr[(i + j) % len(arr)] == sorted_arr[j] for j in range(len(arr))):
            return True
    return False
