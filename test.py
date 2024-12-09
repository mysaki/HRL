import numpy as np
#当相对角度较小的时候，为了使tracker同样能够绕过障碍物，假设一个新的相对角度较大的目的地，使其能够绕开障碍物
def map_new_target(angle, distance):
    if np.abs(angle) > 20:
        return angle,distance
    # 以自身当前位置为原点，估计得到原本的目的地
    ori_target = np.array([np.sin(90+angle)*distance, np.cos(90+angle)*distance])

    # 在角度为四十五的地方生成一个新的目的地，生成一个标记，当到达这个新目的地的时候就把原本目的地的值赋回去
    if angle > 0 :
        new_target = np.array([-np.cos(90+angle)*distance, np.cos(90+angle)*distance])
    elif angle < 0 :
        new_target = np.array([np.cos(90+angle)*distance, np.cos(90+angle)*distance])
    # 得到新目的地和原本的目的地之间的angle以及distance，记录下来
    delta_target = new_target - ori_target
    map_theta = 0
    map_distance = np.linalg.norm(delta_target)
    print("ori_target:",ori_target,"new_target:",new_target)
    print("map_theta:",map_theta,"map_distance:",map_distance)

    return 45,np.linalg.norm(new_target)

# if __name__ == 'main':
print(map_new_target(-10,1))