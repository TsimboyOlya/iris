import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
#python img_data.py ./data/casia_ds_Expert.txt  ./data/casia_ds

def FitToDistr01(amounm_array):
	max_t = max(amounm_array)
	min_t = min(amounm_array)
	init_range = max_t - min_t
	return [(init_val - min_t) / init_range for init_val in amounm_array]


def PersonEyeDistr(same, diff):
	#same = [float(format(i, '.3f')) for i in same]
	uniq_dist_same = list(set(same))
	uniq_dist_same.sort()
	
	#diff = [float(format(i, '.3f')) for i in diff]
	uniq_dist_diff = list(set(diff))
	uniq_dist_diff.sort()
	
	all_mount = len(same) + len(diff)
	same_am = [same.count(i) / all_mount for i in uniq_dist_same]
	diff_am = [diff.count(i) / all_mount for i in uniq_dist_diff]
	
	return uniq_dist_same, FitToDistr01(same_am), uniq_dist_diff, FitToDistr01(diff_am)

same = []
with open('same.txt', 'r') as data_file:
	for line in data_file:
		same += [int(i) for i in line.split()]
diff = []
with open('diff.txt', 'r') as data_file:
	for line in data_file:
		diff += [int(i) for i in line.split()]

same_len = len(same)
diff_len = len(diff)

same = FitToDistr01(same)
diff = FitToDistr01(diff)

same.sort()
diff.sort()

print('same_arr_len -> ', len(same), 'avg ->', sum(i for i in same) / len(same))
print('diff_arr_len -> ', len(diff), 'avg ->', sum(i for i in diff) / len(diff))



same_dist, same_am, diff_dist, diff_am = PersonEyeDistr(same, diff)

plt.figure(figsize=(10,6))
plt.title("Left and Right eyes")
plt.xlabel(r"$\rho$", fontsize=20)
plt.ylabel("amount", fontsize=18)
plt.plot(same_dist, same_am, 'ro', label="same person dist")
plt.plot(diff_dist, diff_am, 'bo', label="diff person dist")
plt.legend(loc="best")
plt.show()