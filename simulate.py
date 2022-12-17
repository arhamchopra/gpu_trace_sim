import sys
import os
import json
import pprint

job_count = len(sys.argv) - 2
stats_dir = "./stats/"
dump_dir = "./dump/"
identifier = sys.argv[1]
jobs = []
for job_file in sys.argv[2:]:
  print("Loading {}".format(job_file))
  with open(job_file, "r") as f:
    jobs += [json.load(f)]

prev_kernel_time = [0] * job_count
kernel_end_time = [0] * job_count
job_pos = [0] * job_count
job_iter_count = [0] * job_count
cur_time = 0
useful_time = 0


basic_concat = sum([[idx]*len(job_seq) for idx, job_seq in enumerate(jobs)], [])
basic_concat_idx = 0

def concat(ready_kernels):
  global basic_concat, basic_concat_idx
  job_idx = basic_concat[basic_concat_idx]
  if (ready_kernels[job_idx] == 1):
    basic_concat_idx += 1
    if basic_concat_idx == len(basic_concat):
      basic_concat_idx = 0
    return job_idx;
  else:
    return -1

def greedy(ready_kernels):
  global kernel_end_time
  min_idx = -1
  min_time = 1e18
  for idx, is_ready in enumerate(ready_kernels):
    if is_ready:
      if min_time > kernel_end_time[idx]:
        min_idx = idx
        min_time = kernel_end_time[idx]
  return min_idx

def short_kernel_first(ready_kernels):
  global prev_kernel_time, kernel_end_time, job_pos, cur_time, useful_time, job_count, jobs, final_kernel_list
  min_idx = -1
  min_time = 1e18
  for idx, is_ready in enumerate(ready_kernels):
    if is_ready:
      kernel_idx = job_pos[idx]
      ktime = jobs[idx][kernel_idx]["ts"]
      if min_time > ktime:
        min_time = ktime
        min_idx = idx
  return min_idx

def short_kernel_first_aged(ready_kernels):
  global prev_kernel_time, kernel_end_time, job_pos, cur_time, useful_time, job_count, jobs, final_kernel_list
  min_idx = -1
  min_time = 1e18
  for idx, is_ready in enumerate(ready_kernels):
    if is_ready:
      kernel_idx = job_pos[idx]
      job = jobs[idx]
      ktime = job[kernel_idx]["ts"] - (cur_time - prev_kernel_time[idx] + job[kernel_idx]["ts"])
      if min_time > ktime:
        min_time = ktime
        min_idx = idx
  return min_idx

def find_next(ready_kernels):
  return short_kernel_first(ready_kernels)

final_kernel_list = []

def simulate():
  global prev_kernel_time, kernel_end_time, job_pos, cur_time, useful_time, job_count, jobs, final_kernel_list
  ready_kernels = [0] * job_count
  to_wait = [0] * job_count
  for idx, job in enumerate(jobs):
    kernel_idx = job_pos[idx]
    if prev_kernel_time[idx] + job[kernel_idx]["ts"] <= cur_time:
      ready_kernels[idx] = 1
    to_wait[idx] = prev_kernel_time[idx] + job[kernel_idx]["ts"] - cur_time


  idx = find_next(ready_kernels)
  #  print("{} {} {}".format(cur_time, to_wait, idx))
  kernel_idx = job_pos[idx]
  if idx == -1:
    min_delta = min([delta for delta in to_wait if delta > 0])
    cur_time += min_delta
    return 0
  else:
    kernel = jobs[idx][kernel_idx]
    final_kernel_list += [{"name": kernel["name"], "ts":cur_time, "dur":kernel["dur"]}]
    prev_kernel_time[idx] = cur_time
    cur_time += jobs[idx][kernel_idx]["dur"]
    kernel_end_time[idx] = cur_time
    useful_time += jobs[idx][kernel_idx]["dur"]
    job_pos[idx] += 1
    if (job_pos[idx] == len(jobs[idx])):
      job_iter_count[idx] += 1
      job_pos[idx] = 0
    #  pprint.pprint(final_kernel_list[-1])
    return 1

while True:
  done_count = 0
  for i in job_iter_count:
    if (i >= 50):
      done_count += 1
  if done_count == job_count: break
  simulate()

for idx in range(len(job_iter_count)):
  job_iter_count[idx] += job_pos[idx] * 1.0 / (len(jobs[idx]))

with open(dump_dir + identifier, "w") as f:
  f.write(json.dumps(final_kernel_list, indent = 4))

with open(stats_dir + identifier, "w") as f:
  dump_dict = {}
  dump_dict["total_time (microsec)"] = cur_time * 1.0
  total_time = cur_time
  dump_dict["(microsec/iter)"] = {}
  for idx, job_file in enumerate(sys.argv[2:]):
    dump_dict["(microsec/iter)"][job_file + "-" + str(idx)] = cur_time * 1.0 / job_iter_count[idx]
  dump_dict["allocation"] = useful_time * 100.0 / cur_time
  f.write(json.dumps(dump_dict, indent = 4))
