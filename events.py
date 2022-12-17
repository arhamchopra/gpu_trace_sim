import os
import sys
import json
import copy
import pprint
from collections import defaultdict

logs_path = "./" + sys.argv[1] + "/"
KERNEL_TYPE = "kernel"
CPU_TYPE = "cpu_op"
CUDA_RUNTIME_TYPE = "cuda_runtime"
GPU_MEMCPY_TYPE = "gpu_memcpy"
GPU_MEMSET_TYPE = "gpu_memset"
PYTHON_FUNCTION_TYPE = "python_function"
USER_ANNOTATION_TYPE = "user_annotation"
FLOW_TYPE = "async_cpu_to_gpu"
USEFUL_PH = ["X"]
USEFUL_CAT = [KERNEL_TYPE, CUDA_RUNTIME_TYPE, USER_ANNOTATION_TYPE]
# KeyFormat = <GPU_Type, No_GPUs, Kernel_Name, Grid, Block, USER_INPUT>

json_data = {}
cpu = []
cuda_runtime = []
gpu_memcpy = []
gpu_memset = []
kernel = []
python_function = []
user_annotation = []
gpu = []
final_result = {}

files = os.listdir(logs_path)

gpu_type = ""
gpu_count = ""
job_name = sys.argv[2]
pattern = sys.argv[3]
dir_name = "./" + job_name + "/" + pattern + "/"
if not (os.path.exists(dir_name)):
  os.makedirs(dir_name)

def make_key(kernel_name, grid, block):
  global gpu_type, gpu_count, job_name
  grid_name = ",".join([str(e) for e in grid])
  block_name = ",".join([str(e) for e in block])
  return "|".join([kernel_name, grid_name, block_name, gpu_type, gpu_count, job_name])

kernel_times = {}
file_idx = 0
min_util = 101
max_util = 0
for filepath in files:
  if (pattern not in filepath): continue
  if (filepath.split(".")[-1] != "json"): continue
  time_unit = 1e-3

  print("Processing " + filepath)
  with open(logs_path + filepath, "r") as f:
    json_data = json.load(f)

  base_filepath = filepath.split(".")[0]
  devices = json_data["deviceProperties"]
  gpu_type = devices[file_idx]["name"]
  gpu_count = str(len(devices))

  useful_events = filter(lambda x: (x["ph"] in USEFUL_PH) and (x["cat"] in USEFUL_CAT),
                         json_data["traceEvents"])
  trace = sorted(useful_events, key = lambda x: x["ts"])
  category_types = list(set([e.get("cat", "-") for e in trace]))
  ph_types = list(set([e.get("ph", "-") for e in trace]))
  pids = list(set([e.get("pid", "-") for e in trace]))
  tids = list(set([e.get("tid", "-") for e in trace]))

  kernel_events = {}
  current_kernel_seq = {}
  step_start = 0
  step_end = 0
  step_idx = 0
  other_kernel_seq = []
  global_kernel_time = 0
  global_total_time = 0

  for ev in trace:
    cat = ev["cat"]
    name = ev["name"]
    ts = ev["ts"]
    pid = ev["pid"]
    if cat == KERNEL_TYPE:
      kernel_events[ev["args"]["External id"]] = ev
      key = make_key(name, ev["args"]["grid"], ev["args"]["block"])
      try:
        kernel_times[key]["time"] += ev["dur"]
        kernel_times[key]["count"] += 1
      except KeyError:
        kernel_times[key] = {}
        kernel_times[key]["time"] = ev["dur"]
        kernel_times[key]["count"] = 1
      global_kernel_time += ev["dur"]
      global_total_time = max(global_total_time, ev["ts"] - trace[0]["ts"])


  for ev in trace:
    #  pprint.pprint(ev)
    cat = ev["cat"]
    name = ev["name"]
    ts = ev["ts"]
    pid = ev["pid"]
    if cat == USER_ANNOTATION_TYPE and "ProfilerStep" in name:
      if step_start != 0:
        seq_key = "_".join([str(file_idx), str(step_idx)])
        for gpu_id, sequence in current_kernel_seq.items():
          # Calculate Occupation Time
          kernel_seq = [kernel_events[eid] for eid in sequence]
          total_time = kernel_seq[-1]["ts"] - kernel_seq[0]["ts"]
          occupation_kernel_time = 0
          for item in kernel_seq:
            occupancy = item["args"]["est. achieved occupancy %"]
            dur = item["dur"]
            occupancy = max(occupancy, 25)
            if dur < 20:
              occupancy = 100
            occupancy /= 100.0
            occupation_kernel_time += occupancy * dur

          prev_ts = kernel_events[sequence[0]]["ts"]
          kernel_seq = []
          for eid in sequence:
            kernel_ev = kernel_events[eid]
            key = make_key(kernel_ev["name"], kernel_ev["args"]["grid"], kernel_ev["args"]["block"])
            kernel_seq += [ { "name": key, "ts": kernel_ev["ts"] - prev_ts, "dur": kernel_ev["dur"] } ]
            prev_ts = kernel_ev["ts"]
          dump_file = dir_name + ".".join([base_filepath, str(gpu_id), str(step_idx), "json"])
          # print("Dumping to {}".format(dump_file))
          with open(dump_file, "w") as f:
            f.write(json.dumps(kernel_seq, indent = 4))
          total_kernel_time = sum([e["dur"] for e in kernel_seq])
          alloc_utilization = total_kernel_time * 1.0 / total_time
          occ_utilization = occupation_kernel_time * 1.0 / total_time
          min_util = min(min_util, total_kernel_time * 1.0 / total_time)
          max_util = max(max_util, total_kernel_time * 1.0 / total_time)
          print("Utilization for GPU:{} Step:{} is Allocation: {:2.2f}, Occupation: {:2.2f}, TotalTime: {:.2f}".format(gpu_id, step_idx, alloc_utilization*100, occ_utilization*100, total_time))
        step_idx += 1
        current_kernel_seq = {}
      print("Starting new step " + name)
      step_start = ts
      step_end = step_start + ev["dur"]
    elif cat == CUDA_RUNTIME_TYPE and name in ["cudaLaunchKernel", "INVALID"]:
        if ts >= step_start and ts <= step_end:
          external_id = ev["args"]["External id"]
          try:
            kernel_ev = kernel_events[external_id]
          except KeyError:
            continue
          gpu_id = kernel_ev["pid"]
          try:
            current_kernel_seq[gpu_id] += [external_id]
          except KeyError:
            current_kernel_seq[gpu_id] = [external_id]
        else:
          print("ERROR: Found an outside boundary CUDA Launch")
          other_kernel_seq += [ev["args"]["External id"]]
  file_idx += 1

print("Per Iteration Utilization is {:2.4f} {:2.4f}".format(min_util*100, max_util*100))
#  print("Global Utilization is {:4.4f}".format(global_kernel_time * 1.0 / global_total_time))
with open(dir_name + "times" + ".json", "w") as f:
  f.write(json.dumps(kernel_times, indent = 4))
