import os
import json
import pprint
from collections import defaultdict

logs_path = "./logs/"
KERNEL_TYPE = "kernel"
CPU_TYPE = "cpu_op"
CUDA_RUNTIME_TYPE = "cuda_runtime"
GPU_MEMCPY_TYPE = "gpu_memcpy"
GPU_MEMSET_TYPE = "gpu_memset"
PYTHON_FUNCTION_TYPE = "python_function"
USER_ANNOTATION_TYPE = "user_annotation"

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

for filepath in files:
  print("Processing " + filepath)
  with open(logs_path + filepath, "r") as f:
    json_data = json.load(f)

  devices = json_data["deviceProperties"]
  trace = sorted(json_data["traceEvents"], key = lambda x: x["ts"])

  event_types = []

  for v in trace:
    try:
      event_types += [(v["ph"], v["cat"])]
      category = v["cat"]
      if (category == CPU_TYPE) :
        cpu += [v]
      elif (category in [KERNEL_TYPE, GPU_MEMCPY_TYPE, GPU_MEMSET_TYPE]) :
        kernel += [v]
        gpu += [v]
      elif (category == CUDA_RUNTIME_TYPE) :
        cuda_runtime += [v]
      elif (category == PYTHON_FUNCTION_TYPE) :
        python_function += [v]
      elif (category == USER_ANNOTATION_TYPE) :
        user_annotation += [v]
        gpu += [v]
    except KeyError as e:
        #  print(v)
        continue

  def get_default_dict() :
    return defaultdict(lambda: 0)

  total_time = {}
  total_use_time_nvidia = {}
  total_use_time_actual = {}
  cur_profiler_step = 0
  cur_total_time = 0
  cur_total_use_time_nvidia = defaultdict(lambda: 0)
  cur_total_use_time_actual = defaultdict(lambda: 0)
  for v in gpu:
    if (v["cat"] == USER_ANNOTATION_TYPE):
      if "ProfilerStep" in v["name"]:
        if cur_profiler_step != 0:
          #  print("Dumping results for step " + str(cur_profiler_step) + " in " + filepath)
          total_time[cur_profiler_step] = v["ts"] - cur_total_time
          total_use_time_nvidia[cur_profiler_step] = dict(cur_total_use_time_nvidia)
          total_use_time_actual[cur_profiler_step] = dict(cur_total_use_time_actual)

        cur_profiler_step = int(v["name"].split("#")[1])
        cur_total_time = v["ts"]
        cur_total_use_time_nvidia = defaultdict(lambda: 0)
        cur_total_use_time_actual = defaultdict(lambda: 0)
        #  print("Starting new step " + str(cur_profiler_step) + " in " + filepath)
      continue
    #  print(v)
    gpu_id = v["pid"]
    duration = v["dur"]
    if v["cat"] in [GPU_MEMCPY_TYPE, GPU_MEMSET_TYPE]:
      nvidia_occupancy = 1
    else:
      nvidia_occupancy = v["args"]["est. achieved occupancy %"]/100.0
    if nvidia_occupancy <= 0.01:
      nvidia_occupancy = 0.1
    cur_total_use_time_nvidia[gpu_id] += duration
    cur_total_use_time_actual[gpu_id] += int(duration*nvidia_occupancy)
    #  print(dict(cur_total_use_time_nvidia))

  total_time[cur_profiler_step] = v["ts"] - cur_total_time
  total_use_time_nvidia[cur_profiler_step] = dict(cur_total_use_time_nvidia)
  total_use_time_actual[cur_profiler_step] = dict(cur_total_use_time_actual)

  final_result[filepath] = {}
  for k in total_time.keys():
    final_result[filepath][k] = {
        "total_time": total_time[k],
        "nvidia_time": total_use_time_nvidia[k],
        "actual_time": total_use_time_actual[k]
        }
    for i in total_use_time_nvidia[k].keys():
      print(filepath + " " + str(i))
      print("Nvidia:" + str(total_use_time_nvidia[k][i]/ total_time[k] * 100))
      print("Actual:" + str(total_use_time_actual[k][i]/ total_time[k] * 100))
  #  pprint.pprint(final_result[filepath])

#  pprint.pprint(final_result)
with open("results.json", "w") as f:
  json.dump(final_result, f)

