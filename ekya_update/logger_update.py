import time, os
from collections import defaultdict
from ekya_update.common import append_to_csv, get_kst_as_string


class LoggerSubstitution:
    def __init__(self, stream_max_len=1000, stream_max_time=30, base_dir=None):
        self.stream_max_time = stream_max_time
        self.stream_max_len = stream_max_len
        self.base_dir = os.path.join(base_dir, "logger_generated")
        
        # Rename previous results
        if os.path.exists(self.base_dir):
            prev_dir_new_path = f"{self.base_dir}_{get_kst_as_string()}"
            os.rename(self.base_dir, prev_dir_new_path)

        # New container for results
        self.log_dicts = defaultdict(lambda: defaultdict(list))
        self.log_dicts_stream_len = defaultdict(int)
        self.stream.previous_global_log_time = time.time()

    def append(self, stream_id, item):
        # 1) Append Logic
        for idx, (key, values) in enumerate(item.items()):
            self.log_dicts[stream_id][key].extend(values)
            if idx == 0:
                self.log_dicts_stream_len[stream_id] += len(values)

        # 2) Flush Logic
        stream_ids_to_flush = set()
        
        # 2-1) Flush based on time (everything)
        current_time = time.time()
        if current_time - self.stream.previous_global_log_time > self.stream_max_time:
            for stream_id, log_items in self.log_dicts.items():
                for key, values in log_items.items():
                    if len(values) > 0:
                        stream_ids_to_flush.add(stream_id)
                        break
        self.stream.previous_global_log_time = current_time

        # 2-2) Flush current stream (current stream)
        if self.log_dicts_stream_len[stream_id] > self.stream_max_len:
            stream_ids_to_flush.add(stream_id)
        
        # 2-3) Actual Flushing Process
        self.flush(stream_ids_to_flush=stream_ids_to_flush)

    def flush(self, stream_ids_to_flush=None):
        # initialize stream_ids_to_flush as all the keys if not given
        if not stream_ids_to_flush:
            stream_ids_to_flush=set(self.log_dicts.keys())
        
        for stream_id in stream_ids_to_flush:
            # Append process
            log_path = os.path.join(self.base_dir, stream_id)
            append_to_csv(self.log_dicts[stream_id], log_path)
            
            # Emptying the queue process
            for key in self.log_dicts[stream_id].keys():
                self.log_dicts[stream_id][key] = []
            self.log_dicts_stream_len[stream_id] = 0
        
    def __del__(self):
        self.flush()
