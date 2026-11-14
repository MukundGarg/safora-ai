import argparse,json,time,resource
from pathlib import Path
import numpy as np
from safora.audio.stream import Yamnet
p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--iterations',type=int,default=30);a=p.parse_args()
m=Yamnet(a.model);x=np.zeros(m.window_samples,np.float32);m.predict(x);times=[]
for _ in range(a.iterations):
    start=time.perf_counter();m.predict(x);times.append(1000*(time.perf_counter()-start))
print(json.dumps({'input':'synthetic silence; no accuracy evaluation','model_bytes':Path(a.model).stat().st_size,'iterations':len(times),'p50_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'host_only_not_android':True},indent=2))
