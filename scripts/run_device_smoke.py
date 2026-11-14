"""Run the compiled instrumented suite on an authorized TCP ADB test device.
Uses the standard ADB protocol directly, allowing writable-directory keys/logs.
Never runs implicitly as part of build. Install adb-shell in a separate dev env.
For USB devices, use normal `adb install` / `am instrument` instead.
"""
import argparse,json,time,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',type=int,default=5555)
p.add_argument('--apk',type=Path,required=True);p.add_argument('--test-apk',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--skip-install',action='store_true');a=p.parse_args()
from adb_shell.adb_device import AdbDeviceTcp
start=time.time();a.output.mkdir(parents=True,exist_ok=True)
artifacts={path.name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in [a.apk,a.test_apk]}
(a.output/'installed-artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
d=AdbDeviceTcp(a.host,a.port,default_transport_timeout_s=600)
try:
    d.connect(auth_timeout_s=15)
    info={name:d.shell('getprop '+prop).strip() for name,prop in {
        'sdk':'ro.build.version.sdk','release':'ro.build.version.release','abi':'ro.product.cpu.abi',
        'model':'ro.product.model','qemu':'ro.kernel.qemu','boot_completed':'sys.boot_completed'}.items()}
    (a.output/'device-info.json').write_text(json.dumps(info,indent=2)+'\n')
    if info['boot_completed']!='1':raise RuntimeError('Android boot is not complete; no tests executed')
    if not a.skip_install:
        for path,remote in [(a.apk,'/data/local/tmp/safora-demo.apk'),(a.test_apk,'/data/local/tmp/safora-test.apk')]:
            print('Pushing',path.name,flush=True);d.push(str(path),remote,read_timeout_s=600)
            try:
                result=d.shell('pm install -r -t '+remote,read_timeout_s=600)
                print(result,flush=True)
                if 'Success' not in result:raise RuntimeError('APK installation failed: '+result)
            finally:d.shell('rm -f '+remote)
    with (a.output/'instrumentation.txt').open('w') as log:
        for chunk in d.streaming_shell('am instrument -w -r org.safora.demo.test/androidx.test.runner.AndroidJUnitRunner',read_timeout_s=600):
            log.write(chunk);log.flush();print(chunk,end='',flush=True)
    result=(a.output/'instrumentation.txt').read_text()
    passed='OK (4 tests)' in result and 'FAILURES!!!' not in result and 'INSTRUMENTATION_FAILED' not in result
    (a.output/'result.json').write_text(json.dumps({'instrumentation_passed':passed,'expected_tests':4,'elapsed_seconds':time.time()-start,'physical_device':info['qemu']!='1','workload':'synthetic fixtures, native assets, manual notification, virtual/live AudioRecord capture; not accuracy evaluation'},indent=2)+'\n')
    if not passed:raise SystemExit(1)
finally:d.close()
