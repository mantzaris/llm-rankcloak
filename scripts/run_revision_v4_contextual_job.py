"""Serialize CUDA jobs and durably charge all wall time against the six-hour replacement-study cap."""
import argparse
import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from rankcloak.revision_v4_stage2_common import ROOT, GPU, atomic_json, read_json, digest
from rankcloak.revision_v4_contextual_coherence import OUT

CAP = 6 * 3600


def compute_processes():
    text = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    return [(int(parts[1]), parts[2]) for line in text.splitlines() if len(parts := [v.strip() for v in line.split(',')]) == 3 and parts[0] == GPU]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',required=True); ap.add_argument('--forecast-seconds',type=float,required=True)
    ap.add_argument('--max-seconds',type=float,required=True); ap.add_argument('command',nargs=argparse.REMAINDER)
    a=ap.parse_args(); command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command or a.forecast_seconds<=0 or a.max_seconds<a.forecast_seconds: raise ValueError('invalid admission')
    folder=OUT/'gpu'; folder.mkdir(parents=True,exist_ok=True)
    with (folder/'stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=folder/'ledger.json'
        ledger=read_json(path) if path.exists() else {'ceiling_seconds':CAP,'jobs':[]}
        if ledger['ceiling_seconds'] != CAP: raise ValueError('budget contract changed')
        for old in ledger['jobs']:
            if old['status']=='running':
                # A lost supervisor is conservatively charged its complete reservation.
                old.update(status='interrupted_supervisor_full_reservation_charged',charged_seconds=old['hard_limit_seconds'])
        used=sum(j['charged_seconds'] for j in ledger['jobs'])
        atomic_json(path,ledger)
        if compute_processes(): raise RuntimeError('GPU busy. No job started.')
        remaining=CAP-used
        if a.forecast_seconds>remaining or a.max_seconds>remaining: raise RuntimeError('forecast or reservation exceeds remaining GPU budget')
        started=time.time(); monotonic=time.monotonic()
        job={'name':a.name,'command':command,'command_sha256':digest(command),'started_epoch':started,
             'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'forecast_seconds':a.forecast_seconds,
             'hard_limit_seconds':a.max_seconds,'charged_seconds':0,'status':'running','gpu_uuid':GPU,'peak_process_memory_mib':0}
        ledger['jobs'].append(job); atomic_json(path,ledger)
        env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                 RANKCLOAK_CONTEXTUAL_GPU_JOB=a.name,CUBLAS_WORKSPACE_CONFIG=':4096:8')
        soft_sent=False; reason=None
        with (folder/(a.name+'.stdout.txt')).open('a') as stdout, (folder/(a.name+'.stderr.txt')).open('a') as stderr:
            try:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                job['pid']=child.pid; atomic_json(path,ledger)
                while child.poll() is None:
                    elapsed=time.monotonic()-monotonic
                    procs=compute_processes()
                    for pid,mem in procs:
                        if pid==child.pid and mem.isdigit():job['peak_process_memory_mib']=max(job['peak_process_memory_mib'],int(mem))
                    if any(pid!=child.pid for pid,_ in procs):reason='unrelated_gpu_job_appeared'
                    if elapsed>=a.max_seconds-60:reason=reason or 'reservation_checkpoint_limit'
                    if reason and not soft_sent:
                        os.killpg(child.pid,signal.SIGUSR1);soft_sent=True;job['checkpoint_requested_reason']=reason
                    if elapsed>=a.max_seconds-10:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    job['charged_seconds']=elapsed;job['heartbeat_epoch']=time.time();atomic_json(path,ledger)
                    time.sleep(2)
                job['exit_code']=child.returncode
                job['status']='completed' if child.returncode==0 else 'checkpointed' if child.returncode==75 else 'failed'
            except BaseException as exc:
                job['status']='supervisor_failed';job['error']=repr(exc)
                if 'child' in locals() and child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                raise
            finally:
                job['charged_seconds']=time.monotonic()-monotonic
                job['finished_epoch']=time.time();atomic_json(path,ledger)
        print({'job':a.name,'status':job['status'],'wall_seconds':job['charged_seconds'],'cumulative_seconds':sum(j['charged_seconds'] for j in ledger['jobs'])},flush=True)
        return 0 if job['status'] in ['completed','checkpointed'] else job.get('exit_code',1)


if __name__=='__main__':sys.exit(main())
