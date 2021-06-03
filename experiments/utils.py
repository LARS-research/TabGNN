import base64
import os
import pickle
import subprocess
import sys
from pprint import pprint
import pdb
import boto3
from __init__ import project_root
from start_training import main as main_training

local_docker_image_id = '<the_id_of_the_docker_image_that_you_build_from_docker/whole_project/Dockerfile>'


def run_script_with_kwargs(script_name, kwargs, session_name, locale='local_tmux', n_gpu=0, n_cpu=1):
    enc_kwargs = base64.b64encode(pickle.dumps(kwargs)).decode()
    container_cmd = f'source activate RDB; \
                      export CUDA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES; \
                      mkdir RDB_code; \
                      sudo mount -o remount,size=100G /dev/shm ;\
                      sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport <your_AWS_EFS_name>:/ ./RDB_code ; \
                      cd RDB_code; \
                      python -m {script_name} {enc_kwargs};'
    if locale == 'local_tmux':
                
        #cmd="tmux new-session -d -s {}".format(session_name)
        #subprocess.run(cmd, shell=True)

        kwargs_filename=os.path.join(project_root, 'experiments/%s_args.pkl'%(session_name))
        pickle.dump(kwargs,open(kwargs_filename,'wb'))

        #subprocess.run("tmux send-keys -t {} 'source activate /home/quanyuhan/rdb2graph_env' C-m ".format(session_name), shell=True, env=os.environ.copy())

        if session_name[-1]=='n':
            gpu_id=str(n_gpu)
        else:
            gpu_id=int(session_name[-1])+2
        #cmd="tmux send-keys -t {} 'sudo {} -m {} {}' C-m ".format(session_name,
        cmd="CUDA_VISIBLE_DEVICES={} python -m {} {}".format(gpu_id, script_name, kwargs_filename)

        subprocess.run(cmd, shell=True, env=os.environ.copy())
        
        #cmd="tmux send-keys -t {} 'exit' C-m ".format(session_name)
        #subprocess.run(cmd, shell=True)
        

    elif locale == 'local_docker':
        subprocess.run("tmux new-session -d -s {}".format(session_name), shell=True)
        subprocess.run(
            "tmux send-keys -t {} 'nvidia-docker run --privileged {} /bin/bash -c \" {} \" ' C-m ".format(session_name,
                                                                                                          local_docker_image_id,
                                                                                                          container_cmd),
            shell=True,
            env=os.environ.copy())
    
    elif locale == 'no_tmux':
        kwargs_filename=os.path.join(project_root, 'experiments/%s_args.pkl'%(session_name))
        pickle.dump(kwargs,open(kwargs_filename,'wb'))

        if session_name[-1]=='n':
            gpu_id=str(n_gpu)
        else:
            gpu_id=int(session_name[-1])+2
        cmd="CUDA_VISIBLE_DEVICES={} python -m {} {} ".format(gpu_id,
                                                                  #sys.executable,
                                                                  script_name,
                                                                  kwargs_filename)
        #pdb.set_trace()
        subprocess.run(cmd, shell=True, env=os.environ.copy())

    else:
        raise ValueError('locale not recognized')
