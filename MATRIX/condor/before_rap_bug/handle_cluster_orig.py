#{{{ imports
import os
import sys
import subprocess
import glob
import time
import signal
import copy
import stat
import getpass
from getpass import getuser
from os.path import join as pjoin
# own modules
from initialize_classes import out, fold
#}}}

#{{{ def: make_executable
def make_executable(file_name):
    st = os.stat(file_name)
    os.chmod(file_name, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
#}}}
#{{{ class: cluster_basic(object)
class cluster_basic(object): # basic class that handles all cluster types
    cluster_name = "basic cluster class"
#{{{ def: __init__(self)
    def __init__(self,config_list,verbose):
        self.config_list = config_list
        self.verbose = verbose
        self.command_kill_job = "*not set*"
        self.command_list_queue = "*not set*"
        self.command_list_queue_with_jobid = "*not set*"
        self.command_list_queue_current_user = "*not set*"
        self.job_status_list = ["*not set*","*not set*","*not set*"]  # structure: [pending,running,(held)]
#}}}
# cluster dependend functions:
#{{{ def: get_list_queue_command_with_multiple_ids(self,job_ids)
    def get_list_queue_command_with_multiple_ids(self,job_ids):
    # this routine returns the cluster-specific command needded when listing jobs with a number of job ids
        out.print_error("Function get_list_queue_command_with_multiple_ids in cluster class for cluster \"%s\" not implemented." % self.cluster_name)
#}}}
#{{{ def: create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path)
    def create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path):
    # this routine creates the cluster-specific batch file for the cluster submit
        out.print_error("Function create_batch_file in cluster class for cluster \"%s\" not implemented." % self.cluster_name)
#}}}
#{{{ def: submit_to_cluster(self,batch_file_name)
    def submit_to_cluster(self,batch_file_name):
    # this routine executes the cluster-specific submit to the cluster
        out.print_error("Function submit_to_cluster in cluster class for cluster \"%s\" not implemented." % self.cluster_name)
#}}}
#{{{ def: get_job_id_from_content_of_submit_output(self,content)
    def get_job_id_from_content_of_submit_output(self,content):
    # this routine extracts the job ID from content, which is the saved on-screen output of the job submit on the cluster
        out.print_error("Function get_job_id_from_content_of_submit_output in cluster class for cluster \"%s\" not implemented." % self.cluster_name)
#}}}
# functions that are needed only for certain clusters:
#{{{ def: create_cd_script(self)
    def create_cd_script(self,job_name,file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path):
    # this routine creates a cd file for clusters, where you cannot change the initial directory via the submission file
        return
#}}}
#{{{ def: create_local_run_script(self)
    def create_local_batch_and_run_script(self,job_name,batch_file_name,file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path,tarball_path):
    # this routine creates a cd file for clusters, where you cannot change the initial directory via the submission file
        local_run_script = open(file_name, 'w')
        scratch_path = self.config_list["cluster_local_scratch_path"]
        scratch_path_variable = "$scratch_path"
        executable_path = pjoin(scratch_path_variable,"bin",os.path.basename(fold.exe_path))
        process_name = os.path.basename(fold.exe_path)
        psg_files = glob.glob(pjoin(os.path.dirname(os.path.dirname(os.path.dirname(fold.run_folder_path))),"psg",process_name,"*.dat"))
        if not psg_files:
            out.print_error("No phase-space files found inside psg/"+process_name+"/, which need to be copied for local run mode.")
        tarball = os.path.basename(tarball_path)
        infile_path = pjoin(scratch_path_variable,os.path.relpath(subprocess_in_path,os.path.dirname(os.path.dirname(fold.run_folder_path))))
        outfile_path = pjoin(scratch_path_variable,os.path.relpath(subprocess_out_path,os.path.dirname(os.path.dirname(fold.run_folder_path))))
        lockfile = "."+os.path.basename(os.path.dirname(fold.run_folder_path))+"-"+os.path.basename(fold.run_folder_path)+".lock"
        run_path_local = pjoin(scratch_path_variable,os.path.relpath(run_path,os.path.dirname(os.path.dirname(fold.run_folder_path))))
        contribution = os.path.basename(os.path.dirname(run_path))
        rundir = os.path.basename(run_path)

        # create batch file with local inputs/settings
        sav = fold.exe_path
        fold.exe_path = executable_path # use sh script as executable, set it back below to previous value
        self.create_batch_file(job_name,batch_file_name,run_path_local,process,subprocess_in_path,subprocess_out_path,subprocess_err_path)
        fold.exe_path = sav

        try:
            local_run_script.write("#!/bin/bash\n")
            local_run_script.write("# Script executed at beginning of MATRIX job submit to run locally on the cluster nodes.\n\n")
            local_run_script.write("# First create/go to scratch and copy executable, unless exists.\n")
            local_run_script.write("scratch_path="+scratch_path+"\n")
            local_run_script.write("if [ ! -d "+scratch_path_variable+" ]; then\n")
            local_run_script.write("    mkdir -p "+scratch_path_variable+"\n")
            local_run_script.write("fi\n")
            local_run_script.write("cd "+scratch_path_variable+"\n")
            local_run_script.write("if [ ! -d bin ]; then\n")
            local_run_script.write("    mkdir -p bin/\n")
            local_run_script.write("fi\n")
            local_run_script.write("cmp --silent "+executable_path+" "+fold.exe_path+" || rsync --progress -avu "+fold.exe_path+" bin/\n")
            local_run_script.write("# Then create psg folder if not exist and copy all phase-space files to there.\n")
            local_run_script.write("if [ ! -d psg/"+process_name+"/ ]; then\n")
            local_run_script.write("    mkdir -p psg/"+process_name+"/\n")
            local_run_script.write("fi\n")
            for psg_file in psg_files:
                psg_file_local_path = pjoin(scratch_path_variable,"psg",process_name,os.path.basename(psg_file))
                local_run_script.write("cmp --silent "+psg_file_local_path+" "+psg_file+" || rsync --progress -avu "+psg_file+" psg/"+process_name+"/\n")
            local_run_script.write("# Then copy and unpack tarball with folder structure for run by only the first job using a lock-file, if the designated input file does not exist.\n")
            local_run_script.write("(\n")
            local_run_script.write("    flock -e 42\n")
            local_run_script.write("    if [ ! -f "+infile_path+" ]; then\n")
            local_run_script.write("        cmp --silent "+tarball+" "+tarball_path+" || rsync --progress -avu "+tarball_path+" .\n\n")
            local_run_script.write("        tar xfz "+tarball+" --keep-newer-files\n")
            local_run_script.write("    fi\n\n")
            local_run_script.write(") 42> %s\n\n" % pjoin(scratch_path_variable,lockfile))
            local_run_script.write("# Now go to unpacked or existing run_folder and start the run locally.\n")
            local_run_script.write("cd "+run_path_local+"\n")
            local_run_script.write(executable_path+" < "+infile_path+"\n")#" > "+outfile_path+"\n")
            local_run_script.write("sleep 10\n\n")
            local_run_script.write("# Finally tar the results and copy them back.\n")
            local_run_script.write("tar cfz results_"+contribution+"_"+rundir+"_"+process+".tar */*"+process+"* */*/*"+process+"* */*/*/*"+process+"*\n")
            local_run_script.write("rsync -av results_"+contribution+"_"+rundir+"_"+process+".tar "+run_path+"\n")
        finally:
            local_run_script.close()
            make_executable(file_name)
#}}}
#{{{ def: get_list_command_for_single_job_id(self,job_id)
    def get_list_command_for_single_job_id(self,job_id):
    # function only needed for clusters where single job_id request has a unexpected behaviour (like printing tons of information), for example SGE
        command = copy.copy(self.command_list_queue_with_jobid)
        command.append(job_id)
        return command
#}}}
#{{{ def: modify_request_by_grepping_job_ids(self,request,job_ids)
    def modify_request_by_grepping_job_ids(self,request,job_ids):
    # function only needed for clusters which do not allow for job-status request with several job ids, for example SGE
        return request
#}}}
# hardly (single command) cluster dependend functions:
#{{{ def: get_jobs_in_cluster_queue(self)
    def get_jobs_in_cluster_queue(self):
    # this function returns the number of jobs that are currently in the cluster queue
        if not self.command_list_queue:
            out.print_error("Cluster command command_list_queue for function get_jobs_in_cluster_queue in cluster class for cluster \"%s\" not implemented." % self.cluster_name)
        command_list_queue = self.command_list_queue

        try:
            p1 = subprocess.Popen(command_list_queue, stdout=subprocess.PIPE, text=True)
            p2 = subprocess.Popen(["grep", "Total for query"], stdin=p1.stdout, stdout=subprocess.PIPE, text=True)
            p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits
            p3 = subprocess.Popen(["awk", "{print $4}"], stdin=p2.stdout, stdout=subprocess.PIPE, text=True)
            p2.stdout.close()
            output, error = p3.communicate()
            nr_jobs_in_cluster = int(output.strip())
        except:
            out.print_error("The command \"%s\" does not appear to work on this cluster. Did you choose the correct cluster in MATRIX_configuration? Exiting..." % self.command_list_queue)

        return nr_jobs_in_cluster
#}}}
#{{{ def: cluster_job_remove(self,job_id)
    def cluster_job_remove(self,job_id):
    # this routine removes a job that is currently running
        try:
            remove = subprocess.Popen([self.command_kill_job,job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
        except:
            try:
                remove = subprocess.Popen([self.command_kill_job,job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
            except:
                try:
                    remove = subprocess.Popen([self.command_kill_job,job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
                except:
                    out.print_error("The command \"%s\" does not appear to work on this cluster. Did you choose the correct cluster in MATRIX_configuration? Exiting..." % self.command_kill_job)
        # remove jobs from active job folder and write it to failed job list
        self.remove_job_from_folder(job_id,"active_jobs") # remove job from active jobs
        self.add_job_id_to_list(job_id,"job_ids_finished.list") # add job to finished job list
#}}}
#{{{ def: terminate_cluster_runs(self)
    def terminate_cluster_runs(self):
        # this function kills all cluster jobs that have been started during this run (in job_ids_started.list)
        job_ids = self.get_job_ids("started")
        for job_id in job_ids:
            try:
                terminate = subprocess.Popen([self.command_kill_job,job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
            except:
                try:
                    terminate = subprocess.Popen([self.command_kill_job,job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
                except:
                    try:
                        terminate = subprocess.Popen([self.command_kill_job,job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
                    except:
                        out.print_error("The command \"%s\" does not appear to work on this cluster. Did you choose the correct cluster in MATRIX_configuration? Exiting..." % self.command_kill_job)
        # 2do: check wether any one of these jobs on the cluster are still running
#}}}
#{{{ def: get_nr_of_cluster_jobs(self,status)
    def get_nr_of_cluster_jobs(self,status):
    # this routine makes a bash request to the cluster and extracts the number of jobs 
    # with ids from the array job_ids (created inside the routine) with the given status
        if not status in ["pending","running"]:
            out.print_error("Status of cluster jobs in get_nr_of_cluster_jobs is \""+status+"\", but can only be \"pending\" or \"running\"")
        job_ids = self.get_job_ids("started") # returns an array of job ids
        if not job_ids: return 0

        command = self.get_list_queue_command_with_multiple_ids(job_ids)
        try:
            request_tmp = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
            request     = self.modify_request_by_grepping_job_ids(request_tmp,job_ids)  # only effective for certain clusters, like SGE
        except:
            request = ""
            out.print_warning("Could not proceed command \"%s\", retrying later..." % " ".join(command))
        if status == "pending":
            nr_of_jobs = request.count(self.job_status_list[0])+request.count(self.job_status_list[2])+request.count(self.job_status_list[3]) # added queued, held, and suspended, as those jobs will/might continue
        elif status == "running":
            nr_of_jobs = request.count(self.job_status_list[1])
        else:
            nr_of_jobs = 0
        return nr_of_jobs
#}}}
#{{{ def: cluster_job_submit(self,subprocess_in_path,subprocess_out_path,subprocess_err_path)
    def cluster_job_submit(self,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path):
    # this routine submits the job with stdin (subprocess_in_path), stdout (subprocess_out_path) and stderr (subprocess_err_path) 
    # to the cluster, saves the corresponding batch file, writes the job_id returned at submit to a file and returns the job_id
        job_name = run_path.strip("/").replace("/","-")+process
        batch_file_name =pjoin(fold.run_folder_path,"cluster","batch_files",job_name+".batch")
        # standard submission in shared file system, always sued for result combination runs
        if int(self.config_list.get("cluster_local_run",0)) == 0 or run_path.startswith("infile.result"):
            self.create_batch_file(job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path)
            self.create_cd_script(job_name,batch_file_name.replace(".batch",".sh"),run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path) # usually does nothing, needed right now only for HTCondor on lxplus
        else: # local run using the scratch on the cluster nodes
            self.create_local_batch_and_run_script(job_name,batch_file_name,batch_file_name.replace(".batch",".sh"),run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path,fold.local_cluster_run_tarball_path) # usually does nothing, needed right now only for HTCondor on lxplus
        submit = self.submit_to_cluster(batch_file_name)
        while True: # waiting for job to be submitted
           if submit.poll() != None:
               break # stops while loop because job submitted
        if not os.path.isfile(batch_file_name+".out"):
            out.print_warning("File "+batch_file_name+".out"+" in function cluster_job_submit does not exist!")
            out.print_warning("Resubmitting job in 15 seconds...")
            time.sleep(15)
            return "resubmit"

        job_id = ""
        time_before = time.time()
#        while True: 
        with open(batch_file_name+".out","r") as out_file:# read job id from here!
                content = out_file.readlines()
                if not content:
                    out.print_warning("Cluster submit failed, since file "+batch_file_name+".out"+" appears to be empty. Cannot read out job_id.")
                    out.print_warning("Resubmitting job in 15 seconds...")
                    time.sleep(15)
                    return "resubmit"
                job_id = self.get_job_id_from_content_of_submit_output(content)
                if job_id == "resubmit":
                    return "resubmit"
            # time_now = time.time()
            # if time_now - time_before > 60:
            #     out.print_warning("Could not open "+batch_file_name+".out"+" to read out job_id within 1 minutes.")
            #     out.print_warning("Resubmitting job in 10 seconds...")
            #     time.sleep(10)
            #     return "resubmit"
            # if job_id:
            #     break
            # else:
            #     time.sleep(20)
        # all job ids are written to one file (later: write finished job file, running job file, aborted job file)
        self.add_job_id_to_list(job_id,"job_ids_started.list")
        # write same job ids to the folder with active jobs; we are using a folder for this, since jobs
        # need to be constantly added and removed, which leads to conflicts with many jobs accessing the same file
        self.add_job_to_folder(job_id,"active_jobs")

        if self.verbose: out.print_info("Job (ID %s, name %s) submitted" % (job_id,job_name)) # remove later
        return job_id
#}}}
#{{{ def: cluster_job_finished(self,job_id)
    def cluster_job_finished(self,job_id):
    # this routine determines wether a job has finished, returns True when finshed, False when still running
        return not self.cluster_job_in_state(job_id,0) and not self.cluster_job_in_state(job_id,1) and not self.cluster_job_in_state(job_id,2)
#}}}
#{{{ def: cluster_job_running(self,job_id)
    def cluster_job_running(self,job_id):
    # this routine determines wether a job is running, returns True when running, False when pending or finished
        return self.cluster_job_in_state(job_id,1) # state=1 ^= "running"
#}}}
#{{{ def: cluster_job_in_state(self,job_id,state)
    def cluster_job_in_state(self,job_id,state):
    # this routine determines wether a job is in a certain state, returns True when in state, False when not
        command = self.get_list_command_for_single_job_id(job_id)
        try:
            request_tmp = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
            request     = self.modify_request_by_grepping_job_ids(request_tmp,[job_id])  # only effective for certain clusters, like SGE
        except:
            request = ""
            out.print_warning("Could not proceed command \"%s\", retrying later..." % " ".join(command))
        nr_of_jobs = request.count(self.job_status_list[state]) # self.job_status_list[0] ^= pending, # self.job_status_list[1] ^= running
        if nr_of_jobs==0:
            return False
        elif nr_of_jobs==1:
            return True
        else:
            out.print_error("Counting the jobs with running status (\"%s\") from command %s on %s cluster with single job_id in cluster_job_running did not return 0 or 1." % (self.job_status_list[state],command,self.cluster_name))
#}}}
# not clusted dependend functions:
#{{{ def: get_job_ids(self,status)
    def get_job_ids(self,status):
    # returns an array of job ids from the list with status
    # problem: active jobs are in a folder, so get them by generating a list from the folder content
        job_ids = []
        if status == "pending" or status == "running": # all these things are actually not needed/used currently, only for status="started" we use this function
            this_status = "active"
            self.create_list_from_folder("active_jobs")
        else:
            this_status = status
        job_ids_list = pjoin(fold.run_folder_path,"cluster","job_ids_"+this_status+".list")
        time_before = time.time()
        while True:
            time_now = time.time()
            if os.path.isfile(job_ids_list):
                with open(job_ids_list, 'r') as in_file:
                    for in_line in in_file:
                        line=in_line.strip() # strip removes all spaces (including tabs and newlines)
                        if not in_line == "":
                            job_ids.append(in_line.strip())
                break
            elif time_now - time_before >= 10:
                if this_status == "started":
#                    out.print_info("Could not open job id file \""+job_ids_list+"\" in order to extract the \""+this_status+"\" job ids in function get_job_ids within 10 seconds. Generating empty list...")
                    open(job_ids_list, 'w').close()
                else:
                    out.print_error("Could not open job id file \""+job_ids_list+"\" in order to extract the \""+this_status+"\" job ids in function get_job_ids within 10 seconds.")
            time.sleep(1)
        return job_ids
#}}}
#{{{ def: add_job_id_to_list(self,job_id,job_id_list)
    def add_job_id_to_list(self,job_id,job_ids_list):
    # this routine remove the job_id from the file job_list
        job_ids_list = pjoin(fold.run_folder_path,"cluster",job_ids_list)
        if os.path.isfile(job_ids_list):
            job_ids_file = open(job_ids_list, 'a') 
        else:
            job_ids_file = open(job_ids_list, 'w') 
        job_ids_file.write(job_id+"\n")
        job_ids_file.close()
#}}}
#{{{ def: add_job_to_folder(self,job_id,folder)
    def add_job_to_folder(self,job_id,folder): # not cluster dependend
    # this routine adds the job_id to the folder as a file with job_id as name
        job_id_file_name = pjoin(fold.run_folder_path,"cluster",folder,job_id)
        if os.path.exists(job_id_file_name):
            out.print_error("File "+job_id_file_name+" already exists, when trying to create it in function add_job_to_folder.")
        with open(job_id_file_name, 'w') as job_id_file:
            job_id_file.write("added")
#}}}
#{{{ def: remove_job_from_folder(self,job_id,folder)
    def remove_job_from_folder(self,job_id,folder): # not cluster dependend
    # this routine removes the job_id from the folder as a file with job_id as name
        job_id_file_name = pjoin(fold.run_folder_path,"cluster",folder,job_id)
        if os.path.exists(job_id_file_name):
            os.remove(job_id_file_name)
        else:
            out.print_warning("File "+job_id_file_name+" does not exist when trying to remove it in function remove_job_from_folder.")
#}}}
#{{{ def: create_list_from_folder(self,folder)
    def create_list_from_folder(self,folder):
        folder_path = pjoin(fold.run_folder_path,"cluster","active_jobs")
        onlyfiles = [ f for f in os.listdir(folder_path) if os.path.isfile(pjoin(folder_path,f)) ]
        job_ids_list = pjoin(fold.run_folder_path,"cluster","job_ids_"+folder.replace("_jobs","")+".list") # all folders should end with "_jobs"
        with open(job_ids_list, 'w') as job_id_file:
            for filename in onlyfiles:
                job_id_file.write(filename+"\n")
#}}}
# 2do: this one is dublicated and should be somewhere else:
#{{{ def: control_c_handler_child(self,signal,frame)
    def control_c_handler_child(self,signal,frame):
        # this handels the ctrl-c for the subprocesses
        sys.exit(0)
#}}}
#}}}
#{{{ class: slurm_cluster(cluster_basic)
class slurm_cluster(cluster_basic): # class for slurm cluster inherited from basic cluster class
    cluster_name = "slurm"
#{{{ def: __init__(self,config_list,verbose)
    def __init__(self,config_list,verbose):
        cluster_basic.__init__(self,config_list,verbose)
        self.command_kill_job = "scancel"
        try:
            self.command_list_queue = ["squeue","-p","%s" % self.config_list["cluster_queue"]]
        except:
            self.command_list_queue = ["squeue"]
        current_user = getpass.getuser()
        self.command_list_queue_current_user = ["squeue","-u",current_user]
        self.command_list_queue_with_jobid = ["squeue","-j"]
#Job state, compact form: PD (pending), R (running), CA (cancelled), CF(configuring), CG (completing), CD (completed), F (failed), TO (timeout), NF (node failure) and SE (special exit state).
        self.job_status_list = [" PD "," R "," *not set* "," *not set* "]  # structure: [pending,running,(held)]
#}}}
#{{{ def: get_list_queue_command_with_multiple_ids(self,job_ids)
    def get_list_queue_command_with_multiple_ids(self,job_ids):
    # this routine returns the cluster-specific command needded when listing jobs with a number of job ids
        all_ids = ""
        for id in job_ids:
            all_ids = all_ids+","+str(id).strip()
        all_ids = all_ids[1:]
        return ["squeue","-j",all_ids]
#}}}
#{{{ def: create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path)
    def create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path):
    # this routine creates the cluster-specific batch file for the cluster submit
        batch_file = open(batch_file_name, 'w')
        try:
            batch_file.write("#!/bin/bash\n")
#            batch_file.write("#SBATCH --exclude=compute010,compute030\n")
            batch_file.write("#SBATCH -J "+job_name+"\n")
            if not run_path.startswith("infile.result"): # for normal runs
                batch_file.write("#SBATCH -i "+subprocess_in_path+"\n")
                batch_file.write("#SBATCH -o "+subprocess_out_path+"\n")
                batch_file.write("#SBATCH -e "+subprocess_err_path+"\n")
            if "cluster_runtime" in self.config_list:
                batch_file.write("#SBATCH -t "+self.config_list["cluster_runtime"]+"\n")
            batch_file.write("#SBATCH -n 1\n")
            if "cluster_queue" in self.config_list:
                batch_file.write("#SBATCH -p "+self.config_list["cluster_queue"]+"\n")
            for submit_line_key in  sorted([key for key in self.config_list if key.startswith("cluster_submit_line")]):
                batch_file.write(self.config_list[submit_line_key]+"\n")
            if run_path.startswith("infile.result"): # for result combination run
                results_type = subprocess_in_path # results typ for result run was saved here
                batch_file.write("srun "+fold.exe_path+" "+results_type+" "+run_path+"\n")
            else: # for normal runs
                if int(self.config_list.get("cluster_local_run",0)) == 0:
                    batch_file.write("srun "+fold.exe_path+"\n")
                else:
                    batch_file.write("srun "+batch_file_name.replace(".batch",".sh")+"\n")
            batch_file.write("wait\n")
# if you want to specify which nodes should be user or exclude certain nodes use:
#            batch_file.write("#SBATCH --exclude=compute00[1-8]\n")
        finally:
            batch_file.close()
#}}}
#{{{ def: submit_to_cluster(self,batch_file_name)
    def submit_to_cluster(self,batch_file_name):
    # this routine executes the cluster-specific submit to the cluster
        with open(batch_file_name+".out",'w') as batch_file_out:
            submit = subprocess.Popen(["sbatch",batch_file_name],stdout=batch_file_out)
        return submit
#}}}
#{{{ def: get_job_id_from_content_of_submit_output(self,content)
    def get_job_id_from_content_of_submit_output(self,content):
    # this routine extracts the job ID from content, which is the saved on-screen output of the job submit on the cluster
        job_id = ""
        for in_line in content:
            line=in_line.strip() # strip removes all spaces (including tabs and newlines)
            if line.split()[2].strip() == "job":
                try:
                    job_id = line.split()[3].strip()
                except:
                    job_id = ""
        if not job_id:
            out.print_warning("Could not read job id from fourth word in returned output.")
            out.print_warning("Resubmitting job in 15 seconds...")
            time.sleep(15)
            return "resubmit"
        return job_id
#}}}
#}}}
#{{{ class: condor_cluster(cluster_basic)
class condor_cluster(cluster_basic): # class for condor cluster inherited from basic cluster class
    cluster_name = "condor"
#{{{ def: __init__(self,config_list,verbose)
    def __init__(self,config_list,verbose):
        cluster_basic.__init__(self,config_list,verbose)
        self.command_kill_job = "condor_rm"
        self.command_list_queue = ["condor_q"]
        self.command_list_queue_with_jobid = ["condor_q"]
        self.job_status_list = [" I "," R "," H "," *not set* "]  # structure: [pending,running,(held)]
#}}}
#{{{ def: get_list_queue_command_with_multiple_ids(self,job_ids)
    def get_list_queue_command_with_multiple_ids(self,job_ids):
    # this routine returns the cluster-specific command needded when listing jobs with a number of job ids
        command = ["condor_q"] 
        for id in job_ids:
            command.append(str(id).strip())
        return command
#}}}
#{{{ def: create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path)
    def create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path):
    # this routine creates the cluster-specific batch file for the cluster submit
        batch_file = open(batch_file_name, 'w')
        try:
            batch_file.write("# MATRIX job submit file for condor vanilla universe\n")
            batch_file.write("Universe   = vanilla\n")
            batch_file.write("Executable = "+fold.exe_path+"\n")
            if not run_path.startswith("infile.result"): # for normal runs
                batch_file.write("input      = "+subprocess_in_path+"\n")
                batch_file.write("log        = "+subprocess_out_path+"\n") # condor log in usual log file ???
                batch_file.write("output     = "+subprocess_out_path+"\n")
                batch_file.write("error      = "+subprocess_err_path+"\n")
            if "cluster_runtime" in self.config_list:
                out.print_warning("No runtime limit supported on condor cluster. Remove \"cluster_runtime\" from MATRIX_configuration to avoid this warning.")
            if "cluster_queue" in self.config_list:
                out.print_warning("No queue selection supported on condor cluster. Remove \"cluster_queue\" from MATRIX_configuration to avoid this warning.")
            if run_path.startswith("infile.result"): # for result combination run
                results_type = subprocess_in_path # results typ for result run was saved here
                batch_file.write("Arguments  = "+results_type+" "+run_path+"\n")
            batch_file.write("Initialdir = "+run_path+"\n")
            batch_file.write("remote_initialdir = "+run_path+"\n")
            batch_file.write("should_transfer_files = no\n")
            for submit_line_key in  sorted([key for key in self.config_list if key.startswith("cluster_submit_line")]):
                batch_file.write(self.config_list[submit_line_key]+"\n")
            batch_file.write("Queue\n")
        finally:
            batch_file.close()
#}}}
#{{{ def: submit_to_cluster(self,batch_file_name)
    def submit_to_cluster(self,batch_file_name):
    # this routine executes the cluster-specific submit to the cluster
        with open(batch_file_name+".out",'w') as batch_file_out:
            submit = subprocess.Popen(["condor_submit",batch_file_name],stdout=batch_file_out)
        return submit
#}}}
#{{{ def: get_job_id_from_content_of_submit_output(self,content)
    def get_job_id_from_content_of_submit_output(self,content):
    # this routine extracts the job ID from content, which is the saved on-screen output of the job submit on the cluster
        job_id = ""
        for in_line in content:
            line=in_line.strip() # strip removes all spaces (including tabs and newlines)
            if "job(s) submitted to cluster" in line:
                try:
                    job_id = line.split()[5].rstrip(".")
                except:
                    job_id = ""
        if not job_id:
            out.print_warning("Could not read job id from returned condor submit output.")
            out.print_warning("Resubmitting job in 15 seconds...")
            time.sleep(15)
            return "resubmit"
        return job_id
#}}}
#}}}
#{{{ class: condor_lxplus_cluster(cluster_basic)
class condor_lxplus_cluster(cluster_basic): # class for HTCondor cluster as used on the lxplus cluster inherited from basic cluster class
                                            # lxplus does not allow for initialdir, thus, use a extra script to cd into the folder to run from.
    cluster_name = "condor_lxplus"
#{{{ def: __init__(self,config_list,verbose)
    def __init__(self,config_list,verbose):
        cluster_basic.__init__(self,config_list,verbose)
        self.command_kill_job = "condor_rm"
#        self.command_list_queue = ["condor_q"]
#        self.command_list_queue_with_jobid = ["condor_q"]
#        self.job_status_list = [" I "," R "," H "]  # structure: [pending,running,(held)]
        self.command_list_queue = ["condor_q", "-af" ,"JobStatus"]
        self.command_list_queue_with_jobid = ["condor_q", "-af" ,"JobStatus"]
        self.job_status_list = ["1","2","5","7"]  # structure: [pending,running,(held)]
#}}}
#{{{ def: get_list_queue_command_with_multiple_ids(self,job_ids)
    def get_list_queue_command_with_multiple_ids(self,job_ids):
    # this routine returns the cluster-specific command needded when listing jobs with a number of job ids
        command = list(self.command_list_queue_with_jobid)
        command[1:1] = [str(id).strip() for id in job_ids]
        return command
#}}}
#{{{ def: get_list_command_for_single_job_id(self,job_id)
    def get_list_command_for_single_job_id(self,job_id):
    # function only needed for clusters where single job_id request has a unexpected behaviour (like printing tons of information), for example SGE
        command = copy.copy(self.command_list_queue_with_jobid)
        # The list of job IDs is required to come before additional arguments for condor cluster
        # I would normally expect the order to not matter, but it may be necessary to have use
        # command.append(job_id) for other clusters if they require the job ID at the end
        command.insert(1, job_id)
        return command
#}}}
#{{{ def: create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path)
    def create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path):
    # this routine creates the cluster-specific batch file for the cluster submit
        batch_file = open(batch_file_name, 'w')
        try:
            batch_file.write("# MATRIX job submit file for condor vanilla universe\n")
            batch_file.write("Universe   = vanilla\n")
            batch_file.write("Executable = "+batch_file_name.replace(".batch",".sh")+"\n")
            batch_file.write("output     = "+subprocess_out_path+"\n")#.replace(".log", ".logout")+"\n")
            batch_file.write("log        = "+subprocess_err_path.replace(".log", ".condor_log")+"\n")
            batch_file.write("error      = "+subprocess_err_path+"\n")#.replace(".log", ".err")+"\n")
            if "cluster_runtime" in self.config_list:
                out.print_warning("No runtime limit supported on condor cluster. Remove \"cluster_runtime\" from MATRIX_configuration to avoid this warning.")
            if "cluster_queue" in self.config_list:
                out.print_warning("No queue selection supported on condor cluster. Remove \"cluster_queue\" from MATRIX_configuration to avoid this warning.")
            if run_path.startswith("infile.result"): # for result combination run
                out.print_error("Result-combination run on the nodes not supported for cluster %s" % self.cluster_name)
            batch_file.write("should_transfer_files = no\n")
            batch_file.write("getenv = true\n")
            for submit_line_key in  sorted([key for key in self.config_list if key.startswith("cluster_submit_line")]):
                batch_file.write(self.config_list[submit_line_key]+"\n")
            batch_file.write("+JobFlavour = \"nextweek\"\n")
            batch_file.write("Queue\n")
        finally:
            batch_file.close()
#}}}
#{{{ def: create_cd_script(self,file_name)
    def create_cd_script(self,job_name,file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path):
    # this routine creates a cd file for clusters, where you cannot change the initial directory via the submission file
        cd_script = open(file_name, 'w')
        try:
            cd_script.write("#!/bin/bash\n")
            cd_script.write("# Script executed at beginning of MATRIX job submit to change directory to different folder and execute the code\n")
            cd_script.write("cd "+run_path+"\n")
            cd_script.write(fold.exe_path+" < "+subprocess_in_path+"\n")#" > "+subprocess_out_path+"\n")
        finally:
            cd_script.close()
            make_executable(file_name)
#}}}
#{{{ def: submit_to_cluster(self,batch_file_name)
    def submit_to_cluster(self,batch_file_name):
    # this routine executes the cluster-specific submit to the cluster
        with open(batch_file_name+".out",'w') as batch_file_out:
            submit = subprocess.Popen(["condor_submit",batch_file_name],stdout=batch_file_out)
        return submit
#}}}
#{{{ def: get_job_id_from_content_of_submit_output(self,content)
    def get_job_id_from_content_of_submit_output(self,content):
    # this routine extracts the job ID from content, which is the saved on-screen output of the job submit on the cluster
        job_id = ""
        for in_line in content:
            line=in_line.strip() # strip removes all spaces (including tabs and newlines)
            if "job(s) submitted to cluster" in line:
                try:
                    job_id = line.split()[5].rstrip(".")
                except:
                    job_id = ""
        if not job_id:
            out.print_warning("Could not read job id from sixth word in second line of returned output of submission.")
            out.print_warning("Resubmitting job in 15 seconds...")
            time.sleep(15)
            return "resubmit"
        return job_id
#}}}
#}}}
#{{{ class: HTcondor_cluster(cluster_basic)
class HTcondor_cluster(cluster_basic): # class for HTCondor cluster as used on the lxplus cluster inherited from basic cluster class
                                            # lxplus does not allow for initialdir, thus, use a extra script to cd into the folder to run from.
    cluster_name = "HTcondor"
#{{{ def: __init__(self,config_list,verbose)
    def __init__(self,config_list,verbose):
        cluster_basic.__init__(self,config_list,verbose)
        self.command_kill_job = "condor_rm"
#
#        self.job_status_list = [" I "," R "," H "]  # structure: [pending,running,(held)]
        self.current_user = getpass.getuser()
        self.command_list_queue = ["condor_q", self.current_user]

        self.command_list_queue_with_jobid = ["condor_q", "-af" ,"JobStatus"]
        self.job_status_list = ["1","2","5","7"]  # structure: [pending,running,(held),suspended]

        self.command_release_held_jobs = ["condor_release", self.current_user]
        self.command_continue_suspended_jobs = ["condor_continue", self.current_user]
#}}}
#{{{ def: cluster_job_finished(self,job_id)
    def cluster_job_finished(self,job_id):
    # this routine determines wether a job has finished, returns True when finshed, False when still running
    # Jaco: I don't want to automatically release held jobs
    #     if self.cluster_job_in_state(job_id,2):
    #         release_held = subprocess.Popen(self.command_release_held_jobs, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
    #         out.print_info("Releasing held condor jos using the command \"%s %s\"" % (self.command_release_held_jobs[0],self.command_release_held_jobs[1]))
        if self.cluster_job_in_state(job_id,3):
            continue_suspended = subprocess.Popen(self.command_continue_suspended_jobs, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode()
            out.print_info("Continuing suspended condor jos using the command \"%s %s\"" % (self.command_continue_suspended_jobs[0],self.command_continue_suspended_jobs[1]))
        return not self.cluster_job_in_state(job_id,0) and not self.cluster_job_in_state(job_id,1) and not self.cluster_job_in_state(job_id,2) and not self.cluster_job_in_state(job_id,3) # held jobs must be released, suspended jobs must be continued
#}}}
#{{{ def: get_list_queue_command_with_multiple_ids(self,job_ids)
    def get_list_queue_command_with_multiple_ids(self,job_ids):
    # this routine returns the cluster-specific command needded when listing jobs with a number of job ids
        command = list(self.command_list_queue_with_jobid)
        command[1:1] = [str(id).strip() for id in job_ids]
        return command
#}}}
#{{{ def: get_list_command_for_single_job_id(self,job_id)
    def get_list_command_for_single_job_id(self,job_id):
    # function only needed for clusters where single job_id request has a unexpected behaviour (like printing tons of information), for example SGE
        command = copy.copy(self.command_list_queue_with_jobid)
        # The list of job IDs is required to come before additional arguments for condor cluster
        # I would normally expect the order to not matter, but it may be necessary to have use
        # command.append(job_id) for other clusters if they require the job ID at the end
        command.insert(1, job_id)
        return command
#}}}
#{{{ def: create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path)
    def create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path):
    # this routine creates the cluster-specific batch file for the cluster submit
        batch_file = open(batch_file_name, 'w')
        try:
            batch_file.write("# MATRIX job submit file for condor vanilla universe\n")
            batch_file.write("Universe   = vanilla\n")
            batch_file.write("Executable = "+batch_file_name.replace(".batch",".sh")+"\n")
            batch_file.write("output     = "+subprocess_out_path+"\n")#.replace(".log", ".logout")+"\n")
            batch_file.write("log        = "+subprocess_err_path.replace(".log", ".condor_log")+"\n")
            batch_file.write("error      = "+subprocess_err_path+"\n")#.replace(".log", ".err")+"\n")
            if "cluster_runtime" in self.config_list:
                out.print_warning("No runtime limit supported on condor cluster. Remove \"cluster_runtime\" from MATRIX_configuration to avoid this warning.")
            if "cluster_queue" in self.config_list:
                out.print_warning("No queue selection supported on condor cluster. Remove \"cluster_queue\" from MATRIX_configuration to avoid this warning.")
            if run_path.startswith("infile.result"): # for result combination run
                out.print_error("Result-combination run on the nodes not supported for cluster %s" % self.cluster_name)
            batch_file.write("should_transfer_files = no\n")
            batch_file.write("getenv = true\n")
            for submit_line_key in  sorted([key for key in self.config_list if key.startswith("cluster_submit_line")]):
                batch_file.write(self.config_list[submit_line_key]+"\n")
            batch_file.write("+JobFlavour = \"nextweek\"\n")
            batch_file.write("Queue\n")
        finally:
            batch_file.close()
#}}}
#{{{ def: create_cd_script(self,file_name)
    def create_cd_script(self,job_name,file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path):
    # this routine creates a cd file for clusters, where you cannot change the initial directory via the submission file
        cd_script = open(file_name, 'w')
        try:
            cd_script.write("#!/bin/bash\n")
            cd_script.write("# Script executed at beginning of MATRIX job submit to change directory to different folder and execute the code\n")
            cd_script.write("cd "+run_path+"\n")
            cd_script.write(fold.exe_path+" < "+subprocess_in_path+"\n")#" > "+subprocess_out_path+"\n")
        finally:
            cd_script.close()
            make_executable(file_name)
#}}}
#{{{ def: submit_to_cluster(self,batch_file_name)
    def submit_to_cluster(self,batch_file_name):
    # this routine executes the cluster-specific submit to the cluster
        with open(batch_file_name+".out",'w') as batch_file_out:
            submit = subprocess.Popen(["condor_submit",batch_file_name],stdout=batch_file_out)
        return submit
#}}}
#{{{ def: get_job_id_from_content_of_submit_output(self,content)
    def get_job_id_from_content_of_submit_output(self,content):
    # this routine extracts the job ID from content, which is the saved on-screen output of the job submit on the cluster
        job_id = ""
        for in_line in content:
            line=in_line.strip() # strip removes all spaces (including tabs and newlines)
            if "job(s) submitted to cluster" in line:
                try:
                    job_id = line.split()[5].rstrip(".")
                except:
                    job_id = ""
        if not job_id:
            out.print_warning("Could not read job id from sixth word in second line of returned job output.")
            out.print_warning("Resubmitting job in 15 seconds...")
            time.sleep(15)
            return "resubmit"
        return job_id
#}}}
#}}}
#{{{ class: lsf_cluster(cluster_basic)
class lsf_cluster(cluster_basic): # class for LSF cluster inherited from basic cluster class
    cluster_name = "LSF"
#{{{ def: __init__(self,config_list,verbose)
    def __init__(self,config_list,verbose):
        cluster_basic.__init__(self,config_list,verbose)
        self.command_kill_job = "bkill"
        try:
            self.command_list_queue = ["bjobs","-q","%s" % self.config_list["cluster_queue"],"-u","all"]
        except:
            self.command_list_queue = ["bjobs","-u","all"]
        self.command_list_queue_with_jobid = ["bjobs"]
        self.job_status_list = [" PEND "," RUN "," *not set* "," *not set* "]  # structure: [pending,running,(held)]
#}}}
#{{{ def: get_list_queue_command_with_multiple_ids(self,job_ids)
    def get_list_queue_command_with_multiple_ids(self,job_ids):
    # this routine returns the cluster-specific command needded when listing jobs with a number of job ids
        command = ["bjobs"]
        for id in job_ids:
            command.append(str(id).strip())
        return command
#}}}
#{{{ def: create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path)
    def create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path):
    # this routine creates the cluster-specific batch file for the cluster submit
        batch_file = open(batch_file_name, 'w')
        try:
            batch_file.write("#!/bin/bash\n")
            batch_file.write("#BSUB -J "+job_name+"\n")
            if not run_path.startswith("infile.result"): # for normal runs
                batch_file.write("#BSUB -o "+subprocess_out_path+"\n")
                batch_file.write("#BSUB -e "+subprocess_err_path+"\n")
            if "cluster_runtime" in self.config_list:
                batch_file.write("#BSUB -t "+self.config_list["cluster_runtime"]+"\n")
            batch_file.write("#BSUB -n 1\n")
            if "cluster_queue" in self.config_list:
                batch_file.write("#BSUB -q "+self.config_list["cluster_queue"]+"\n")
            for submit_line_key in  sorted([key for key in self.config_list if key.startswith("cluster_submit_line")]):
                batch_file.write(self.config_list[submit_line_key]+"\n")
            if run_path.startswith("infile.result"): # for result combination run
                out.print_error("Result combination on cluster \"%s\" only working in local mode. Cannot be submitted to cluster. Exiting..." % self.cluster_name)
                # results_type = subprocess_in_path # results type (result/distribution) for result run was saved here
                # batch_file.write("srun "+fold.exe_path+" "+results_type+" "+run_path+"\n")
            else: # for normal runs
                # on LSF need to go to correct dir
                batch_file.write("cd %s\n" % run_path)
                if int(self.config_list.get("cluster_local_run",0)) == 0:
                    batch_file.write(fold.exe_path+" < "+subprocess_in_path+"\n")
                else:
                    batch_file.write(batch_file_name.replace(".batch",".sh")+"\n")
            batch_file.write("wait\n")
        finally:
            batch_file.close()
#}}}
#{{{ def: submit_to_cluster(self,batch_file_name)
    def submit_to_cluster(self,batch_file_name):
    # this routine executes the cluster-specific submit to the cluster
        with open(batch_file_name,'r') as batch_file:
          with open(batch_file_name+".out",'w') as batch_file_out:
            submit = subprocess.Popen(["bsub"],stdin=batch_file,stdout=batch_file_out)
        return submit
#}}}
#{{{ def: get_job_id_from_content_of_submit_output(self,content)
    def get_job_id_from_content_of_submit_output(self,content):
    # this routine extracts the job ID from content, which is the saved on-screen output of the job submit on the cluster
        job_id = ""
        for in_line in content:
            line=in_line.strip() # strip removes all spaces (including tabs and newlines)
            if "is submitted to" in line:
                try:
                    job_id = line.lstrip("Job <").split(">")[0]
                except:
                    job_id = ""
        if not job_id:
            out.print_warning("Could not read job id within \"<\" and \">\" in returned submission output.")
            out.print_warning("Resubmitting job in 15 seconds...")
            time.sleep(15)
            return "resubmit"
        return job_id
#}}}
#}}}
#{{{ class: pbs_cluster(cluster_basic)
class pbs_cluster(cluster_basic): # class for pbs cluster inherited from basic cluster class
    cluster_name = "pbs" # works for Portable Batch System (PBS) and the new Torque (OpenPBS) cluster
#{{{ def: __init__(self,config_list,verbose)
    def __init__(self,config_list,verbose):
        cluster_basic.__init__(self,config_list,verbose)
        self.command_kill_job = "qdel"
        self.command_list_queue = ["qstat"]
        self.command_list_queue_with_jobid = ["qstat"]
        self.job_status_list = [" Q "," R "," *not set* "," *not set* "]  # structure: [pending,running,(held)]
#}}}
#{{{ def: get_list_queue_command_with_multiple_ids(self,job_ids)
    def get_list_queue_command_with_multiple_ids(self,job_ids):
    # this routine returns the cluster-specific command needded when listing jobs with a number of job ids
        command = ["qstat"]
        for id in job_ids:
            command.append(str(id).strip())
        return command
#}}}
#{{{ def: create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path)
    def create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path):
    # this routine creates the cluster-specific batch file for the cluster submit
        batch_file = open(batch_file_name, 'w')
        try:
            batch_file.write("#!/bin/bash\n")
            batch_file.write("#PBS -N "+job_name+"\n")
            if not run_path.startswith("infile.result"): # for normal runs
                batch_file.write("#PBS -o "+subprocess_out_path+"\n")
                batch_file.write("#PBS -e "+subprocess_err_path+"\n")
            if "cluster_runtime" in self.config_list:
                out.print_warning("No runtime limit supported on condor cluster. Remove \"cluster_runtime\" from MATRIX_configuration to avoid this warning.")
#            batch_file.write("#PBS -n 1\n") # causes problems on many clusters
            if "cluster_queue" in self.config_list:
                batch_file.write("#PBS -q "+self.config_list["cluster_queue"]+"\n")
            for submit_line_key in  sorted([key for key in self.config_list if key.startswith("cluster_submit_line")]):
                batch_file.write(self.config_list[submit_line_key]+"\n")
            if run_path.startswith("infile.result"): # for result combination run
                out.print_error("Result combination on cluster \"%s\" only working in local mode. Cannot be submitted to cluster. Exiting..." % self.cluster_name)
                # results_type = subprocess_in_path # results type (result/distribution) for result run was saved here
                # batch_file.write("srun "+fold.exe_path+" "+results_type+" "+run_path+"\n")
            else: # for normal runs
                # on LSF need to go to correct dir
                batch_file.write("cd %s\n" % run_path)
                if int(self.config_list.get("cluster_local_run",0)) == 0:
                    batch_file.write(fold.exe_path+" < "+subprocess_in_path+"\n")
                else:
                    batch_file.write(batch_file_name.replace(".batch",".sh")+"\n")
            batch_file.write("wait\n")
        finally:
            batch_file.close()
#}}}
#{{{ def: submit_to_cluster(self,batch_file_name)
    def submit_to_cluster(self,batch_file_name):
    # this routine executes the cluster-specific submit to the cluster
        with open(batch_file_name,'r') as batch_file:
          with open(batch_file_name+".out",'w') as batch_file_out:
            submit = subprocess.Popen(["qsub"],stdin=batch_file,stdout=batch_file_out)
        return submit
#}}}
#{{{ def: get_job_id_from_content_of_submit_output(self,content)
    def get_job_id_from_content_of_submit_output(self,content):
    # this routine extracts the job ID from content, which is the saved on-screen output of the job submit on the cluster
        job_id = ""
        job_id = content[0].strip()
        if not job_id:
            out.print_warning("Could not read job id within \"<\" and \">\" in returned submission output.")
            out.print_warning("Resubmitting job in 15 seconds...")
            time.sleep(15)
            return "resubmit"
        return job_id
#}}}
#}}}
#{{{ class: sge_cluster(cluster_basic)
class sge_cluster(cluster_basic): # class for Sun Grid Engine (SGE)  cluster inherited from basic cluster class
    cluster_name = "sge"
#{{{ def: __init__(self,config_list,verbose)
    def __init__(self,config_list,verbose):
        cluster_basic.__init__(self,config_list,verbose)
        self.command_kill_job = "qdel"
        self.command_list_queue = ["qstat"]
        self.command_list_queue_with_jobid = ["qstat", "-j"]
        self.job_status_list = [" qw "," r "," *not set* "," *not set* "]  # structure: [pending,running,(held)]
#}}}
#{{{ def: get_list_queue_command_with_multiple_ids(self,job_ids)
    def get_list_queue_command_with_multiple_ids(self,job_ids):
    # this routine returns the cluster-specific command needded when listing jobs with a number of job ids
        command = self.command_list_queue
        return command
#}}}
#{{{ def: get_list_command_for_single_job_id(self,job_id)
    def get_list_command_for_single_job_id(self,job_id):
    # function only needed for clusters where single job_id request has a unexpected behaviour (like printing tons of information), for example SGE
        command = self.command_list_queue
        return command
#}}}
#{{{ def: create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path)
    def create_batch_file(self,job_name,batch_file_name,run_path,process,subprocess_in_path,subprocess_out_path,subprocess_err_path):
    # this routine creates the cluster-specific batch file for the cluster submit
        batch_file = open(batch_file_name, 'w')
        try:
            batch_file.write("#!/bin/bash\n")
            batch_file.write("#$ -N "+job_name+"\n")
            batch_file.write("#$ -l h_vmem=2G\n")
            if not run_path.startswith("infile.result"): # for normal runs
                batch_file.write("#$ -o "+subprocess_out_path+"\n")
                batch_file.write("#$ -e "+subprocess_err_path+"\n")
            if "cluster_runtime" in self.config_list:
                out.print_warning("No runtime limit supported on condor cluster. Remove \"cluster_runtime\" from MATRIX_configuration to avoid this warning.")
#            batch_file.write("#PBS -n 1\n")
            if "cluster_queue" in self.config_list:
                batch_file.write("#$ -q "+self.config_list["cluster_queue"]+"\n")
            for submit_line_key in  sorted([key for key in self.config_list if key.startswith("cluster_submit_line")]):
                batch_file.write(self.config_list[submit_line_key]+"\n")
            if run_path.startswith("infile.result"): # for result combination run
                out.print_error("Result combination on cluster \"%s\" only working in local mode. Cannot be submitted to cluster. Exiting..." % self.cluster_name)
                # results_type = subprocess_in_path # results type (result/distribution) for result run was saved here
                # batch_file.write("srun "+fold.exe_path+" "+results_type+" "+run_path+"\n")
            else: # for normal runs
                # on LSF need to go to correct dir
                batch_file.write("cd %s\n" % run_path)
                if int(self.config_list.get("cluster_local_run",0)) == 0:
                    batch_file.write(fold.exe_path+" < "+subprocess_in_path+"\n")
                else:
                    batch_file.write(batch_file_name.replace(".batch",".sh")+"\n")
            batch_file.write("wait\n")
        finally:
            batch_file.close()
#}}}
#{{{ def: submit_to_cluster(self,batch_file_name)
    def submit_to_cluster(self,batch_file_name):
    # this routine executes the cluster-specific submit to the cluster
        with open(batch_file_name,'r') as batch_file:
          with open(batch_file_name+".out",'w') as batch_file_out:
            submit = subprocess.Popen(["qsub"],stdin=batch_file,stdout=batch_file_out)
        return submit
#}}}
#{{{ def: get_job_id_from_content_of_submit_output(self,content)
    def get_job_id_from_content_of_submit_output(self,content):
    # this routine extracts the job ID from content, which is the saved on-screen output of the job submit on the cluster
        job_id = ""
        for in_line in content:
           line=in_line.strip() # strip removes all spaces (including tabs and newlines)
           if "Your job" in line:
              try:
                job_id = line.split()[2].strip()
              except:
                job_id = ""
        if not job_id:
            out.print_warning("Could not read job id within \"<\" and \">\" in returned submission output.")
            out.print_warning("Resubmitting job in 15 seconds...")
            time.sleep(15)
            return "resubmit"
        return job_id
#}}}
#{{{ def: modify_request_by_grepping_job_ids(self,request,job_ids)
    def modify_request_by_grepping_job_ids(self,request,job_ids):
    # function greps for job ids in request and returns all lines containing one of the job_ids
        request_tmp = ""
        for line in request.splitlines():
            for id in job_ids:
                if id in line:
                    request_tmp += line+'\n'
        request = request_tmp
        return request
#}}}
#}}}

get_cluster_class_from_name = {'condor': condor_cluster,'HTcondor': HTcondor_cluster,'condor_lxplus': condor_lxplus_cluster, 'LSF': lsf_cluster, 'slurm': slurm_cluster, 'PBS': pbs_cluster, 'Torque': pbs_cluster, 'SGE': sge_cluster}
