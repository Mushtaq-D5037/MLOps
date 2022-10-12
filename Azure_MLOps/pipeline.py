# importing libraries
from azureml.core import Workspace, Datastore, Dataset
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep

ws = Workspace.from_config()
datastore =  ws.get_default_datastore() # to store pipeline data output, mandatory to define a default datastore 
compute   = 'ML-Pipeline-Cluster'
# compute = 'M-CLUSTER'
# directory = 'churn-prediction' 

# Create an Azure ML experiment in your workspace
# experiment = Experiment(workspace=ws, name="Experiments_Training")

# dataset name as given in 'Data Tab'
dataset_eu   = 'CHURN_PREDICTION_BGNBD-EU_DATA-PROD'
dataset_la   = 'CHURN_PREDICTION_BGNBD-LA_DATA-PROD'
dataset_apac = 'CHURN_PREDICTION_BGNBD-APAC_DATA-PROD'


# loading data from Dataset
df_eu   = Dataset.get_by_name(workspace=ws, name= dataset_eu)
df_la   = Dataset.get_by_name(workspace=ws, name= dataset_la)
df_apac = Dataset.get_by_name(workspace=ws, name= dataset_apac)

# Initializing Pipeline Parameters
# Countries List w.r.t to TENANT
# make sure there in only one space in between and no extra space in beginnig and ending
EU_TENANT   = 'DE ES FR GB IE IT PT'  
LA_TENANT   = 'MX BR CA'
APAC_TENANT = 'AU NZ'

pipeline_param_EU_TENANT   = PipelineParameter(name="EU Tenant countries",   default_value=EU_TENANT)
pipeline_param_LA_TENANT   = PipelineParameter(name="LA Tenant countries",   default_value=LA_TENANT)
pipeline_param_APAC_TENANT = PipelineParameter(name="APAC Tenant countries", default_value=APAC_TENANT)

print(f'Compute: {compute}')
print(f'Workspace:{ws.name}')
print(f'Default Datastore: {datastore.name}')
print(f'Dataset1 : {dataset_eu}')
print(f'Dataset2 : {dataset_apac}')
print(f'Dataset3 : {dataset_la}')
print(f'EU TENANT: {EU_TENANT}')
print(f"LA TENANT: {LA_TENANT}")
print(f"APAC TENANT:{APAC_TENANT}")

# creating re-usable Environment
# creating an environment
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration

# create python environment for experiment
# bgnbd_env = Environment('test_bgnbd_pipeline_env')

# Create a Python environment for the experiment (from a .yml file)
# experiment_env = Environment.from_conda_specification("experiment_env", experiment_folder + "/experiment_env.yml")

# Create a Python environment from defining packages
# bgnbd_packages = CondaDependencies.create(conda_packages=['scikit-learn','pandas','numpy','pip', 'pyodbc','sqlalchemy'],
#                                           pip_packages=['azureml-defaults','lifetimes'])

# # adding dependencies to the environment
# bgnbd_env.python.conda_dependencies = bgnbd_packages

# register the environment
# bgnbd_env.register(workspace=ws)
reg_env = Environment.get(ws, 'test_bgnbd_pipeline_env')

# create a run config object for the pipeline
pipeline_runconfig = RunConfiguration()

# use the compute target
pipeline_runconfig.target = compute

# assigning the run configuration to the envrionment
pipeline_runconfig.environment = reg_env

print('RunConfiguration created')

# creating a output folder to store intermediate output from the pipeline
pre_process_output_folder = PipelineData(name='pre_process_output', datastore=datastore)
prediction_output_folder  = PipelineData(name='prediction_output',  datastore=datastore)

   
# creating pipeline steps
pre_process_step = PythonScriptStep(name = 'step 1: Data Preparation', 
                                    script_name='pre_process.py', 
                                    arguments= [
                                                '--input_data_eu'  , df_eu.as_named_input('raw_data_eu'), 
                                                '--input_data_la'  , df_la.as_named_input('raw_data_la'),
                                                '--input_data_apac', df_apac.as_named_input('raw_data_apac'),
                                                '--eu_tenant',pipeline_param_EU_TENANT,
                                                '--la_tenant',pipeline_param_LA_TENANT,
                                                '--apac_tenant',pipeline_param_APAC_TENANT,
                                                '--output', pre_process_output_folder],   
                                    outputs  = [pre_process_output_folder],
                                    compute_target=compute, 
                                    runconfig=pipeline_runconfig, 
                                    allow_reuse=False, 
                                    source_directory=None)

prediction_step = PythonScriptStep(name = 'step 2: Training & Prediction', 
                                    script_name='prediction.py', 
                                    arguments= ['--input_data',pre_process_output_folder, '--output', prediction_output_folder], 
                                    inputs   = [pre_process_output_folder], 
                                    outputs  = [prediction_output_folder], 
                                    compute_target=compute, 
                                    runconfig=pipeline_runconfig, 
                                    allow_reuse=False, 
                                    source_directory=None) 

overlay_step = PythonScriptStep(name = 'step 3: Overlaying Features ', 
                                script_name='overlay.py', 
                                arguments= ['--raw_data', pre_process_output_folder, 
                                            '--main_data', pre_process_output_folder, 
                                            '--result_data',prediction_output_folder,
                                            '--eu_tenant',pipeline_param_EU_TENANT,
                                            '--la_tenant',pipeline_param_LA_TENANT,
                                            '--apac_tenant',pipeline_param_APAC_TENANT
                                           ], 
                                inputs   = [pre_process_output_folder,prediction_output_folder],  
                                compute_target=compute, 
                                runconfig=pipeline_runconfig, 
                                allow_reuse=False,
                                source_directory=None)

pipeline = Pipeline(workspace=ws, steps=[pre_process_step,prediction_step,overlay_step]) # 
pipeline.validate()
# print('pipeline steps created')

# Run the pipeline as an experiment #HCO-Churn-Prediction
pipeline_run = Experiment(ws, 'HCO-Churn-Prediction').submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)

# publishing pipeline
published_pipeline = pipeline_run.publish_pipeline(name='churn_prediction_batch',
                                                   description='Churn prediction Batch pipeline',
                                                   version='1.0')


rest_endpoint = published_pipeline.endpoint

# invoking pipeline via endpoint
from azureml.core.authentication import InteractiveLoginAuthentication
import requests

# Authentication
interactive_authentication = InteractiveLoginAuthentication()
auth_header = interactive_authentication.get_authentication_header()
print('authentication header ready')

response = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={"ExperimentName": "Experiments_Training"})
run_id = response.json()["Id"]

# Scheduling pipeline
from azureml.pipeline.core import ScheduleRecurrence, Schedule

weekly = ScheduleRecurrence(frequency='Week', interval=1)
pipeline_schedule = Schedule.create(ws, name='Weekly Predictions',
                                        description='batch inferencing',
                                        pipeline_id=published_pipeline.id,
                                        experiment_name='Batch_Prediction',
                                        recurrence=weekly)

# Getting list of all Schedule
# to get all scheduled set active_only=False
schedules = Schedule.list(ws, active_only=True) 
print("Your workspace has the following schedules set up:")
for schedule in schedules:
    print(f"\nSchedule ID: {schedule.id}, Published pipeline ID: {schedule.pipeline_id}")
    # getting details of Schedule
    print(Schedule.get(ws, schedule.id))
    
# Disabling Schedule
# Set the wait_for_provisioning flag to False if you do not want to wait  
# for the call to provision the schedule in the backend.

# schedule_id = '3f340eee-d4a5-4aa6-bb90-22989582f934'
# fetched_schedule = Schedule.get(ws, schedule_id)
# fetched_schedule.disable(wait_for_provisioning=True)

# fetched_schedule_new = Schedule.get(ws, schedule_id)
# print(f"Disabled schedule: {fetched_schedule_new.id}. New status is: {fetched_schedule_new.status}")

# re-enable Schedule
# fetched_schedule.enable(wait_for_provisioning=True)

# update/change schedule
# recurrence = ScheduleRecurrence(frequency="Month", interval=1) # Runs every one month
# fetched_schedule.update(name="My_Updated_Schedule", 
#                         description="Updated_Schedule_Run", 
#                         status='Active', 
#                         wait_for_provisioning=True,
#                         recurrence=recurrence)

# fetched_schedule
