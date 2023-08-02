import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from datetime import datetime
# import seaborn as sns
# sns.set()
# import libraries
#Azure
from azureml.core import Run
#python
import os
import pandas as pd
import numpy as np
import argparse
import pyodbc
import urllib
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
#sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import  silhouette_score
#sqlalchemy
from sqlalchemy import create_engine 



 # CALCULATING ENGAGEMENT SCORE
def cal_eng(x, y):
    if x==0:
        v = 0
    else:
        v = y/x
    return v
    

def email_module_eng_bucket(df, col):
    df[f'{col}_BUCKET'] = np.nan
    for idx, v in enumerate(df[col]):
        
        if v == 0.0:
            df[f'{col}_BUCKET'].loc[idx] = 'No'
        
        elif v>0.0 and v<=0.25:
            df[f'{col}_BUCKET'].loc[idx] = 'Low'
        
        elif v>0.25 and v<=0.5:
            df[f'{col}_BUCKET'].loc[idx] = 'Medium'
            
        elif v>0.5 and v<=0.75:
            df[f'{col}_BUCKET'].loc[idx] = 'AboveAverage'
            
        elif v>0.75 and v<1:
            df[f'{col}_BUCKET'].loc[idx] = 'High'
        elif v==1.0:
            df[f'{col}_BUCKET'].loc[idx] = '100_prcnt'
                  
    return df

def engagement_bucket2(df, col):
    df[f'{col}_BUCKET'] = np.nan
    for idx, v in enumerate(df[col]):
        
        if v == 0.0:
            df[f'{col}_BUCKET'].loc[idx] = '0.0'
        
        elif v>0.0 and v<=1:
            df[f'{col}_BUCKET'].loc[idx] = '>0.0 - 1.0'
                   
    return df
    
def normalize_dr1(df,pca_components):
    
    '''Normalizing & Dimensionality Reduction '''
    min_max_scaler1 = MinMaxScaler()
    
    # transform data
    min_max_scaled1 = min_max_scaler1.fit_transform(df)
    print(min_max_scaled1)

    # PCA
    pca   = PCA(n_components = pca_components, random_state=1)
    X_pca = pca.fit_transform(min_max_scaled1)
    
    return min_max_scaler1, min_max_scaled1, X_pca 


def data_drift(df_train, df_pred, col):
    
    train_data = df_train[col]
    pred_data   = df_pred[col]
    bins = np.histogram_bin_edges(list(train_data) + list(pred_data), bins="sturges")
    
    # binning data - to make train & test equal length
    train_counts = np.histogram(train_data, bins)[0] 
    pred_counts  = np.histogram(pred_data, bins)[0] 
    
    # Relative Frequency
    train_percents = train_counts/ len(train_data)
    pred_percents  = pred_counts / len(pred_data)
    
    # replacing zeroes with number close to zero
    if_zeroes : bool =True
    if if_zeroes:
        np.place(
            train_percents,
            train_percents == 0,
            min(train_percents[train_percents != 0]) / 10**6
            if min(train_percents[train_percents != 0]) <= 0.0001
            else 0.0001,
        )
        np.place(
            pred_percents,
            pred_percents == 0,
            min(pred_percents[pred_percents != 0]) / 10**6
            if min(pred_percents[pred_percents != 0]) <= 0.0001
            else 0.0001,
        )
    
    # calculating KL_DIVERGENCE
    kl_div_value = stats.entropy(train_percents, pred_percents)
    kl_div_value = np.round(kl_div_value,4)
    print(f'column:{col}, KL(P||Q):{kl_div_value}')
   
    try: 
        plt.title(f'KL(P||Q) = {kl_div_value}')
        a = df_train[col].unique().tolist()
        b = df_pred[col].unique().tolist()
        xticks  = a+b
        ax =  df_train[col].plot.density(color='green', label='Train', xticks= xticks)
        df_pred[col].plot.density(ax=ax, color='orange', label='Pred',)
        plt.xlabel(col)
        plt.legend()
        plt.show()
        
    except np.linalg.LinAlgError as err:
        if 'singular matrix' in str(err):
            print(f'Density plot cannot be plotted for column:{col}, as it has only 0')
     
    
    return kl_div_value, plt
           
def cluster_monitoring(df, cluster, df_m):
    eng_cut0 = [0.0]
    eng_cut1 = [1.0]
    eng_cut2 = [0.0, 1.0]
    value = np.nan
    
    print('ClusterName:', cluster)

    df2 = df[df['CLUSTER']==cluster] 
    cluster_size = len(df2)
    email_eng  = df2['EMAIL_ENGAGEMENT'][~df2['EMAIL_ENGAGEMENT'].isin(eng_cut2)].count()
    learn_eng  = df2['LEARNING_ENGAGEMENT'][~df2['LEARNING_ENGAGEMENT'].isin(eng_cut0)].count()
    rc  = df2['REMOTE_CALLS'][df2['REMOTE_CALLS']>0].count()
    f2f = df2['FACE_TO_FACE_APP'][df2['FACE_TO_FACE_APP']>0].count()
    
    min_email_sent   =  df2['EMAIL_SENT'].min()
    email_percent = np.round((email_eng/ cluster_size)*100, 2)
    learn_percent = np.round((learn_eng/ cluster_size)*100, 2) 
    rc_percent    = np.round((rc/ cluster_size)*100, 2) 
    f2f_percent   = np.round((f2f/cluster_size)*100, 2) 

    # for Digital Cluster
    email_eng1  = df2['EMAIL_ENGAGEMENT'][df2['EMAIL_ENGAGEMENT'].isin(eng_cut1)].count()
    email_percent1 = np.round((email_eng1/ cluster_size)*100, 2) 
    
    if cluster == 'NON_DIGITAL' and cluster_size>0:
        value  = f"ClusterSize={cluster_size},\nREMOTE_CALLS={rc_percent}%,\nF2F_APP={f2f_percent}%,\nEMAIL_ENG={email_percent}%,\nLEARNING_ENG={learn_percent}%"
        if email_percent > 10 or learn_percent > 10:
            status = 'misclassified'
        else:
            status = 'aligned with train results'

    elif cluster =='DIGITAL' and cluster_size>0:
        value = f"ClusterSize={cluster_size},\nEMAIL_ENG={email_percent1}%,\nLEARNING_ENG={learn_percent}%"
        
        if email_percent1 < 90 or learn_percent > 10:
            status = 'misclassified'
        else:
            status = 'aligned with train results'

    elif cluster == 'SELECTIVELY_ENGAGED' and cluster_size>0:
        value = f"ClusterSize={cluster_size},\nMIN_EMAIL_SENT={min_email_sent},\nEMAIL_ENG between 0 to 1={email_percent}%,\nLEARNING_ENG={learn_percent}%"
        if min_email_sent < 2 or email_percent < 90 or learn_percent > 10:
            status = 'misclassified'
        else:
            status = 'aligned with train results'
    elif cluster == 'PRO_KNOWLEDGE' and cluster_size>0:
        value= f"ClusterSize={cluster_size},\nLEARNING_ENG={learn_percent}%"
        if learn_percent < 90:
            status = 'misclassified'
        else:
            status = 'aligned with train results'
        
    elif cluster == 'NOT_ENGAGED' and cluster_size>0:
        value = f"ClusterSize={cluster_size},\nLEARNING_ENG={learn_percent}%,\nEMAIL_ENG={email_percent}%,\nREMOTE_CALLS={rc_percent}%,\nF2F_APP={f2f_percent}%"
        if email_percent > 10 or learn_percent > 10:
            status = 'misclassified'
        else:
            status = 'aligned with train results'
        
    df_m[f'CLUSTER_{cluster}'] = value
    
    return df_m, value, status
    


if __name__ =='__main__':

    #Initializing run object
    run = Run.get_context()

    parser = argparse.ArgumentParser()
    #input
    parser.add_argument("--training_output_eu"  , type=str, help="EU training output data")
    parser.add_argument("--training_output_la", type=str, help="LA training output data")
    parser.add_argument("--training_output_apac"  , type=str, help="APAC training output data") 
    parser.add_argument("--predictions_eu"  , type=str, help="EU predictions output data")
    parser.add_argument("--predictions_la", type=str, help="LA predictions output data")
    parser.add_argument("--predictions_apac"  , type=str, help="APAC predictions output data") 
    #output
    parser.add_argument("--output", type=str, help="output directory")

    args = parser.parse_args()

    # converting to dataframes
    df_train_eu   = run.input_datasets['train_output_data_eu'].to_pandas_dataframe()
    df_train_la   = run.input_datasets['train_output_data_la'].to_pandas_dataframe()
    df_train_apac = run.input_datasets['train_output_data_apac'].to_pandas_dataframe()
    df_pred_eu   = run.input_datasets['pred_output_data_eu'].to_pandas_dataframe()
    df_pred_la   = run.input_datasets['pred_output_data_la'].to_pandas_dataframe()
    df_pred_apac = run.input_datasets['pred_output_data_apac'].to_pandas_dataframe()

    print('EU Dataframe Shape:', df_pred_eu.shape)
    print('EU Countries:', df_pred_eu['COUNTRY'].unique().tolist())

    print('LA Dataframe Shape:', df_pred_la.shape)
    print('LA Countries:', df_pred_la['COUNTRY'].unique().tolist())

    print('APAC Dataframe Shape:', df_pred_apac.shape)
    print('APAC Countries:', df_pred_apac['COUNTRY'].unique().tolist())

    # concatenating
    df_train_output = pd.concat([df_train_eu, df_train_la, df_train_apac])
    df_pred = pd.concat([df_pred_eu, df_pred_la, df_pred_apac])


    print('Train Data Output Shape', df_train_output.shape)
    print('Countries in Trained Output:', df_train_output['COUNTRY'].unique().tolist())
    print(df_train_output.head())

    print('Prediction Output Shape', df_pred.shape)
    print(df_pred.head())
    
    print('Calculating Engagement')
    df_pred['EMAIL_ENGAGEMENT']  = df_pred.apply(lambda x: cal_eng(x.EMAIL_SENT, x.EMAIL_OPENED), axis=1)
    df_pred['MEETING_ENGAGEMENT']= df_pred.apply(lambda x: cal_eng(x.MEETING_INVITED, x.MEETING_ATTENDED), axis=1)
    df_pred['EVENT_ENGAGEMENT']  = df_pred.apply(lambda x: cal_eng(x.WEBINAR_INVITED, x.WEBINAR_ATTENDED), axis=1)
    df_pred['LEARNING_ENGAGEMENT'] = df_pred['MODULE_STARTED'].apply(lambda x: 1 if x>20 else(0 if x == 0 else x/20))
    

    # MODEL COLUMNS
    cols = ['EMAIL_ENGAGEMENT','MODULE_STARTED','LEARNING_ENGAGEMENT',
            'EMAIL_SENT','FACE_TO_FACE_APP','WEBINAR_INVITED', 'REMOTE_CALLS',
            'EVENT_ENGAGEMENT', 'MEETING_INVITED', 'MEETING_ENGAGEMENT',]
    # KL DIVERGENCE
    features = []
    kl_value = []
    
    # KS CURVE
    p_value = 0.05
   
    for col in cols:
        # KL DIVERGENCE
        kl_div,plt = data_drift(df_train_output, df_pred, col)
        features.append(col)
        kl_value.append(kl_div)
        
        # logging image
        run.log_image(name=f"{col}",plot=plt)
        plt.close()
        
        # KS TEST
        test_results = stats.ks_2samp(df_train_output[col], df_pred[col])
        confidence = np.round(test_results[1], 4)
        if confidence < p_value:
             run.log_row('Feature Drift', Feature=col, KL_Divergence=kl_div, KOLMOGROV_SMIRNOV_P_VALUE=confidence, Drift='DETECTED')

        else:
            run.log_row('Feature Drift', Feature=col, KL_Divergence=kl_div, KOLMOGROV_SMIRNOV_P_VALUE=confidence, Drift='NOT_DETECTED')
 
    # logging month
    exe_date = df_pred['Execution_Date'].unique().tolist()[0]
    exe_date = pd.to_datetime(exe_date)
    last_six_months = exe_date + timedelta(days=-180)
    month =  last_six_months.strftime('%Y-%m-%d') + ' to '+ exe_date.strftime('%Y-%m-%d') 
    print('Execution Date', exe_date)
    print('Last six month', last_six_months)
    run.log('PREDICTION DATA (MONTH)', month)

    # model version
    model_version = df_pred['MODEL_VERSION'].unique().tolist()[0]


    # ------ KOLMOGROV SMIRNOV TEST ---------
    # p_value = 0.05
    # rejected = 0

    # for col in cols:
        
    #     test_results = stats.ks_2samp(df_orig[col], df_test[col])
    #     confidence = np.round(test_results[1], 2)
    #     if confidence < p_value:
             
    #          rejected += 1
    #          print(f'column rejected: {col} | p-value:{confidence}')
    #     else:
    #         print(f'Fail to reject columns: {col} | p-value:{confidence}')
    
    # print("We rejected",rejected,"columns in total")


    # ------ CALCULATING SILHOUETTE --------------
    
    #  CREATING CLUSTER COLUMN
    label = {9999:'NOT_ENGAGED', 0:'DIGITAL', 1:'NON_DIGITAL', 2:'PRO_KNOWLEDGE', 3:'SELECTIVELY_ENGAGED'}
    df_pred['LABELS']  = df_pred['LABELS'].fillna(9999)
    df_pred['LABELS']  = df_pred['LABELS'].apply(lambda x: int(x))
    df_pred['CLUSTER'] = df_pred['LABELS'].map(label)
    df_pred = df_pred.reset_index(drop=True)
    print('Prediction')
    print(df_pred['CLUSTER'].value_counts())
   
    # COPYING DATAFRAME
    df_score = df_pred.copy()

    df_score = df_score[df_score['CLUSTER']!='NOT_ENGAGED']
    print(df_score['CLUSTER'].value_counts())
    df_score = df_score.reset_index(drop=True)
    

    # BUCKETING 
    eng_score_cols = ['EMAIL_ENGAGEMENT','EVENT_ENGAGEMENT','MEETING_ENGAGEMENT','LEARNING_ENGAGEMENT']
    
    for col in eng_score_cols:
        print(col)
        if col == 'EMAIL_ENGAGEMENT':
            df_score = email_module_eng_bucket(df_score, col)
        else:
            df_score = engagement_bucket2(df_score, col)

    # ONE-HOT ENCODING
    dummy_col = ['EMAIL_ENGAGEMENT_BUCKET', 'LEARNING_ENGAGEMENT_BUCKET',
                 'EVENT_ENGAGEMENT_BUCKET', 'MEETING_ENGAGEMENT_BUCKET']

    df_score = pd.get_dummies(df_score, columns=dummy_col)


    ## TAKING TRAINED COLUMNS
    # K-Means model clustering columns
    kmeans_training_columns =  [
                                  'LEARNING_ENGAGEMENT_BUCKET_0.0',
                                  'LEARNING_ENGAGEMENT_BUCKET_>0.0 - 1.0',
                                  'EMAIL_ENGAGEMENT_BUCKET_100_prcnt',
                                  'EMAIL_ENGAGEMENT_BUCKET_AboveAverage', 
                                  'EMAIL_ENGAGEMENT_BUCKET_High',
                                  'EMAIL_ENGAGEMENT_BUCKET_Low', 
                                  'EMAIL_ENGAGEMENT_BUCKET_Medium',
                                  'EMAIL_ENGAGEMENT_BUCKET_No', 
                                  'MEETING_ENGAGEMENT_BUCKET_>0.0 - 1.0', 
                                  'EVENT_ENGAGEMENT_BUCKET_>0.0 - 1.0'
                                ]

    trained_cols_not_in_test = []
    df_score2 = df_score.copy()
    for col in kmeans_training_columns:
        if col not in  df_score2.columns:
            df_score2[col] = 0
            trained_cols_not_in_test.append(col)

    print(f"Trained Columns not in Prediction Data:{trained_cols_not_in_test}")            
    df_score2 = df_score2[kmeans_training_columns]
    df_score2.drop(trained_cols_not_in_test, axis=1, inplace=True)

    # normalzing
    scaler, df_score_norm, _ = normalize_dr1(df_score2, 2)
    
    # LABEL ENCODING
    le = LabelEncoder()
    y =le.fit_transform(df_score['CLUSTER'])
    
    # silhouette co-eff
    predicted_clusters = len(df_score['CLUSTER'].unique().tolist())
    
    if predicted_clusters > 1:
        silhouette_test = silhouette_score(df_score_norm, y).round(3)
        print(f'Silhoutte Score: {silhouette_test}')
        run.log('Prediction Silhouette Co-eff', silhouette_test)

    else:
        silhouette_test = np.nan
        run.log('Clusters Predicted', predicted_clusters)
        run.log('Silhouette Co-efficient is calcutated only if predicted clusters are more than 1', silhouette_test)
   
    # TRAIN ACCURACY
    run.log('Train Silhouette Co-eff', 0.84)


    # CLUSTER MONITORING
    # CREATING DATAFRAME
    df_monitoring = pd.DataFrame(data = [kl_value], columns = features)
    # adding other columns
    df_monitoring['DRIFT_TEST'] = 'KL(P||Q)'
    df_monitoring['MONTH']=  month
    df_monitoring['MODEL_VERSION'] = model_version
    df_monitoring['ACCURACY'] = silhouette_test
    df_monitoring['EXECUTION_DATE'] = datetime.now()

    cluster_list = df_pred['CLUSTER'].unique().tolist()
    print('Predicted ClusterS:',cluster_list)
    # desired clusters
    cluster_desired = ['DIGITAL', 'NON_DIGITAL','SELECTIVELY_ENGAGED', 'PRO_KNOWLEDGE', 'NOT_ENGAGED']
    for c in cluster_desired:
        if c not in cluster_list:
            print('Cluster not formed:', c)
            cluster_list.append(c)

    print('Calculating Cluster Characterstics')
    status_list = []
    for cluster in cluster_list:
        df_monitoring, value, status = cluster_monitoring(df_pred, cluster, df_monitoring)
        status_list.append(status)
        run.log_row('Cluster Monitoring', Cluster=cluster, Characteristics=value, Status=status)

    if 'misclassified' in status_list:
        df_monitoring['MODEL_PERFORMANCE'] = 'Dropped'
        run.log('MODEL PERFORMANCE', 'DROPPED')
    else:
        df_monitoring['MODEL_PERFORMANCE'] = 'Stable'
        run.log('MODEL PERFORMANCE', 'STABLE')

    
    print(df_monitoring.columns)
    print(df_monitoring.head())

    # RE-ARRANGING COLUMNS
    col_order = ['MONTH','ACCURACY','CLUSTER_DIGITAL', 'CLUSTER_NON_DIGITAL', 'CLUSTER_SELECTIVELY_ENGAGED',    'CLUSTER_PRO_KNOWLEDGE', 'CLUSTER_NOT_ENGAGED', 'DRIFT_TEST', 'REMOTE_CALLS','FACE_TO_FACE_APP', 'EMAIL_SENT', 'WEBINAR_INVITED', 'MODULE_STARTED', 'MEETING_INVITED','EMAIL_ENGAGEMENT','LEARNING_ENGAGEMENT', 'EVENT_ENGAGEMENT', 'MEETING_ENGAGEMENT','MODEL_VERSION', 'EXECUTION_DATE','MODEL_PERFORMANCE'
                ]
    df_monitoring = df_monitoring[col_order]
    
    # saving monitoring data
    if not (args.output  is None):
        os.makedirs(args.output, exist_ok=True)
        print(f"created output folder: {args.output}")

    save_path1  = os.path.join(args.output , f'df_monitoring.csv')
    df_monitoring.to_csv (save_path1, header=True, index=False)
    
    run.complete()


    
    
