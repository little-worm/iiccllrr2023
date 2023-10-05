import boto3
import os
import logging
import datetime
from boto3.session import Session
cfd = os.path.dirname(__file__)

bucketName = 'hcp-openaccess'
prefix = 'HCP_1200'
outputPath = cfd+'/HCP_dataset'
access_key = 'AKIAXO65CT57HVRCTMH4'# [你的 aws_access_key]
secret_key = 'XA6zzMixA9ci15pEZ24zjgLCOuoiWdiSRUdaPDkv' # [你的 aws_secret_key]
bucketName = 'hcp-openaccess'
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
session = Session(aws_access_key_id=access_key,aws_secret_access_key=secret_key)
s3 = session.resource('s3')
theTime = datetime.datetime.now().strftime('%Y_%m_%d-%H-%M_%S')
#os.makedirs(theTime)

logger = logging.getLogger('script')
formatter = logging.Formatter('%(asctime)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger.setLevel(level = logging.DEBUG)
logger.propagate = False

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

bucket = s3.Bucket(bucketName)
logger.info('Bucket built!')

with open(cfd+'/HCP_dataset/subjects.txt', 'r') as fr:
        for subject_number in fr.readlines():
            subject_number = subject_number.strip()
            keyList = bucket.objects.filter(Prefix = prefix + '/{}/MNINonLinear/Results/tfMRI'.format(subject_number))
            keyList = [key.key for key in keyList]
            keyList = [x for x in keyList if '_LR.nii.gz' in x ]
            totalNumber = len(keyList)
            for idx,tarPath in enumerate(keyList):
                downloadPath = os.path.join(outputPath,tarPath)
                #downloadDir = os.path.dirname(downloadPath)
                downloadPath1 = os.path.join(outputPath,subject_number+'_'+tarPath.split('/')[-1].split('_')[1]+'.nii.gz')
                #if not os.path.exists(downloadDir):
                 #   os.makedirs(downloadDir)
                try:
                    if not os.path.exists(downloadPath1):
                        bucket.download_file(tarPath,downloadPath1)
                        logger.info('%s: %s downloaded! %d/%d',subject_number,tarPath.split('/')[-1],idx+1,totalNumber)
                    else :
                        logger.info('%s: %s already exists! %d/%d',subject_number,tarPath.split('/')[-1],idx+1,totalNumber)
                except Exception as exc:
                    logger.error('{}'.format(str(exc)))
            logger.info('%s completed!', subject_number)
