#!/usr/bin/env bash

# download data from kaggle
kaggle datasets download -d landlord/multilingual-disaster-response-messages -p data/
# unzip data archive
unzip data/multilingual-disaster-response-messages -d data/
# delete zip file
rm data/multilingual-disaster-response-messages.zip
# merge files into one file
sed '2,$!d' data/disaster_response_messages_validation.csv >> data/disaster_response_messages_training.csv 
sed '2,$!d' data/disaster_response_messages_test.csv >> data/disaster_response_messages_training.csv 

# # rename file
mv -v data/disaster_response_messages_training.csv data/disaster_response_messages.csv

# remove extra files
rm data/disaster_response_messages_test.csv 
rm data/disaster_response_messages_validation.csv
