#!/usr/bin/env bash

# download data from kaggle
kaggle datasets download -d landlord/multilingual-disaster-response-messages -p data/
# unzip data archive
unzip data/multilingual-disaster-response-messages -d data/
# delete zip file
rm data/multilingual-disaster-response-messages.zip
# merge files into one file
# add a blank row at the end of training csv
echo -e "\n" >> data/disaster_response_messages_training.csv
# append validation csv to training csv, removing header
sed '2,$!d' data/disaster_response_messages_validation.csv >> data/disaster_response_messages_training.csv 
# add a blank row at the end of training csv
echo -e "\n" >> data/disaster_response_messages_training.csv
# append test csv to training csv, removing header
sed '2,$!d' data/disaster_response_messages_test.csv >> data/disaster_response_messages_training.csv 
# remove blank rows
sed -i '' '/^$/d' data/disaster_response_messages_training.csv

# # rename file
mv -v data/disaster_response_messages_training.csv data/disaster_response_messages.csv

# remove extra files
rm data/disaster_response_messages_test.csv 
rm data/disaster_response_messages_validation.csv
