# Plant-Production-Automation-Deployment-

This GitHub repository contains the complete project. From preprocessing the data to deriving EDA insights to making the Model to fine tuning it to deployment.In order to run this, you must download them in one folder and run the app.py file. 

The project is done on real industrial data. The data size is large with 72 columns and close to six thousand rows. 

The app.py file is the Python file which contains the main deployment. The .pkl (pickle) files are the actual models and graphs being saved so that the model can be deployed and run on a website. 

This file has 2 main models being saved: 1) Multioutput boiler loss prediction which predicts production losses, and 2) XGBoost_Boiler which predicts production efficiency.  
