# Yelp-data-challenge
This is repo for sentence classfication on Yelp data set using deep learning. <br/>
If there is any question or bug you found please contact: zzjiang@bu.edu

# Preprocess steps
The original dataset can be obtained from : <br/>
https://www.yelp.com/dataset <br/>
However you can download our preprocessed datasets directly at : <br/>
Two label training set : https://drive.google.com/file/d/1BOLf-vgiYSGGCWaXP-vq5wsEQzuM2lY3/view?usp=sharing <br/>
Two label testing set : https://drive.google.com/file/d/1I4pnJxaz8c69a89SzOtPfRAaQwNuNkEZ/view?usp=sharing <br/>
Five label training set :https://drive.google.com/file/d/1XFlEPieaJklN8gP7pgUmeRR4NRTT9jGM/view?usp=sharing <br/>
Five label testing set :https://drive.google.com/file/d/1w0FN0p9lsQN9ti964att6HJK5fyTgYfw/view?usp=sharing <br/>

# Two label classification 
There are total 8 models we built for two label classification , they can be found at balanced_model/2_label <br/>
All the python files can executed directly on the SCC computing cloud(with your own data directory of course) <br/>
yoon_kim_original.py is the original model <br/>
yoon_kim_batch_norm.py is the model "yoon_kim_2" in our report <br/>
The rest includes the baseline.py(baseline model), cnn_gru.py, lstm_cnn.py, rnn_bilstm.py (bi-directional lstm) and rnn_attention.py <br/>

# Five label classification 
There are total 7 models we built for two label classification , they can be found at balanced_model/5_label <br/>
All the python files can executed directly on the SCC computing cloud(with your own data directory of course) <br/>
# Simulation
The simulation files are at report/simulation, in which includes simulation for dropout, batch norm and vanishing gradient (not described in report since we are going beyond the page limit)
# Plots
The plot.py and plot_compare.py contains the code we used for the figures. To run them, you need to read in  the history of models(included for all models at end of each file for 2-label models and 5-label models) 
