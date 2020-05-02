# Machine Learning (ML) for Additive manufacturing (AM)
The values of thermal conductivity (k), density and heat capacity (cp) of an alloy depends on the composition of the elements that constitutes the particular alloy and the temperature of measurement. In order to model the relationship between these values, artificial neural networks can be used. We have used a model with two hidden layers with 20 and 5 neurons and the output layer with one neuron. The input neurons can be varied based on the number of elements to train for. 

## Required packages

numpy  
matplotlib  
torch  
tqdm  
pandas  
os    
time  

## Training dataset format

The training data should be a csv file with each row as a datapoint to train. Each row should comprise of the element composition for chosen number of elements in percentage, followed by temperature in celcius and the last column as the value to be regressed. 

## To train the model with the given dataset

The dataset is available in the folder data. To train, use the following command,

**python main.py --data_path data/data_k.csv --ch k --save_cp True**

The above command is to train for thermal conductivity. To train for density change datapath as data/data_density.csv and replace k with density. Similarly, to train for heat capacity replace path as data/data_cp and k with cp. save_cp is set to true to save the checkpoints during training.

## To test for a single data point

To predict the thermal conductivity, heat capacity and density for certain composition and temperature, we need to make changes in test.py file. Replace the value of variable 'a' with the data that needs prediction as an array with element composition followed by temperature. Change the value of ch as per the value that needs to be predicted ('k' - Thermal conductivity, 'cp' - heat capacity , 'density' - density). After modifications run the command,

**python test.py**
