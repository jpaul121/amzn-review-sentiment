# <h1>Classifying Amazon clothing & accessory reviews</h1>
Predict whether or not customers expressed a positive or negative opinion of a product based on the wording of their review on Amazon. The text file containing the raw data used in this model is the property of Stanford University's Network Analysis Project and can be found [here](http://snap.stanford.edu/data/web-Amazon-links.html). 

# <h1>Installation</h1>
## Download the data
* Clone this repo to your computer. 
* Download the raw text file containing the data for this project. 
    * You can find the data [here](http://snap.stanford.edu/data/web-Amazon-links.html) under _Clothing_&_Accessories.txt.gz_
    * Please be sure to download the data directly from the page in the link, as opposed to following the link navigating you to an updated version of the data on another page. 
* Downloading from the page linked to above should give you a file archived in the `.gz` format. 
    * Decompress this file using 7zip or any similar program to access the text file containing the data. 
* Make sure the text file containing the data is located inside the folder containing this project. 
## Install the requirements
* Install the requirements for running this project using `pip install -r requirements.txt`. 
    * Make sure you're using Python 3. Replace `pip` with `pip3` in the command above if the requirements do not seem to show up when you begin using this project. 
    * On Windows, you'll need to run `python -m pip install -r requirements.txt`. 
    * You may want to use a virtual environment for this. 
# <h1>Usage</h1>
* Navigate to the folder containing this project in your shell. 
* Run `python process.py`. 
    * This will create a clean, streamlined version of the data in the raw text file in a file named `clothing_reviews.csv`. 
* Run `python train_and_test.py`. 
    * This will train the model and test it on holdout data, printing accuracy scores inside the shell throughout. 