import json
import pickle
import numpy as np
__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        # finding the index f our location
        # if not found list throughs an error
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    # fill our x with zeros
    X = np.zeros(len(__data_columns))
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    
    if loc_index >= 0:
        X[loc_index] = 1
        
    return round(__model.predict([X])[0], 2)
def get_location_names():
    return __locations

def load_saved_artifacts():
    global __data_columns
    global __locations
    global __model
    with open("columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    
    with open("home_price.pickle", "rb") as f:
        __model = pickle.load(f)
        
if __name__ == "__main__":
    load_saved_artifacts()