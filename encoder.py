"""
 * @file encoder.py
 * @authors Ben Bellerose
 * @date Novemebr 21 2019
 * @modified Novemebr 22 2019
 * @modifiedby BB
 * @brief class for encoding data to be processed by machine learing models
 */
"""
import numpy as np
import pandas as pd

class _private():
    """Input: data - dataframe/series/list contaning desired data
       Function: convert item to list for processing
       Output: list containing the desired data"""
    def create_list(data,**args):
        if isinstance(data,pd.DataFrame):
            if 'header' in args:
                l = data[args['header']].to_list() #Convert column to list
            else:
                raise Exception('DataFrame found but no header set, please use "header" argument to set a column.')
        elif isinstance(data,pd.Series):
            l = data.to_list() #Convert to list
        elif isinstance(data,list):
            l = data #Set list as data
        else:
            raise Exception('Data variable not an acceptable type, please make sure you are using a DataFrame/Series/List.')
        return l

class encode_data():
    """Input: item - string containing the item type
              data - dataframe/series/list contaning all item types
       Function: encode item type using one hot encoding method
       Output: list containing boolean representation of item"""
    def one_hot(item,data,**args):
        l = _private.create_list(data,**args)
        if item in l:
            index = l.index(item) #Find ordinal location
            result = [1 if i == index else 0 for i,val in enumerate(l)] #Build vector
            return result
        else:
            result = [0]*len(l)
            return result

    """Input: item - string containing the item type
              data - dataframe/series/list contaning all item types
       Function: encode item type using a ordinal value
       Output: integer containing the encoded item type"""
    def ordinal(item,data,**args):
        l = _private.create_list(data,**args)
        if item in l:
            result = l.index(item) #Find ordinal location
            return result
        else:
            return 'NA'

    """Input: item - string containing the item type
              data - dataframe/series/list contaning all item types
       Function: encode item type using a binary vector
       Output: integer containing the encoded item type"""
    def binary(item,data,**args):
        l = _private.create_list(data,**args)
        if item in l:
            index = l.index(item) #Find ordinal location
            if 'bit' in args:
                result = list(np.binary_repr(index,args['bit']))
            else:
                x = 8 #Number of bits to use
                while True:
                    if len(item) <= 2**x:
                        result = list(np.binary_repr(index,x))
                        break
                    else:
                        x += 8 #Add another byte to x
            return result
        else:
            return 'NA'

    """Input: word - string containing the item type
              data - dataframe/series/list contaning all item types
              model - tensorflow model thats been trained to find the semantics of the data's contents
       Function: encode item type using a word vector
       Output: numpy array containing the word vector for the desired word"""
    def word_vector_2nd_layer(word,data,model,**args):
        l = _private.create_list(data,**args)
        w2 = model.get_weights()[4] #Extract second weight in model
        word_index = l.index(word) #Find index of the word in the data
        word_vector = np.array([row[word_index] for row in w2])
        return word_vector

if __name__ == "__main__":
    print(encode_data.one_hot('cat',['The','cat','is','fat']))
    print(encode_data.ordinal('cat',['The','cat','is','fat']))
    print(encode_data.binary('cat',['The','cat','is','fat']))
