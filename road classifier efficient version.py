import sys
import numpy as np
import nnfs
import math
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import random
from multiprocessing import Pool, Process, Queue

images_tested = 0

def load_images_from_folder(folder):
    images = []
    folderlist = os.listdir(folder)

    for i in range(0,10):
        ranimg = random.randint(0,5000)-1
        img = mpimg.imread(os.path.join(folder,folderlist[ranimg]))
        if img is not None:
            images.append(img)
    return images

## DATA
def load_imgs(one_hot):
    center_list = load_images_from_folder("C:/Users/riyad/OneDrive/Desktop/Neural network/test drive/Data/train/center")
    right_list = load_images_from_folder("C:/Users/riyad/OneDrive/Desktop/Neural network/test drive/Data/train/right")
    left_list = load_images_from_folder("C:/Users/riyad/OneDrive/Desktop/Neural network/test drive/Data/train/left")
    #center_list = load_images_from_folder("C:/Users/Server/Downloads/archive/Data/train/center")
    #right_list = load_images_from_folder("C:/Users/Server/Downloads/archive/Data/train/left")
    #left_list = load_images_from_folder("C:/Users/Server/Downloads/archive/Data/train/right")

    images_list = [left_list, center_list, right_list]

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], np.array([0.2989, 0.5870, 0.1140]))

    grays = []

    for set in images_list:
        for image in set:
            gray = rgb2gray(image)
            grays.append(gray.transpose())


    X = grays

        ## SHUFFLING 
    shuffling_array = []
    for i in range(0,30):
        shuffling_array.append(i)
    shuffling_array = np.array(shuffling_array)
    np.random.shuffle(shuffling_array)

    for i in range(0,30):
        one_hot[i] = one_hot[shuffling_array[i]]
        grays[i] = grays[shuffling_array[i]]
    return X, one_hot


#plt.imshow(gray, cmap=plt.get_cmap('gray'))
#plt.show()


##WEIGHTS

## BEGNNING OF NEW NEURON IN PATH
#how best tune weights and bias to achieve desired output



class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons).astype(np.float32) ## weights
        self.biases = 0.10*np.random.randn(1,n_neurons).astype(np.float32)
    def forward(self, inputs):
        self.output = np.dot(inputs,self.weights) + self.biases


class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs).astype(np.float32)

class Convolutional_Filter:
    def __init__(self,width,height):
        self.weights = 0.10*np.random.randn(width,height).astype(np.float32) # weights
        self.bias = 0.10*np.random.randn(1,1).astype(np.float32)
        self.width = width
        self.height = height
    def forward(self,input):
        input = input.astype(np.float32)/10000000.0
        x = -1
        feature_map = [[]]
        
        for i in range(0,len(input)-self.width+1,64): # rows -1
            x+=1
            for j in range(0,(len(input[i])-self.height+1),36): # column -1 omitted
                section = input[i:i+self.width,j:j+self.height]
                convulsion = np.sum(np.inner(section,self.weights)) + self.bias
                convulsion = float(convulsion)
                feature_map[x].append(convulsion)
            feature_map.append([])
        feature_map.pop()
        npft_map = np.asarray(feature_map)
        self.output = npft_map

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True)).astype(np.float32)
        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True).astype(np.float32)
        self.output = probabilities

class Loss:
    def calculate(self,output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses).astype(np.float32)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,10^-7,(1-(10^-7))).astype(np.float32)



        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1).astype(np.float32)

        negative_log_likelihoods = -np.log(correct_confidences).astype(np.float32)
        return negative_log_likelihoods

def filter2(input_array): # searches 5by5 grid

    final_array = []
    for i in range(0,len(input_array),5):
        for j in range(0,(len(input_array[i])),5):
            section = input_array[i:i+5,j:j+5]
            nmax = np.max(section).astype(np.float32)
            final_array.append(nmax)
    final_array = np.array(final_array)
    return final_array

def derive(y2,y1,dx):
    return (y2-y1)/dx


lossfx = Loss_CategoricalCrossentropy()



def initialize(layers):
    pass

## vars

denses = []
activations = []

## initialize layers

denses.append(Layer_Dense(16,32))
denses.append(Layer_Dense(32,64))
denses.append(Layer_Dense(64,128))
denses.append(Layer_Dense(128,256))
denses.append(Layer_Dense(256,128))
denses.append(Layer_Dense(128,64))
denses.append(Layer_Dense(64,3))

activations.append(Activation_ReLU())
activations.append(Activation_ReLU())
activations.append(Activation_ReLU())
activations.append(Activation_ReLU())
activations.append(Activation_ReLU())
activations.append(Activation_ReLU())
activations.append(Activation_Softmax())

## FILTER REQUIREMENTS
filter1 = Convolutional_Filter(960,540)
activation1 = Activation_ReLU()


# load denses
curfile = os.getcwd()
for i in range(0,len(denses)):
    try: 
        weight_save = np.load(curfile + "\\" + 'Dense ' + str(i) + ".npy") 
        denses[i].weights = weight_save
    except: weight_save = None

    try: 
        bias_save = np.load(curfile + "\\" + 'Dense Bias ' + str(i) + ".npy") 
        denses[i].biases = bias_save
    except: bias_save = None
        

try: 
    filter_save = np.load(curfile + "\\" + 'Filter ' + str(0) + ".npy") 
    filter1.weights = filter_save
except: 
    filter_save = None
try: 
    filterbias_save = np.load(curfile + "\\" + 'Filter Bias ' + str(0) + ".npy") 
    filter1.biases = filterbias_save
except: 
    filterbias_save = None



# save denses
def save():
    for i in range(0,len(denses)):
        print('Dense ' + str(i))
        weight_save = np.save('Dense ' + str(i), denses[i].weights)
        bias_save = np.save('Dense Bias' + str(i), denses[i].biases)

    filter_save = np.save('Filter ' + str(0), filter1.weights)
    filterbias_save = np.save('Filter Bias ' + str(0), filter1.bias)



precision = 0.01
## training functions
one_hot_org = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]) # EXPECTED OUTPUT, 0 LEFT, 1 CENTER, 2 RIGHT



def run(input):

    for i in range(0,len(denses)):
        if i == 0:
            denses[i].forward(input)
        else:
            denses[i].forward(activations[i-1].output)
        activations[i].forward(denses[i].output)
    loss = lossfx.calculate(activations[len(denses)-1].output,one_hot)
    return loss

def run_s(prevout,layerum):
    denses[layerum].forward(prevout)
    for i in range(layerum, len(denses)):
        denses[i].forward(activations[i-1].output)
        activations[i].forward(denses[i].output)
    loss = lossfx.calculate(activations[len(denses)-1].output,one_hot)
    return loss


def adjust_val(medium,input, layernum, prevout):
        org_medium = medium.copy()
        orgloss = run(input)
        for h in range(0,len(medium)):
            for l in range(0,len(medium[h])):
                if layernum == 0:
                    loss1 = run(input)
                    medium[h][l]+=precision
                    loss2 = run(input)
                    medium[h][l]-=precision
                    diff = derive(loss2,loss1,precision) ### FIND SLOPE OF TANGENT HERE

                    original = medium[h][l]
                    #lose1 = run(input)
                    if diff > 0:
                        medium[h][l]-=precision
                    elif diff < 0:
                        medium[h][l]+=precision
                    '''lose2 = run(input)
                    if lose2 >= lose1:
                        medium[h][l] = original'''

                else:

                    loss1 = run_s(prevout,layernum)
                    medium[h][l]+=precision
                    loss2 = run_s(prevout,layernum)
                    medium[h][l]-=precision
                    diff = derive(loss2,loss1,precision) ### FIND SLOPE OF TANGENT HERE

                    #lose1 = run_s(prevout,layernum)
                    if diff > 0:
                        medium[h][l]-=precision
                    elif diff < 0:
                        medium[h][l]+=precision
                    '''lose2 = run_s(prevout,layernum)
                    if lose2 >= lose1:
                        medium[h][l] = original'''
            #CHANGE BIAs

        final_loss = run(input)
        if final_loss > orgloss:
            medium = org_medium


                




def adj_filter(medium, rawinput,orginput):
    org_medium = medium.copy()
    orgloss = run(orginput)
    for h in range(0,len(medium),192): # 10 TIMES, 96
        #print(h/len(medium))
        for l in range(0,len(medium[h]),108): # 10 TIMES, 54
###############################
            filter1.forward(rawinput)
            activation1.forward(filter1.output)
            org_input = filter2(activation1.output) ## NEW INPUT 9

            loss1 = run(org_input)
            medium[h:h+192][l:l+108]+=precision

            filter1.forward(rawinput)
            activation1.forward(filter1.output)
            new_input = filter2(activation1.output) ## NEW INPUT 9

            loss2 = run(new_input)
            medium[h:h+192][l:l+108]-=precision
            diff = derive(loss2,loss1,precision)
####################
            if diff > 0:
                medium[h:h+192][l:l+108]-=precision
            elif diff < 0:
                medium[h:h+192][l:l+108]+=precision
    filter1.forward(rawinput)
    activation1.forward(filter1.output)
    new_input = filter2(activation1.output)
    final_loss = run(new_input)
    if final_loss > orgloss:
        print(" FILTER WORSE")
        medium = org_medium
            

last_loss = 0


losses = []
accuracies = []

def worker(data_img, one_hot):

    ## ADJUST FILTER
    filter1.forward(data_img)
    activation1.forward(filter1.output)
    max_pool = filter2(activation1.output) ## NEW INPUT 9
    #print("ADJUSTING FILTER")
    adj_filter(filter1.weights,data_img,max_pool)

    ## ADJUST BACKEND NEURAL NETWORK WEIGHTS FOR 9 INPUT
    #print("ADJUSTING WEIGHTS")
    for i in range(0,len(denses)):
        prevout = None
        if i != 0:
            prevout = activations[i-1].output
        adjust_val(denses[i].weights,max_pool,i,prevout)
        adjust_val(denses[i].biases,max_pool,i,prevout)
    losses.append(run(max_pool))
    softmax_out = activations[len(denses)-1].output
    predictions = np.argmax(softmax_out,axis=1)

    accuracy = np.mean(predictions == one_hot) #  y is targets
    accuracies.append(accuracy)
    #print(accuracy)


while True:
    losses = []
    accuracies = []
    one_hot = one_hot_org.copy()
    # GEN NEW IMG
    X, one_hot = load_imgs(one_hot)

    

    iteration = -1
    ## BACKPROPOGATION
    for imagedata in X:
        worker(imagedata,one_hot)
        
        iteration +=1
        print("Image ", iteration, "/30")
    losses = np.array(losses)
    print(np.mean(losses))
    
    print(np.mean(accuracies))
    save()
    images_tested+=30
    print(images_tested)
    





#### TRAINING ########################################################

