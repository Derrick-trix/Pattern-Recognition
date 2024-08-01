
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


#------------------------------------------------------------------------------------------------------------------------------------------

#Defining all the functions required

def char_encoder(file_name):
    # read and convert to lower case
    raw_text = open(file_name, "r", encoding="utf-8").read()
    raw_text = raw_text.lower()
    chars = sorted(list(set(raw_text)))
    # create mapping of unique chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    # creating  the decoder for above map
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    vocab_size = len(chars)

    # List to hold character integers
    list_of_character_ints =[]
    
    for char in raw_text:
        list_of_character_ints.append(char_to_int[char])
    
    int_array= np.array(list_of_character_ints).reshape(-1, 1)
    # Encode each int value
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int8)
    one_hot_encoded_matrix = encoder.fit_transform(int_array)
    return one_hot_encoded_matrix, vocab_size, char_to_int, int_to_char


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.in_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in_to_output = nn.Linear(input_size + hidden_size, output_size)
        
    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), 1)
        # I always provide both input character encoded and the hidden state(previous) encoded
        # I get two different outputs from the network
        # 1: The predicted character encoded
        # 2: The next hidden state encoded 
        hidden = torch.sigmoid(torch.nn.functional.linear(combined,self.in_to_hidden.weight.clone() , self.in_to_hidden.bias))
        output = torch.nn.functional.linear(combined, self.in_to_output.weight.clone(), self.in_to_output.bias)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


def train_rnn(hidden_state,one_hot_mat, model, num_epochs, optimizer, loss_fn):

    train_data_temp = one_hot_mat[:-1]
    train_labels_temp = one_hot_mat[1:]
    #considering just 200 characters for training since it takes so long time. It can seen the model performs good just by learning initial characters.
    train_data = torch.tensor(train_data_temp[:200])
    train_labels = torch.tensor(train_labels_temp[:200])

    # Creating a dataloader
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=1)

    # Training loop
    #hidden_state = model.init_hidden()
    #print(hidden_state)
    for n in range(num_epochs):
        loss_e =0
        preds =[]
        for train_d, test_d in train_loader:
            y_pred, hidden_state = model(train_d.float(), hidden_state)
            preds.append(y_pred)
            #print(y_pred)
            #print(hidden_state)
            loss = loss_fn(y_pred, test_d.float())
            # print(loss)
            loss_e += loss.item()*train_d.size(0)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print(f"loss at epoch {n+1}: {loss_e/len(train_loader.sampler)}")

    return(train_data,train_labels,preds)

def get_prediction(model, input_tensor):
    #hidden_state= model.init_hidden()
    y_pred,_= model(input_tensor.unsqueeze(0).float(), hidden_state)
    return y_pred;


def  evaluate_per_sequence(model, train_data, train_labels, random_index,seq_len): 
    
    #hidden_state= model.init_hidden()
    
    accuracy_test_data = train_data[random_index:random_index+seq_len]
    accuracy_test_labels = train_labels[random_index:random_index+seq_len]
    
    predicted_ints =[]
    predicted_letters = []
    #print("accuracy_test_data", accuracy_test_data)
    #print("accuracy_test_labels", accuracy_test_labels)
    for test_d in accuracy_test_data:
        y_pred,_= model(test_d.unsqueeze(0).float(), hidden_state)

        # Pluck the higest output value's position as the predicted encoding 
        encoded_letter = np.argmax(y_pred.detach().numpy())
        predicted_ints.append(encoded_letter)

        # Covert to a character
        decoded_letter = int_to_char[encoded_letter]
        predicted_letters.append(decoded_letter)

    actual_ints = []
    actual_letters =[]
    for test_l in accuracy_test_labels:
        actuals_array = test_l.detach().numpy()
        encoded_int = np.argmax(actuals_array)
        actual_ints.append(encoded_int)
    
        decoded_actual = int_to_char[encoded_int]
        actual_letters.append(decoded_actual)

    print(f"Model performance for randomly selected index {random_index}")
   
    pred_sentence = ""
    for pred_letter in predicted_letters:
        pred_sentence += pred_letter
    print("Predicted sentence: ", pred_sentence)
    
    actual_sentence = ""
    for act_letter in actual_letters:
        actual_sentence += act_letter
    print("Actual sentence: ", actual_sentence)

    accuracy = (1-(np.count_nonzero(np.array(predicted_ints) - np.array(actual_ints))/len(actual_ints))) *100
    print("Accuracy : ", accuracy)   


def evaluate_per_pure_text(model,train_data,train_labels,random_index): 

    input_tensor = train_data[random_index]
    pred_letter = get_prediction(model,input_tensor)
    #predicted_int = np.argmax(pred_letter.detach().numpy())
    
    preds = ""
    predicted_ints =[]
    for i in range(50):
        predicted_int = np.argmax(pred_letter.detach().numpy())
        predicted_ints.append(predicted_int)
        predicted_char = int_to_char[predicted_int]
        preds += predicted_char
        new_vector = np.zeros(vocab_size)
        new_vector[predicted_int] = 1
        pred_letter = get_prediction(model, torch.tensor(new_vector))
    print("predicted sequence: ", preds)
    
    actual_encoded = train_labels[random_index: random_index+50]
    actuals =""
    actual_ints =[]
    for label in actual_encoded:
        actuals_array = label.detach().numpy()
        encoded_int = np.argmax(actuals_array)
        actual_ints.append(encoded_int)
        decoded_actual = int_to_char[encoded_int]
        actuals += decoded_actual
    
    print("Actual sequence: " , actuals)
    accuracy = (1-(np.count_nonzero(np.array(predicted_ints) - np.array(actual_ints))/len(actual_ints))) *100
    print("Accuracy : ", accuracy)

#------------------------------------------------------------------------------------------------------------------------------------------


#Training and testing for file 1


one_hot_mat, vocab_size, char_to_int, int_to_char = char_encoder("abcde.txt")

#defining model paramters
hidden_state_size = 14
learning_rate = 0.05
num_epochs = 10

#defining the model
rnn = RNN(input_size=vocab_size, hidden_size= hidden_state_size, output_size=vocab_size)
#just initialising the hidden states once and training for bothe the files which will capture the details about about both files
hidden_state = rnn.init_hidden()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
print("Model architecture: \n\n",rnn)

print("\nTraining with file 1: abcde.txt\n")
#training the model
train_data, train_labels, _ = train_rnn(hidden_state, one_hot_mat, rnn, num_epochs, optimizer, loss_fn)


# Chose index 120 as my random selection for now to test for a sequence of inputs
print("\nTesting for a fixed sequence\n")
random_index = 0
seq_len = 50
evaluate_per_sequence(rnn, train_data, train_labels, random_index,seq_len)


# Evaluate for random index 120
print("\nTesting for pure text generation\n")
evaluate_per_pure_text(rnn,train_data,train_labels,120)


#------------------------------------------------------------------------------------------------------------------------------------------


#Training and testing for file 2

print("\nTraining with file 2: abcde_edcba.txt\n")

#encoding text
one_hot_mat, vocab_size, char_to_int, int_to_char = char_encoder("abcde_edcba.txt")

#setting up model parameters
hidden_state_size = 14
learning_rate = 0.05
num_epochs = 10

#defining model
#rnn = RNN(input_size=vocab_size, hidden_size= hidden_state_size, output_size=vocab_size)
#loss_fn = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

#training the model
train_data, train_labels, _ = train_rnn(hidden_state,one_hot_mat, rnn, num_epochs, optimizer, loss_fn)

# Chose index 120 as my random selection for now to test for a sequence of inputs
print("\nTesting for a sequence\n")
random_index = 120
seq_len = 50
evaluate_per_sequence(rnn, train_data, train_labels, random_index,seq_len)

# Evaluate for random index 120
print("\nTesting for pure text generation\n")
evaluate_per_pure_text(rnn, train_data,train_labels,120)
