import torch 
import torch.nn as nn
import math
import torch.nn.functional as F
import pickle
import torch.optim as optim
from gensim.models import Word2Vec, KeyedVectors
import nltk
from nltk.util import ngrams
import numpy as np


d_model = 10
new_dim = 5


class feedforward_Encoder(nn.Module):
    def __init__(self):
        super(feedforward_Encoder,self).__init__()
        self.l1 = nn.Linear(10,40)
        self.l2 = nn.Linear(40,10)
        
    def forward(self,x):
        y = F.relu(self.l1(x))
        y = self.l2(y)
        return y
 

# Encoder
#######################################################################################
r = 10
c = 5
query_weights = abs(torch.rand((r,c)))
key_weights = abs(torch.rand((r,c)))
value_weights = abs(torch.rand((r,c)))

params = []
params = params + list(query_weights) + list(key_weights) + list(value_weights)

class Encoder():
    def __init__(self):
        self.d_model = d_model
        self.new_dim = new_dim
        self.positional_encodings = None
        self.first_sublayer_output = None
        self.keys = None
        self.values = None
        


    def PositionalEncoding(self,wordVecs):
        for pos in range(wordVecs.shape[0]):
            for i in range(wordVecs[pos].shape[0]):
                if i%2 == 0:
                    wordVecs[pos][i] = wordVecs[pos][i] + math.sin(pos/(10000**(2*i/self.d_model)))
                else:
                    wordVecs[pos][i] = wordVecs[pos][i] + math.cos(pos/(10000**(2*i/self.d_model))) 
                    
        self.positional_encodings = wordVecs
        return wordVecs
    
    
    
    def qkvs(self,vectorMatrix, new_dim):
        return torch.matmul(vectorMatrix, query_weights), torch.matmul(vectorMatrix, key_weights), \
        torch.matmul(vectorMatrix, value_weights) 
        # Check for transposeness in matrix multiplication
    
    
    def qk_dotproducts(self,queries, keys):
        dotproduct_matrix = torch.Tensor([])
        for i in queries:
            dotproduct_vector = torch.Tensor([])
            for j in keys:
                dotproduct_vector = torch.cat([dotproduct_vector, torch.dot(i,j).reshape(-1)])
            dotproduct_matrix = torch.cat([dotproduct_matrix, dotproduct_vector.reshape(1,-1)])
        return dotproduct_matrix
    
    
    def getSoftmaxed_qkdp(self,qk_dotproductmatrix):
        sm = nn.Softmax(dim = 0)
        sm_matrix = torch.tensor([])
        for i in qk_dotproductmatrix:
            sm_matrix = torch.cat([sm_matrix, sm(i).reshape(1,-1)])
        return sm_matrix
    
    
    def getSoftmaxWeightedValues(self,softmaxed_qkdp, values):
        dim2_mat = torch.tensor([])
        dim3_mat = torch.tensor([])
        outer_loop_range = softmaxed_qkdp.shape[0]
        inner_loop_range = values.shape[0]
        for i in range(outer_loop_range):
            for j in range(inner_loop_range):
                dim2_mat = torch.cat([dim2_mat, (softmaxed_qkdp[i][j]*values[j]).reshape(-1)])
            dim3_mat = torch.cat([dim3_mat, dim2_mat.reshape(1,values.shape[0],values.shape[1])])
            dim2_mat = torch.tensor([]) 
        return dim3_mat
    
    
    
    def getWeightedSum(self,softmax_weighted_values):
        next_layer_input = torch.tensor([])
        for i in softmax_weighted_values:
            transposed_i = i.t()
            new_word_representation = torch.tensor([])
            for j in transposed_i:
                rowsum = j.sum()
                new_word_representation = torch.cat([new_word_representation, rowsum.reshape(-1)])
            next_layer_input = \
            torch.cat([next_layer_input, new_word_representation.reshape(1,new_word_representation.shape[0])])    
        return next_layer_input
        
    
    
    def returnRepresentation(self, vectorRepresentations):
        pos_encoded = self.PositionalEncoding(vectorRepresentations)
        new_dim = self.new_dim
        queries, keys, values = self.qkvs(pos_encoded, new_dim)
        qk_dotproductmatrix = self.qk_dotproducts(queries, keys)
        d_k = keys.shape[1] # to be changed later to square root of 'key' vector dimension
        qk_dotproductmatrix/=(d_k**0.5)
        softmaxed_qkdp = self.getSoftmaxed_qkdp(qk_dotproductmatrix)
        softmax_weighted_values = self.getSoftmaxWeightedValues(softmaxed_qkdp, values)
        weightedSum = self.getWeightedSum(softmax_weighted_values)
        return weightedSum  
    
    
    def getW0(self):
        self.t = torch.randn(self.d_model, self.d_model).float()
        return self.t
    
    
    
    def multiHeadAttention(self, vectorRepresentations, heads=2):
        listOfHeads = []
        op = torch.tensor([])
        for i in range(heads):
            temp = self.returnRepresentation(vectorRepresentations)
            listOfHeads.append(temp)
    
        outputRepresentation = torch.tensor([])
        for i in range(listOfHeads[0].shape[0]):
            outputRepresentation = torch.cat([listOfHeads[0][i],listOfHeads[1][i]])
            op = torch.cat([op, outputRepresentation.reshape(1,outputRepresentation.shape[0])])
        
        W0 = self.getW0()
        projected_attention_vecs = torch.matmul(op, W0) 
        #Layer Normalisation
        layer_norm_one = nn.LayerNorm(projected_attention_vecs.size()[1])
        add_and_norm = layer_norm_one(projected_attention_vecs+self.positional_encodings)
        ##############   
        self.first_sublayer_output = add_and_norm
        return add_and_norm
    
    
    def ff_and_addnorm(self, vectorRepresentations):
        received_representations = self.multiHeadAttention(vectorRepresentations)
        activationlist = []
        activations = torch.tensor([])
        for i in received_representations:
            ffobj = feedforward_Encoder()
            activationlist.append(ffobj)
            activations = torch.cat([activations, activationlist[-1](i).reshape(1,received_representations.\
                                                                                shape[1])])
         
        layer_norm_two = nn.LayerNorm(activations.size()[1])
        add_and_norm = layer_norm_two(activations + self.first_sublayer_output)
        return add_and_norm
        
         
    def forward(self, vectorRepresentations):
        return self.ff_and_addnorm(vectorRepresentations)

########################################################################################



# Decoder
#########################################################################################
r = 10
c = 5
decoder_query_weights = abs(torch.rand((r,c)))
decoder_key_weights = abs(torch.rand((r,c)))
decoder_value_weights = abs(torch.rand((r,c)))


decoder_masked_query_weights = abs(torch.rand((r,c)))
decoder_masked_key_weights = abs(torch.rand((r,c)))
decoder_masked_value_weights = abs(torch.rand((r,c)))

W0 = torch.randn(d_model, d_model).float()
W1 = torch.randn(d_model, d_model).float()


encoder_keys = torch.randn(5,5)
encoder_values = torch.randn(5,5)


class Decoder():
    
    def __init__(self):
        self.vectorRepresentations = None
        self.positional_encodings = None
        self.d_model = d_model
        self.new_dim = new_dim
        self.maskedMultiHeadAttentionOutputVectors = None
        
        
        
    def PositionalEncoding(self,wordVecs):
        for pos in range(wordVecs.shape[0]):
            for i in range(wordVecs[pos].shape[0]):
                if i%2 == 0:
                    wordVecs[pos][i] = wordVecs[pos][i] + math.sin(pos/(10000**(2*i/self.d_model)))
                else:
                    wordVecs[pos][i] = wordVecs[pos][i] + math.cos(pos/(10000**(2*i/self.d_model))) 
                    
        self.positional_encodings = wordVecs
        return wordVecs
    
    

    def qkvs_Attention(self,vectorMatrix):
        return torch.matmul(vectorMatrix, decoder_query_weights)
    
    
    
    def qkvs_maskedAttention(self,vectorMatrix):
        return torch.matmul(vectorMatrix, decoder_masked_query_weights), torch.matmul(vectorMatrix, decoder_masked_key_weights), \
        torch.matmul(vectorMatrix, decoder_masked_value_weights) 
    
    
    def maskedMatrix(self,m,ind):
        returnMatrix = torch.tensor([]).float()
        for i in range(m.shape[0]):
            if i<=ind:
                returnMatrix = torch.cat([returnMatrix,m[i].unsqueeze(0)])
            else:
                returnMatrix = torch.cat([returnMatrix,torch.tensor([-float('Inf') for k in range(m.shape[1])]).float().unsqueeze(0)])
        
        return returnMatrix
    
    
    def dotProductMaskedMatrix(self,l,m2):
        returnMatrix = torch.tensor([]).float()
        for i in range(m2.shape[0]):
            returnMatrix = torch.cat([returnMatrix,torch.dot(l,m2[i]).reshape(-1)])
       
        
        return returnMatrix
    
    
    def qk_dotproducts_maskedAttention(self,queries, keys):
        finalMatrix = torch.Tensor([])
        for i in range(queries.shape[0]):
            b = self.maskedMatrix(queries,i)
            c = self.dotProductMaskedMatrix(b[i],b)
            finalMatrix = torch.cat([finalMatrix,c.unsqueeze(0)])
    
        return finalMatrix
    
    
    def qk_dotproducts_Attention(self,queries, keys):
        dotproduct_matrix = torch.Tensor([])
        for i in queries:
            dotproduct_vector = torch.Tensor([])
            for j in keys:
                dotproduct_vector = torch.cat([dotproduct_vector, torch.dot(i,j).reshape(-1)])
            dotproduct_matrix = torch.cat([dotproduct_matrix, dotproduct_vector.unsqueeze(0)])
            
        return dotproduct_matrix
    
    

    
 
    
     
    def getSoftmaxWeightedValues(self,softmaxed_qkdp, values):
        dim2_mat = torch.tensor([])
        dim3_mat = torch.tensor([])
        outer_loop_range = softmaxed_qkdp.shape[0]
        inner_loop_range = values.shape[0]
        for i in range(outer_loop_range):
            for j in range(inner_loop_range):
                dim2_mat = torch.cat([dim2_mat, (softmaxed_qkdp[i][j]*values[j]).reshape(-1)])
            dim3_mat = torch.cat([dim3_mat, dim2_mat.reshape(1,values.shape[0],values.shape[1])])
            dim2_mat = torch.tensor([]) 
        return dim3_mat
    
    
    
    def getWeightedSum(self,softmax_weighted_values):
        return softmax_weighted_values.sum(dim=0)
    
    
    
    
    def returnMaskedRepresentation(self, vectorRepresentations):
        pos_encoded = self.PositionalEncoding(vectorRepresentations)
        new_dim = self.new_dim
        queries, keys, values = self.qkvs_maskedAttention(pos_encoded)
        
        qk_dotproductmatrix = self.qk_dotproducts_maskedAttention(queries, keys)
        
        d_k = keys.shape[1] # to be changed later to square root of 'key' vector dimension
        qk_dotproductmatrix/=(d_k**0.5)
        

        
        qk_dotproductmatrix[:] = nn.Softmax(dim=1)(qk_dotproductmatrix)
    
    
        softmax_weighted_values = self.getSoftmaxWeightedValues(qk_dotproductmatrix, values)
        weightedSum = self.getWeightedSum(softmax_weighted_values)
        return weightedSum 
        
 
    
    def maskedMultiHeadAttention_add_norm(self, vectorRepresentations, heads=2):
        listOfHeads = []
        op = torch.tensor([])
        
        #Multiple Heads
        for i in range(heads):
            temp = self.returnMaskedRepresentation(vectorRepresentations)
            listOfHeads.append(temp)
    
        outputRepresentation = torch.tensor([])
        for i in range(listOfHeads[0].shape[0]):
            outputRepresentation = torch.cat([listOfHeads[0][i],listOfHeads[1][i]])
            op = torch.cat([op, outputRepresentation.reshape(1,outputRepresentation.shape[0])])
            
        
    
        projected_attention_vecs = torch.matmul(op, W0) 
        #Layer Normalisation
        layer_norm_one = nn.LayerNorm(projected_attention_vecs.size()[1])
        add_and_norm_one = layer_norm_one(projected_attention_vecs+self.positional_encodings)
        ##############   
        self.first_sublayer_output = add_and_norm_one
        return add_and_norm_one
        
     
    
    def returnRepresentation(self, vectorRepresentations):
        inp_vectors = self.maskedMultiHeadAttention_add_norm(vectorRepresentations)
        self.maskedMultiHeadAttentionOutputVectors = inp_vectors
        

        queries = self.qkvs_Attention(inp_vectors)
        
        keys = torch.matmul(Encoder_output, decoder_query_weights)
        values = torch.matmul(Encoder_output, decoder_value_weights)
        
        qk_dotproductmatrix = self.qk_dotproducts_Attention(queries, keys)
        d_k = keys.shape[1] # to be changed later to square root of 'key' vector dimension
        qk_dotproductmatrix/=(d_k**0.5) #In paper we divide by sqrt(d_k) that is sqrt(64) = 8
        

        
        softmaxed_qkdp = nn.Softmax(dim = 1)(qk_dotproductmatrix)
        
        softmax_weighted_values = self.getSoftmaxWeightedValues(softmaxed_qkdp, values)
        weightedSum = softmax_weighted_values.sum(dim = 0)
        return weightedSum  
    
    
    
    
    def multiHeadAttention_add_norm(self, vectorRepresentations, heads=2):
        
        listOfHeads = []
        op = torch.tensor([])
        
        #Multiple Heads
        for i in range(heads):
            temp = self.returnRepresentation(vectorRepresentations)
            listOfHeads.append(temp)
            
   
            
        outputRepresentation = torch.tensor([])
        for i in range(listOfHeads[0].shape[0]):
            outputRepresentation = torch.cat([listOfHeads[0][i],listOfHeads[1][i]])
            op = torch.cat([op, outputRepresentation.reshape(1,outputRepresentation.shape[0])])
        
    
      
        

        projected_attention_vecs = torch.matmul(op, W1) 
      
    
        #Layer Normalisation
        layer_norm_two = nn.LayerNorm(projected_attention_vecs.size()[1])
        add_and_norm_two = layer_norm_two(projected_attention_vecs + self.maskedMultiHeadAttentionOutputVectors)
        ##############   
        self.first_sublayer_output = add_and_norm_two
        return add_and_norm_two
    
    
    
    def ff_and_addnorm(self, vectorRepresentations):
        received_representations = self.multiHeadAttention_add_norm(vectorRepresentations)
        
        
        
        activations = torch.tensor([])
        ffobj = feedforward_Encoder()
        for i in received_representations:
            activations = torch.cat([activations, ffobj(i).unsqueeze(0)])
         
        
        
        layer_norm_three = nn.LayerNorm(activations.size()[1])
        add_and_norm_three = layer_norm_three(activations + received_representations)
        return add_and_norm_three
    
    
    def forward(self, vectorRepresentations):
        return self.ff_and_addnorm(vectorRepresentations)
    

#################################################################################################


# Doing this because input sentences have to be limited to corpus 
##################################################################################################
germanSens = pickle.load(open(f'subsampledGermanSens.pkl', 'rb'))
englishSens = pickle.load(open(f'subsampledEnglishSens.pkl', 'rb'))


# Padding english and german sentences to same length
def initialPadding(germanSens, englishSens):
    germanVecs = [nltk.word_tokenize(sentence.lower()) for sentence in germanSens]
    englishVecs = [nltk.word_tokenize(sentence.lower()) for sentence in englishSens]
    
    germanVecs = [['sos']+i+['eos'] for i in germanVecs]
    englishVecs = [i+['eos'] for i in englishVecs]
    
    newgermanVecs = []
    newenglishVecs = []
    
    for i in range(100):
        if len(germanVecs[i])<=30 and len(englishVecs[i])<=30:
            newgermanVecs.append(germanVecs[i])
            newenglishVecs.append(englishVecs[i])
    
    
    for i in range(len(newenglishVecs)):
        len_en = len(newenglishVecs[i])
        len_de = len(newgermanVecs[i])
        
        
        for j in range(32 - len_de):
            newgermanVecs[i].append('ppd')
        
        for j in range(32 - len_en):
            newenglishVecs[i].append('ppd')
                
           
    return newgermanVecs, newenglishVecs





germanVecs, englishVecs = initialPadding(germanSens, englishSens)
modelEnglish = Word2Vec(englishVecs, min_count=1, size=10)
modelGerman = Word2Vec(germanVecs, min_count=1, size=10)

#creating word2index for german with 'sos'
word2index_german = {}
ind = 0

for k in modelGerman.wv.vocab:
    word2index_german[k] = ind
    ind+=1

###########################################################################################3

def getWordVecs(listOfTokens, lang):
    wvecs = torch.tensor([]).float()
    
    for i in listOfTokens:
        if lang == 'en':wvecs = torch.cat([wvecs, torch.from_numpy(modelEnglish.wv[i]).unsqueeze(0)])
        else: wvecs = torch.cat([wvecs, torch.from_numpy(modelGerman.wv[i]).unsqueeze(0)])
        
    return wvecs


#final projection to vocabulary in decoder
mapToVocab = nn.Sequential(
             nn.Linear(10,len(word2index_german)-1),
             nn.ReLU(),
             nn.Softmax(dim = 0)
)



# Inference
eng_example = input('Enter Sentence in English\n')
englishVec = nltk.word_tokenize(eng_example) + ['eos']



englishVec = getWordVecs(englishVec,'en')
each_sentence_length = englishVec.shape[0]

germ = getWordVecs(['sos' for _ in range(englishVec.shape[0])],'de')
temp = germ.clone()

e = Encoder()
d = Decoder()

for i in range(each_sentence_length):
    Encoder_output = e.forward(englishVec)
    Decoder_output = d.forward(germ)
    
    temp[i] = Decoder_output[i]
    germ = temp.clone()
    # print(germ)
    





#creating a clone of word2index removing 'sos'
w2i = word2index_german.copy()
w2i.pop('sos')


predOneHotVector_inference = torch.zeros(len(w2i)).long()
ind = 32
translated_string = ''
#print(eng_example)
print()    
mappings = [torch.argmax(i) for i in germ]
for i in mappings:
    predOneHotVector_inference[i] = 1
    for k in w2i:
        if w2i[k] == i:
            translated_string+=(k+' ')
 

# Finally printing translation           
print(translated_string)

