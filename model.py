#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import torch
from parameters import *
import random

class Encoder(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        x = torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
        #print("encoder x", x.shape)
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
    
    def __init__(self, input_size, embedding_size, hidden_size, word2ind , unkToken, padToken, endToken):
        super(Encoder, self).__init__()
        self.word2ind = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(input_size, embedding_size)
        self.lstm =  torch.nn.LSTM(embedding_size, hidden_size)
        self.dropout = torch.nn.Dropout(0)

    def forward(self, input):
        #print(input)
        embedded = self.dropout(self.embedding(input))

        embedding = self.embedding(input)
        
        output, (hidden, cell) = self.lstm(embedding)
        return output, hidden

class Decoder(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        x = torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
        #print("decoder x", x.shape)
        return x
    
    def __init__(self, input_size, embedding_size, hidden_size, output_size,word2ind,unkToken, padToken, endToken):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]
        self.embedding =  torch.nn.Embedding(input_size, embedding_size)
        print(embedding_size, input_size, output_size)
        self.lstm =  torch.nn.LSTM(embedding_size, hidden_size , num_layers=1 )
        self.fc =  torch.nn.Linear(hidden_size, output_size)


    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        #shape of x:(N) but we want (1,N)
        #print("input shape", input.shape)
        embedding = self.embedding(input)
        #print("embed shape", embedding.shape)
        hiddenLastLayer = (hidden.shape)[0]-1
        hidden = hidden[hiddenLastLayer]
        hidden = hidden.unsqueeze(0)
        #print(hidden.shape)
        #print("emb",embedding.shape)
        output, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        #print("output shape", output.shape)
        predictions = self.fc(output)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell



class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName):
        self.load_state_dict(torch.load(fileName))
    
    def __init__(self, encoder, decoder, targetWord2Ind, sourceWord2Ind):
        super(NMTmodel, self).__init__()  
        self.encoder = encoder
        self.decoder = decoder
        self.targetWord2Ind = targetWord2Ind
        self.sourceWord2Ind = sourceWord2Ind
    
    def forward(self, source, target, teacher_force_ratio=0.5):
        source = self.encoder.preparePaddedBatch(source, self.sourceWord2Ind )
        target = self.decoder.preparePaddedBatch(target, self.targetWord2Ind)
        batch_size = source.shape[1]
        target_len = target.shape[0]
        #print(batch_size, target_len)
        target_vocab_size = len(self.targetWord2Ind)
        #print(target_vocab_size)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        #print(source)
        hidden, cell = self.encoder(source)
        x=target[0]
        
        for t in range( 1, target_len):
            output, hidden, cell = self.decoder (x,hidden, cell)
            #print(output.shape)
            #print()
            outputs[t]= output

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess
        
        output = outputs[1:].reshape(-1, outputs.shape[2])
        #print("output shape ", output.shape)
        target = target[1:].reshape(-1)
        #print("target shape", target.shape)
        output = torch.nn.functional.cross_entropy(output,target,ignore_index=self.decoder.padTokenIdx)
        return output

    def translateSentence(self, sentence, limit=1000):
        return result
