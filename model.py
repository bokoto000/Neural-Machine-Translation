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
        return output, (hidden , cell)

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
        self.word2ind = word2ind
        self.embedding =  torch.nn.Embedding(input_size, embedding_size)
        #print(embedding_size, input_size, output_size)
        self.lstm =  torch.nn.LSTM(hidden_size+embedding_size, hidden_size , num_layers=1 )
        self.fc =  torch.nn.Linear(hidden_size, output_size)
        self.energy = torch.nn.Linear(hidden_size * 2, 1024)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, input, encoder_states, hidden, cell):
        x = input.unsqueeze(0)
        # x: (1, N) where N is the batch size

        embedding = self.embedding(x)
        # embedding shape: (1, N, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)
        #print(h_reshaped.shape, encoder_states.shape)
        cat = torch.cat((h_reshaped, encoder_states), dim=2)
        #print(cat.shape)
        en = self.energy(cat)
        #print(en.shape)
        energy = self.relu(self.energy(cat))
        # energy: (seq_length, N, 1)

        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)
        #print(context_vector[-1].unsqueeze(0).shape, embedding.shape)
        rnn_input = torch.cat((context_vector[-1].unsqueeze(0), embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)
        #print(rnn_input.shape)
        outputs, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, hidden_size)

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
        encoder_states,(hidden,  cell) = self.encoder(source)
        x=target[0]
        
        for t in range( 1, target_len):
            output, hidden, cell = self.decoder (x, encoder_states, hidden,cell)
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

    def translateSentence(self, sentence, max_length=1000):
    # print(sentence)

    # sys.exit()
        def getWordFromIdx(dictionary, idx):
            if idx in dictionary.keys():
                return dictionary[idx]
            return 2
    # Load german tokenizer
        english = self.encoder.word2ind
        bulgarian = self.decoder.word2ind
        #print(english)
        # Create tokens using spacy and everything in lower case (which is what our vocab is)
        #print(english["IMF"])
        #toLower = lambda s: s[:1].lower() + s[1:] if s else ''
        tokens = [getWordFromIdx(english, word) for word in sentence]
        # print(tokens)
        #print(tokens)
        # sys.exit()
        # Add <SOS> and <EOS> in beginning and end respectively
        tokens.insert(0, english["<S>"])
        tokens.append(english["</S>"])

        # Convert to Tensor
        sentence_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)

        # Build encoder hidden, cell state
        with torch.no_grad():
            enc_res, (hidden, cell) = self.encoder(sentence_tensor)

        outputs = [bulgarian[startToken]]

        for _ in range(max_length):
            previous_word = torch.LongTensor([outputs[-1]]).to(device)

            with torch.no_grad():
                output, hidden, cell = self.decoder(previous_word,enc_res,  hidden, cell)
                best_guess = output.argmax(1).item()

            outputs.append(best_guess)

            # Model predicts it's the end of the sentence
            if output.argmax(1).item() == english["</S>"]:
                break
        #print(outputs)
        #print(bulgarian)
        revBulgarian ={v:k for k, v in bulgarian.items()}
        translated_sentence = [revBulgarian[idx] for idx in outputs]

    # remove start token
        return translated_sentence[1:]
