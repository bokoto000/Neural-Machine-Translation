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
import torch.nn as nn
import torch.nn.functional as F

class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind, unkTokenIdx, padTokenIdx ):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w,unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName):
        self.load_state_dict(torch.load(fileName))
    
    def __init__(self, embed_size, hidden_size, word2indBg, word2indEn, startToken, padToken, unkToken, endToken, lstm_layers=2, dropout=0.3 ):
        super(NMTmodel, self).__init__()        

        #dictionaries
        self.word2indBg = word2indBg
        self.word2indEn = word2indEn

        #bulgarian tokens
        self.unkTokenIdxBg = word2indBg[unkToken]
        self.padTokenIdxBg = word2indBg[padToken]
        self.endTokenIdxBg = word2indBg[endToken]
        self.startTokenIdxBg = word2indBg[startToken]

        #english tokens
        self.unkTokenIdxEn = word2indEn[unkToken]
        self.padTokenIdxEn = word2indEn[padToken]
        self.endTokenIdxEn = word2indEn[endToken]
        self.startTokenIdxEn = word2indEn[startToken]

        #embed functions
        self.embedBg = nn.Embedding(len(word2indBg), embed_size)
        self.embedEn = nn.Embedding(len(word2indEn), embed_size)

        #encoder decoder lstm
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers=lstm_layers)
        self.decoder = nn.LSTM(embed_size, hidden_size, num_layers=lstm_layers)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, len(word2indBg))
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, source, target):
        #embeddings
        enBatch = self.preparePaddedBatch(source, self.word2indEn, self.unkTokenIdxEn, self.padTokenIdxEn)
        enEmbedded = self.embedEn(enBatch)
        bgBatch = self.preparePaddedBatch(target, self.word2indBg, self.unkTokenIdxBg, self.padTokenIdxBg)
        bgEmbedded = self.embedBg(bgBatch[:-1])

        #encode
        sourceLengths = [len(s) for s in source]
        output, (h, c) = self.encoder(
            torch.nn.utils.rnn.pack_padded_sequence(enEmbedded, sourceLengths, enforce_sorted=False))
        outputSource, _ = torch.nn.utils.rnn.pad_packed_sequence(output)

        #decode
        targetLengths = [len(t) - 1 for t in target]
        output, (h, c) = self.decoder(
            torch.nn.utils.rnn.pack_padded_sequence(bgEmbedded, targetLengths, enforce_sorted=False),
            (h, c))
        outputTarget, _ = torch.nn.utils.rnn.pad_packed_sequence(output)

        #attention
        attentionWeights = F.softmax((torch.bmm(outputSource.permute(1, 0, 2), outputTarget.permute(1, 2, 0))), dim=1)
        contextVector = torch.bmm(outputSource.permute(1, 2, 0), attentionWeights).permute(2, 0, 1)
        outputTarget = self.attention(torch.cat((contextVector, outputTarget), dim=-1))
        outputTarget = self.projection(self.dropout(outputTarget.flatten(0, 1)))

        return F.cross_entropy(outputTarget, bgBatch[1:].flatten(0, 1), ignore_index=self.padTokenIdxBg)

    def translateSentence(self, sentence, limit=1000):
        return result
