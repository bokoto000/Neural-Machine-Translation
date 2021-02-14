import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'

corpusDataFileName = 'corpusData'
wordsDataFileName = 'wordsData'
modelFileName = 'NMTmodel'

startToken = '<S>'
endToken = '</S>'
unkToken = '<UNK>'
padToken = '<PAD>'

#device = torch.device("cuda:0")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


parameter1 = 1
parameter2 = 2
parameter3 = 3
parameter4 = 4

uniform_init = 0.1
learning_rate = 0.005
clip_grad = 5.0
learning_rate_decay = 0.5

batchSize = 8

embedding_size=32
hidden_size=512

maxEpochs = 3
log_every = 10
test_every = 2000

max_patience = 5
max_trials = 5
