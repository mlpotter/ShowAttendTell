#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision.models as models

import torch
from torch import nn


# In[2]:


from torchvision import transforms


# In[3]:


import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
from PIL import Image


# In[4]:


class LSTM(nn.Module):
    """
    embedding_size = m
    vocabulary_size (number of unique words) = K
    hidden_size (LSTM dimensionality) = n
    context_size (context vector size) = D
    num_context_vec (number of context vectors) = L
    
    """
    def __init__(self,vocabulary_size,embedding_size=512,hidden_size=512,context_size=512,attention_dim=512,num_context_vec=14*14):
        super(LSTM,self).__init__()
        
        self.embedding_size=embedding_size
        self.vocabulary_size=vocabulary_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_context_vec = num_context_vec
        self.attention_dim = attention_dim
        
        
        # Embedding matrices
        self.Lo = nn.Linear(embedding_size,vocabulary_size)
        self.Lh = nn.Linear(hidden_size,embedding_size)
        self.Lz = nn.Linear(context_size,embedding_size)
        self.E = nn.Linear(vocabulary_size,embedding_size)
        print("Initialized embedding matrices for p(y | a, y_prev)")
    
    
        # initial memory state and hidden state
        self.f_init_c = nn.Linear(context_size,hidden_size)
        self.f_init_h = nn.Linear(context_size,hidden_size)
        print("Initialized memory state and hidden state fc layers for LSTM")
        
        # soft attention
        #self.f_att = nn.Linear(hidden_size+context_size,1)
        self.f_att_dec = nn.Linear(hidden_size,attention_dim)
        self.f_att_enc = nn.Linear(context_size,attention_dim)
        self.f_full_att = nn.Linear(attention_dim,1)
        
        print("Initialized soft version of attention mechanism")
        
        # Beta for object focus
        self.gate_scalar = nn.Linear(hidden_size,1)
        print("Beta Initialized")
    
        self.LSTM = nn.LSTM(input_size=context_size+hidden_size+embedding_size,hidden_size=hidden_size,batch_first=True)
        print("Initialized LSTM")
        
    def init_hidden(self,a):
        c0 = self.f_init_c(torch.mean(a,dim=2).unsqueeze(dim=1))
        h0 = self.f_init_h(torch.mean(a,dim=2).unsqueeze(dim=1))
        return h0,c0
        
    def forward(self,a_i,input,hn_prev,cn_prev):
        # initial hidden state and cell state
        
        # attention model biased on previous hidden state     
        e_ti1 = self.f_att_enc(a_i.permute(0,2,1))
        e_ti2 = self.f_att_dec( hn_prev )
        att = self.f_full_att( torch.relu(e_ti1 + e_ti2) ).squeeze(2).unsqueeze(1)
        alpha_ti = torch.softmax(att,dim=2)
        
        # gate output
        beta = torch.sigmoid( self.gate_scalar(hn_prev) )

        # context vector
        z_expectation = beta * torch.sum(alpha_ti*a_i,dim=2).unsqueeze(dim=1)
       
        #word embedding
        w_embedding = self.E(input)
        
        _,(hn,cn) = self.LSTM(torch.cat( (w_embedding,hn_prev,z_expectation) ,dim=2),(hn_prev.squeeze(1).unsqueeze(0),cn_prev.squeeze(1).unsqueeze(0)))
        
        # nt
        #p_yt = torch.softmax( torch.exp( self.Lo( w_embedding + self.Lh(hn.squeeze(0).unsqueeze(1)) + self.Lz(z_expectation) ) ) ,dim=2)
        p_yt = self.Lo( w_embedding + self.Lh(hn.squeeze(0).unsqueeze(1)) + self.Lz(z_expectation) )
        
        return p_yt,hn.squeeze(0).unsqueeze(1),cn.squeeze(0).unsqueeze(1),alpha_ti
        
    def __str__(self):
        return "Dimension information:\nm={}\nK={}\nn={}\nD={}L={}\n".format(self.embedding_dim,self.vocabulary_size,self.hidden_size,self.context_size)

# In[7]:


# In[7]:


class VGG:
    def __init__(self):
        self.vgg19 = models.vgg19(pretrained=True).features[:35]
        for param in self.vgg19.parameters():
            param.requires_grad=False
        
    def __call__(self,input):
        with torch.no_grad():
            return self.vgg19(input).view(-1,512,14*14)


# In[ ]:


if __name__ == "__main__":
    print("Running main script")
    
    from torchvision import transforms
    import urllib
    url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)
    from PIL import Image
        
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    lstm = LSTM(128,200,64,512,14*14)
    vgg = VGG()
    ai = vgg(input)
    
    c0,h0 = lstm.init_hidden(ai)
    yt = torch.randn((1,1,lstm.vocabulary_size))
    f = lstm.forward(ai,yt,c0,h0)
                      
    for obj in f:
        print(obj.size())
                      
    criterion = nn.NLLLoss()
    label = torch.tensor([2])
    loss = criterion(f[0].squeeze(0),label)
    loss.backward()

