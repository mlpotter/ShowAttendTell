#!/usr/bin/env python
# coding: utf-8

# In[597]:


import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import os
from PIL import Image


# In[355]:


class flickr8000(Dataset):
    def __init__(self,IMAGE_PATH=r'dataset\images',TEXT_PATH=r'dataset\text'):
        
        self.IMAGE_PATH = IMAGE_PATH
        self.TEXT_PATH = TEXT_PATH
        
        self.dataset = pd.read_csv(os.path.join(self.TEXT_PATH,'captions.txt'))
        self.len = self.dataset.shape[0]

        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        list_of_words = sorted(list(set(" ".join(self.dataset['caption'].values.tolist()).split())))
        list_of_words.append("<START>")
        
        self.dictionary = dict.fromkeys(list_of_words)
        
        for i,word in enumerate(list(self.dictionary.keys())):
            one_hot_encoding = torch.zeros((1,len(self.dictionary)))
            one_hot_encoding[0,i] = 1
            self.dictionary[word] = one_hot_encoding
            
        self.dictionary_len = {}
        for i,text in enumerate(self.dataset['caption']):
            if text[-1] != ".":
                text = text + " ."
            text_size = len(text.split())
            if text_size not in self.dictionary_len:
                self.dictionary_len[text_size] = [i]
            else:
                self.dictionary_len[text_size].append(i)
                                
    def __getitem__(self,index):
        indices = self.dictionary_len[index]
        
        #image 
        image = Image.open(os.path.join(self.IMAGE_PATH,self.dataset.iloc[indices[0]]['image']))
        if self.transforms:
            image = self.transforms(image).unsqueeze(0)
        
        # text   
        text = self.dataset.iloc[indices[0]]['caption']
        if text[-1] != ".":
            text = text + " ."    
        text = text.split()
        text_encoded = self.dictionary[text[0]]
        for word in text[1:]:
            text_encoded = torch.cat((text_encoded,self.dictionary[word]),0)
        caption = text_encoded.unsqueeze(0)
                    
        for idx in indices[1:]:
            # image
            img = Image.open(os.path.join(self.IMAGE_PATH,self.dataset.iloc[idx]['image'])) 
            if self.transforms:
                img = self.transforms(img).unsqueeze(0)
            image = torch.cat( (image,img) ,0)
            
            # text
            text = self.dataset.iloc[idx]['caption']
            if text[-1] != ".":
                text = text + " ."
            text = text.split()
            text_encoded = self.dictionary[text[0]]
            for word in text[1:]:
                text_encoded = torch.cat((text_encoded,self.dictionary[word]),0)
            text_encoded.unsqueeze_(0)
            caption = torch.cat( (caption,text_encoded) ,0)
        
        print(int(len(indices)/index))
        print(len(indices))
        print(index)
        caption_labels = torch.cat((caption != 0).nonzero()[:,2:3].chunk(int(len(indices))),1).transpose(0,1)
        
        return image,caption,caption_labels
    
    def __len__(self):
        return self.len


# In[596]:


class flickr8000_subset(Dataset):
    def __init__(self,IMAGE_PATH=r'dataset\images',TEXT_PATH=r'dataset\text',sentence_len=11):
        
        self.IMAGE_PATH = IMAGE_PATH
        self.TEXT_PATH = TEXT_PATH
        
        self.dataset = pd.read_csv(os.path.join(self.TEXT_PATH,'captions.txt'))
        # fix the end character to always be "."
        self.dataset['caption'] = self.dataset['caption'].apply(lambda text: text+" ." if text[-1] != "." else text)
        # split the caption into a list of words
        self.dataset['caption'] = self.dataset['caption'].apply(str.split)
        # keep only caption and images of sentence length specified
        self.dataset = self.dataset[self.dataset['caption'].apply(len)==sentence_len].reset_index(drop=True)

        self.len = self.dataset.shape[0]
        self.sentence_len = sentence_len

        self.transforms =  transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        list_of_words = sorted(list(set(" ".join(self.dataset['caption'].apply(" ".join).values.tolist()).split())))
        list_of_words.append("<START>")
        self.dictionary = dict.fromkeys(list_of_words)
        
        for i,word in enumerate(list(self.dictionary.keys())):
            one_hot_encoding = torch.zeros((1,len(self.dictionary)))
            one_hot_encoding[0,i] = 1
            self.dictionary[word] = one_hot_encoding        
        
        self.dataset = self.dataset.values
                                
    def __getitem__(self,index):
        if not isinstance(index,slice):
            row = self.dataset[index,:]
            img = Image.open(os.path.join(self.IMAGE_PATH,row[0])) 
            if self.transforms:
                img = self.transforms(img)
            img.unsqueeze_(0)
            
            text = row[1]
            text_encoded = self.dictionary[text[0]]
            for word in text[1:]:
                text_encoded = torch.cat((text_encoded,self.dictionary[word]),0)
            text_encoded.unsqueeze_(0)
            
            labels = text_encoded.nonzero()[:,2].reshape(-1,self.sentence_len)
                        
            return img,text_encoded,labels
            
            
        images = []
        captions = []
        for row in self.dataset[index,:]:
            # load images
            img = Image.open(os.path.join(self.IMAGE_PATH,row[0])) 
            if self.transforms:
                img = self.transforms(img)
            img.unsqueeze_(0)
            images.append(img)
        
            # load captions
            text = row[1]
            text_encoded = self.dictionary[text[0]]
            for word in text[1:]:
                text_encoded = torch.cat((text_encoded,self.dictionary[word]),0)
            text_encoded.unsqueeze_(0)
            captions.append(text_encoded)
            
        images = torch.cat(images,0)
        captions = torch.cat(captions,0)
        labels = (captions !=0 ).nonzero()[:,2].reshape(-1,self.sentence_len)
                     
        return images,captions,labels
    
    def __len__(self):
        return self.len
