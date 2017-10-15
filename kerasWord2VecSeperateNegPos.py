#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the GNU Affero General Public License, version 3 - http://www.gnu.org/licenses/agpl-3.0.html
# based on code from https://github.com/SimonPavlik/word2vec-keras-in-gensim/blob/keras106/word2veckeras/word2veckeras.py

import sys
from Queue import Queue

import gensim.models.word2vec
import numpy as np
from keras.layers import Input, Embedding, Lambda, merge, Reshape
from keras.layers.core import Activation
from keras.models import Model






def preprocess_sentences(model,sents,neg_dict,pos_dict):
    neg_sentences = []
    pos_sentences = []
    for sentence in sents:
        word_vocabs = [w for w in sentence if w in model.vocab]
        neg_words = [word for word in word_vocabs if word in neg_dict]
        pos_words = [word for word in word_vocabs if word in pos_dict]
        if len(neg_words) > 0 :
            neg_sentences.append(sentence)
        elif len(pos_words) > 0 :
            pos_sentences.append(sentence)
    with open('./neg_sentences_new_dic.lst','wb') as f:
        pickle.dump(neg_sentences,f)
    with open('./pos_sentences_new_dic.lst','wb') as f:
        pickle.dump(pos_sentences,f)


def train_batch_sg_our_counter(model, sentences, alpha=None, work=None, sub_batch_size=256, batch_size=256):
    batch_count = 0
    sub_batch_count = 0
    train_x0 = np.zeros((batch_size, sub_batch_size), dtype='int32')
    train_x1 = np.zeros((batch_size, sub_batch_size), dtype='int32')
    train_y = np.zeros((batch_size, sub_batch_size), dtype='int8')
    neg_dict = model.neg_dict
    pos_dict = model.pos_dict
    batch_num = 0
    for sentence in sentences:
        words = [w for w in sentence if w in model.vocab]
        neg_words = [word for word in words if word in neg_dict]
        pos_words = [word for word in words if word in pos_dict]

        if len(neg_words) > 0 or len(pos_words) > 0 :

            for pos, word in enumerate(words):
                if word in neg_words:
                    label =0
                elif word in pos_words:
                    label =1
                else:
                    continue
                w_vocab = model.vocab[word]
                reduced_window = model.random.randint(model.window)

                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)
                for pos2, word2 in enumerate(words[start:(pos + model.window + 1 - reduced_window)], start):
                    w2_vocab = model.vocab[word2]
                    # don't train on the `word` itself
                    if pos2 != pos:
                        train_x0[batch_count][sub_batch_count] = w_vocab.index
                        train_x1[batch_count][sub_batch_count] = w2_vocab.index
                        train_y[batch_count][sub_batch_count] = label
                        sub_batch_count += 1
                        if sub_batch_count >= sub_batch_size:
                            batch_count += 1
                            sub_batch_count = 0
                        if batch_count >= batch_size:
                            batch_num +=1
                            batch_count = 0
    print "total of batch", batch_num



def train_batch_sg_our(model, sentences, alpha=None, work=None, sub_batch_size=256, batch_size=256):
    batch_count = 0
    sub_batch_count = 0
    train_x0 = np.zeros((batch_size, sub_batch_size), dtype='int32')
    train_x1 = np.zeros((batch_size, sub_batch_size), dtype='int32')
    train_y = np.zeros((batch_size, sub_batch_size), dtype='int8')
    neg_dict = model.neg_dict
    pos_dict = model.pos_dict
    batch_num = 0
    while 1:
        for sentence in sentences:
            words = [w for w in sentence if w in model.vocab]
            neg_words = [word for word in words if word in neg_dict]
            pos_words = [word for word in words if word in pos_dict]

            if len(neg_words) > 0 or len(pos_words) > 0 :

                for pos, word in enumerate(words):
                    if word in neg_words:
                        label =0
                    elif word in pos_words:
                        label =1
                    else:
                        continue
                    w_vocab = model.vocab[word]
                    reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code

                    # now go over all words from the (reduced) window, predicting each one in turn
                    start = max(0, pos - model.window + reduced_window)
                    for pos2, word2 in enumerate(words[start:(pos + model.window + 1 - reduced_window)], start):
                        w2_vocab = model.vocab[word2]
                        # don't train on the `word` itself
                        if pos2 != pos:
                            train_x0[batch_count][sub_batch_count] = w_vocab.index
                            train_x1[batch_count][sub_batch_count] = w2_vocab.index
                            train_y[batch_count][sub_batch_count] = label
                            sub_batch_count += 1
                            if sub_batch_count >= sub_batch_size:
                                batch_count += 1
                                sub_batch_count = 0
                            if batch_count >= batch_size:
                                batch_num +=1
                                yield ({'index': train_x0, 'point': train_x1}, {'code': train_y})
                                batch_count = 0

def build_keras_model_sg(index_size,vector_size,
                         context_size,
                         #code_dim,
                         sub_batch_size=256,
                         learn_vectors=True,learn_hidden=True,
                         model=None):
    point_input = Input(shape=(sub_batch_size,), dtype='int32', name='point')
    index_input = Input(shape=(sub_batch_size,), dtype='int32', name='index')

    point_embedding = Embedding(
        input_dim=context_size, output_dim=vector_size, input_length=sub_batch_size,
        weights=[model.keras_syn1], name='embedpoint')(point_input)
    index_embedding = Embedding(
        input_dim=index_size, output_dim=vector_size, input_length=sub_batch_size,
        weights=[model.syn0], name='embedding')(index_input)

    merged_vectors = merge([index_embedding, point_embedding], mode='mul', name='merge')
    average = Lambda(lambda x: x.sum(2), output_shape=(sub_batch_size,))(merged_vectors)

    output = Activation(activation='sigmoid', name='code')(average)

    kerasmodel = Model(input=[point_input, index_input], output=output)
    kerasmodel.compile(optimizer='rmsprop', loss={'code': 'mse'})

    return kerasmodel




def copy_word2vec_instance_from_to(w2v,w2v_to,sentences=None,documents=None):# ,dm=None, **kwargs):
        if hasattr(w2v,'dm'):
            if w2v.dm is None :
                w2v_to.sg = w2v.sg
            else:
                w2v_to.sg=(1+w2v.dm) % 2
        else:
            w2v_to.sg = w2v.sg
                
        w2v_to.window = w2v.window 
        w2v_to.min_count =w2v.min_count
        w2v_to.sample =w2v.sample
        w2v_to.cbow_mean=w2v.cbow_mean
        
        w2v_to.negative = w2v.negative
        w2v_to.hs=w2v.hs
            
        w2v_to.alpha = w2v.alpha 

        w2v_to.vector_size=w2v.vector_size
        
        if hasattr(w2v,'dm_concat') and hasattr(w2v_to,'dm_concat'):
            if not w2v_to.dm_concat:
                w2v_to.layer1_size= w2v.layer1_size


        w2v_to.index2word=w2v.index2word
        w2v_to.sorted_vocab = w2v.sorted_vocab

        w2v_to.vocab=w2v.vocab

        w2v_to.max_vocab_size = w2v.max_vocab_size

        if hasattr(w2v,'dm'):
            docs=documents
            for document_no, document in enumerate(docs):
                document_length = len(document.words)
                for tag in document.tags:
                    w2v_to.docvecs.note_doctag(tag, document_no, document_length)
        w2v_to.reset_weights()

        w2v_to.syn0=w2v.syn0

        return w2v_to



def train_prepossess(model):
    
    vocab_size=len(model.vocab)
    if model.negative>0 and model.hs :
        model.keras_context_negative_base_index=len(model.vocab)
        model.keras_context_index_size=len(model.vocab)*2
        model.keras_syn1=np.vstack((model.syn1,model.syn1neg))
    else:
        model.keras_context_negative_base_index=0
        model.keras_context_index_size=len(model.vocab)
        if model.hs :
            model.keras_syn1=model.syn1
        else:
            model.keras_syn1=model.syn1neg

    model.neg_labels = []
    if model.negative > 0:
        model.neg_labels = np.zeros(model.negative + 1,dtype='int8')
        model.neg_labels[0] = 1

    trim_rule=None
    if len(model.vocab) == 0 :
        print 'build_vocab'
        model.build_vocab(sentences, trim_rule=trim_rule)


    model.word_context_size_max=0
    if model.hs :
        model.word_context_size_max += max(len(model.vocab[w].point) for w in model.vocab if hasattr(model.vocab[w],'point'))
    if model.negative > 0:
        model.word_context_size_max += model.negative + 1
        
class Word2VecKeras(gensim.models.word2vec.Word2Vec):

    def __init__(self,neg_dict,pos_dict,*args,**kwargs):
        super(Word2VecKeras,self).__init__(*args,**kwargs)
        self.neg_dict = neg_dict
        self.pos_dict = pos_dict

    def compare_w2v(self,w2v2):
        return np.mean([np.linalg.norm(self[w]-w2v2[w]) for w in self.vocab if w in w2v2.vocab])

    def train(self, sentences, total_words=None, word_count=0,
               total_examples=None, queue_factor=2, report_delay=1,
               batch_size=128
               ,sub_batch_size=16
              ):
        train_prepossess(self)
        



        vocab_size=len(self.vocab)
        
        sub_batch_size_update=False
        if hasattr(self,'sub_batch_size'):
            if self.sub_batch_size != sub_batch_size :
                sub_batch_size_update=True
                self.sub_batch_size=sub_batch_size

        if self.sg:
            samples_per_epoch=max(1,int((self.iter*self.window*2*sum(map(len,sentences)))/(sub_batch_size)))

            if not hasattr(self, 'kerasmodel') or sub_batch_size_update:
                self.kerasmodel=build_keras_model_sg(index_size=vocab_size,vector_size=self.vector_size,
                                                     context_size=self.keras_context_index_size,
                                                     #code_dim=vocab_size,
                                                     sub_batch_size=sub_batch_size,
                                                     model=self
                                                     )

            print "start to train"
            self.gen=train_batch_sg_our(self, sentences, sub_batch_size=sub_batch_size,batch_size=batch_size)
            self.kerasmodel.get_layer(name='embedding').set_weights([self.syn0])
            self.kerasmodel.fit_generator(self.gen,samples_per_epoch=35200, nb_epoch=self.iter, verbose=1)

        self.syn0 = self.kerasmodel.get_layer(name='embedding').get_weights()[0]
        if self.negative > 0 and self.hs:
            syn1tmp = self.kerasmodel.get_layer(name='embedpoint').get_weights()[0]
            self.syn1 = syn1tmp[0:len(self.vocab)]
            self.syn1neg = syn1tmp[len(self.vocab):2 * len(self.vocab)]
        elif self.hs:
            self.syn1 = self.kerasmodel.get_layer(name='embedpoint').get_weights()[0]
        else:
            self.syn1neg = self.kerasmodel.get_layer(name='embedpoint').get_weights()[0]

    def load_embbedding(self,weights):
        self.syn0 = weights

if __name__ == "__main__":


    status = 'export'


    if status == 'train':
        import pickle
        from nltk.corpus import movie_reviews
        from scipy import spatial

        station = 'desktop'
        if station == 'desktop':
            path = '/home/yaniv/src/datasciense-NLP/'
        else:
            path = './'
        with open(path +"negative_list.lst", "rb") as f1, open(
            path + "positive_list.lst", "rb") as f2:
            neg_words = pickle.load(f1)
            pos_words = pickle.load(f2)

        v_iter = 0
        v_size = 300
        sg_v = 1
        topn = 4
        hs = 0
        negative = 5





        sents = movie_reviews.sents()







        # Load Google's pre-trained Word2Vec model.
        vs1 = gensim.models.Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

        vsk1 = Word2VecKeras(neg_words,pos_words,hs=hs,negative=negative,sg=sg_v,size=v_size,iter=v_iter)

        vsk1 = copy_word2vec_instance_from_to(vs1,vsk1)
        vsk1.sg = sg_v
        preprocess_sentences(vsk1, sents, neg_words,pos_words)

		with open('./neg_sentences_new_dic.lst','rb') as f:
			neg_sentences = pickle.load(f)
		with open('./pos_sentences_new_dic.lst','rb') as f:
			pos_sentences = pickle.load(f)

        result = 1 - spatial.distance.cosine(vsk1['good'], vsk1['bad'])
        print "the similarity between good and bad " , result
        print " the euclidean distance between good and bad" , spatial.distance.euclidean(vsk1['good'], vsk1['bad'])
        sentences = pos_sentences+neg_sentences
        train_batch_sg_our_counter(vsk1, sentences, sub_batch_size=16, batch_size=128)
        vsk1.iter=1
		vsk1.train(sentences,batch_size=128,sub_batch_size=16)
		with open('./trained_weights_one_epoch.npy','wb') as f:
			np.save(f,vsk1.syn0)
		vsk1.kerasmodel.fit_generator(vsk1.gen, samples_per_epoch=batch_size*num_of_batches, nb_epoch=9, verbose=1)
		with open('./trained_weights_10_epoch.npy','wb') as f:
			np.save(f,vsk1.syn0)
		vsk1.kerasmodel.fit_generator(vsk1.gen, samples_per_epoch=batch_size * num_of_batches, nb_epoch=20, verbose=1)
		with open('./trained_weights_30_epoch.npy', 'wb') as f:
			np.save(f, vsk1.syn0)
		result = 1 - spatial.distance.cosine(vsk1['good'], vsk1['bad'])
		print "the similarity between good and bad ", result
		print "the euclidean distance betwwen good and bad" , spatial.distance.euclidean(vsk1['good'], vsk1['bad'])

    elif status == 'export':

        from gensim.models.keyedvectors import KeyedVectors
        def load_keyedvectors(weights,vocab,index2word,vector_size):
            kv = KeyedVectors()
            kv.vocab = vocab
            kv.index2word = index2word
            kv.vector_size = vector_size
            kv.syn0 = weights
            return kv


        vs1 = gensim.models.Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        with open('./trained_weights.npy','rb') as f:
            weights = np.load(f)
        vocab = vs1.vocab
        index2word = vs1.index2word
        vector_size = vs1.vector_size
        keyVector = load_keyedvectors(weights,vocab,index2word,vector_size)
        keyVector.save('saved_keywords.bin')
    sys.exit()
    
