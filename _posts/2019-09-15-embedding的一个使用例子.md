---
tag: 机器学习 机器学习实战
---







在处理一些文本类型的数据的时候，常常需要用到Embedding，这里对Embedding做一个实际的应用。







在最开始接触的时候，因为涉及到大量的数据，刚开始的做法是：单独为某一类特征（文本类型）train一个NN，然后通过`keras.backend.function`建立一个keras函数来直接获得某一层的输出，意图将该输出先保存下来。待全部的特征都分别被保存下来以后，然后再统一投到另一个NN中去作为input。







例如，建立一个Model如下：





```python
model = Sequential()
model.add(Embedding(len(enc_for_123.categories_[0])+5,3,input_length = 1))
model.add(Flatten())
model.add(Dense(units = 4,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 8,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(6,activation = 'softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
```



在fit（train）完这个model以后，通过keras函数获得某一层输出，在这里是Embedding layer，即第0层。







```python
from keras import backend
get_embedding_layer_output = backend.function([model.layers[0].input,backend.learning_phase()],[model.layers[1].output]) //这里这个backend.learning_phase()参数是为了区别训练和predict。
embedding_layer_output = get_embedding_layer_output([X,0])[0]
embedding_layer_output = list(embedding_layer_output.reshape(3))  //embedding layer output会多一层
ret_dict[line[0]] = embedding_layer_output
```





这里需要reshape是因为embedding layer的output会多一个维度。



 如果模型在训练和测试两种模式下不完全一致，例如你的模型中含有Dropout layer，批规范化（BatchNormalization）等layer，就需要在函数中传递一个learning_phase的标记， 然后在传递参数的时候设置为0，就像在这个例子中一样。如果训练和预测不涉及这些，则可以直接忽略掉这一个参数。







这样的做法优点有，但是不详，不过显然不会取得很好的performance。缺点很多，例如，需要耗费大量的时间来分别给每个特征训练NN；各个特征（包含数值数据）之间的配合不能体现；不能很好配合模型的其他部分；代码太复杂等。









---







Keras其实提供了另外一种模型的写法，用两个括号的形式来表示输入输出。这样可以很灵活地来构建模型，包括RNN等复杂结构模型。在下面另外一个例子中，直接使用了这种方法来构建Embedding。







例如，在这个例子中，构建的模型结构如下，`model.summary()`：



```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_3 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_4 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_5 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_6 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_7 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_8 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_9 (InputLayer)            (None, 1)            0                                            
__________________________________________________________________________________________________
input_10 (InputLayer)           (None, 1)            0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 1, 5)         60          input_2[0][0]                    
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 1, 5)         15          input_3[0][0]                    
__________________________________________________________________________________________________
embedding_3 (Embedding)         (None, 1, 5)         20          input_4[0][0]                    
__________________________________________________________________________________________________
embedding_4 (Embedding)         (None, 1, 5)         10          input_5[0][0]                    
__________________________________________________________________________________________________
embedding_5 (Embedding)         (None, 1, 5)         10          input_6[0][0]                    
__________________________________________________________________________________________________
embedding_6 (Embedding)         (None, 1, 5)         10          input_7[0][0]                    
__________________________________________________________________________________________________
embedding_7 (Embedding)         (None, 1, 5)         15          input_8[0][0]                    
__________________________________________________________________________________________________
embedding_8 (Embedding)         (None, 1, 5)         60          input_9[0][0]                    
__________________________________________________________________________________________________
embedding_9 (Embedding)         (None, 1, 5)         20          input_10[0][0]                   
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 5)            0           embedding_1[0][0]                
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 5)            0           embedding_2[0][0]                
__________________________________________________________________________________________________
flatten_3 (Flatten)             (None, 5)            0           embedding_3[0][0]                
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 5)            0           embedding_4[0][0]                
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 5)            0           embedding_5[0][0]                
__________________________________________________________________________________________________
flatten_6 (Flatten)             (None, 5)            0           embedding_6[0][0]                
__________________________________________________________________________________________________
flatten_7 (Flatten)             (None, 5)            0           embedding_7[0][0]                
__________________________________________________________________________________________________
flatten_8 (Flatten)             (None, 5)            0           embedding_8[0][0]                
__________________________________________________________________________________________________
flatten_9 (Flatten)             (None, 5)            0           embedding_9[0][0]                
__________________________________________________________________________________________________
input_1 (InputLayer)            (None, 7)            0                                            
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 52)           0           flatten_1[0][0]                  
                                                                 flatten_2[0][0]                  
                                                                 flatten_3[0][0]                  
                                                                 flatten_4[0][0]                  
                                                                 flatten_5[0][0]                  
                                                                 flatten_6[0][0]                  
                                                                 flatten_7[0][0]                  
                                                                 flatten_8[0][0]                  
                                                                 flatten_9[0][0]                  
                                                                 input_1[0][0]                    
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 88)           4664        concatenate_1[0][0]              
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 88)           0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 128)          11392       dropout_1[0][0]                  
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 2)            258         dense_2[0][0]                    
==================================================================================================
Total params: 16,534
Trainable params: 16,534
Non-trainable params: 0
__________________________________________________________________________________________________
None
```





可以看到，这里在一个NN中同时为9个不同的文本类型feature构建了embedding layer。embedding layer通常会接一个flatten layer。





同时，这个NN的输入还有7个数值型的feature，使用Concatenate layer将数值型特征（7维）和9个flatten layer拼接起来。然后再将拼接后的output传入NN，直接进行训练。



代码如下：





```python
NN_nums_input = layers.Input(shape = (7,))
input_1 = layers.Input(shape = (1,))
input_2 = layers.Input(shape = (1,))
input_3 = layers.Input(shape = (1,))
input_4 = layers.Input(shape = (1,))
input_5 = layers.Input(shape = (1,))
input_6 = layers.Input(shape = (1,))
input_7 = layers.Input(shape = (1,))
input_8 = layers.Input(shape = (1,))
input_9 = layers.Input(shape = (1,))


embedding_1 = layers.Embedding(len(enc_for_str.categories_[0]), 5, input_length = 1)(input_1)
embedding_2 = layers.Embedding(len(enc_for_str.categories_[1]), 5, input_length = 1)(input_2)
embedding_3 = layers.Embedding(len(enc_for_str.categories_[2]), 5, input_length = 1)(input_3)
embedding_4 = layers.Embedding(len(enc_for_str.categories_[3]), 5, input_length = 1)(input_4)
embedding_5 = layers.Embedding(len(enc_for_str.categories_[4]), 5, input_length = 1)(input_5)
embedding_6 = layers.Embedding(len(enc_for_str.categories_[5]), 5, input_length = 1)(input_6)
embedding_7 = layers.Embedding(len(enc_for_str.categories_[6]), 5, input_length = 1)(input_7)
embedding_8 = layers.Embedding(len(enc_for_str.categories_[7]), 5, input_length = 1)(input_8)
embedding_9 = layers.Embedding(len(enc_for_str.categories_[8]), 5, input_length = 1)(input_9)


flatten_for_embeding_1 = layers.Flatten()(embedding_1)
flatten_for_embeding_2 = layers.Flatten()(embedding_2)
flatten_for_embeding_3 = layers.Flatten()(embedding_3)
flatten_for_embeding_4 = layers.Flatten()(embedding_4)
flatten_for_embeding_5 = layers.Flatten()(embedding_5)
flatten_for_embeding_6 = layers.Flatten()(embedding_6)
flatten_for_embeding_7 = layers.Flatten()(embedding_7)
flatten_for_embeding_8 = layers.Flatten()(embedding_8)
flatten_for_embeding_9 = layers.Flatten()(embedding_9)


gather = layers.Concatenate()([flatten_for_embeding_1,flatten_for_embeding_2,flatten_for_embeding_3,flatten_for_embeding_4,flatten_for_embeding_5,flatten_for_embeding_6,flatten_for_embeding_7,flatten_for_embeding_8,flatten_for_embeding_9,NN_nums_input])


x = layers.Dense(units = 88, activation = 'relu',kernel_regularizer = regularizers.l2(l = 0.05))(gather)
x = layers.Dropout(rate = 0.2)(x)
x = layers.Dense(units = 128, activation = 'relu', kernel_regularizer = regularizers.l2(l = 0.05))(x)
embedding_output = layers.Dense(units = 2, activation = 'softmax')(x)


embedding_model = Model(input = [input_1,input_2,input_3,input_4,input_5,input_6,input_7,input_8,input_9,NN_nums_input], output = embedding_output)


embedding_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
```









在实际使用的时候，基本上都是采用下面这种方法。