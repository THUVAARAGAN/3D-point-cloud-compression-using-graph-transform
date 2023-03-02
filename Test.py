import pickle
import numpy as np
import  pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.layers import (Dense, 
                                     BatchNormalization, 
                                     LeakyReLU, 
                                     Reshape, 
                                     Conv2DTranspose,
                                     Conv2D,
                                     Dropout,
                                     Flatten)

BUFFER_SIZE = 60000
BATCH_SIZE = 256

path ="/home/heethanjan/THENUKAN/Few_Shot_Distribution_Calibration/checkpoints/miniImagenet/WideResNet28_10_S2M2_R/last/base_features.plk"


#print("###### shape ###########")
L=[]
with open(path, 'rb') as f:
    data = pickle.load(f)
    for key in data.keys():
        #print(key)
        feature = np.array(data[key])
        fe_tensor = tf.convert_to_tensor(feature)
        L.append(fe_tensor)
        #print(fe_tensor.shape)  

add=L[0]
for i in range(1,len(L)):
    add=tf.concat([add, L[i]], 0)

#print(add.shape)
#print(add[1])


BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(add).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)



def make_generator_model():
    model = tf.keras.Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(640,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    model.add(Reshape((1,784)))
    assert model.output_shape == (None,1,784)
    model.add(Dense(640))

    return model


generator = make_generator_model()

#generator.summary()

noise = tf.random.normal([1, 640])
generated_image = generator(noise, training=False)

#print("###### generated_image.shape ###########")
#print(generated_image.shape)

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(Reshape((32, 20, 1)))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model


discriminator = make_discriminator_model()

decision = discriminator(generated_image)
#print("###### decision ###########")
#print (decision)
#discriminator.summary()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

import os

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 3000
noise_dim = 640

def train_step(images):
  
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    # 2 - Generate images and calculate loss values
    # GradientTape method records operations for automatic differentiation.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, 
                                               generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, 
                                                discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

import time

def train(dataset, epochs):

  for epoch in range(epochs):
    start = time.time()
    for image_batch in dataset:
      train_step(image_batch)
    if (epoch + 1) % 5 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

print("GPU", "available (Yess!!!!" if tf.config.list_physical_devices("GPU") else "not available")

checkpoint.restore(tf.train.latest_checkpoint("/home/heethanjan/Heetha/other/CLIP-main/CLIP-main/training_checkpoints/ckpt-313.data-00000-of-00001"))

model = generator


path ="/home/heethanjan/THENUKAN/Few_Shot_Distribution_Calibration/checkpoints/miniImagenet/WideResNet28_10_S2M2_R/last/novel_features.plk"

#print("###### shape ###########")
L=[]
dict1={}
with open(path, 'rb') as f:
    data = pickle.load(f)
    for key in data.keys():
        #print(key)
        feature = np.array(data[key])
        fe_tensor = tf.convert_to_tensor(feature)
        L.append(fe_tensor)
        predictions = model(fe_tensor, training=False)

        list1=[]
        for i in range (len(predictions)):
            list1.append(predictions[i][0])
        feature_list = np.array(list1)
        fe_tensor_list = tf.convert_to_tensor(feature_list)
        
        sum_ten=tf.add(fe_tensor, fe_tensor_list)
        sum_ten=tf.divide(sum_ten, 2,0.00001)
        dict1[key]=sum_ten
        #print(fe_tensor.shape) 
        #print(fe_tensor_list.shape)  
    print(fe_tensor-predictions)
filename = '/home/heethanjan/THENUKAN/Few_Shot_Distribution_Calibration/checkpoints/miniImagenet/WideResNet28_10_S2M2_R/last/sum_ten.plk'        
outfile = open(filename,'wb')
pickle.dump(dict1,outfile)
outfile.close()
