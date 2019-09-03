import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import cv2
import random
from random import shuffle
from models.VAE_Network import VAE_Network
from utils.config import *


def get_test_data(data):
    # data: [batch, OBS_WIDTH, OBS_WIDTH, 4]
    length = len(data)
    rand_indice = np.arange(length)
    np.random.shuffle(rand_indice)
    test_data = data[rand_indice[:TEST_IMG_LEN], :, :, :]
    # test_data: [TEST_IMG_LEN, OBS_WIDTH, OBS_WIDTH, 4]
    return test_data


def set_demo(data_dir):
    demo_list = os.listdir(data_dir)
    sample = random.sample(demo_list, 100)
    expert_states = []
    for name in sample:
        demo = np.load(data_dir + name)
        states = demo['state']
        expert_states.append(states)
    expert_states = np.concatenate(expert_states, axis=0)
    return expert_states


def save_original(data, i):
    filename = './test_img/ori_%.5d.jpg' % i
    #data : (TEST_IMG_LEN, OBS_WIDTH, OBS_WIDTH, 3)
    frame = data
    frame = np.uint8(frame*255)
    frame = frame.reshape((OBS_WIDTH*TEST_IMG_LEN, OBS_WIDTH, 3))
    cv2.imwrite(filename, frame)


def save_reconstructed(data, i):
    filename = './test_img/out_%.5d.jpg' % i
    #data : (TEST_IMG_LEN, OBS_WIDTH, OBS_WIDTH, 3)
    frame = data
    frame = np.uint8(frame*255)
    frame = frame.reshape((OBS_WIDTH*TEST_IMG_LEN, OBS_WIDTH, 3))
    cv2.imwrite(filename, frame)
        

if __name__ == '__main__':
    BATCH_SIZE = 128
    ROTATE = 10
    SAVE_RATE = 100
    EPOCHS = 9999999
    TEST_IMG_LEN = 6
    data_dir = 'data/Navi-v1/'
    model_dir = 'weights/vaecnn/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists('test_img'):
        os.makedirs('test_img')
    try:
        dummy_data = np.ones([1] + STATE_SHAPE, dtype=np.float32)
        vae = VAE_Network()
        vae(dummy_data)
#        vae.load('vae')
        losses = []
        indice = []
        
        for epoch in range(EPOCHS):
            print('epoch ', epoch, end='\t')
            loss_stat = open('loss_stat.csv', 'w', newline='')
            wr = csv.writer(loss_stat)
            running_loss = 0
            if epoch % ROTATE == 0:
                data = set_demo(data_dir)
            if epoch % SAVE_RATE == 0:
                test_data = get_test_data(data)
                save_original(test_data, epoch)
            for i in range(len(data)//BATCH_SIZE):
                running_loss += vae.update(data[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            average_loss = running_loss/(len(data)//BATCH_SIZE)
            print(average_loss)
            wr.writerow([epoch, average_loss])
            loss_stat.close()
            losses.append(average_loss)
            indice.append(epoch)
            if epoch % SAVE_RATE == 0:
                plt.plot(indice, losses)
                vae.save(model_dir, 'vae')
                vae.encoder.save(model_dir, 'encoder')
                plt.savefig('loss_stat.png')
                out, _, _ = vae(test_data, False)
                save_reconstructed(out.numpy(), epoch)
    except Exception as e:
        print(e)
        vae.save(model_dir,'vae')
        vae.encoder.save(model_dir, 'encoder')
        

