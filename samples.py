#coding=utf-8

import numpy as np 
import os
import math


import threading
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import pickle
import random

input_command = ""
isStop = True
label = -1

def thread_send_signal():
    global isStop
    global label

    path_signal = ['/media/zjchen/5b8cbd15-7a4b-45cd-bfde-b139ad4b725b/zjchen/data2/20190826-lab_signals_test/735M_GNU_ask', '/media/zjchen/5b8cbd15-7a4b-45cd-bfde-b139ad4b725b/zjchen/data2/20190826-lab_signals_test/735M_GNU_fsk', '/media/zjchen/5b8cbd15-7a4b-45cd-bfde-b139ad4b725b/zjchen/data2/20190826-lab_signals_test/735M_GNU_msk', '/media/zjchen/5b8cbd15-7a4b-45cd-bfde-b139ad4b725b/zjchen/data2/20190826-lab_signals_test/735M_GNU_ofdm', '/media/zjchen/5b8cbd15-7a4b-45cd-bfde-b139ad4b725b/zjchen/data2/20190826-lab_signals_test/735M_GNU_bpsk', '/media/zjchen/5b8cbd15-7a4b-45cd-bfde-b139ad4b725b/zjchen/data2/20190826-lab_signals_test/735M_GNU_dpsk', '/media/zjchen/5b8cbd15-7a4b-45cd-bfde-b139ad4b725b/zjchen/data2/20190826-lab_signals_test/735M_GNU_16qam', '/media/zjchen/5b8cbd15-7a4b-45cd-bfde-b139ad4b725b/zjchen/data2/20190826-lab_signals_test/735M_GNU_64qam']

    while 1:
        if input_command == "q":
            break
        if isStop == False :
            label = random.randint(0, 7)
            #label = threading.current_thread().name
            command = "hackrf_transfer -t " + str(path_signal[int(label)]) + " -f 735000000 -x 47 -a 1 -p 1 -s 8000000 -b 8000000 -d 0000000000000000644064dc344993cd"
            print('[Thread]: Send signal thread')

            print("-------------------------- Signal:" + str(label) + " Start  --------------------------")
            print(command)
            os.system(command)
            print("-------------------------- Signal:" + str(label) + " End  --------------------------")
            isStop = True

# 为线程定义一个函数
def thread_input_command():
    print(threading.current_thread().name)

    np.set_printoptions(threshold=np.inf)

    start_freq = 732000000  #732MHz
    stop_freq = 739000000 #739MHz
    step_freq = 6000000    #6MHz

    #folder = 0
    #f = open('./config_count', 'w')
    #f.write(str(folder))
    #f.close()
    f = open('./config_count', 'r')
    folder = int(f.read())
    f.close()
    '''
    f = open('./config_freq', 'r')
    start_freq = float(f.read())
    f.close()
    '''

    Ew_array = []
    Cp_array = []
    Max_Ew_array = []
    isSignal = True
    Tw = 80406.93

    x_Ew = []
    y_Cp = []
    z_max_cp = []

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x_max = 1000
    y_max = 1000
    z_max = 100

    label_score_array = np.zeros((1, 8)) 
    label_score_array = [[1057., 806., 1156., 731., 573., 1181., 585., 1140.]]
    #data_array_p = np.zeros((x_max, y_max, z_max))
    #data_array_p_g = np.zeros((2, x_max, y_max))
    data_array_p_g = np.zeros((x_max, y_max))
    data_array_p = np.zeros((x_max, y_max))
    #data_array_P = pickle.load(open('/media/zjchen/5b8cbd15-7a4b-45cd-bfde-b139ad4b725b/zjchen/data2/20190826-lab_signals_test/array_1_1000', 'rb'))
    #data_array_P = data_array_P * 1000
    result_mc_list = []
    result_mc_list = pickle.load(open('/home/zjchen/data2/result_mc', 'rb'))

    data_score_display = 0 
    data_score_display = result_mc_list[9239] * int(folder - 1)

    g_array = np.zeros((x_max, y_max))
    xpos = []
    ypos = []
    zpos = []
    '''
    print('[MC] Geting Data (=_=).o0O')
    data_array_0 = pickle.load(open('/media/zjchen/5b8cbd15-7a4b-45cd-bfde-b139ad4b725b/zjchen/data2/20190826-lab_signals_test/array_0_0', 'rb'))
    print('[MC] Geted Data (^_=).o0O')
    '''

    learn_per = np.ones((1, 8)) 
    '''
    learn_per[0][3] = learn_per[0][3] + 0.1
    learn_per[0][4] = learn_per[0][4] + 0.11
    learn_per[0][5] = learn_per[0][5] + 0.1
    learn_per[0][6] = learn_per[0][6] + 0.15
    learn_per[0][7] = learn_per[0][7] + 0.09
    '''
    learn_per[0][2] = learn_per[0][2] + 0.05
    learn_per[0][6] = learn_per[0][6] + 0.05
    learn_per[0][7] = learn_per[0][7] + 0.05
    error_label = np.zeros((8, 8)) 
    error_label = [[0., 0., 0., 0., 2., 0., 0., 60.],
 [0., 0.,309., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 227., 88., 99., 32.],
 [0., 0., 0., 218., 0., 21., 339., 2.],
 [0., 0., 0., 7., 0., 0., 0., 0.],
 [0., 0., 0., 125., 152., 269., 0., 0.],
 [0., 0., 0., 23., 1., 0., 41., 0.]]
    global isStop
    global label

    '''
    print(result_mc_list[9239])
    print(int(folder - 1))
    print(data_score_display)
    print(error_label)
    print(label_score_array[0])
    label_score_array[0][0] = label_score_array[0][0] + 1
    print(label_score_array[0])
    exit()
    '''
    ####################### while start #######################
    while 1:
        
        ####################### for start #######################
        for count in range(1, int(math.ceil((stop_freq - start_freq) / step_freq))):

            
            while 1 :
                if isStop :
                    label = -1
                    isStop = False
                    time.sleep(5)
                    break
                else:
                    continue
            
            str_start = str(start_freq + step_freq * (count - 1))
            str_stop = str(start_freq + step_freq * count)
            #TEST
            print(str(count) + " " + str_start + " " + str_stop)
            
            command = "soapy_power -r 6000000 -B 100000 -O scan_freq -D constant -F rtl_power_fftw -n 60 -d serial=0000000000000000a27466e627215b0f -f " + str_start + ":" + str_stop + "-"
            #command = "soapy_power -r 6000000 -B 100000 -O scan_freq -D constant -F rtl_power_fftw -n 60 -d serial=0000000000000000866863dc395732cf -f " + str_start + ":" + str_stop + "-"
            print("")
            print("------------------- Count:" + str(folder) + ", Freq:" + str_start + "Hz Sart ------------------------")
            print(command)
            os.system(command)
            print("-------------------------- Freq:" + str_start + "Hz End  --------------------------")
               
            ### xyz : Get IQ 
            xpos = pickle.load(open('/home/zjchen/data2/xpos', 'rb'))
            ypos = pickle.load(open('/home/zjchen/data2/ypos', 'rb'))
            zpos = pickle.load(open('/home/zjchen/data2/zpos', 'rb'))

            print('##################')
            data_score = []

            for l in range(0, 8):
            #if 1:
                
                data_array_s = np.array(pickle.load(open('/media/zjchen/5b8cbd15-7a4b-45cd-bfde-b139ad4b725b/zjchen/data2/20190826-lab_signals_test/data/array_g/' + str(l) + '_1d_1x1000', 'rb')))
                data_array_p = np.zeros((x_max, y_max))
                #print(sorted(data_array_s.reshape(1, x_max * y_max)))
                #print(data_array_s)
                #exit()
                data_score_tmp = 0
                
                for i in range(0, len(xpos)): 
                    
                    ### XYZ 
                    #print('x:%d, y:%d, z:%d' % (int(xpos[i]), int(ypos[i]), int(zpos[i])))   
                    #data_array_p[int(xpos[i])][int(ypos[i])][int(zpos[i])] = data_array_p[int(xpos[i])][int(ypos[i])][int(zpos[i])] + 1

                    ### XY
                    #print('x:%d, y:%d, z:%d' % (int(xpos[i]), int(ypos[i]), int(zpos[i])))  
                    #print(data_array_p[int(xpos[i])][int(ypos[i])])
                    data_array_p[int(xpos[i])][int(ypos[i])] = data_array_p[int(xpos[i])][int(ypos[i])] + int(zpos[i])
                    #data_array_p[int(xpos[i])][int(ypos[i])] = data_array_p[int(xpos[i])][int(ypos[i])] + int(zpos[i])
                    #data_array_p[int(xpos[i])][int(ypos[i])] = max(data_array_p[int(xpos[i])][int(ypos[i])], int(zpos[i]))
                    #print(1 / np.abs(data_array_s[int(xpos[i])][int(ypos[i])] - zpos[i]))
                    #print(data_array_p[int(xpos[i])])
                    #print(int(zpos[i]))
                    #exit()
                    '''
                    if zpos[i] > 10:
                        continue
                    if np.abs(data_array_s[int(xpos[i])][int(ypos[i])] - zpos[i]) == 0:
                        data_score = data_score + 10
                    else:
                        data_score = data_score + float(1 / np.abs(data_array_s[int(xpos[i])][int(ypos[i])] - zpos[i]))
                    '''
                    #data_score = data_score + np.abs(data_array_s[int(xpos[i])][int(ypos[i])] - zpos[i])
                    '''
                    ##############################
                    index = np.where(data_array_s[int(xpos[i])][int(ypos[i])] > 0)
                    #index = np.where(np.sum(data_array_s[int(xpos[i])][int(ypos[i])]) == 10)
                    #exit()
                    if len(index[0]) > 0 :
                        #print(index)
                        #print(index[0])
                        #print(int(zpos[i]))
                        #print(data_score_tmp)
                        #print(data_array_s[int(xpos[i])][int(ypos[i])])
                        #print(np.sum(data_array_s[int(xpos[i])][int(ypos[i])]))
                        
                        #exit()
                        for d in range(0, z_max): 
                            if data_array_s[int(xpos[i])][int(ypos[i])][int(d)] > 0 :
                                data_score_tmp = data_score_tmp + (1 - data_array_s[int(xpos[i])][int(ypos[i])][int(d)] / 100) * np.abs(int(d) - int(zpos[i]))
                            #print(data_array_s[int(index[0][d])][int(index[1][d])][int(index[2][d])])
                            #print(np.abs(index[0][d] - int(zpos[i])))
                        #exit()
                        
                        #data_score_tmp = data_score_tmp + 1
                    '''
                    #data_score_tmp = data_score_tmp + data_array_s[int(xpos[i])][int(ypos[i])]
                
                #print(data_array_p.shape)
                '''
                if len(np.where(data_array_p_g > 0)) == 0 :
                    data_array_p_g = np.array(np.gradient(data_array_p))
                else:
                    data_array_p_g = np.array(np.gradient(data_array_p))
                '''
                data_array_p_g = np.array(np.gradient(data_array_p)[1])
                #print(data_array_p_g.shape)
                #print(data_array_s.shape)
                
                distance_tmp = np.abs(data_array_p_g - data_array_s)
                data_score_tmp = np.sum(distance_tmp)
                
                #if l == 6 :
                #    data_score_tmp = 1.01 * data_score_tmp
                
                #elif l == 7 :
                #    data_score_tmp = 1.05 * data_score_tmp
                
                data_score_tmp = learn_per[0][l] * data_score_tmp
                #exit()
                
                #for g_x in range(0, x_max):
                #    for g_y in range(0, y_max):
                #       if g_x == 0 and g_y == 0 :
                #           continue
                #       g_array[g_x][g_y] = max(g_array[g_x][g_y], g_array[g_x - 1][g_y], g_array[g_x][g_y - 1], g_array[g_x - 1][g_y - 1]) - min(g_array[g_x][g_y], g_array[g_x - 1][g_y], g_array[g_x][g_y - 1], g_array[g_x - 1][g_y - 1])

                #print(len(np.argwhere(data_array_p > 1)))
                #print(sorted(data_array_p.reshape(1, x_max * y_max).tolist()))
                #print(data_array_p[int(xpos[i])][int(ypos[i])])
                #exit()
                
                ##############################
                #print('@@@')
                print(data_score_tmp)
                data_score.append(data_score_tmp)
                #print(np.sum(np.abs(data_array_s - g_array)))
                #print(np.sum(data_array_s))
                #print(data_score)
                
            
            ##############################
            print('Total Learn per: 8')
            print(learn_per)

            data_score_array = np.array(data_score)
            #label = 0
            index = np.where(data_score_array == min(data_score_array))[0][0]
            print('Result: %d, Label: %d' % (index, label))
            if label < 0 :
                print('[Thread]: label error !')
                exit()
            if index == int(label):
                data_score_display += 1
                label_score_array[0][label] = label_score_array[0][label] + 1
            else :
                error_label[label][index] = error_label[label][index] + 1

            print(error_label) 
            result_mc_data = float(data_score_display / folder)
            result_mc_list.append(result_mc_data)
            if folder % 10 == 0 :
                pickle.dump(result_mc_list, open('/home/zjchen/data2/result_mc', 'wb'))
            print('Score: %f' % result_mc_data)
            print(label_score_array[0])
            
            
            #print(data_array.shape)


            '''
            #################################################
            ### Y : Get Cp 
            f = open('/home/zjchen/data2/trainss_data2', 'r') 
            data2 = f.read()
            f.close()
            Cp_array = np.append(Cp_array, float(data2))
            ##
            ### X : Get Ew 
            f = open('/home/zjchen/data2/trainss_data', 'r')
            data = f.read()
            f.close()
            if len(data) != 0:
                Ew_array = np.append(Ew_array, float(data))
            ##
            
            ### Z : Get Test 
            f = open('/home/zjchen/data2/trainss_data3', 'r')
            data_max_ew = f.read()
            f.close()
            Max_Ew_array = np.append(Max_Ew_array, float(data_max_ew))
            ##


            z_max_cp.append(float(data_max_ew))
            y_Cp.append(float(data2))
            x_Ew.append(float(data))


            ### Z : Get Test 
            ### 5 Ew
            print('[Samples]: Locations low Ew:%f' % (np.mean(Max_Ew_array)))
            max_ew_data_file = open('/home/zjchen/data2/trainss_data_test', 'a')
            max_ew_data_file.write(data_max_ew + " ")
            max_ew_data_file.close()
            '''

            '''
            ### Max Ew
            print('[Samples]: Locations Max Ew:%f' % (np.mean(Max_Ew_array)))
            max_ew_data_file = open('/home/zjchen/data2/trainss_data_max_ew_locations', 'a')
            max_ew_data_file.write(data_max_ew + " ")
            max_ew_data_file.close()

            '''

            #print('[Samples]: data %d' % int(data))
 
            ### calculate Pd
            #Ew_array = np.array(Ew)
            #print('[Samples]: Num:%d , Good:%d , Pd:%.2f%%' % (len(Ew_array), np.sum(Ew_array > Tw), (np.sum(Ew_array > Tw) / len(Ew_array)) * 100))
            #time.sleep(1)
            ###
           
            ### save data
            #print(Ew_array)
            #print('[Samples]: average Cp:%d , Ew:%d ' % (np.mean(Cp_array), np.mean(Ew_array)))
            '''
            if isSignal:
                print('[Samples]: Total:%d , Good:%d , Pd(signal):%.2f%% !!!' % (len(Ew_array), np.sum(Ew_array < Tw), (np.sum(Ew_array < Tw) / len(Ew_array)) * 100))
            else:
                print('[Samples]: Total:%d , Good:%d , Pd(noise):%.2f%% @@@' % (len(Ew_array), np.sum(Ew_array > Tw), (np.sum(Ew_array > Tw) / len(Ew_array)) * 100))
            '''
            '''
            #################################################
            ### X : Get Ew 
            ew_data_file = open('/home/zjchen/data2/trainss_data_ew', 'a')
            ew_data_file.write(data + " ")
            ew_data_file.close()
            ### Y : Get Cp 
            cp_data_file = open('/home/zjchen/data2/trainss_data_cp', 'a')
            cp_data_file.write(data2 + " ")
            cp_data_file.close()
            ###
            '''
            print(input_command)
            if input_command == "q":
                #plt.plot(x_Ew, y_Cp, 'b.')
                '''
                ax.plot(x_Ew, y_Cp, z_max_cp, 'b.')
                ax.set_xlabel('X(Cir in Sum)')
                ax.set_ylabel('Y(Cp)')
                ax.set_zlabel('Z(Cir in Num)')
                '''
                #plt.show()
                break
        ####################### for end #######################
       
        folder = folder + 1
        f = open('./config_count', 'w')
        f.write(str(folder))
        f.close()
        '''
        if folder > 19:
            exit()
        '''
        ####################### if start #######################
        if folder > 10000 or input_command == "q":
            #plt.plot(x_Ew, y_Cp, 'b.')
            '''
            ax.plot(x_Ew, y_Cp, z_max_cp, 'b.')
            ax.set_xlabel('X(Cir in Sum)')
            ax.set_ylabel('Y(Cp)')
            ax.set_zlabel('Z(Cir in Num)')
            '''

            ax.plot(xpos, ypos, zpos, 'b.')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            result_mc_list = pickle.load(open('/home/zjchen/data2/result_mc', 'rb'))
            plt.figure()
            print(result_mc_list)
            plt.plot(result_mc_list)

            print('[MC] Count:%d' % int(folder - 1))
            print('[MC] Saving Data (=_=).o0O')
            '''
            ### XYZ 
            data_array = data_array / (folder)
            #data_array = (data_array + data_array_0) / 2
            pickle.dump(data_array, open('/media/zjchen/5b8cbd15-7a4b-45cd-bfde-b139ad4b725b/zjchen/data2/20190826-lab_signals_test/array_0', 'wb'), protocol = 4)
            '''
            ### XY 
            #data_array_p = (data_array_p) / (folder - 1)
            #print(data_array_p)
            #print(len(np.argwhere(data_array_p > 0)))
            #print((np.sum(data_array_p)))
            #print(sorted(data_array_p.reshape(1, x_max * y_max)))
            #data_array_p_g = np.array(np.gradient(data_array_p))
            #pickle.dump(data_array_p_g, open('/media/zjchen/5b8cbd15-7a4b-45cd-bfde-b139ad4b725b/zjchen/data2/20190826-lab_signals_test/data/array_g/7_1d_1x1000_2', 'wb'), protocol = 4)
            #pickle.dump(g_array, open('/media/zjchen/5b8cbd15-7a4b-45cd-bfde-b139ad4b725b/zjchen/data2/20190826-lab_signals_test/array_3_2', 'wb'), protocol = 4)
            print('[MC] Saved Data (^_^).o0O')
            #plt.show()
            break
        ####################### if end #######################


    print("[Thread]: " + str(start_freq) + "-" + str(stop_freq) + " All Done")
    plt.show()
    ####################### while end #######################


if __name__ == '__main__':
    
    print("##########################################################################")
    print("Function: Scan frequency. Judge signal or noise. Save IQ samples")


    S2 = threading.Thread(target = thread_send_signal, name = "[Thread]: Sned signal thread")
    S2.start()

    S1 = threading.Thread(target = thread_input_command, name = "[Thread]: Enter scan thread")
    S1.start()

    #print(time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    #exit()


    while 1:
        input_command = input()
        if input_command == "q":
            print("[Main]: Quiting...")
            exit()







