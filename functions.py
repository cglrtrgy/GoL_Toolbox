import sys

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
import torch.nn.functional as F
from intvalpy import lineqs

import cvxpy as cp

import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import time

import pickle


import multiprocessing
from multiprocessing import Pool

import pandas as pd

def get_polytope_list(first_sample_bit_vector, indexes_for_active_bits_first_sample, input_dimension=2, model=None):
     
    polytope_list = []
    active_bit_count_list = []
    run = True 
    
    polytope_list.append(first_sample_bit_vector)
    active_bit_count_list.append(len(indexes_for_active_bits_first_sample))
    
    
    for index in indexes_for_active_bits_first_sample:
    #     print(index)

        copy_of_first_sample_bit_vector = first_sample_bit_vector.copy()

        if copy_of_first_sample_bit_vector[index]==0:
            copy_of_first_sample_bit_vector[index]=1
        else:
            copy_of_first_sample_bit_vector[index]=0

    #     print(copy_of_first_sample_bit_vector)

        if any(np.array_equal(copy_of_first_sample_bit_vector, polytope) for polytope in polytope_list):
            continue
        else:
            polytope_list.append(copy_of_first_sample_bit_vector)
            
    #counter for polytopes 
    i = 1
    while i < len(polytope_list):
        
        bit_string = polytope_list[i]

        #change this based on how many hidden layers we have
        #if we have different number of nodes for each hidden layer, I need to
        #update the code
        split_vectors =np.array_split(bit_string,2)
        A, c = get_inequalities(model,[split_vectors],is_input_sample=False,return_bit_vec_list=False, input_dimension=input_dimension)
        indexes_for_active_bits,_ = active_bits_index_2 (A[0],c[0].reshape(-1))
        
        active_bit_count_list.append(len(indexes_for_active_bits))

        for index in indexes_for_active_bits:

            copy_of_bit_string= bit_string.copy()

            if copy_of_bit_string[index]==0:
                copy_of_bit_string[index]=1
            else:
                copy_of_bit_string[index]=0

            if any(np.array_equal(copy_of_bit_string, polytope) for polytope in polytope_list):
                continue
            else:
                polytope_list.append(copy_of_bit_string)

        if i % 100 == 0:
            print(f"Polytope no {i}")
        i += 1
        
      
    return polytope_list, active_bit_count_list


def get_weights_and_biases (model):
    # weights = np.array([[]])
    # weight_sizes = np.array([[]])
    # biases = np.array([[]])
    # biases_sizes = np.array([[]])
    
    weight_output_dict = {}
    bias_output_dict = {}
    
    counter_weight = 0
    counter_bias = 0


    for name, para in model.named_parameters():
    #     print (list(para.shape))
    


        if 'weight' in name:
            #print(para.detach().numpy())
#             weights = np.append(weights, para.detach().numpy())
#             weight_sizes = np.append(weight_sizes, list(para.shape))
            
            name_weight = "Layer_"+str(counter_weight)
            weight_output_dict[name_weight] = para.detach().numpy()
            counter_weight = counter_weight + 1
            
        if 'bias' in name:
            #print(para.detach().numpy())
#             biases = np.append(biases, para.detach().numpy())
#             biases_sizes = np.append(biases_sizes, list(para.shape))
            
            name_bias = "Layer_"+str(counter_bias)
            bias = para.detach().numpy()
            bias_output_dict[name_bias] = bias.reshape(-1,1)
            
            counter_bias = counter_bias + 1

    #return weights, their shapes, biases and their shapes
    return weight_output_dict, bias_output_dict


    # A simple hook class that returns the input and output of a layer during forward/backward pass
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()


def bit_vector_from_relus(model, input_vectors, stack_layers=False, verbose = False, get_unique_bit_vectors_only = True):
    
    bit_vector_output_dict = {}

    # register hooks if layer is Relu, otherwise skip
    hookF = [Hook(layer[1]) for layer in list(model._modules.items()) if isinstance(layer[1], nn.ReLU)]
    all_bit_vector_list = []
    for i in range(input_vectors.shape[1]):
        
        if verbose:
            if i%1000==0:
                print("Input no: ", i+1, "out of",input_vectors.shape[1] )
        
        an_input_vector = input_vectors[:,i]
        
        # run a data batch
        out=model(torch.tensor(np.transpose(an_input_vector), dtype=torch.float32))
        # backprop once to get the backward hook results

        #initiliaze empty numpy vector
        bit_list_for_a_vector = []
        counter_relu = 0
        for hook in hookF:

#             name = "Relu_" + str(counter_relu)

            relu_output = hook.output.detach().numpy().reshape(-1,1)

            bit_vector_output = np.where(relu_output > 0, 1, 0)
            bit_list_for_a_vector.append(bit_vector_output)
                

#             bit_vector_output_dict[name] = bit_vector_output

            counter_relu = counter_relu + 1
        
        if get_unique_bit_vectors_only:
            if any(np.array_equal(bit_list_for_a_vector, vector) for vector in all_bit_vector_list):
                continue
        
        if stack_layers:
            bit_list_for_a_vector = np.concatenate(bit_list_for_a_vector)
#                 stacked_bit_vector_list = [np.concatenate(element, axis=0) for element in bit_vector_list_of_samples]
        
        all_bit_vector_list.append(bit_list_for_a_vector) 
                   
    #list(each sample) of list of layers
    #all_bit_vector_list[0] --> list of bit vectors (np.array) for each layer for sample_0
    #len(all_bit_vector_list) --> number of samples
    #len(all_bit_vector_list[0]) --> number of relu layers in the model

    return all_bit_vector_list

def stack_bit_vectors(dct):
    vectors = [value for value in dct.values()]
    return np.vstack(vectors)


def get_inequalities(model, test, is_input_sample=True, verbose=False, return_bit_vec_list=True, input_dimension=None ):
    
    
    if is_input_sample:
        bit_vector_list_of_samples = bit_vector_from_relus(model, test)
        input_dimension = test.shape[0]
        
    else:
        bit_vector_list_of_samples = test
    
    #This is to store all the samples in a list
    A_list = []
    c_list =[]
    
    
    for a_bit_vector_list in bit_vector_list_of_samples:

        #These are for one single sample
        A_for_a_bit_vector = np.empty((0, input_dimension))
        
        c_for_a_bit_vector = np.empty((0, 1))
        

        weight_output_dict, bias_output_dict = get_weights_and_biases(model)

        #layer_1
        bit_str = list(a_bit_vector_list)[0].reshape(-1,1)
        
        if verbose:
            print("bit_str: ", bit_str, bit_str.shape)

        sign_str = np.where(bit_str == 0, 1, -1)
        if verbose:
            print("sign_str: ", sign_str, sign_str.shape)

        #W_1_hat = W_1
        W = list(weight_output_dict.values())[0]
        if verbose:
            print("Weight: ",W, W.shape)

        #b_1_hat = b_1
        b = list(bias_output_dict.values())[0]
        if verbose:
            print("b_1: ",b, b.shape)

        #A1 = -np.matmul(np.diag(sign_str1), W1)

        #if we dont do reshape to a column vector like bit_str we cannot get diagonal matrix
        A = np.diag(sign_str.reshape(-1,))@ W
        if verbose:
            print("A_1: " ,A, A.shape)
        # print("diag: ",np.diag(sign_str.reshape(-1,)),np.diag(sign_str.reshape(-1,)).shape)

        # #this is element wise multiply
        # #c1 = np.multiply(b1, sign_str1)
        c = -b*sign_str
        if verbose:
            print("c_1: " ,c, c.shape)
        
        A_for_a_bit_vector = np.concatenate((A_for_a_bit_vector, A))
        c_for_a_bit_vector = np.concatenate((c_for_a_bit_vector, c))

        for i in range(1,(len(a_bit_vector_list))):

            W_next = list(weight_output_dict.values())[i] #.transpose()
            b_next = list(bias_output_dict.values())[i]

            W_next_hat = np.matmul(W_next, np.matmul(np.diag(bit_str.reshape(-1,)), W)) #1 gets updated
            b_next_hat = np.matmul(W_next, np.matmul(np.diag(bit_str.reshape(-1,)), b))+b_next #b gets updated

            bit_str_next = list(a_bit_vector_list)[i].reshape(-1,1)
            if verbose:
                print("bit_str_next: ", bit_str_next)
            sign_str_next = np.where(bit_str_next == 0, 1, -1)
            if verbose:
                print("sign_str_next: ", sign_str_next)



            A_next = np.diag(sign_str_next.reshape(-1,)) @ W_next_hat
            if verbose:
                print("A_next: ", A_next)
            c_next = -b_next_hat*sign_str_next
            A = np.concatenate((A, A_next))
            c = np.concatenate((c, c_next))

            W = W_next_hat
            b = b_next_hat
            bit_str = bit_str_next
            
            A_for_a_bit_vector = np.concatenate((A_for_a_bit_vector, A_next))
            c_for_a_bit_vector = np.concatenate((c_for_a_bit_vector, c_next))
        
        A_list.append(A_for_a_bit_vector)
        c_list.append(c_for_a_bit_vector)
            #final shape of A for a single sample needs to be (total_nodes x num_features) e.g.,(3+3=6, 2)
            
    if return_bit_vec_list:
        # Stack the elements separately
        stacked_bit_vector_list = [np.concatenate(element, axis=0) for element in bit_vector_list_of_samples]
        return A_list, c_list, stacked_bit_vector_list  
        
    return A_list, c_list 


    # # A_del denote the matrix by deleting the ith row from A, and c_del denote the vector by deleting the ith
# element from c, checking whether max (a_i x)<= c_i satisfying constraint A_del x<= c_del, if yes,
# the ith bit is inactive, if max (a_i x)> c_i or min(-a_i x)<-c_i, the ith bit is active
#  store the active bit index in active_bit_index

#https://www.cvxpy.org/api_reference/cvxpy.problems.html
# https://www.cvxpy.org/api_reference/cvxpy.problems.html
def active_bits_index(A_random_sample,c_random_sample, verbose=False, n_jobs=1):
    global A_active
    global c_active

    active_bits_index = []

    A_active = A_random_sample[0]
    c_active = c_random_sample[0].reshape(-1)
    num_bits = A_active.shape[0]
    

    # create a process pool that uses all cpus
    with multiprocessing.Pool(n_jobs) as pool:
        items = np.arange(num_bits)
        progress_count = 0  # Track the iteration count

        for index in pool.imap(solve_problem, items):
            if index != -1:
                active_bits_index.append(index)
            if verbose and ((progress_count % 250) == 0):
                print(f"bit index {progress_count + 1}/{num_bits}")
                # print(f"active bits: {active_bits_index}")
            progress_count += 1

    return np.array(active_bits_index)


def solve_problem(i):

    A = A_active
    c = c_active

    dim_input = A.shape[1]

    A_del = np.delete(A, i, 0)
    c_del = np.delete(c, i, 0)
    # solving the LP problem
    x = cp.Variable(dim_input)

    prob = cp.Problem(cp.Minimize((-A[i, :]) @ x), [A_del @ x <= c_del])
    prob.solve()
    if prob.value < -c[i]:
        return i
    else:
        return -1

def calculate_center_of_mass(vertices):
    num_vertices = len(vertices)
    sum_x = sum(vertex[0] for vertex in vertices)
    sum_y = sum(vertex[1] for vertex in vertices)
    center_x = sum_x / num_vertices
    center_y = sum_y / num_vertices
    return center_x, center_y

def clip_vertices(vertices, bound):
    clipped_vertices = []
    for i in range(len(vertices)):
        curr_vertex = vertices[i]
        next_vertex = vertices[(i + 1) % len(vertices)]  # Get the next vertex (wraps around to the first vertex)
        
        x1, y1 = curr_vertex[0], curr_vertex[1]
        x2, y2 = next_vertex[0], next_vertex[1]

        # Check if the line segment intersects with the boundaries
        if (-bound <= x1 <= bound and -bound <= y1 <= bound):
            clipped_vertices.append(curr_vertex)

        # Check if the line segment intersects with the x-axis boundaries
        if (y1 < -bound and y2 > -bound) or (y1 > -bound and y2 < -bound):
            intersect_x = x1 + ((-bound - y1) * (x2 - x1)) / (y2 - y1)
            if -bound <= intersect_x <= bound:
                clipped_vertices.append([intersect_x, -bound])

        if (y1 < bound and y2 > bound) or (y1 > bound and y2 < bound):
            intersect_x = x1 + ((bound - y1) * (x2 - x1)) / (y2 - y1)
            if -bound <= intersect_x <= bound:
                clipped_vertices.append([intersect_x, bound])

        # Check if the line segment intersects with the y-axis boundaries
        if (x1 < -bound and x2 > -bound) or (x1 > -bound and x2 < -bound):
            intersect_y = y1 + ((-bound - x1) * (y2 - y1)) / (x2 - x1)
            if -bound <= intersect_y <= bound:
                clipped_vertices.append([-bound, intersect_y])

        if (x1 < bound and x2 > bound) or (x1 > bound and x2 < bound):
            intersect_y = y1 + ((bound - x1) * (y2 - y1)) / (x2 - x1)
            if -bound <= intersect_y <= bound:
                clipped_vertices.append([bound, intersect_y])

    return clipped_vertices

def active_bits_index_2(A, c):
    
    active_bits_index = []
    num_bits = A.shape[0]
    dim_input = A.shape[1]
    for i in np.arange(num_bits):
        A_del = np.delete(A, i, 0)
        c_del = np.delete(c, i, 0)
        # solving the LP problem    
        x = cp.Variable(dim_input)
        prob = cp.Problem(cp.Minimize((-A[i, :])@x),[A_del @ x <= c_del])
        prob.solve()
        if prob.value < - c[i]:
            active_bits_index.append(i)
    return np.array(active_bits_index), prob.solver_stats