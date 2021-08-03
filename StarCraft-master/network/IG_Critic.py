#-*- coding: UTF-8 -*- 
import torch.nn as nn
import torch
import torch.nn.functional as F
#import torch.nn.functional as f
'''
输入当前的状态、当前agent的obs、其他agent执行的动作、当前agent的编号对应的one-hot向量、所有agent上一个timestep执行的动作
输出当前agent的所有可执行动作对应的联合Q值——一个n_actions维向量
advantage在外面进行计算，不在这个coma_critic中，但是可以认为是coma_critic的一部分
'''

class IGCritic(nn.Module):
    def __init__(self, input_shape,args):
        super(IGCritic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,32)
        self.fc4 = nn.Linear(32, args.n_actions)
        

    def forward(self, inputs):
        #print(inputs.size())
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        #q_tot = self.fc2(x)

        #q_tot = self.layers(inputs)
        
        return x

# add relprop() method to each layer
########################################
class Linear(nn.Linear):
    def __init__(self, linear,args):
        super(nn.Linear, self).__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.args = args
        
    def relprop(self, R):
        V = torch.clamp(self.weight, min=0)
        #V = self.weight
        #print(self.X.size())
        #self.X = self.X.unsqueeze(0)
        #print(V.size())
        Z = torch.mm(self.X, torch.transpose(V,0,1)) + self.args.e
        
        #print(R.size())
        #print(Z.size())
        S = R / Z
        
        C = torch.mm(S, V)
        R = self.X * C
        return R

    #gamma rule
    def relprop2(self, R):
        gamma = 0.1
        pos = torch.clamp(self.weight, min=0)*gamma
        V = self.weight + pos
        #print(V.size())
        #print(self.X.size())
        Z = torch.mm(self.X, torch.transpose(V,0,1)) + self.args.e
        
        #print(Z.size())
        #exit()
        #print("X")
        #print(R.sum())
        S = R / Z
        #print(S.size())
        
        C = torch.mm(S, V)
        #print(C.size())
        R = self.X * C
        #print(R)
        #exit()
        return R

    def relprop3(self, R):
        alpha = 1
        beta = 0

        sel = self.weight> 0
        zeros = torch.zeros_like(self.weight)

        weights_pos = torch.where(sel,  self.weight, zeros)
        weights_neg = torch.where(~sel, self.weight, zeros)

        X1  = torch.where(self.X >  0, self.X, torch.zeros_like(self.X))
        X2  = torch.where(self.X <= 0, self.X, torch.zeros_like(self.X))

        def f(X1,X2,W1,W2):
            #get positives
            W1 = weights_pos
            Z1 = torch.mm(self.X, torch.transpose(W1,0,1))
            
            #get negatives
            W2 = weights_neg
            Z2 = torch.mm(self.X, torch.transpose(W2,0,1))

            Z = Z1 + Z2
            S = R / (Z+(Z==0)*1e-6)

            C1 = torch.mm(S, W1)
            C2 = torch.mm(S, W2)

            R1 = C1*X1
            R2 = C2*X2

            return R1 + R2

        R_pos = f(X1,X2,weights_pos,weights_neg)
        R_neg = f(X2,X1,weights_pos,weights_neg)
        
        return R_pos * alpha - R_neg * beta

class ReLU(nn.ReLU):   
    def relprop(self, R): 
        return R

class GRUCell(nn.GRUCell):
    def __init__(self, grucell,args):
        #print(grucell.input_size)
        super(GRUCell, self).__init__(64,64)
        self.in_features = grucell.input_size
        self.weight_ih = grucell.weight_ih
        self.weight = self.weight_ih[128:,:]#grucell.weight_ih
        self.bias = grucell.bias
        self.args = args
        
    def relprop(self, R):
        V = torch.clamp(self.weight, min=0)
        V = V.cuda()
        #exit()
        Z = torch.mm(self.X, torch.transpose(V,0,1)) + self.args.e
        S = R / Z
        C = torch.mm(S, V)
        R = self.X * C
        return R

class extractor(nn.Module):
    def __init__(self, input_shape,f_size):
        super(extractor, self).__init__()

        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64,f_size)


    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        #x = self.fc2(x)
        features = x
        
        return features


class GRU_extractor(nn.Module):
    def __init__(self, input_shape,f_size,args):
        super(GRU_extractor, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, f_size)


    def forward(self, obs,hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        features = self.fc2(h)
        
        return features,h
    

class LRPCritic(nn.Module):
    def __init__(self, input_shape,args):
        super(LRPCritic, self).__init__()

        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32, 1)


        self.layers = nn.Sequential(
        Linear(self.fc1,args),
        ReLU(),
        Linear(self.fc2,args),
        ReLU(),
        Linear(self.fc3,args)
        )

    def forward(self, inputs):

        q_tot = self.layers(inputs)
        
        return q_tot
    
    def relprop(self, R):
        #print(len(self.layers))
        for l in range(len(self.layers), 0, -1):
            R = self.layers[l-1].relprop(R)
        return R

class LRP_GRUCritic(nn.Module):
    def __init__(self, input_shape,args):
        super(LRP_GRUCritic, self).__init__()

        
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.args= args
        #print(self.rnn)
        #exit()
        
        
        self.layers = nn.Sequential(
        Linear(self.fc1,args),
        ReLU(),
        GRUCell(self.rnn,args),
        ReLU(),
        Linear(self.fc2,args)
        )

    def forward(self, inputs,hidden_state):
        
        """
        x = ReLU(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        
        
        return q,h
        """
        q_tot = self.layers(inputs)
        #print(q_tot.size())
        #exit()
        

        return q_tot
    
    def relprop(self, R):
        #print(len(self.layers))
        for l in range(len(self.layers), 0, -1):
            R = self.layers[l-1].relprop(R)
        return R

class GruIGCritic(nn.Module):
    def __init__(self, input_shape,args):
        super(GruIGCritic, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        

    def init_hidden(self):
        # make hidden states on same device as model
        #print("INIT")
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()


    def forward(self, inputs,hidden_state):

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        #x = F.relu(self.fc2(x))
        q = self.fc2(h)
        return q, h

class RNN(nn.Module):
    # obs_shape应该是obs_shape+n_actions+n_agents，还要输入当前agent的上一个动作和agent编号，这样就可以只使用一个神经网络
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, obs, hidden_state):
        
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h