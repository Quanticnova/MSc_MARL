import torch.nn as nn
import torch.nn.functional as f
import torch
from .pfrnns import PFGRUCell as pfrnn
class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class PFRNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(PFRNN, self).__init__()
        self.args = args
        self.agent_batch = args.batch_size*args.n_agents
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        #self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.rnn = pfrnn(15,args.rnn_hidden_dim,args.rnn_hidden_dim,32,32,0.5)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h0,p0 = hidden_state
        #exit()
        h0 = h0.cuda()
        p0 = p0.cuda()

        hidden_state = (h0,p0)
        #h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h,p = self.rnn(x, hidden_state)
        #q = self.fc2(h)

        # for now use mean aggregator
        #p = p.view(15,-1,1)
        #h = h.view(15,-1,self.args.rnn_hidden_dim)
        
        #mean = torch.sum(h * torch.exp(p),dim=0)
        if h0.size()[0] == self.agent_batch:

            #self.eval_hidden = torch.zeros((episode_num,15, self.n_agents, self.args.rnn_hidden_dim))
            #self.eval_p = torch.zeros((episode_num,15,self.n_agents,1))
            h = h.view(self.args.batch_size,15, self.args.n_agents, self.args.rnn_hidden_dim)
            p = p.view(self.args.batch_size,15, self.args.n_agents, 1)
            #print(h0.size())
            #print(p.size())
            #exit()
            #print(h.size())
            #print(p.size())
            #exit()
            mean = torch.sum(h * torch.exp(p), dim=1) 
            mean = mean.view(self.args.batch_size*self.args.n_agents,self.args.rnn_hidden_dim) 
            #mean  =
            #print(mean.size())
            #exit()
        else:
            #print(h.size())
            #print(p.size())
            #exit()
            mean = torch.sum(h * torch.exp(p), dim=0)
            #print(mean.size())
            #exit()
           
        q = self.fc2(mean)
        q = q.unsqueeze(0)
       
        return q, h,p

class DRNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(DRNN, self).__init__()
        self.args = args
        self.n_atom = 51
        self.v_min = 0
        self.v_max = 20
        self.v_range = torch.linspace(self.v_min,self.v_max)
        self.v_step = ((self.v_max-self.v_min)/(self.n_atom-1))

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions*self.n_atom)

    def forward(self, obs, hidden_state):
        #print(obs.size())
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        q = q.view(-1,self.args.n_actions,self.n_atom)
        #print(q.size())
        q = torch.softmax(q,dim=1)
        #print(q.size())
        #exit()
        return q, h

# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class CFV_RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(CFV_RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        #policy head
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        #critic head
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        cfv = self.fc3(h)
        return q, h,cfv

class shape(nn.Module):
    # obs_shape应该是obs_shape+n_actions+n_agents，还要输入当前agent的上一个动作和agent编号，这样就可以只使用一个神经网络
    def __init__(self, input_shape, args):
        super(shape, self).__init__()
        self.args = args
        
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, obs):
        x = f.relu(self.fc1(obs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q