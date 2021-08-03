#-*- coding: UTF-8 -*- 
import torch
import os
from network.base_net import RNN
from network.vdn_net import VDNNet
from network.IG_Critic import LRPCritic
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from common.utils import td_lambda_target
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

class VDN:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape*args.stack
        self.obs_shape = args.obs_shape*args.stack
        c_obs = args.obs_shape
        input_shape = self.obs_shape
        #print(input_shape)
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
            c_obs += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents
            #c_obs += self.n_agents

        #print(input_shape)
        #Declare Networks
        self.critic_in = (int(c_obs)*self.n_agents)
        #print(self.critic_in)
        #exit()
        self.eval_rnn = RNN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = RNN(input_shape, args)
        self.eval_vdn_net = VDNNet()  # 把agentsQ值加起来的网络
        self.target_vdn_net = VDNNet()
        self.ig_net = LRPCritic(self.critic_in,args)
        self.target_ig_net = LRPCritic(self.critic_in,args)
        self.ins = input_shape
        #print((input_shape-self.n_agents)*self.n_agents)
        #exit()
        self.args = args

        """
        def forward_hook(self, input, output):
            self.X = input[0]
            self.Y = output
            

        for i in range(0, len(self.ig_net.layers)):
            self.ig_net.layers[i].register_forward_hook(forward_hook)
            self.target_ig_net.layers[i].register_forward_hook(forward_hook)
        """

        #exit()
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_vdn_net.cuda()
            self.target_vdn_net.cuda()
            self.ig_net.cuda()
            self.target_ig_net.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        """
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_vdn = self.model_dir + '/vdn_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_vdn_net.load_state_dict(torch.load(path_vdn, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_vdn))
            else:
                raise Exception("No model!")
        """

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())
        self.target_ig_net.load_state_dict(self.ig_net.state_dict())

        self.eval_parameters = list(self.eval_rnn.parameters())
        self.ig_parameters = list(self.ig_net.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
            #self.c_optimizer = torch.optim.RMSprop(self.ig_parameters, lr=(args.lr))
            self.c_optimizer = torch.optim.Adam(self.ig_parameters, lr=(args.lr))

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        self.target_hidden_p = None
        print('Init alg LRP')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        #print(train_step)
        episode_num = batch['o'].shape[0]
        #print(episode_num)
        #print(self.args.batch_size)
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        # TODO pymarl中取得经验没有取最后一条，找出原因
        u, r, avail_u, avail_u_next, terminated = batch['u'], batch['r'],  batch['avail_u'], \
                                                  batch['avail_u_next'], batch['terminated']
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        if self.args.cuda:
            u = u.cuda()
            r = r.cuda()
            mask = mask.cuda()
            terminated = terminated.cuda()

        #******TRAIN CRITIC******
        
        Q_tots,inputs,Q_targets,inputs_next,_ = self.get_q_tot(batch, max_episode_len,train_step,False,terminated)
        Q_targets = r + self.args.gamma *Q_targets* (1 - terminated)
        Q_loss = ( Q_tots - Q_targets )#* mask
        Q_loss = (Q_loss**2).mean()#.sum()/mask.sum()
        
        self.c_optimizer.zero_grad()
        Q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ig_parameters, self.args.grad_norm_clip)
        self.c_optimizer.step()
        #print('Loss V is ', Q_loss)
        LRP_list = []
        
        scaled_list = []
        ig_list = []
        steps = 5

        


        #####TRAIN POLICY#####
        #INTEGRATED GRADIENT CALCULATIONS
        #generate unrolled paths
        unrolled_path = []
        for i in range(len(inputs)):
            input = inputs[i].cpu().detach().numpy()
            b = inputs_next[i].cpu().detach().numpy()
            step_sizes = (b - input)/steps
            #step_sizes = (inputs[i].cpu().detach().numpy() - inputs_next[i].cpu().detach().numpy())/steps
            each_unrolled_path = [step_sizes*i_step + input for i_step in range(steps)]
            
            #each_unrolled_path = [input + (float(j)/steps)*(input - b) for j in range(steps)]
            unrolled_path += each_unrolled_path

        unrolled_path.append(inputs[len(inputs)-1].cpu().detach().numpy())#terminated state
        step_sizes = np.asarray(unrolled_path[0:-1]) - np.asarray(unrolled_path[1:])
        #step_sizes = torch.stack(unrolled_path[0:-1]) - torch.stack(unrolled_path[1:])
        #exit()
        unrolled_path = unrolled_path[0:-1]
        full_step_size = step_sizes#.cpu().detach().numpy()

        #grads = torch.autograd(self.ig_net,unrolled_path)
        #exit()
        ex = self.calculate_outputs_and_gradients(unrolled_path,self.target_ig_net,self.args.cuda)
        #ex = self.calculate_outputs_and_gradients(unrolled_path,self.ig_net,self.args.cuda)
        #print(ex.shape)
        ex = full_step_size*ex
        #print(ex.shape)
        #exit()
        full_step = len(inputs)
        for loc in range(full_step):
            agent_ex = ex[loc*steps:]
            #print(agent_ex.shape)
            agent_ex = np.sum(agent_ex,axis=0) # sum across trajectories
            #print(agent_ex.shape)
            #exit()
            agent_ex = np.reshape(agent_ex, (self.args.n_agents, -1))#np.reshape(agent_ex,(self.args.batch_size,self.n_agents,self.critic_in)) 
            agent_ex = agent_ex.sum(axis = 1) # sum relevance values
            ig_list.append(agent_ex)
    

        if self.args.cuda:
                #ig_list.cuda()
                torch_device = torch.device('cuda:0')

        ig_list = torch.tensor(ig_list,dtype=torch.float32,device=torch_device)#.unsqueeze(0)
    
        
        if self.args.cuda == True:
            ig_list = ig_list.cuda()
            mask = mask.cuda()
        
        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents，n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        #print(q_evals.size())
        #exit()
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]
        


        targets = ig_list
        targets = targets.unsqueeze(0)
        #print(targets.size())
        #print(q_evals.size())
        #print(mask.size())
        #exit()
        #targets = targets.permute(1,0,2)
        #targets = targets#*mask
       
        td_error =  (q_evals - targets)
       
        td_error = td_error.sum(dim=2)
        td_error = td_error.unsqueeze(2)

       #print()
        #mask = (1 - batch["padded"].float().repeat(1, 1, self.n_agents)).cuda()
        masked_td_error =  td_error*mask  # 抹掉填充的经验的td_error
        loss = (masked_td_error ** 2).sum() / mask.sum()
        #loss = masked_td_error.mean()
        #print(loss.size())
        #exit()
        #print('Loss Q is ', loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        ###test decomposition###
        """
        final_q = self.ig_net(inputs_next[len(inputs_next)-1])
        a,b = [], []
        model = self.ig_net
        #print(len(inputs))
        #exit()
        for i in range(len(inputs)):
            #a.append(Q_targets[:,i,:])
            #b1 = self.ig_net(inputs[i]) - final_q
            #b.append(b1)
            o1 = model(inputs[i])
            #print(o1.size())
            #b1 = model(inputs_next[i])
            b1 = final_q
            ig = ig_list[i].detach().cpu()
            #print(b1.size())
            #print(ig_list[i].size())
            #exit()
            
            b.append(np.array((o1-b1).sum().detach().cpu()))
            a.append(np.array(ig).sum()) 
            
            #b.append(np.array((o1[i]-b1[i]).sum().detach().cpu()))
            #a.append(np.array(ig[i].sum())) 
        #np.set_printoptions(formatter={''}) 
        np.set_printoptions(suppress=True)

        plt.figure()
        plt.plot(np.array(a))
        plt.plot(np.array(b))
        plt.savefig('plot', format='png')
        print(np.array(a))
        print(np.array(b))
        exit()
        """
        

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())
            self.target_ig_net.load_state_dict(self.ig_net.state_dict())
    
    def calculate_outputs_and_gradients(self,inputs, model,cuda):
        # do the pre-processing
        #tf.gradients(self.target_q_tot, self.mixer_target_s_a)
        gradients = []
        model = model.cpu()
        for input in inputs:
            #input.cuda()
            #input.require_grad=True
            #if cuda:
                #torch_device = torch.device('cuda:0')
            #else:
                #torch_device = torch.devive('cpu') 
            
            with torch.autograd.set_detect_anomaly(True):
                #print(input.size())
                #exit()
                input = torch.Tensor(input)
                obs = input.clone().detach().requires_grad_(True)
                #print(obs)
                #obs = torch.tensor(input, dtype=torch.float32, requires_grad=True,device=torch_device)
                output = model(obs)

            output = output.sum()
            # clear grad
            model.zero_grad()
            output.backward()
            gradient = obs.grad.detach().cpu().numpy()
            gradients.append(gradient)
        gradients = np.array(gradients)
        if cuda:
            model = model.cuda()
        return gradients

    def _get_returns(self, r, mask, terminated, max_episode_len):
        r = r.squeeze(-1)
        mask = mask.squeeze(-1)
        terminated = terminated.squeeze(-1)
        terminated = 1 - terminated
        n_return = torch.zeros_like(r)
        n_return[:, -1] = r[:, -1] * mask[:, -1]
        for transition_idx in range(max_episode_len - 2, -1, -1):
            n_return[:, transition_idx] = (r[:, transition_idx] + self.args.gamma * n_return[:, transition_idx + 1] * terminated[:, transition_idx]) * mask[:, transition_idx]
        return n_return.unsqueeze(-1)
    
    def get_q_tot_in(self, batch, max_episode_len,train_step):
        episode_num = batch['o'].shape[0]
        input_list = []
        input_list_next = []
        for transition_idx in range(max_episode_len):
            
            inputs,inputs_next= self._get_Q_tot_inputs(batch, transition_idx,train_step,max_episode_len)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
    
            input_list.append(inputs)
            input_list_next.append(inputs_next)
        
        return input_list,input_list_next
    
    def _get_Q_tot_inputs(self, batch, transition_idx,train_step,max_ep_len):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条

        #print(self.args.stack)
        #exit()

        #obs = batch['o'][:,0]
        #if transition_idx >= self.args.stack-1:
        #    obs, obs_next, u_onehot = batch['o'][:, transition_idx-self.args.stack+1:transition_idx+1], \
        #                            batch['o_next'][:, transition_idx-self.args.stack+1:transition_idx+1], batch['u_onehot'][:]
            
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        
        inputs, inputs_next = [], []
        episode_num = obs.shape[0]
        inputs.append(obs)
        inputs_next.append(obs_next)
        if self.args.last_action:
            inputs.append(u_onehot[:, transition_idx])

            if transition_idx < max_ep_len-1:
                inputs_next.append(u_onehot[:, transition_idx+1])
            else:
                inputs_next.append(torch.zeros_like(u_onehot[:, transition_idx]))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * 1, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * 1, -1) for x in inputs_next], dim=1)

        inputs = inputs.cuda()
        inputs_next = inputs_next.cuda()
        #print(inputs.size())
        #exit()
  
        return inputs, inputs_next

    def get_q_tot(self, batch, max_episode_len,train_step,rel_calc,terminated):
        episode_num = batch['o'].shape[0]
        terminated = terminated
        #ig = IntegratedGradients(self.ig_net)
        q_evals, q_targets = [], []
        revelence = []
        input_list = []
        input_list_next = []
        input_grads = []
        hook_err = True
        lr_input,lr_inputs_next= self._get_Q_tot_inputs(batch, max_episode_len-1,train_step,max_episode_len)
        
        terminated = terminated.permute(1,0,2)

        for transition_idx in range(max_episode_len):
            
            inputs,inputs_next= self._get_Q_tot_inputs(batch, transition_idx,train_step,max_episode_len)  # 给obs加last_action、agent_id

            input_list.append(inputs)
            input_list_next.append(inputs_next)
            q_eval = self.ig_net(inputs)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target = self.target_ig_net(inputs_next)
            
            
            if rel_calc:
                #temp = self.ig_net(lr_input)
                if hook_err == False:
    
                    temp = q_eval#*(1-terminated[transition_idx])
                    #temp = temp.squeeze(2)
                    #print(temp.size())
                    #exit()
                    LRP = self.ig_net.relprop(temp)
                    LRP = LRP.view(self.args.batch_size,self.n_agents,(self.ins-self.n_agents)).detach()
                    #print(LRP.size())
                    #exit()
                    LRP = LRP.sum(dim=2)
                    #if LRP.sum() != q_eval.sum():
                        #hook_err = True

                
                
                if hook_err == True:
                    #hook_err = False
                    mini = torch.zeros((self.args.batch_size,(self.ins-self.n_agents)*self.n_agents))
                    for k in range(self.args.batch_size):
                        #LRP = self.target_ig_net.relprop(q_target[k])
                        #ratio = q_target[k].sum()/LRP.sum()
                        LRP = self.ig_net.relprop(q_eval[k])
                        ratio = q_eval[k].sum()/LRP.sum()
                        LRP = LRP.sum(dim = 0)*ratio
                        mini[k] = LRP
                    mini = mini.view(self.args.batch_size,self.n_agents,(self.ins-self.n_agents)).detach()
                    mini = mini.sum(dim=2)
                    LRP = mini
                    #print(LRP.sum())
                    #print("")

                    LRP = LRP.cuda()
                
                revelence.append(LRP)
            #print(LRP.sum())
            #print(q_eval.sum())
            #exit()

            q_evals.append(q_eval)
            q_targets.append(q_target)
            
        
        if rel_calc:
            #exit()
            revelence = torch.stack(revelence)
            revelence = revelence.view(self.args.batch_size,max_episode_len,self.args.n_agents)
    
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        
        return q_evals,input_list,q_targets,input_list_next,revelence

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs = batch['o'][:,0]
        #if transition_idx >= self.args.stack-1:
        #    obs, obs_next, u_onehot = batch['o'][:, transition_idx-self.args.stack+1:transition_idx+1], \
        #                            batch['o_next'][:, transition_idx-self.args.stack+1:transition_idx+1], batch['u_onehot'][:]
        
       
        first = batch['o'][:,0]
        first_next = batch['o_next'][:,0]
        u_onehot = batch['u_onehot'][:]

        #preprep stack
        obs_list = first.repeat(1,1,self.args.stack)
        obs_next_list = first_next.repeat(1,1,self.args.stack)

        for i in range(self.args.stack):
            if i < transition_idx:
                pos = self.args.stack-1-i
                stride = int(self.obs_shape/self.args.stack)
                obs_list[:,:, pos*stride: (pos+1)*stride ] = batch['o'][:,transition_idx-i] 
                obs_next_list[:,:,pos*stride: (pos+1)*stride] = batch['o_next'][:,transition_idx-i]
            #else:
                #obs_list.append(batch['o'][:,transition_idx])
                #obs_next_list.append(batch['o_next'][:,transition_idx])

            obs = obs_list
            obs_next = obs_next_list
            #obs = torch.stack(obs_list,dim=1)
            #obs_next = torch.stack(obs_next_list,dim=1)
            #print(obs.size())
            #exit()

        episode_num = obs.shape[0]
        inputs, inputs_next = [], []

        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        #print(obs.size())
        #exit()
        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode，agent，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成episode_num*n_agents条数据
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
                #self.target_hidden_p = self.target_hidden_p.cuda()
            #print(inputs.size())
            #print(self.eval_rnn)
            #exit()
            #print(self.eval_hidden.size())
            #exit()
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # 得到的q_eval维度为(episode_num*n_agents, n_actions)
            
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # 把q_eval维度重新变回(episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_vdn_net.state_dict(), self.model_dir + '/' + num + '_vdn_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
