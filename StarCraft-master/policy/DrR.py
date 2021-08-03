import torch
import os
from network.base_net import RNN
from network.commnet import CommNet
from network.g2anet import G2ANet
from network.coma_critic import DrRCritic as ComaCritic
from common.utils import td_lambda_target2 as td_lambda_target
from torch.distributions import Categorical
import copy

class DrReinforce:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape*args.stack
        actor_input_shape = self.obs_shape  # actor网络输入的维度，和vdn、qmix的rnn输入维度一样，使用同一个网络结构
        critic_input_shape = self._get_critic_input_shape()  # critic网络输入的维度
        
        print(critic_input_shape)
        # 根据参数决定RNN的输入维度
        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_network:
            actor_input_shape += self.n_agents
        args.cuda=False
        self.args = args
        #self.args.cuda=False
        # 神经网络
        # 每个agent选动作的网络,输出当前agent所有动作对应的概率，用该概率选动作的时候还需要用softmax再运算一次。
        if self.args.alg == 'DrR':
            print('Init alg coma')
            self.eval_rnn = RNN(actor_input_shape, args)
        elif self.args.alg == 'coma+commnet':
            print('Init alg coma+commnet')
            self.eval_rnn = CommNet(actor_input_shape, args)
        elif self.args.alg == 'coma+g2anet':
            print('Init alg coma+g2anet')
            self.eval_rnn = G2ANet(actor_input_shape, args)
        else:
            raise Exception("No such algorithm")

        # 得到当前agent的所有可执行动作对应的联合Q值，得到之后需要用该Q值和actor网络输出的概率计算advantage
        self.eval_critic = ComaCritic(critic_input_shape, self.args)
        self.target_critic = ComaCritic(critic_input_shape, self.args)

        if self.args.cuda:
            #print("CUDA")
            #exit()
            self.eval_rnn.cuda()
            self.eval_critic.cuda()
            self.target_critic.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_params.pkl'):
                path_rnn = self.model_dir + '/rnn_params.pkl'
                path_coma = self.model_dir + '/critic_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_critic.load_state_dict(torch.load(path_coma, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_coma))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_critic.load_state_dict(self.eval_critic.state_dict())

        self.rnn_parameters = list(self.eval_rnn.parameters())
        self.critic_parameters = list(self.eval_critic.parameters())

        if args.optimizer == "RMS":
            self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=args.lr_critic)
            self.rnn_optimizer = torch.optim.Adam(self.rnn_parameters, lr=args.lr_actor)
        self.args = args

        self.eval_hidden = None

    def _get_critic_input_shape(self):
        # state
        input_shape = self.state_shape   # 48
        # obs
        input_shape += self.obs_shape  # 30
        # agent_id
        input_shape += self.n_agents  # 3
        # 所有agent的当前动作和上一个动作
        input_shape += self.n_actions * self.n_agents * 2  # 54

        print(input_shape)
        return input_shape
    
    def _get_Q_tot_inputs(self, batch, transition_idx,max_ep_len):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        #obs = obs[train_step].unsqueeze(0)
        #obs_next = obs_next[0]
        #u_onehot = u_onehot[train_step].unsqueeze(0)

        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        #print(obs)
        inputs_next.append(obs_next)
        
        """
        if self.args.last_action:
            inputs.append(u_onehot[:, transition_idx])

            if transition_idx < max_ep_len-1:
                inputs_next.append(u_onehot[:, transition_idx+1])
            else:
                inputs_next.append(torch.zeros_like(u_onehot[:, transition_idx]))
        """
        
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * 1, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * 1, -1) for x in inputs_next], dim=1)
        
        if self.args.cuda:
            inputs = inputs.cuda()
            inputs_next = inputs_next.cuda()
  
        return inputs, inputs_next

    def learn(self, batch, max_episode_len, train_step, epsilon):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            #print(key)
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        #exit()
        u, r, avail_u, terminated = batch['u'], batch['r'],  batch['avail_u'], batch['terminated']
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        if self.args.cuda:
            u = u.cuda()
            mask = mask.cuda()

        q_values = self._train_critic(batch, max_episode_len, train_step)  # 训练critic网络，并且得到每个agent的所有动作的Ｑ值  
        
        returns = self._get_returns(q_values,mask,terminated,max_episode_len)
        #print(returns.size())
    

        action_prob = self._get_action_prob(batch, max_episode_len, epsilon)  # 每个agent的所有动作的概率
        dist = Categorical(action_prob)
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)  # 每个agent的选择的动作对应的概率
        pi_taken[mask == 0] = 1.0  # 因为要取对数，对于那些填充的经验，所有概率都为0，取了log就是负无穷了，所以让它们变成1
        log_pi_taken = torch.log(pi_taken)                  

        loss1 = -(returns*log_pi_taken)
        loss = (loss1*mask).sum()/mask.sum()
        
        self.rnn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()

    def _get_returns(self, r, mask, terminated, max_episode_len):
        episode_num = r.size()[0]
        r = r.squeeze(-1)
        mask = mask.squeeze(-1)
        terminated = 1 - terminated
        terminated = terminated.repeat(1,1,self.n_agents)
        n_return = torch.zeros_like(r)
        if self.args.cuda:
            r = r.cuda()
            mask = mask.cuda()
            terminated = terminated.cuda()
            n_return = n_return.cuda()
        
        n_return[:, -1,:] = r[:, -1,:] * mask[:, -1,:]
        #n_return = n_return.cuda()
        #print(n_return.size())
        #exit()
        for transition_idx in range(max_episode_len - 2, -1, -1):
            n_return[:, transition_idx] = (r[:, transition_idx,:] + self.args.gamma * n_return[:, transition_idx + 1,:] * terminated[:, transition_idx,:]) * mask[:, transition_idx,:]
        
        return n_return#.unsqueeze(-1).expand(-1, -1, self.n_agents)

    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        # 取出所有episode上该transition_idx的经验
        #print("INPUTS GET")
        obs, obs_next, s, s_next = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx],\
                                   batch['s'][:, transition_idx], batch['s_next'][:, transition_idx]
        episode_num = obs.shape[0]
        u_onehot = batch['u_onehot'][:, transition_idx]
        u_onehot_train = u_onehot.view((episode_num, 1, -1))
        if transition_idx != max_episode_len - 1:
            u_onehot_next = batch['u_onehot'][:, transition_idx + 1]
        else:
            u_onehot_next = torch.zeros(*u_onehot.shape)
        # s和s_next是二维的，没有n_agents维度，因为所有agent的s一样。其他都是三维的，到时候不能拼接，所以要把s转化成三维的
        #print(s.size())
        s_train = s.view((episode_num, 1, -1))
        s = s.unsqueeze(1).expand(-1, self.n_agents, -1)
        
        s_next = s_next.unsqueeze(1).expand(-1, self.n_agents, -1)
        
        if self.args.cuda:
            s = s.cuda()
            s_train = s_train.cuda()
            s_next = s_next.cuda()
            u_onehot = u_onehot.cuda()
            u_onehot_next = u_onehot_next.cuda()
            u_onehot_train = u_onehot_train.cuda()
            
            obs = obs.cuda()
            obs_next =obs_next.cuda()

            
        
        episode_num = obs.shape[0]
        # 因为coma的critic用到的是所有agent的动作，所以要把u_onehot最后一个维度上当前agent的动作变成所有agent的动作
        
        u_onehot = u_onehot.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
        u_onehot_next = u_onehot_next.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)

        
        if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
            u_onehot_last = torch.zeros_like(u_onehot)
            if self.args.cuda:
                u_onehot_last = u_onehot_last.cuda()
        else:
            u_onehot_last = batch['u_onehot'][:, transition_idx - 1]
            u_onehot_last = u_onehot_last.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
            if self.args.cuda:
                u_onehot_last = u_onehot_last.cuda()

        inputs, inputs_next = [], []
        inputs_train = []
        # append central state
        inputs.append(s)
        inputs_train.append(s_train)
        inputs_next.append(s_next)
        inputs_train.append(u_onehot_train)
        
        #append local obs
        inputs.append(obs)
        inputs.append(u_onehot_last)

        #print(u_onehot.size)
        # 添加当前动作
        '''
        因为coma对于当前动作，输入的是其他agent的当前动作，不输入当前agent的动作，为了方便起见，每次虽然输入当前agent的
        当前动作，但是将其置为0相量，也就相当于没有输入。
        '''
        action_mask = (1 - torch.eye(self.n_agents))  
        action_mask = action_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        if self.args.cuda:
            action_mask = action_mask.cuda()
        #action_mask = action_mask.unsqueeze(0)
        #print(u_onehot.size())
        #print(action_mask.size())
        #exit()
        inputs.append(u_onehot * action_mask.unsqueeze(0))
        inputs_next.append(u_onehot_next * action_mask.unsqueeze(0))

        test = u_onehot * action_mask
        #print(test)
        #print(u_onehot_train)
       

        # 添加agent编号对应的one-hot向量
        '''
        因为当前的inputs三维的数据，每一维分别代表(episode编号，agent编号，inputs维度)，直接在后面添加对应的向量
        即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
        agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
        '''
        if self.args.cuda:
            inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1).cuda())
            inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1).cuda())
        else:
            inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        # 要把inputs中的5项输入拼起来，并且要把其维度从(episode_num, n_agents, inputs)三维转换成(episode_num * n_agents, inputs)二维
        #for x in inputs:
            #print(x.size())
        #exit()
        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next], dim=1)

        #for x in inputs_train:
            #print(x.size())
        inputs_train = torch.cat([x for x in inputs_train], dim=2)
        #print(inputs_train.size())
        #print(inputs.size())
        #exit()
        return inputs, inputs,inputs

    def _get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next,inputs_train = self._get_critic_inputs(batch, transition_idx,max_episode_len)
            #print(inputs.size())
            #print(inputs_train.size())
            #inputs_train = inputs_train.unsqueeze(0)
            #exit()
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                inputs_train = inputs_train.cuda()
            # 神经网络输入的是(episode_num * n_agents, inputs)二维数据，得到的是(episode_num * n_agents， n_actions)二维数据
            with torch.no_grad():
                q_eval = self.eval_critic(inputs)
            q_train = self.eval_critic(inputs_train)
            #q_target = self.target_critic(inputs_next)

            # 把q值的维度重新变回(episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_train = q_train.view(episode_num,self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_train)
        # 得的q_evals和q_targets是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1).squeeze(-1)
        q_targets = torch.stack(q_targets, dim=1).squeeze(-1)
        #print(q_evals.size())
        #print(q_targets.size())
        #exit()
        return q_evals, q_targets

    def _get_actor_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)
        # 给inputs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_network:
            # 因为当前的inputs三维的数据，每一维分别代表(episode编号，agent编号，inputs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把inputs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_action_prob(self, batch, max_episode_len, epsilon):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']  # coma不用target_actor，所以不需要最后一个obs的下一个可执行动作
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
            outputs, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            # 把q_eval维度重新变回(8, 5,n_actions)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)
        # 得的action_prob是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        action_prob = torch.stack(action_prob, dim=1).cpu()

        action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])   # 可以选择的动作的个数
        action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
        action_prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        # 因为上面把不能执行的动作概率置为0，所以概率和不为1了，这里要重新正则化一下。执行过程中Categorical会自己正则化。
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        # 因为有许多经验是填充的，它们的avail_actions都填充的是0，所以该经验上所有动作的概率都为0，在正则化的时候会得到nan。
        # 因此需要再一次将该经验对应的概率置为0
        action_prob[avail_actions == 0] = 0.0
        if self.args.cuda:
            action_prob = action_prob.cuda()
        return action_prob

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def _train_critic(self, batch, max_episode_len, train_step):
        u, r, avail_u, terminated = batch['u'], batch['r'], batch['avail_u'], batch['terminated']
        u_next = u[:, 1:]
        padded_u_next = torch.zeros(*u[:, -1].shape, dtype=torch.long).unsqueeze(1)
        u_next = torch.cat((u_next, padded_u_next), dim=1)
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        if self.args.cuda:
            u = u.cuda()
            u_next = u_next.cuda()
            mask = mask.cuda()
            r = r.cuda()
        q_evals, q_train = self._get_q_values(batch, max_episode_len)
        #print(q_train.size())
        q_evals = torch.gather(q_evals,dim=3,index=u).squeeze(3)
        q_train = torch.gather(q_train,dim=3,index=u).squeeze(3)
        #print(q_train.size())
        #exit()
        r =r.repeat(1,1,self.n_agents)
        #q_train = q_train.sum(dim=2)
        #q_train = q_train.unsqueeze(-1)
        
        #print(r.size())
        #print(q_train.size())
        #print(mask.size())
        #exit()
        surr1 = r - q_train   #surr1

        #td_error = torch.min(surr1,surr2).squeeze(-1)
        td_error = surr1
        #l2 = self.old_vals + 
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        #print(loss)
        # print('Loss is ', loss)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
        self.critic_optimizer.step()
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_critic.load_state_dict(self.eval_critic.state_dict())

        #exit()
        #get counterfactual values


        return q_evals

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_critic.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')