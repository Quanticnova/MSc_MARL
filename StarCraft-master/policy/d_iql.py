import torch
import os
from network.base_net import DRNN as RNN
from network.vdn_net import VDNNet


class DIQL:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape*args.stack
        self.obs_shape = args.obs_shape*args.stack
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        # 神经网络
        self.eval_rnn = RNN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = RNN(input_shape, args)
        self.eval_vdn_net = VDNNet()  # 把agentsQ值加起来的网络
        self.target_vdn_net = VDNNet()
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_vdn_net.cuda()
            self.target_vdn_net.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
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

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

        self.eval_parameters = list(self.eval_vdn_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        self.n_atom = 51
        self.v_min = 0
        self.v_max = 20
        self.v_range = torch.linspace(self.v_min,self.v_max,self.n_atom)
        self.v_step = ((self.v_max-self.v_min)/(self.n_atom-1))

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg DVDN')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
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
        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents，n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        
        ###GET DISTRIBUTION
        delta_z = float(self.v_max- self.v_min)/(self.n_atom -1)
        support  = torch.linspace(self.v_min,self.v_max,self.n_atom)

        au = u
        avail_u_next = avail_u_next.unsqueeze(-1)
        avail_u_next = avail_u_next.repeat(1,1,1,1,self.n_atom)
        q_targets[avail_u_next == 0.0] = - 9999999
        self.v_range = self.v_range.view(1,1,1,1,self.n_atom).cuda()
        
        q_targets_mean = torch.sum(q_targets*self.v_range.expand_as(q_targets),dim=4)
        best_targets = q_targets_mean.argmax(dim=3)
        best_targets = best_targets.unsqueeze(-1)
        best_targets = best_targets.unsqueeze(-1)
        best_targets = best_targets.repeat(1,1,1,1,self.n_atom)
    
        qa_probs = torch.gather(q_targets,dim=3,index=best_targets).squeeze(3)
        #qa_probs = qa_probs.prod(dim=2)
        qa_probs = qa_probs.sum(dim=2)#/self.n_agents
        qa_probs = qa_probs.squeeze(0).cuda()

        #print(r.size())
        #print(qa_probs.size())
        #exit()
        r = r.expand_as(qa_probs)
        #exit()

        #print(r.size())
        #print(self.v_range.size())
        #self.v_range = self.v_range.view(1,51,1)
        self.v_range = torch.linspace(self.v_min,self.v_max,self.n_atom).cuda()
        #r = r.squeeze(2)
        #sexit()
      
        t1 = terminated.expand_as(r)
        #print(terminated.size())
        #print(r.size())
        #print(t1.size())
        #exit()
        Tz = (r +  self.args.gamma*self.v_range.unsqueeze(0).expand_as(r))#.unsqueeze(0).expand_as(r)
        Tz = Tz.clamp(min=self.v_min,max=self.v_max) # Bellman operator
        b = (Tz - self.v_min)/self.v_step
        l  = b.floor().long()
        u  = b.ceil().long()

        #fix dissapearing prob mass
        #l[(u > 0) * (l == u)] -= 1
        #u[(l < (self.n_atom - 1)) * (l == u)] += 1

        #distribute probs
        m = torch.zeros(max_episode_len,self.args.batch_size,self.n_atom).cuda()#.type(dtype.FT)
        offset = torch.linspace(0,(max_episode_len-1)*self.n_atom,max_episode_len).long().unsqueeze(1).expand(max_episode_len,self.n_atom).cuda()
        
        #print(qa_probs.size())
        m.view(-1).index_add_(0, (l + offset).view(-1),(qa_probs * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1),(qa_probs * (b - l.float())).view(-1))

        au = au.unsqueeze(-1)
        au = au.repeat(1,1,1,1,self.n_atom)
        
        q_evals = q_evals.clamp(min=1e-3)
        dist = torch.gather(q_evals, dim=3, index=au).squeeze(3)
        #dist = dist.prod(dim=2)
        dist = dist.sum(dim=2)#/self.n_agents

        #print(dist.size())
        #print(m.size())
        #exit()
        dist =dist.squeeze(0)
        m = m.permute(1,0,2)
        #print(dist.size())
        #print(m.size())
        #print(terminated.size())
        #exit()
        
        terminated = terminated.squeeze(-1)
        td_error = (m * (-torch.log(dist))).sum(dim=2) * (1-terminated)#(dim=1)
        #print(td_error.size())
        #exit()
        #exit()
        mask = mask.squeeze()
        #print(mask.size())
        #print(td_error.size())
        #exit()
        #print(td_error.size())
        #print(mask.size())
        #exit()
        masked_td_error = td_error * mask  # 抹掉填充的经验的td_error
        #loss = masked_td_error.mean()#masked_td_error.sum()/mask.sum()
        loss = masked_td_error.sum()/mask.sum()
        # loss = masked_td_error.pow(2).mean()
        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        #loss = (masked_td_error ** 2).sum() / mask.sum()
        # print('Loss is ', loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

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

            #print(inputs.size())
            #exit()
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # 得到的q_eval维度为(episode_num*n_agents, n_actions)
            #print(q_eval.size())
            #exit()
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # 把q_eval维度重新变回(episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, self.n_actions,self.n_atom)
            #print(q_eval.size())
            q_target = q_target.view(episode_num, self.n_agents, self.n_actions,self.n_atom)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        #print(q_evals.size())
        #exit()
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