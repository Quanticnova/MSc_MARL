import torch
import os
from network.base_net import RNN
from network.vdn_net import VDNNet
from network.IG_Critic import IGCritic
from network.base_net  import CFV_RNN
from common.utils import td_lambda_target
import copy
class ARM:
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

        # Declare Neural networks
        self.eval_rnn = CFV_RNN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = CFV_RNN(input_shape, args)
        self.eval_vdn_net = VDNNet()  # 把agentsQ值加起来的网络
        self.target_vdn_net = VDNNet()

        # Set old critic and policy to be copies of old networks
        self.old_policy = copy.deepcopy(self.eval_rnn)
        self.args = args

        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_vdn_net.cuda()
            self.target_vdn_net.cuda()
            self.old_policy.cuda()

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
        #self.critic_parameters = list(self.critic_net.parameters())

        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
            #self.c_optimizer = torch.optim.RMSprop(self.critic_parameters, lr=args.lr*10)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        self.old_eval_hidden = None
        self.old_target_hidden = None
        print('Init alg VDN')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
    
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
        
        
        #Get q_values and q_targets 
        q_evals, q_targets,v_evals,v_targets,td_lambda = self.get_q_values(batch, max_episode_len)
        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]

        #self.init_hidden(episode_num)
        with torch.no_grad():
            old_q_evals,old_v_evals = self.get_old_q_values(batch, max_episode_len)
            
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        #q_evals = q_evals.sum(dim=2)

        #print(v_evals.size())
        #print(v_targets.size())
        #print(q_evals.size())
        #print(q_targets.size())
        #exit()
        v_evals = v_evals.squeeze(3)
        v_targets = v_targets.squeeze(3)
        old_q_evals = torch.gather(old_q_evals, dim=3, index=u).squeeze(3)
        old_v_evals = old_v_evals.squeeze(3)

        if train_step == 0:
            theta = 0*old_q_evals
        else:
            theta = torch.clamp(old_q_evals - old_v_evals,min=0)
        
        self.old_policy = copy.deepcopy(self.eval_rnn)
        if self.args.cuda:
            
            self.old_policy.cuda()
            td_lambda =td_lambda.cuda()

        #exit()
        mask = mask.squeeze(2)
        terminated = terminated.squeeze(2)
        
        #Critic_loss
        #r = r.repeat(1, 1, self.n_agents)

        #exit()
        #c_target = r + self.args.gamma* v_targets

        #q_evals = torch.clamp
        cfv = torch.clamp(q_evals- v_evals.detach(),min=0)
        q1 = (cfv).sum(dim=2)
        #q1 = (q_evals - v_evals.detach()).sum(dim=2)
        v_evals = v_evals.sum(dim=2)
        v_targets = v_targets.sum(dim=2)
        r = r.squeeze(2)

        #v_targets = r + self.args.gamma*v_targets
        #exit()
        #td_lambda = td_lambda.sum(dim=2)
        #exit()
        td_lambda = td_lambda.squeeze(2)
        v_targets = td_lambda
        #print(td_lambda.size())
        #print(v_evals.size())
        #exit()
        v_loss = v_targets - v_evals
        #v_loss = td_lambda - v_evals#c_target - v_evals
        masked_v_loss = mask*v_loss
        #exit()
        v_loss = 0.5*(masked_v_loss ** 2).sum() / mask.sum()
        #self.c_optimizer.zero_grad()
        #v_loss.backward
        #torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
        #self.c_optimizer.step()

        
        #exit()
        if self.args.cuda:
            theta.cuda()
            v_targets.cuda()
            r.cuda()        
        #targets = (old_q_evals + self.args.gamma*v_targets.detach() + r)* (1 - terminated)
        #exit()
        
       
        

        #q2 = theta.sum(dim=2)
        old_q_evals = torch.clamp(old_q_evals,min = 0)

        q2 = old_q_evals.sum(dim=2)
       #theta = torch.clamp(old_q_evals - old_v_evals,min=0)
       
        
        #exit()
        old_q_evals
        targets = (q2.detach() + self.args.gamma*v_targets.detach() +r)* (1 - terminated)
        #targets = (old_q_evals.sum(dim=2).detach() + self.args.gamma*v_targets +r)* (1 - terminated)
        td_error = q1 - targets
        
      
        #td_error = q_evals.sum(dim=2) - q2 - self.args.gamma*v_targets - r
        #td_error = q_evals.sum(dim=2) - old_q_evals.sum(dim=2) - self.args.gamma*v_targets - r
        #td_error = targets.detach() - q_evals
        #td_error = torch.clamp(q_evals,min=0) - targets
        #exit()
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # loss = masked_td_error.pow(2).mean()
        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = 0.5*(masked_td_error ** 2).sum() / mask.sum()
        loss += v_loss
        #exit()

        #loss += v_loss

        # print('Loss is ', loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        target_params = self.target_rnn.parameters()
        params = self.eval_rnn.parameters()

        for target, net in zip(target_params,params):
            target.data.add_(0.01*(net.data - target.data))

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            #self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
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
        v_evals, v_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden,v_eval = self.eval_rnn(inputs, self.eval_hidden)  # 得到的q_eval维度为(episode_num*n_agents, n_actions)
            q_target, self.target_hidden,v_target = self.target_rnn(inputs_next, self.target_hidden)

            #print(q_eval.size())
            #exit()

            # 把q_eval维度重新变回(episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)

            v_eval = v_eval.view(episode_num, self.n_agents, -1)
            v_target = v_target.view(episode_num, self.n_agents, -1)
            v_evals.append(v_eval)
            v_targets.append(v_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        v_evals = torch.stack(v_evals, dim=1)
        v_targets = torch.stack(v_targets, dim=1)


        q2 = torch.sum(v_targets,dim =2,keepdim=False )#q_targets.squeeze(3)
        q2 = q2.squeeze(0)
        td_lambda = td_lambda_target(batch, max_episode_len, q2.cpu(), self.args)

        return q_evals, q_targets,v_evals,v_targets,td_lambda


    def get_old_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        v_evals = []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.old_eval_hidden = self.eval_hidden.cuda()
                self.old_target_hidden = self.target_hidden.cuda()
            #print(inputs.size())
            #exit()
            q_eval, self.old_eval_hidden,v_eval = self.old_policy(inputs, self.old_eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            
            #q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # 把q_eval维度重新变回(8, 5,n_actions)
            #print(q_eval.size())
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            v_eval = v_eval.view(episode_num, self.n_agents, -1)
            
            q_evals.append(q_eval)
            v_evals.append(v_eval)
            #q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        v_evals = torch.stack(v_evals, dim=1)
        #print(q_eval.size())
        #exit()
        #q_targets = torch.stack(q_targets, dim=1)
        return q_evals,v_evals
        

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.old_eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.old_target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_vdn_net.state_dict(), self.model_dir + '/' + num + '_vdn_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')