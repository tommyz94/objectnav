from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Categorical

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.auxiliary_tasks.aux_utils import (
    ACTION_EMBEDDING_DIM, RolloutAuxTask
)

@baseline_registry.register_aux_task(name="ROOM_SIMPLE")
class Room_Simple(RolloutAuxTask):
    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_size, 32),  # Frame + belief
            nn.ReLU(),
            nn.Linear(32, 31)
        )  # query and perception
        self.loss = nn.CrossEntropyLoss(reduction='none')

    @torch.jit.export
    def get_loss(self,
                 observations: Dict[str, torch.Tensor],
                 actions,
                 sensor_embeddings: Dict[str, torch.Tensor],
                 final_belief_state,
                 belief_features,
                 metrics: Dict[str, torch.Tensor],
                 n: int,
                 t: int,
                 env_zeros: List[List[int]]
                 ):
        belief_features = belief_features.view(t * n, -1).unsqueeze(0)
        prob = F.softmax(self.classifier(belief_features), dim=1).squeeze()
        with torch.no_grad():
            one_gt = observations['roomcat'].to(torch.long).squeeze(1)
        loss = self.loss(prob, one_gt)
        return self.masked_sample_and_scale(loss)

@baseline_registry.register_aux_task(name="ROOM")
class Room(RolloutAuxTask):

    def __init__(self, cfg, aux_cfg, task_cfg, device, observation_space=None, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.classifier = nn.Linear(aux_cfg.hidden_size, 31)
        self.action_embedder = nn.Embedding(num_actions + 1, ACTION_EMBEDDING_DIM)
        self.state_projector = nn.Linear(self.aux_hidden_size, aux_cfg.hidden_size)
        self.query_gru = nn.GRU(ACTION_EMBEDDING_DIM, aux_cfg.hidden_size)
        self.k = self.aux_cfg.num_steps
    @torch.jit.export
    def get_loss(self,
        observations: Dict[str, torch.Tensor],
        actions,
        sensor_embeddings: Dict[str, torch.Tensor],
        final_belief_state,
        belief_features,
        metrics: Dict[str, torch.Tensor],
        n: int,
        t: int,
        env_zeros: List[List[int]]
    ):
        belief_features = belief_features.view(t*n, -1)
        k = self.k
        query_in = self.state_projector(belief_features)

        action_embedding = self.action_embedder(actions) # t n -1
        action_padding = torch.zeros(self.k - 1, n, action_embedding.size(-1), device=self.device)
        action_padded = torch.cat((action_embedding, action_padding), dim=0) # (t+k-1) x n x -1
        # t x n x -1 x k
        action_seq = action_padded.unfold(dimension=0, size=self.k, step=1).permute(3, 0, 1, 2)\
            .view(self.k, t*n, action_embedding.size(-1))
        # for each timestep, predict up to k ahead -> (t,k) is query for t + k (+ 1) (since k is 0-indexed) i.e. (0, 0) -> query for 1
        out_all, _ = self.query_gru(action_seq, query_in.unsqueeze(0))
        query_all = out_all.view(self.k, t, n, -1).permute(1, 0, 2, 3) # t x k x n -1
        room_preds = self.classifier(query_all)
        # Cut off the last step, it's entirely padding (should ideally be cut earlier, but not messing with that right now)
        # Targets: predict k steps for each starting timestep
        room_true = observations['roomcat'].view(t, n)
        room_padded = torch.cat((room_true[1:], torch.zeros(self.k, n, dtype=torch.long, device=self.device)), dim=0)# (t+k) x n
        # Offset by 1 because our predictions take the belief at the end of timestep t
        room_true = room_padded.unfold(dimension=0, size=self.k, step=1).permute(0, 2, 1).unsqueeze(-1)# t x k x n
        # coverage_at_t_plus_1[t] describes a slice of k-steps starting at time t+1
        # to get deltas, subtract coverage[t+1]

        # Masking
        # Note which timesteps [1, t+k+1] could have valid queries, at distance (k) (note offset by 1)
        valid_modeling_queries = torch.ones(
            t + k, k, n, 1, device=self.device, dtype=torch.bool
        )  # (padded) timestep predicted x prediction distance x env
        valid_modeling_queries[
        t - 1:] = False  # >= t is past rollout, and t is index t - 1 here
        for j in range(1, k + 1):  # for j-step predictions
            valid_modeling_queries[:j - 1,
            j - 1] = False  # first j frames cannot be valid for all envs (rollout doesn't go that early)
            for env in range(n):
                has_zeros_batch = env_zeros[env]
                # in j-step prediction, timesteps z -> z + j are disallowed as those are the first j timesteps of a new episode
                # z-> z-1 because of modeling_queries being offset by 1
                for z in has_zeros_batch:
                    valid_modeling_queries[z - 1: z - 1 + j, j - 1,
                    env] = False

        # instead of the whole range, we actually are only comparing a window i:i+k for each query/target i - for each, select the appropriate k
        # we essentially gather diagonals from this full mask, t of them, k long
        valid_diagonals = [torch.diagonal(valid_modeling_queries, offset=-i)
                           for i in
                           range(t)]  # pull the appropriate k per timestep
        valid_mask = torch.stack(valid_diagonals, dim=0).permute(0, 3, 1,
                                                                 2)# t x n x 1 x k -> (t-1) x k x n x 1

        room_true = torch.masked_select(room_true, valid_mask)

        # [t, k, n, c] -> [t * k * n, c]
        room_preds = room_preds.flatten(end_dim=2)
        valid_mask_flat = valid_mask.flatten() # [t * k * n]
        room_preds = room_preds[valid_mask_flat]
        # TODO We'll add soft labels if this is hard?
        loss = F.cross_entropy(
            room_preds,
            room_true.long().detach(),
            reduction='none'
        )

        return self.masked_sample_and_scale(loss)

@baseline_registry.register_aux_task(name="ObjDist")
class ObjDist(RolloutAuxTask):
    """ GID, predicting distribution. Easier than counts
        - feed starting belief and ending frame into an initial hidden state
            - h_t, phi_t+k -> action probs
        - KL distribution loss over actions t-> t+k-1
    """

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        self.num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.PREDICTION = 1
        self.RECALL = 0
        self.mode = self.PREDICTION
        self.k = aux_cfg.num_steps # wow, it can be negative!
        if self.k < 0:
            self.mode = self.RECALL
            self.k = -self.k
        self.decoder = nn.Sequential(
            nn.Linear(cfg.hidden_size + self.aux_hidden_size, 32), # MLP probe for action distribution to be taken
            nn.ReLU(),
            nn.Linear(32, 21)
        )

    # @torch.jit.export
    # Categorical not supported with script
    # https://github.com/pytorch/pytorch/issues/18094
    def get_loss(self,
        observations: Dict[str, torch.Tensor],
        actions,
        sensor_embeddings: Dict[str, torch.Tensor],
        final_belief_state,
        belief_features,
        metrics: Dict[str, torch.Tensor],
        n: int,
        t: int,
        env_zeros: List[List[int]]
    ):
        vision = sensor_embeddings["all"]

        # Going to do t' = t-k of these
        if self.mode == self.PREDICTION: # forward
            belief_features = belief_features[:-self.k]
            end_frames = vision[self.k:] # t' x n x -1
        else: # backward
            belief_features = belief_features[self.k:]
            end_frames = vision[:-self.k]

        init_input = torch.cat([belief_features, end_frames], dim=-1).view((t - self.k) * n, -1)
        obj_pred = self.decoder(init_input)
        obj_pred = Categorical(F.softmax(obj_pred, dim=1))
        action_seq = actions[:-1].unfold(dimension=0, size=self.k, step=1) # t' x n x k (this is the target)
        action_seq = action_seq.view((t-self.k) * n, self.k) # (t' x n) over k # needs to be over 4
        # Count the numbers by scattering into one hot and summing
        action_gt = torch.zeros(*action_seq.size(), self.num_actions, device=self.device)
        action_gt.scatter_(dim=-1, index=action_seq.unsqueeze(-1), value=1) # A t indices specified by action seq (actions taken), scatter 1s
        action_gt = action_gt.sum(-2) # (t'*n) x num_actions now, turn to a distribution
        action_gt = Categorical(action_gt.float() / action_gt.sum(-1).unsqueeze(-1))
        pred_loss = kl_divergence(action_gt, obj_pred).view(t-self.k, n) # t' x n
        # Masking - reject up to k-1 behind a border cross (z-1 is last actual obs)
        valid_modeling_queries = torch.ones(
            t, n, device=self.device, dtype=torch.bool
        )
        for env in range(n):
            has_zeros_batch = env_zeros[env]
            for z in has_zeros_batch:
                if self.mode == self.PREDICTION:
                    valid_modeling_queries[z-self.k: z, env] = False
                else: # recall? Mask first k frames
                    valid_modeling_queries[z:z+self.k, env] = False
        if self.mode == self.PREDICTION:
            valid_modeling_queries = valid_modeling_queries[:-self.k]
        else:
            valid_modeling_queries = valid_modeling_queries[self.k:]
        return self.masked_sample_and_scale(pred_loss, mask=valid_modeling_queries)

@baseline_registry.register_aux_task(name="CPCS")
class CPCS(RolloutAuxTask):
    """ Action-conditional CPC - up to k timestep prediction
        From: https://arxiv.org/abs/1811.06407
    """

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        self.classifier = nn.Sequential(
            nn.Linear(41 + self.aux_hidden_size, 32),  # Frame + belief
            nn.ReLU(),
            nn.Linear(32, 1)
        )  # query and perception
        self.action_dim = ACTION_EMBEDDING_DIM
        self.action_embedder = nn.Embedding(num_actions + 1, self.action_dim)
        self.query_gru = nn.GRU(self.action_dim, self.aux_hidden_size)
        self.dropout = nn.Dropout(aux_cfg.dropout)
        self.k = self.aux_cfg.num_steps

    def get_positives(self, observations: Dict[str, torch.Tensor],
                      sensor_embeddings: Dict[str, torch.Tensor], n: int,
                      t: int):
        objs_list = []
        for idx, sem in enumerate(observations['semantic']):
            objs = torch.unique(sem).long()
            if objs[0] == -1:
                objs = objs[1:]
            tmp = torch.zeros((41))
            tmp[objs] = 1
            objs_list.append(tmp)
        positives=torch.stack(objs_list).to(self.device)
        return positives.view(t, n, -1)

    def get_negatives(self, positives, t: int, n: int):
        negative_inds = torch.randperm(t * n, device=self.device,
                                       dtype=torch.int64)
        return torch.gather(
            positives.view(t * n, -1),
            dim=0,
            index=negative_inds.view(t * n, 1).expand(t * n,
                                                      positives.size(-1)),
        ).view(t, n, -1)

    # def add_noise(self, tensor, prob=0.1):
    #     probs = torch.rand(tensor.shape, dtype=torch.float, device=self.device)
    #     tensor = torch.where(probs < torch.FloatTensor([prob/2.]), tensor, torch.zeros(tensor.shape,
    #                                                            dtype=torch.float,
    #                                                            device=self.device))
    #     probs = torch.rand(tensor.shape, dtype=torch.float, device=self.device)
    #     tensor = torch.where(probs < torch.FloatTensor([prob/2.]), tensor, torch.ones(tensor.shape,
    #                                                            dtype=torch.float,
    #                                                            device=self.device))
    #     return tensor

    @torch.jit.export
    def get_loss(self,
                 observations: Dict[str, torch.Tensor],
                 actions,
                 sensor_embeddings: Dict[str, torch.Tensor],
                 final_belief_state,
                 belief_features,
                 metrics: Dict[str, torch.Tensor],
                 n: int,
                 t: int,
                 env_zeros: List[List[int]]
                 ):
        k = self.k  # up to t

        belief_features = belief_features.view(t * n, -1).unsqueeze(0)
        positives = self.get_positives(observations, sensor_embeddings, n, t)
        negatives = self.get_negatives(positives, t, n)
        #positives = self.add_noise(positives)
        #negatives = self.add_noise(negatives)
        action_embedding = self.action_embedder(actions)  # t n -1
        action_padding = torch.zeros(k - 1, n, action_embedding
                                     .size(-1), device=self.device)
        action_padded = torch.cat((action_embedding, action_padding),
                                  dim=0)  # (t+k-1) x n x -1
        # t x n x -1 x k
        action_seq = action_padded.unfold(dimension=0, size=k, step=1).permute(
            3, 0, 1, 2) \
            .view(k, t * n, action_embedding.size(-1))

        # for each timestep, predict up to k ahead -> (t,k) is query for t + k (+ 1) (since k is 0-indexed) i.e. (0, 0) -> query for 1
        out_all, _ = self.query_gru(action_seq, belief_features)
        query_all = out_all.view(k, t, n, -1).permute(1, 0, 2, 3)

        # Targets: predict k steps for each starting timestep
        positives_padded = torch.cat((positives[1:],
                                      torch.zeros(k, n, positives.size(-1),
                                                  device=self.device)),
                                     dim=0)  # (t+k) x n
        positives_expanded = positives_padded.unfold(dimension=0, size=k,
                                                     step=1).permute(0, 3, 1,
                                                                     2)  # t x k x n x -1
        positives_logits = self.classifier(
            torch.cat([positives_expanded, query_all], -1))
        negatives_padded = torch.cat((negatives[1:],
                                      torch.zeros(k, n, negatives.size(-1),
                                                  device=self.device)),
                                     dim=0)  # (t+k) x n x -1
        negatives_expanded = negatives_padded.unfold(dimension=0, size=k,
                                                     step=1).permute(0, 3, 1,
                                                                     2)  # t x k x n x -1
        negatives_logits = self.classifier(
            torch.cat([negatives_expanded, query_all], -1))

        # Masking
        # Note which timesteps [1, t+k+1] could have valid queries, at distance (k) (note offset by 1)
        valid_modeling_queries = torch.ones(
            t + k, k, n, 1, device=self.device, dtype=torch.bool
        )  # (padded) timestep predicted x prediction distance x env
        valid_modeling_queries[
        t - 1:] = False  # >= t is past rollout, and t is index t - 1 here
        for j in range(1, k + 1):  # for j-step predictions
            valid_modeling_queries[:j - 1,
            j - 1] = False  # first j frames cannot be valid for all envs (rollout doesn't go that early)
            for env in range(n):
                has_zeros_batch = env_zeros[env]
                # in j-step prediction, timesteps z -> z + j are disallowed as those are the first j timesteps of a new episode
                # z-> z-1 because of modeling_queries being offset by 1
                for z in has_zeros_batch:
                    valid_modeling_queries[z - 1: z - 1 + j, j - 1,
                    env] = False

        # instead of the whole range, we actually are only comparing a window i:i+k for each query/target i - for each, select the appropriate k
        # we essentially gather diagonals from this full mask, t of them, k long
        valid_diagonals = [torch.diagonal(valid_modeling_queries, offset=-i)
                           for i in
                           range(t)]  # pull the appropriate k per timestep
        valid_mask = torch.stack(valid_diagonals, dim=0).permute(0, 3, 1,
                                                                 2)  # t x n x 1 x k -> t x k x n x 1

        positives_masked_logits = torch.masked_select(positives_logits,
                                                      valid_mask)
        negatives_masked_logits = torch.masked_select(negatives_logits,
                                                      valid_mask)
        positive_loss = F.binary_cross_entropy_with_logits(
            positives_masked_logits, torch.ones_like(positives_masked_logits),
            reduction='none'
        )

        subsampled_positive = self.masked_sample_and_scale(positive_loss)
        negative_loss = F.binary_cross_entropy_with_logits(
            negatives_masked_logits, torch.zeros_like(negatives_masked_logits),
            reduction='none'
        )
        subsampled_negative = self.masked_sample_and_scale(negative_loss)

        return subsampled_positive + subsampled_negative


@baseline_registry.register_aux_task(name="CPCS_A")
class CPCS_A(CPCS):
    pass

@baseline_registry.register_aux_task(name="CPCS_B")
class CPCS_B(CPCS):
    pass

@baseline_registry.register_aux_task(name="CPCS_C")
class CPCS_C(CPCS):
    pass

@baseline_registry.register_aux_task(name="CPCS_D")
class CPCS_D(CPCS):
    pass
