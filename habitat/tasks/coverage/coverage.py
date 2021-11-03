#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pickle
from typing import Any, List, Optional, Type

import numpy as np

from habitat.config import Config
from habitat.core.dataset import Episode, Dataset
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import (
    Simulator,
    ShortestPathPoint,
)

from habitat.tasks.nav.nav import (
    merge_sim_episode_config,
    EpisodicGPSSensor
)


class CoverageEpisode(Episode):
    """Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    start_room: Optional[str]
    shortest_paths: Optional[List[ShortestPathPoint]]

    def __init__(
        self,
        start_room: Optional[str] = None,
        shortest_paths: Optional[List[ShortestPathPoint]] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # Bounce all info, we don't use it


@registry.register_measure
class Coverage(Measure):
    """Coverage
    Number of visited squares in a gridded environment
    - this is not exactly what we want to reward, but visual coverage is very slow.
    """
    cls_uuid: str = "coverage"

    def __init__(self, sim, config, **kwargs: Any):
        self._sim = sim
        self._config = config

        self._visited = None  # number of visits
        self._mini_visited = None
        self._step = None
        self._reached_count = None
        self._mini_reached = None
        self._mini_delta = 0.5
        self._grid_delta = config.GRID_DELTA
        self.priors = pickle.load(open('./pretrained_models/p_o1_o2.pkl', 'rb'))
        self.MP3D_cat = [3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 18, 19, 20, 22, 23,
                         25, 26, 27, 33, 34, 38]
        self.id_to_cat = {v: k for k, v in {"chair": 3, "table": 5, "picture": 6,
                                         "cabinet": 7, "cushion": 8,
                                         "sofa": 10, "bed": 11,
                                         "chest_of_drawers": 13, "plant": 14,
                                         "sink": 15, "toilet": 18, "stool": 19,
                                         "towel": 20, "tv_monitor": 22,
                                         "shower": 23, "bathtub": 25,
                                         "counter": 26, "fireplace": 27,
                                         "gym_equipment": 33, "seating": 34,
                                         "clothes": 38}.items()}
        self.objnav_to_id = {v: k for k, v in {"chair": 0, "table": 1,
                                              "picture": 2, "cabinet": 3,
                                              "cushion": 4, "sofa": 5,
                                              "bed": 6, "chest_of_drawers": 7,
                                              "plant": 8, "sink": 9,
                                              "toilet": 10,
                                              "stool": 11, "towel": 12,
                                              "tv_monitor": 13, "shower": 14,
                                              "bathtub": 15, "counter": 16,
                                              "fireplace": 17,
                                              "gym_equipment": 18,
                                              "seating": 19, "clothes": 20
                                              }.items()}
        self.id_to_objnav = {"chair": 0, "table": 1,
                                              "picture": 2, "cabinet": 3,
                                              "cushion": 4, "sofa": 5,
                                              "bed": 6, "chest_of_drawers": 7,
                                              "plant": 8, "sink": 9,
                                              "toilet": 10,
                                              "stool": 11, "towel": 12,
                                              "tv_monitor": 13, "shower": 14,
                                              "bathtub": 15, "counter": 16,
                                              "fireplace": 17,
                                              "gym_equipment": 18,
                                              "seating": 19, "clothes": 20
                                              }
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _to_grid(self, delta, sim_x, sim_y, sim_z=0):
        # * Note we actually get sim_x, sim_z in 2D case but that doesn't affect logic
        grid_x = int((sim_x) / delta)
        grid_y = int((sim_y) / delta)
        grid_z = int((sim_z) / delta)
        return grid_x, grid_y, grid_z

    def get_reward_for_current_obs(self, observations):
        sem_objs = np.unique(observations['semantic'])
        best_prob = 0.
        objgoal_cat = self.objnav_to_id[observations['objectgoal'][0]]
        for obj_id in sem_objs:
            if obj_id in self.MP3D_cat:
                obj_cat = self.id_to_cat[obj_id]
                if obj_cat == objgoal_cat:
                    return 0.1
                rew = self.priors[obj_cat][objgoal_cat]
                if rew > best_prob:
                    best_prob = rew
        return best_prob/10.






    def reset_metric(self, episode, task, observations, *args: Any, **kwargs: Any):
        # ! EGOCENTRIC will ASSUME sensor available
        self._visited = {}
        self._visited_reward_value = {}
        self._mini_visited = {}
        self._reached_count = 0
        # Used for coverage prediction (not elegant, I know)
        self._mini_reached = 0
        self._step = 0  # Tracking episode length
        current_visit, rew = self._visit(task, observations)
        self._metric = {
            "reached": self._reached_count,
            "mini_reached": self._mini_reached,
            "visit_count": current_visit,
            "step": self._step,
            "curr_rew": rew
        }

    def _visit(self, task, observations):
        """ Returns number of times visited current square """
        self._step += 1
        if self._config.EGOCENTRIC:
            global_loc = observations[EpisodicGPSSensor.cls_uuid]
        else:
            global_loc = self._sim.get_agent_state().position.tolist()

        mini_loc = self._to_grid(self._mini_delta, *global_loc)
        if mini_loc in self._mini_visited:
            self._mini_visited[mini_loc] += 1
        else:
            self._mini_visited[mini_loc] = 1
            self._mini_reached += 1

        grid_loc = self._to_grid(self._grid_delta, *global_loc)
        rew = self.get_reward_for_current_obs(observations)
        if grid_loc in self._visited:
            self._visited[grid_loc] += 1
            self._visited_reward_value[grid_loc] = 0.1 + max(rew, self._visited_reward_value[grid_loc]-0.1)
            return self._visited[grid_loc], self._visited_reward_value[grid_loc]
        self._visited[grid_loc] = 1
        self._visited_reward_value[grid_loc] = 0.1 + rew
        self._reached_count += 1
        return self._visited[grid_loc], self._visited_reward_value[grid_loc]

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, observations, **kwargs: Any
    ):
        current_visit, rew = self._visit(task, observations)
        #print(observations['semantic'])
        self._metric = {
            "reached": self._reached_count,
            "mini_reached": self._mini_reached,
            "visit_count": current_visit,
            "step": self._step,
            "curr_rew": rew
        }


@registry.register_measure
class CoverageExplorationReward(Measure):
    # Parallels ExploreRLEnv in `environments.py`

    cls_uuid: str = "coverage_explore_reward"

    def __init__(self, sim, config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._attenuation_penalty = 1.0
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.cls_uuid,
            [
                Coverage.cls_uuid,
            ],
        )
        self._attenuation_penalty = 1.0
        self._metric = 0
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        self._attenuation_penalty *= self._config.ATTENUATION
        visit = task.measurements.measures[
            Coverage.cls_uuid
        ].get_metric()["visit_count"]
        rew = task.measurements.measures[
            Coverage.cls_uuid
        ].get_metric()["curr_rew"]
        self._metric = self._attenuation_penalty * rew / (visit ** self._config.VISIT_EXP)

@registry.register_task(name="Coverage-v0")
class CoverageTask(EmbodiedTask):
    def __init__(
        self,
        config: Config,
        sim: Simulator,
        dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)
