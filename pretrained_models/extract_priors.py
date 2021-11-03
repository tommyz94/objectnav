import pickle
import numpy as np
from collections import defaultdict

MP3D_OBJS = ['chair', 'table', 'picture', 'cabinet', 'cushion', 'sofa', 'bed', 'chest_of_drawers', 'plant', 'sink',
                 'toilet', 'stool', 'towel', 'tv_monitor', 'shower', 'bathtub', 'counter', 'fireplace', 'gym_equipment',
                 'seating', 'clothes']
COCO_OBJS = ['chair', 'sofa', 'plant', 'bed', 'toilet', 'tv_monitor', 'table', 'sink']

# Probahility of being in a room given an object, P(r_i|o_j)
def p_o_r(info_dict, mode=0):
    if mode == 0:
        POSSIBLE_OBJS = MP3D_OBJS
    else:
        POSSIBLE_OBJS = COCO_OBJS

    por = {}
    for obj in POSSIBLE_OBJS:
        por[obj] = {}
        for apartment in info_dict:
            for level in info_dict[apartment]['levels'].values():
                for scene in level['regions'].values():
                    por[obj][scene['category']] = []



    for apartment in info_dict:
        for level in info_dict[apartment]['levels'].values():
            for scene_values in level['regions'].values():

                scene_name = scene_values['category']
                objs_in_room = set()

                for obj in scene_values['objs'].values():
                    if obj['category'] in POSSIBLE_OBJS:
                        objs_in_room.add(obj['category'])
                for obj in POSSIBLE_OBJS:
                    por[obj][scene_name].append(0)
                for obj in objs_in_room:
                    por[obj][scene_name][-1] = 1

    for obj in por.keys():
        for scene in por[obj]:
            if len(por[obj][scene]) == 0:
               res = 0
            else:
                res = sum(por[obj][scene]) / len(por[obj][scene])
            por[obj][scene] = res

    return por

# Probahility of being in a room given an object, P(r_i|o_j)
def p_r_o(info_dict, mode=0):
    if mode == 0:
        POSSIBLE_OBJS = MP3D_OBJS
    else:
        POSSIBLE_OBJS = COCO_OBJS

    pro = {}

    for apartment in info_dict:
        for level in info_dict[apartment]['levels'].values():
            for scene in level['regions'].values():
                pro[scene['category']] = {}

    for scene in pro.keys():
        for obj in POSSIBLE_OBJS:
            pro[scene][obj] = []

    for apartment in info_dict:
        for level in info_dict[apartment]['levels'].values():
            for scene_values in level['regions'].values():

                scene_name = scene_values['category']
                objs_to_add = set()

                for obj in scene_values['objs'].values():
                    if obj['category'] in POSSIBLE_OBJS:
                        objs_to_add.add(obj['category'])

                for obj in objs_to_add:
                    pro[scene_name][obj].append(1)

                for obj in POSSIBLE_OBJS:
                    if obj not in objs_to_add:
                        pro[scene_name][obj].append(0)

    for scene in pro:
        for obj in pro[scene]:
            res = sum(pro[scene][obj]) / len(pro[scene][obj])
            pro[scene][obj] = res

    return pro


# Proability of an object o1 given an object o2 P(o1|o2)
def p_o1_o2(info_dict, mode=0):
    if mode == 0:
        POSSIBLE_OBJS = MP3D_OBJS
    else:
        POSSIBLE_OBJS = COCO_OBJS
    p_o1_o2 = {obj1: {obj2: [] for obj2 in POSSIBLE_OBJS if obj2 != obj1} for obj1 in POSSIBLE_OBJS}

    for apartment in info_dict:
        for level in info_dict[apartment]['levels'].values():
            for scene_values in level['regions'].values():

                obj_in_scene = set([o['category'] for o in scene_values['objs'].values()
                                    if o['category'] in POSSIBLE_OBJS])

                for obj1 in obj_in_scene:
                    for obj2 in obj_in_scene:
                        if obj1 != obj2:
                            p_o1_o2[obj1][obj2].append(1)

                # for obj2 in [obj for obj in POSSIBLE_OBJS if obj not in obj_in_scene]:
                for obj1 in obj_in_scene:
                    for obj2 in [obj for obj in POSSIBLE_OBJS if obj not in obj_in_scene]:
                        if obj1 != obj2:
                            p_o1_o2[obj1][obj2].append(0)

    for obj1 in p_o1_o2:
        for obj2 in p_o1_o2[obj1]:
            p_o1_o2[obj1][obj2] = sum(p_o1_o2[obj1][obj2]) / len(p_o1_o2[obj1][obj2])

    return p_o1_o2


# Average room dimension
def room_avg(info_dict, mode=0):
    if mode == 0:
        POSSIBLE_OBJS = MP3D_OBJS
    else:
        POSSIBLE_OBJS = COCO_OBJS
    adr = defaultdict(list)

    for apartment in info_dict:
        for level in info_dict[apartment]['levels'].values():
            for scene_values in level['regions'].values():
                scene_name = scene_values['category']
                scene_dim = scene_values['dims'][0] * scene_values['dims'][2]
                adr[scene_name].append(scene_dim)

    for scene in adr:
        adr[scene] = sum(adr[scene]) / len(adr[scene])

    return adr

# Average room dimension
def p_r_in_building(info_dict, mode=0):
    if mode == 0:
        POSSIBLE_OBJS = MP3D_OBJS
    else:
        POSSIBLE_OBJS = COCO_OBJS
    pr = {}
    for apartment in info_dict:
        for level in info_dict[apartment]['levels'].values():
            for scene_values in level['regions'].values():
                scene_name = scene_values['category']
                pr[scene_name] = []
    tot_room = 0
    for apartment in info_dict:
        for level in info_dict[apartment]['levels'].values():
            for scene in pr:
                pr[scene].append(0)
            for scene_values in level['regions'].values():
                scene_name = scene_values['category']
                pr[scene_name][-1] = 1
    tot_prob = 0.0
    for scene in pr:
        pr[scene] = sum(pr[scene]) / len(pr[scene])
        tot_prob += pr[scene]
    print(tot_prob)

    return pr

# p(R)
def p_r(info_dict, mode=0):
    if mode == 0:
        POSSIBLE_OBJS = MP3D_OBJS
    else:
        POSSIBLE_OBJS = COCO_OBJS
    pr = {}
    for apartment in info_dict:
        for level in info_dict[apartment]['levels'].values():
            for scene_values in level['regions'].values():
                scene_name = scene_values['category']
                pr[scene_name] = 0
    tot_room = 0
    for apartment in info_dict:
        for level in info_dict[apartment]['levels'].values():
            for scene_values in level['regions'].values():
                scene_name = scene_values['category']
                pr[scene_name] += 1
                tot_room += 1
    tot_prob = 0.0
    for scene in pr:
        pr[scene] = pr[scene] / tot_room
        tot_prob += pr[scene]
    print(tot_prob)

    return pr

# Average distance of two objects in same room
def obj_avg(info_dict, mode=0):
    if mode == 0:
        POSSIBLE_OBJS = MP3D_OBJS
    else:
        POSSIBLE_OBJS = COCO_OBJS
    avgdist_o1_o2 = {obj1: {obj2: [] for obj2 in POSSIBLE_OBJS if obj2 != obj1} for obj1 in POSSIBLE_OBJS}

    for apartment in info_dict:
        for level in info_dict[apartment]['levels'].values():
            for scene_values in level['regions'].values():

                obj_in_scene = [(o['category'], o['center']) for o in scene_values['objs'].values()
                                if o['category'] in POSSIBLE_OBJS]

                for obj1_cat, obj1_cen in obj_in_scene:
                    for obj2_cat, obj2_cen in obj_in_scene:
                        if obj1_cat != obj2_cat:
                            o1_pos = np.array([obj1_cen[0], obj1_cen[2]])
                            o2_pos = np.array([obj2_cen[0], obj2_cen[2]])
                            avgdist_o1_o2[obj1_cat][obj2_cat].append(np.linalg.norm(o1_pos - o2_pos))

    for obj1 in avgdist_o1_o2:
        for obj2 in avgdist_o1_o2[obj1]:
            if len(avgdist_o1_o2[obj1][obj2]) > 0:
                avgdist_o1_o2[obj1][obj2] = sum(avgdist_o1_o2[obj1][obj2]) / len(avgdist_o1_o2[obj1][obj2])
            else:
                avgdist_o1_o2[obj1][obj2] = np.nan

    return avgdist_o1_o2


# def parse_pracml(info_dict):
#     obj_list = []
#     rooms_list = []
#     in_list = []
#     obj_set = set()
#     rooms_set = set()
#     in_set = ['In(obj, room)']
#     obj_set2 = set()
#     obj_set3 = set()
#     rooms_set2 = set()
#     in_set2 = ['In(x, y)', 'In(z, y)']
#     room_i = 0
#     obj_i = 0
#     for apartment in info_dict:
#         for level in info_dict[apartment]['levels'].values():
#             for scene_values in level['regions'].values():
#                 scene_name = scene_values['category']
#                 if scene_name == 'toilet':
#                     scene_name = 'toilet_room'
#                 rooms_list += [scene_name.capitalize().replace(' ', '_').replace('/', '-') + '(Room_{})'.format(room_i)]
#                 rooms_set.add(scene_name.capitalize().replace(' ', '_').replace('/', '-') + '(room)')
#                 rooms_set2.add(scene_name.capitalize().replace(' ', '_').replace('/', '-') + '(y)')
#
#                 for obj in scene_values['objs'].values():
#                     if obj['category'] in POSSIBLE_OBJS:
#                         obj_cat = obj['category']
#                         obj_list += [obj_cat.capitalize().replace(' ', '_').replace('/', '-') + '(Obj_{})'.format(obj_i)]
#                         obj_set.add(obj_cat.capitalize().replace(' ', '_').replace('/', '-') + '(obj)')
#                         obj_set2.add(obj_cat.capitalize().replace(' ', '_').replace('/', '-') + '(x)')
#                         obj_set3.add(obj_cat.capitalize().replace(' ', '_').replace('/', '-') + '(z)')
#                         in_list += ['In(Obj_{}, Room_{})'.format(obj_i, room_i)]
#                         obj_i += 1
#
#                 room_i += 1
#     outfile = open('train.db', 'w')
#     outfile.write('// Objs predicates\n')
#     for elem in obj_list:
#         outfile.write(elem + '\n')
#     outfile.write('\n')
#     outfile.write('// Rooms predicates\n')
#     for elem in rooms_list:
#         outfile.write(elem + '\n')
#     outfile.write('\n')
#     outfile.write('// In predicates\n')
#     for elem in in_list:
#         outfile.write(elem + '\n')
#     outfile.close()
#
#
#     outfile = open('priors.mln', 'w')
#     outfile.write('// Objs predicates\n')
#     for elem in obj_set:
#         outfile.write(elem + '\n')
#     outfile.write('\n')
#     outfile.write('// Rooms predicates\n')
#     for elem in rooms_set:
#         outfile.write(elem + '\n')
#     outfile.write('\n')
#     outfile.write('// In predicates\n')
#     for elem in in_set:
#         outfile.write(elem + '\n')
#     outfile.write('\n')
#     outfile.write('// Room inference rules\n')
#     for obj in obj_set2:
#         for room in rooms_set2:
#             outfile.write('0 ' + obj + ' ^ ' + in_set2[0] + ' => ' + room + '\n')
#     for room in rooms_set2:
#         for obj in obj_set2:
#             outfile.write('0 ' + room + ' ^ ' + in_set2[0] + ' => ' + obj + '\n')
#     for obj in obj_set2:
#         for obj2 in obj_set3:
#             outfile.write('0 ' + in_set2[0] + ' ^ ' + in_set2[1] + ' ^ ' + obj + ' => ' + obj2 + '\n')
#
#
#     outfile.close()

# def parse_pracml_small(info_dict):
#     objs_allow = ['toilet', 'sink']
#     room_allow = ['bathroom', 'kitchen']
#     obj_list = []
#     rooms_list = []
#     in_list = []
#     obj_set = set()
#     rooms_set = set()
#     in_set = ['In(obj, room)']
#     obj_set2 = set()
#     obj_set3 = set()
#     rooms_set2 = set()
#     in_set2 = ['In(x, y)', 'In(z, y)']
#     room_i = 0
#     obj_i = 0
#     for apartment in info_dict:
#         for level in info_dict[apartment]['levels'].values():
#             for scene_values in level['regions'].values():
#
#                 scene_name = scene_values['category']
#                 if scene_name in room_allow:
#                     if scene_name == 'toilet':
#                         scene_name = 'toilet_room'
#                     rooms_list += [scene_name.capitalize().replace(' ', '_').replace('/', '-') + '(Room_{})'.format(room_i)]
#                     rooms_set.add(scene_name.capitalize().replace(' ', '_').replace('/', '-') + '(room)')
#                     rooms_set2.add(scene_name.capitalize().replace(' ', '_').replace('/', '-') + '(y)')
#
#                     for obj in scene_values['objs'].values():
#                         if obj['category'] in POSSIBLE_OBJS and obj['category'] in objs_allow:
#                             obj_cat = obj['category']
#                             obj_list += [obj_cat.capitalize().replace(' ', '_').replace('/', '-') + '(Obj_{})'.format(obj_i)]
#                             obj_set.add(obj_cat.capitalize().replace(' ', '_').replace('/', '-') + '(obj)')
#                             obj_set2.add(obj_cat.capitalize().replace(' ', '_').replace('/', '-') + '(x)')
#                             obj_set3.add(obj_cat.capitalize().replace(' ', '_').replace('/', '-') + '(z)')
#                             in_list += ['In(Obj_{}, Room_{})'.format(obj_i, room_i)]
#                             obj_i += 1
#
#                     room_i += 1
#     outfile = open('train_small.db', 'w')
#     outfile.write('// Objs predicates\n')
#     for elem in obj_list:
#         outfile.write(elem + '\n')
#     outfile.write('\n')
#     outfile.write('// Rooms predicates\n')
#     for elem in rooms_list:
#         outfile.write(elem + '\n')
#     outfile.write('\n')
#     outfile.write('// In predicates\n')
#     for elem in in_list:
#         outfile.write(elem + '\n')
#     outfile.close()
#
#
#     outfile = open('priors_small.mln', 'w')
#     outfile.write('// Objs predicates\n')
#     for elem in obj_set:
#         outfile.write(elem + '\n')
#     outfile.write('\n')
#     outfile.write('// Rooms predicates\n')
#     for elem in rooms_set:
#         outfile.write(elem + '\n')
#     outfile.write('\n')
#     outfile.write('// In predicates\n')
#     for elem in in_set:
#         outfile.write(elem + '\n')
#     outfile.write('\n')
#     outfile.write('// Room inference rules\n')
#     for obj in obj_set2:
#         for room in rooms_set2:
#             outfile.write('0 ' + obj + ' ^ ' + in_set2[0] + ' => ' + room + '\n')
#     for room in rooms_set2:
#         for obj in obj_set2:
#             outfile.write('0 ' + room + ' ^ ' + in_set2[0] + ' => ' + obj + '\n')
#     for obj in obj_set2:
#         for obj2 in obj_set3:
#             outfile.write('0 ' + in_set2[0] + ' ^ ' + in_set2[1] + ' ^ ' + obj + ' => ' + obj2 + '\n')
#
#
#     outfile.close()


if __name__ == '__main__':
    info_dict = pickle.load(open('stats_dict.pkl', 'rb'))
    mode = 0
    if mode == 1:
        mode_str = '_coco'
    else:
        mode_str = ''
    # # parse_pracml_small(info_dict)
    # pro = p_r_o(info_dict, mode)
    # pickle.dump(pro, open('p_r_o{}.pkl'.format(mode_str), "wb"))
    #
    # adr = room_avg(info_dict, mode)
    # pickle.dump(adr, open('room_avg_dim{}.pkl'.format(mode_str), "wb"))

    po1o2 = p_o1_o2(info_dict, mode)
    pickle.dump(po1o2, open('p_o1_o2{}.pkl'.format(mode_str), "wb"))
    #
    # por = p_o_r(info_dict, mode)
    # pickle.dump(por, open('p_o_r{}.pkl'.format(mode_str), "wb"))
    #
    # pr = p_r_in_building(info_dict, mode)
    # pickle.dump(pr, open('p_r_per_building{}.pkl'.format(mode_str), "wb"))
    #
    # obj_avg = obj_avg(info_dict, mode)
    # pickle.dump(obj_avg, open('object_avg_dist{}.pkl'.format(mode_str), "wb"))
    # print('ciao')
