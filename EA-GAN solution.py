import numpy as np
import pandas as pd
import json
import random
from operator import itemgetter
import copy
import math
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
from numpy.random import randn
from IPython import display
import torch

# import torch 
# print(torch.__version__)

def set_random_seed(seed, deterministic=False): # Adapted from open-mmlab (2019), https://www.programcreek.com/python/?code=open-mmlab%2Fmmdetection%2Fmmdetection-master%2Fmmdet%2Fapis%2Ftrain.py 
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 


seed = 13 # 13
set_random_seed(seed, deterministic=False)

# EA and SH model adapted from NDresevic (2019) https://github.com/NDresevic/timetable-generator
class Module:

    def __init__(self, groups, teacher, subject, type, duration, rooms):
        self.groups = groups
        self.teacher = teacher
        self.subject = subject
        self.type = type
        self.duration = duration
        self.rooms = rooms

    def __str__(self):
        return "Groups {} | Teacher '{}' | Subject '{}' | Type {} | {} Weeks | Rooms {} \n"\
            .format(self.groups, self.teacher, self.subject, self.type, self.duration, self.rooms)

    def __repr__(self):
        return str(self)


class Room:

    def __init__(self, name, type):
        self.name = name
        self.type = type

    def __str__(self):
        return "{} - {} \n".format(self.name, self.type)

    def __repr__(self):
        return str(self)


class Data:

    def __init__(self, groups, teachers, modules, rooms):
        self.groups = groups
        self.teachers = teachers
        self.modules = modules
        self.rooms = rooms
        
        
        
        
        
        



def subjects_order_cost(subjects_order):
    """
    Calculates percentage of soft constraints - order of modules (Lectures, Labs).
    :param subjects_order: dictionary where key = (name of the module, index of the group), value = [int, int, int]
    where ints represent start times (row in matrix) for types of modules (Lectures and Labs) respectively. If start time is -1
    it means that that subject does not have that type of module.
    :return: percentage of satisfied constraints
    """
    # number of subjects not in right order
    cost = 0
    # number of all orders of subjects
    total = 0

    for (subject, group_index), times in subjects_order.items():

        if times[0] != -1 and times[1] != -1:
            total += 1
            # Lectures after Labs
            if times[0] > times[1]:
                cost += 1


    # print(cost, total)
    return 100 * (total - cost) / total 


def empty_space_groups_cost(groups_empty_space):
    """
    Calculates total empty space of all groups for total weeks, maximum empty space in a week and average empty space for a all
    weeks per group.
    :param groups_empty_space: dictionary where key = group index, values = list of rows where it is in
    :return: total cost, maximum per week, average cost
    """
    # total empty space of all groups for the whole year (total weeks)
    cost = 0
    # max empty space in one week for some group
    max_empty = 0

    for group_index, times in groups_empty_space.items():
        times.sort()
        # empty space for each week for current group
        empty_per_week = {0: 0}

        for i in range(1, len(times) - 1):
            a = times[i-1]
            b = times[i]
            diff = b - a
            # modules are in the same week if their time div 30is the same (30 slots/week)
            if a // 31 == b // 31 and diff > 1:
                empty_per_week[a // 31] += diff - 1
                cost += diff - 1

        # compare current max with empty spaces per week for current group
        for key, value in empty_per_week.items():
            if max_empty < value:
                max_empty = value

    return cost, max_empty, cost / len(groups_empty_space)


def empty_space_teachers_cost(teachers_empty_space):
    """
    Calculates total empty space of all teachers for all weeks, maximum empty space in a week and average empty space for all
    week per teacher.
    :param teachers_empty_space: dictionary where key = name of the teacher, values = list of rows where it is in
    :return: total cost, maximum per day, average cost
    """
    # total empty space of all teachers for the whole 31 weeks
    cost = 0
    # max empty space in one week for some teacher
    max_empty = 0

    for teacher_name, times in teachers_empty_space.items():
        times.sort()
        # empty space for each week for current teacher
        empty_per_week = {0: 0}

        for i in range(1, len(times) - 1):
            a = times[i - 1]
            b = times[i]
            diff = b - a
            # modules are in the same week if their time div 31 is the same (31 weeks in total - treat weeks as hours)
            if a // 31 == b // 31 and diff > 1:
                empty_per_week[a // 31] += diff - 1
                cost += diff - 1

        # compare current max with empty spaces per week for current teacher
        for key, value in empty_per_week.items():
            if max_empty < value:
                max_empty = value

    return cost, max_empty, cost / len(teachers_empty_space)


def free_slot(matrix):
    """
    Checks if there is a slot without a module in a week. If so, returns it in format 'week: slot', otherwise -1.
    """
    weeks = ['Week#']
    slots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
             28, 29, 30, 31]

    for i in range(len(matrix)):
        exists = True
        for j in range(len(matrix[i])):
            field = matrix[i][j]
            if field is not None:
                exists = False

        if exists:
            return '{}: {}'.format(weeks[i // 31], slots[i % 31])

    return -1



def check_zero(x):
    if x == 0:
        return 1
    else:
        return 0



def hard_constraints_cost(matrix, data):
    """
    Calculates total cost of hard constraints: in every room is at most one module at a time, every module is in one
    of it's possible rooms, every teacher holds at most one room at a time and every group attends at most one
    module at a time.
    For everything that does not satisfy these constraints, one is added to the cost.
    :return: total cost, cost per module, cost of teachers, cost of rooms, cost of groups
    """
    # cost_class: dictionary where key = index of a module, value = total cost of that module
    cost_module = {}
    for c in data.modules:
        cost_module[c] = 0

    cost_rooms = 0
    cost_teacher = 0
    cost_group = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            field = matrix[i][j]                                        # for every field in matrix
            if field is not None:
                if field < len(data.modules):
                    c1 = data.modules[field]                                # take class from that field

                # calculate loss for classroom
                if j not in c1.rooms:
                    cost_rooms += 1
                    cost_module[field] += 1

                for k in range(j + 1, len(matrix[i])):                  # go through the end of row
                    next_field = matrix[i][k]
                    if next_field is not None:
                        c2 = data.modules[next_field]                   # take class of that field

                        # calculate loss for teachers
                        if c1.teacher == c2.teacher:
                            cost_teacher += 1
                            cost_module[field] += 1

                        # calculate loss for groups
                        g1 = c1.groups
                        g2 = c2.groups
                        for g in g1:
                            if g in g2:
                                cost_group += 1
                                cost_module[field] += 1

    total_cost = cost_teacher + cost_rooms + cost_group
    return total_cost, cost_module, cost_teacher, cost_rooms, cost_group


def check_hard_constraints(matrix, data):
    """
    Checks if all hard constraints are satisfied, returns number of overlaps with modules, lecture rooms, teachers and
    groups.
    """
    overlaps = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            field = matrix[i][j]                                    # for every field in matrix
            if field is not None:
                c1 = data.modules[field]                            # take class from that field

                # calculate loss for room
                if j not in c1.rooms:
                    overlaps += 1

                for k in range(len(matrix[i])):                     # go through the end of row
                    if k != j:
                        next_field = matrix[i][k]
                        if next_field is not None:
                            c2 = data.modules[next_field]           # take class of that field

                            # calculate loss for teachers
                            if c1.teacher == c2.teacher:
                                overlaps += 1

                            # calculate loss for groups
                            g1 = c1.groups
                            g2 = c2.groups
                            # print(g1, g2)
                            for g in g1:
                                if g in g2:
                                    overlaps += 1

    return overlaps












def load_data(file_path, teachers_empty_space, groups_empty_space, subjects_order):
    """
    Loads and processes input data, initialises helper structures.
    :param file_path: path to file with input data
    :param teachers_empty_space: dictionary where key = name of the teacher, values = list of rows where it is in
    :param groups_empty_space: dictionary where key = group index, values = list of rows where it is in
    :param subjects_order: dictionary where key = (name of the subject, index of the group), value = [int, int, int]
    where ints represent start times (row in matrix) for types of modules lectures and labs respectively. If start time is -1
    it means that that subject does not have that type of module.
    :return: Data(groups, teachers, modules, rooms)
    """
    with open(file_path) as file:
        data = json.load(file)

    # modules: dictionary where key = index of a module, value = module
    modules = {}
    # rooms: dictionary where key = index, value = room name
    rooms = {}
    # teachers: dictionary where key = teachers' name, value = index
    teachers = {}
    # groups: dictionary where key = name of the group, value = index
    groups = {}
    module_list = []

    for cl in data['Modules']:
        new_group = cl['Groups']
        new_teacher = cl['Teacher']

        # initialise for empty space of teachers
        if new_teacher not in teachers_empty_space:
            teachers_empty_space[new_teacher] = []

        new = Module(new_group, new_teacher, cl['Subject'], cl['Type'], cl['Duration'], cl['Room'])
        # add groups
        for group in new_group:
            if group not in groups:
                groups[group] = len(groups)
                # initialise for empty space of groups
                groups_empty_space[groups[group]] = []

        # add teacher
        if new_teacher not in teachers:
            teachers[new_teacher] = len(teachers)
        module_list.append(new)

    # shuffle mostly because of teachers
    random.shuffle(module_list)
    # add rooms
    for cl in module_list:
        modules[len(modules)] = cl

    # every module is assigned a list of rooms it can be in as indexes (later columns of matrix)
    for type in data['Rooms']:
        for name in data['Rooms'][type]:
            new = Room(name, type)
            rooms[len(rooms)] = new

    # every module has a list of groups marked by its index, same for rooms
    for i in modules:
        cl = modules[i]

        room = cl.rooms
        index_rooms = []
        # add rooms
        for index, c in rooms.items():
            if c.type == room:
                index_rooms.append(index)
        cl.rooms = index_rooms

        module_groups = cl.groups
        index_groups = []
        for name, index in groups.items():
            if name in module_groups:
                # initialise order of subjects
                if (cl.subject, index) not in subjects_order:
                    subjects_order[(cl.subject, index)] = [-1, -1, -1]
                index_groups.append(index)
        cl.groups = index_groups

    return Data(groups, teachers, modules, rooms)


def set_up(num_of_columns):
    """
    Sets up the timetable matrix and dictionary that stores free fields from matrix.
    :param num_of_columns: number of rooms
    :return: matrix, free
    """
    w, h = num_of_columns, 31   # 23 rooms * 31 (weeks) 
    matrix = [[None for x in range(w)] for y in range(h)]
    free = []

    # initialise free dict as all the fields from matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            free.append((i, j))
    return matrix, free


def show_timetable(matrix):
    """
    Prints timetable matrix.
    """
    weeks = ['Week#']
    slots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
             28, 29, 30, 31]
    
    # print heading for rooms
    for i in range(len(matrix[0])):
        if i == 0:
            print('{:17s} R{:6s}'.format('', '0'), end='')
        else:
            print('R{:6s}'.format(str(i)), end='')
    print()

    d_cnt = 0
    h_cnt = 0
    for i in range(len(matrix)):
        week = weeks[d_cnt]
        slot = slots[h_cnt]
        print('{:10s} {:2d} ->  '.format(week, slot), end='')
        for j in range(len(matrix[i])):
            print('{:6s} '.format(str(matrix[i][j])), end='')
        print()
        h_cnt += 1
        if h_cnt == 31:
            h_cnt = 0
            d_cnt += 1
            print()


def write_solution_to_file(matrix, data, filled, filepath, groups_empty_space, teachers_empty_space, subjects_order, i):
    """
    Writes statistics and schedule to file.
    """
    f = open('solution_files/valid_timetable_{}_'.format(i) + filepath, 'w')

    f.write('-------------------------- STATISTICS --------------------------\n')
    cost_hard = check_hard_constraints(matrix, data)
    if cost_hard == 0:  
        f.write('\nHard constraints satisfied: 100.00 %\n')
    else:
        f.write('Hard constraints NOT satisfied, cost: {}\n'.format(cost_hard))
    f.write('Soft constraints satisfied: {:.02f} %\n\n'.format(subjects_order_cost(subjects_order)))


    empty_groups, max_empty_group, average_empty_groups = empty_space_groups_cost(groups_empty_space)
    f.write('TOTAL empty space for all GROUPS and all weeks: {}\n'.format(empty_groups))
    f.write('MAX empty space for GROUP in a week: {}\n'.format(max_empty_group))
    f.write('AVERAGE empty space for GROUPS per week: {:.02f}\n\n'.format(average_empty_groups))

    empty_teachers, max_empty_teacher, average_empty_teachers = empty_space_teachers_cost(teachers_empty_space)
    f.write('TOTAL empty space for all TEACHERS and all weeks: {}\n'.format(empty_teachers))
    f.write('MAX empty space for TEACHER in day: {}\n'.format(max_empty_teacher))
    f.write('AVERAGE empty space for TEACHERS for all week: {:.02f}\n\n'.format(average_empty_teachers))

    f_slot = free_slot(matrix)
    if f_slot != -1:
        f.write('Free term -> {}\n'.format(f_slot))
    else:
        f.write('NO weekly slots without modules.\n')

    groups_dict = {}
    for group_name, group_index in data.groups.items():
        if group_index not in groups_dict:
            groups_dict[group_index] = group_name
    weeks = ['Week#']
    slots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
             28, 29, 30, 31]

    f.write('\n--------------------------- SCHEDULE ---------------------------')
    for module_index, times in filled.items():
        c = data.modules[module_index]
        groups = ' '
        for g in c.groups:
            groups += groups_dict[g] + ', '
        f.write('\n\nModule {}\n'.format(module_index))
        f.write('Teacher: {} \nSubject: {} \nGroups:{} \nType: {} \nDuration: {} week(s)'
                .format(c.teacher, c.subject, groups[:len(groups) - 2], c.type, c.duration))
        room = str(data.rooms[times[0][1]])
        f.write('\Room: {:2s}\nTime: {}'.format(room[:room.rfind('-')], weeks[times[0][0] // 31]))
        for time in times:
            f.write(' {}'.format(slots[time[0] % 31]))
    f.close()

is_valid = int()



def show_statistics(matrix, data, subjects_order, groups_empty_space, teachers_empty_space):
    """
    Prints statistics.
    """
    global is_valid
    cost_hard = check_hard_constraints(matrix, data)
    loss_before, cost_modules, cost_teachers, cost_rooms, cost_groups = hard_constraints_cost(matrix, data)
    m = check_zero(sum(cost_modules.values()))
    t = check_zero(cost_teachers)
    r = check_zero(cost_rooms)
    g = check_zero(cost_groups)
    
    percent_hard_constraints = (m+t+r+g) / 4
    
    if cost_hard == 0:
        is_valid = 1

        print('Hard constraints satisfied: 100.00 %')
    else:
        is_valid = 0
        print('Hard constraints NOT satisfied, cost: {}'.format(cost_hard))
    print('Soft constraints satisfied: {:.02f} %\n'.format(subjects_order_cost(subjects_order)))

    empty_groups, max_empty_group, average_empty_groups = empty_space_groups_cost(groups_empty_space)
    print("#Modules {}, #Teachers {}, #Groups {}, Rooms {}, #Weeks {}".format(len(data.modules), len(data.teachers), len(data.groups), 
                                                                                       len(data.rooms), 31))
    
    
    print('TOTAL empty space for all GROUPS and all weeks: ', empty_groups)
    print('MAX empty space for GROUP in a week: ', max_empty_group)
    print('AVERAGE empty space for GROUPS for all week: {:.02f}\n'.format(average_empty_groups))

    empty_teachers, max_empty_teacher, average_empty_teachers = empty_space_teachers_cost(teachers_empty_space)
    print('TOTAL empty space for all TEACHERS and all weeks: ', empty_teachers)
    print('MAX empty space for TEACHER in a week: ', max_empty_teacher)
    print('AVERAGE empty space for TEACHERS per all weeks: {:.02f}\n'.format(average_empty_teachers))

    f_slot = free_slot(matrix)
    if f_slot != -1:
        print('Free slot ->', f_slot)
    else:
        print('NO weekly slots without modules.')
        


    return is_valid,  percent_hard_constraints








def initial_population(data, matrix, free, filled, groups_empty_space, teachers_empty_space, subjects_order):
    """
    Sets up initial timetable for given modules by inserting in free fields such that every module is in its fitting
    room.
    """
    modules = data.modules

    for index, modules in modules.items():
        ind = 0

        # ind = random.randrange(len(free) - int(modules.duration))
        while True:
            if ind < len(free):
                start_field = free[ind]
            else:
                break
            # check if module won't start one week and end on after two weeks 
            start_time = start_field[0]
            end_time = start_time + int(modules.duration) - 1
            if start_time % 2 > end_time % 2:
                ind += 1
                continue

            found = True
            # check if whole block for the module is free
            for i in range(1, int(modules.duration)):
                field = (i + start_time, start_field[1])
                if field not in free:
                    found = False
                    ind += 1
                    break

            # secure that room fits
            if start_field[1] not in modules.rooms:
                ind += 1
                continue

            if found:
                for group_index in modules.groups:
                    # add order of the subjects for group
                    insert_order(subjects_order, modules.subject, group_index, modules.type, start_time)
                    # add times of the module for group
                    for i in range(int(modules.duration)):
                        groups_empty_space[group_index].append(i + start_time)

                for i in range(int(modules.duration)):
                    filled.setdefault(index, []).append((i + start_time, start_field[1]))        # add to filled
                    free.remove((i + start_time, start_field[1]))                                # remove from free
                    # add times of the class for teachers
                    teachers_empty_space[modules.teacher].append(i + start_time)
                break

    # fill the matrix
    for index, fields_list in filled.items():
        for field in fields_list:
            matrix[field[0]][field[1]] = index
            
    return matrix, free


def insert_order(subjects_order, subject, group, type, start_time):
    """
    Inserts start time of the module for given subject, group and type of module.
    """
    times = subjects_order[(subject, group)]
    if type == 'Lecture':
        times[0] = start_time
    else:
        times[1] = start_time
    subjects_order[(subject, group)] = times


def exchange_two(matrix, filled, ind1, ind2):
    """
    Changes places of two modules with the same duration in timetable matrix.
    """
    
    if ind1 in filled:
        if ind2 in filled:
            
            fields1 = filled[ind1]
            filled.pop(ind1, None)
            fields2 = filled[ind2]
            filled.pop(ind2, None)
            
            if len(matrix) < len(fields1):
                for i in range(len(fields1)):
                    t = matrix[fields1[i][0]][fields1[i][1]]
                    
                    if i < len(fields2):
                    
                        matrix[fields1[i][0]][fields1[i][1]] = matrix[fields2[i][0]][fields2[i][1]]
                        matrix[fields2[i][0]][fields2[i][1]] = t

                filled[ind1] = fields2
                filled[ind2] = fields1

    return matrix


def valid_teacher_group_row(matrix, data, index_module, row):
    """
    Returns if the module can be in that row because of possible teacher or groups overlaps.
    """
    c1 = data.modules[index_module]
    for j in range(len(matrix[row])):
        if matrix[row][j] is not None:
            c2 = data.modules[matrix[row][j]]
            # check teacher
            if c1.teacher == c2.teacher:
                return False
            # check groups
            for g in c2.groups:
                if g in c1.groups:
                    return False
    return True


def mutate_ideal_spot(matrix, data, ind_module, free, filled, groups_empty_space, teachers_empty_space, subjects_order):
    """
    Function that tries to find new fields in matrix for module index where the cost of the module is 0 (taken into
    account only hard constraints). If optimal spot is found, the fields in matrix are replaced.
    """

    # find rows and fields in which the module is currently in
    rows = []
    if ind_module in filled:
        fields = filled[ind_module]
        for f in fields:
            rows.append(f[0])
    
        modules = data.modules[ind_module]
        ind = 0
        while True:
            # ideal spot is not found, return from function
            if ind >= len(free):
                return
            start_field = free[ind]
    
            # check if module won't start one week and end 3 weeks after
            start_time = start_field[0]
            end_time = start_time + int(modules.duration) - 1
            if start_time % 2 > end_time % 2:
                ind += 1
                continue
    
            # check if new room is suitable
            if start_field[1] not in modules.rooms:
                ind += 1
                continue
    
            # check if whole block can be taken for new module and possible overlaps with teachers and groups
            found = True
            for i in range(int(modules.duration)):
                field = (i + start_time, start_field[1])
                if field not in free or not valid_teacher_group_row(matrix, data, ind_module, field[0]):
                    found = False
                    ind += 1
                    break
    
            if found:
                # remove current module from filled dict and add it to free dict
                filled.pop(ind_module, None)
                for f in fields:
                    free.append((f[0], f[1]))
                    matrix[f[0]][f[1]] = None
                    # remove empty space of the group from old place of the module
                    for group_index in modules.groups:
                        if f[0] in groups_empty_space[group_index]:
                            groups_empty_space[group_index].remove(f[0])
                        else:
                            continue
                    # remove teacher's empty space from old place of the module
                    if f[0] in teachers_empty_space[modules.teacher]:
                        teachers_empty_space[modules.teacher].remove(f[0])
                    else:
                        continue
    
                # update order of the subjects and add empty space for each group
                for group_index in modules.groups:
                    insert_order(subjects_order, modules.subject, group_index, modules.type, start_time)
                    for i in range(int(modules.duration)):
                        groups_empty_space[group_index].append(i + start_time)
    
                # add new term of the module to filled, remove those fields from free dict and insert new block in matrix
                for i in range(int(modules.duration)):
                    filled.setdefault(ind_module, []).append((i + start_time, start_field[1]))
                    free.remove((i + start_time, start_field[1]))
                    matrix[i + start_time][start_field[1]] = ind_module
                    # add new empty space for teacher
                    teachers_empty_space[modules.teacher].append(i+start_time)
                break


def evolutionary_algorithm(matrix, data, free, filled, groups_empty_space, teachers_empty_space, subjects_order):
    """
    Evolutionary algorithm that tries to find schedule such that hard constraints are satisfied.
    It uses (1+1) evolutionary strategy with Stifel's notation.
    """
    n = 3
    sigma = 2
    run_times = 5
    max_stagnation = 200

    for run in range(run_times):
        print('Run {} | sigma = {}'.format(run + 1, sigma))

        t = 0
        stagnation = 0
        cost_stats = 0
        while stagnation < max_stagnation:

            # check if optimal solution is found
            loss_before, cost_modules, cost_teachers, cost_rooms, cost_groups = hard_constraints_cost(matrix, data)
            if loss_before == 0 and check_hard_constraints(matrix, data) == 0:
                print('Found optimal solution: \n')
                show_timetable(matrix)
                break

            # sort modules by their loss, [(loss, module index)]
            costs_list = sorted(cost_modules.items(), key=itemgetter(1), reverse=True)

            # 10*n
            for i in range(len(costs_list) // 4):
                # mutate one to its ideal spot
                if random.uniform(0, 1) < sigma and costs_list[i][1] != 0:
                    mutate_ideal_spot(matrix, data, costs_list[i][0], free, filled, groups_empty_space,
                                      teachers_empty_space, subjects_order)
                else:
                    # exchange two who have the same duration
                    r = random.randrange(len(costs_list))
                    c1 = data.modules[costs_list[i][0]]
                    c2 = data.modules[costs_list[r][0]]
                    if r != i and costs_list[r][1] != 0 and costs_list[i][1] != 0 and c1.duration == c2.duration:
                        exchange_two(matrix, filled, costs_list[i][0], costs_list[r][0])

            loss_after, _, _, _, _ = hard_constraints_cost(matrix, data)
            if loss_after < loss_before:
                stagnation = 0
                cost_stats += 1
            else:
                stagnation += 1

            t += 1
            # Stifel for (1+1)-ES
            if t >= 10*n and t % n == 0:
                s = cost_stats
                if s < 2*n:
                    sigma *= 0.85
                else:
                    sigma /= 0.85
                cost_stats = 0

        print('Number of iterations: {} \nCost: {} \nModule cost: {} | Teachers cost: {} | Groups cost: {} | Rooms cost:'
              ' {}'.format(t, loss_after,  sum(cost_modules.values()), cost_teachers, cost_groups, cost_rooms))

    
    return matrix, data, free, filled, groups_empty_space, teachers_empty_space, subjects_order

def simulated_hardening(matrix, data, free, filled, groups_empty_space, teachers_empty_space, subjects_order, file):
    """
    Algorithm that uses simulated hardening with geometric decrease of temperature to optimize timetable by satisfying
    soft constraints as much as possible (empty space for groups and existence of a weekly slot in which there is no module).
    """
    # number of iterations
    iter_count = 259
    # temperature
    t = 0.5
    _, _, curr_cost_group = empty_space_groups_cost(groups_empty_space)
    _, _, curr_cost_teachers = empty_space_teachers_cost(teachers_empty_space)
    curr_cost = curr_cost_group  # + curr_cost_teachers
    if free_slot(matrix) == -1:
        curr_cost += 1

    for i in range(iter_count):
        rt = random.uniform(0, 1)
        t *= 0.99                # geometric decrease of temperature

        # save current results
        old_matrix = copy.deepcopy(matrix)
        old_free = copy.deepcopy(free)
        old_filled = copy.deepcopy(filled)
        old_groups_empty_space = copy.deepcopy(groups_empty_space)
        old_teachers_empty_space = copy.deepcopy(teachers_empty_space)
        old_subjects_order = copy.deepcopy(subjects_order)

        # try to mutate 1/4 of all modules
        for j in range(len(data.modules) // 4):
            index_class = random.randrange(len(data.modules))
            mutate_ideal_spot(matrix, data, index_class, free, filled, groups_empty_space, teachers_empty_space,
                              subjects_order)
        _, _, new_cost_groups = empty_space_groups_cost(groups_empty_space)
        _, _, new_cost_teachers = empty_space_teachers_cost(teachers_empty_space)
        new_cost = new_cost_groups  # + new_cost_teachers
        if free_slot(matrix) == -1:
            new_cost += 1

        if new_cost < curr_cost or rt <= math.exp((curr_cost - new_cost) / t):
            # take new cost and continue with new data
            curr_cost = new_cost
        else:
            # return to previously saved data
            matrix = copy.deepcopy(old_matrix)
            free = copy.deepcopy(old_free)
            filled = copy.deepcopy(old_filled)
            groups_empty_space = copy.deepcopy(old_groups_empty_space)
            teachers_empty_space = copy.deepcopy(old_teachers_empty_space)
            subjects_order = copy.deepcopy(old_subjects_order)
        if i % 50== 0:
            print('Iteration: {:4d} | Average cost: {:0.8f}'.format(i, curr_cost))
            


    
    # print('TIMETABLE AFTER HARDENING')
    # show_timetable(matrix)
#     show_statistics(matrix, data, subjects_order, groups_empty_space, teachers_empty_space)
    loss_before, cost_modules, cost_teachers, cost_rooms, cost_groups = hard_constraints_cost(matrix, data)
    print(cost_rooms, cost_groups, sum(cost_modules.values()), cost_teachers)
    
    return matrix, data, free, filled, groups_empty_space, teachers_empty_space, subjects_order

    
def append_missing(miss_list, val_list):
    for i in range(len(miss_list)):
        val_list = np.append(val_list, val_list[i])
    return val_list
    
def delete_duplicated(val_list, miss_list, dup_list):
    c = 0
    for i in val_list:
        if c < len(miss_list):
            if i in dup_list:
                x = np.where(val_list == i)[0] 
                val_list = np.delete(val_list, x[0])
                c += 1
            
    return val_list


def fix_unique_range(val_list):

    a = np.unique(val_list).tolist()
    b = list(range(385))

    set1 = set(b)
    set2 = set(a)

    missing = list(sorted(set1 - set2))

    val_list= np.append(val_list, missing)
    return val_list, missing


def form_duplicated_list(val_list):
    duplicated_list = []
    seen = []
    for i in val_list:
        if i in seen:
            duplicated_list.append(i)
        else:
            seen.append(i)
    return duplicated_list
        

        
def check_all_modules(matrix):    
    c = 0
    s = []
    for i in range(385):
        d = np.sum(any(i in sl for sl in matrix))
        if d == 0:
            s.append(i) 
        else:
            c += d
            
    if c == 385:
        return True
    else:
        return False, s
    
def assign_free_slots(val_list, dup_list, num_of_slots):
    r = 0
    seen = []
    for i in val_list:
        if r < num_of_slots:
            if i in dup_list:
                if i not in np.unique(seen):
                    x = np.where(val_list == i)[0] 
                    val_list = val_list.tolist()
                    val_list[x[0]] = None
                    val_list = np.array(val_list)
                    r += 1
        seen.append(i)  
    return val_list

def process_list(values):
    cache = {}
    for i in range(len(values)):
        for j in range(len(values[i])):
            if values[i][j] == None:
                continue
            if values[i][j] not in cache:
                cache[values[i][j]] = [(i,j)]
            else:
                arr = cache[values[i][j]]
                arr.append((i,j))
                cache[values[i][j]] = arr
    return cache


# def main():
#     """
#     free = [(row, column)...] - list of free fields (row, column) in matrix
#     filled: dictionary where key = index of the module, value = list of fields in matrix

#     subjects_order: dictionary where key = (name of the subject, index of the group), value = [int, int, int]
#     where ints represent start times (row in matrix) for types of modules lectures and labs respectively
#     groups_empty_space: dictionary where key = group index, values = list of rows where it is in
#     teachers_empty_space: dictionary where key = name of the teacher, values = list of rows where it is in

#     matrix = columns are rooms, rows are times, each field has index of the module or it is empty
#     data = input data, contains modules, rooms, teachers and groups
#     """


#     seed = 109

#     for i in range(1):
        
#         seed += 1
#         random.seed(seed)

    
#         filled = {}
#         subjects_order = {}
#         groups_empty_space = {} 
#         teachers_empty_space = {}
#         file = 'SampleWMGschedule2.txt'
    
    


#         data = load_data('test_files/' + file, teachers_empty_space, groups_empty_space, subjects_order)
#         matrix, free = set_up(len(data.rooms))
#         initial_population(data, matrix, free, filled, groups_empty_space, teachers_empty_space, subjects_order)
#         print("Is solution valid: {}".format(is_valid))

#         total, _, _, _, _ = hard_constraints_cost(matrix, data)
#         print('Initial cost of hard constraints: {}'.format(total))

#         #  Generator - a solution is generated (matrix, data etc.)
#         #  generate -> evolution -> harden -> check the solution meets hard constraints 
#         # -> return 0 or 1 to the generator accordingly
        
    
        
        
#         evolutionary_algorithm(matrix, data, free, filled, groups_empty_space, teachers_empty_space, subjects_order)
#         print('STATISTICS')
#         show_statistics(matrix, data, subjects_order, groups_empty_space, teachers_empty_space)
#         matrix, data, filled, file, groups_empty_space, teachers_empty_space, subjects_order \
#         = simulated_hardening(matrix, data, free, filled, groups_empty_space, teachers_empty_space, subjects_order, file, i)
        
#         write_solution_to_file(matrix, data, filled, file, groups_empty_space, teachers_empty_space, subjects_order, i)


#         z = pd.DataFrame(matrix)
#         z.to_excel("output_solution_{}.xlsx".format(i))
        
        
import torch
from torch import nn
import torch.optim as optim


filled = {}
subjects_order = {}
groups_empty_space = {} 
teachers_empty_space = {}
file = 'Week-Room Sample Module Data.txt'

    


data = load_data('test_files/' + file, teachers_empty_space, groups_empty_space, subjects_order)
matrix, free = set_up(len(data.rooms))



# device = "cuda" if torch.cuda.is_available() else "cpu"
device =  "cpu"

lr = 3e-5
batch_size = 15
num_epochs = 10


# General Adversarial Network Code adapted from Conor Lazarou (2020) https://towardsdatascience.com/pytorch-and-gans-a-micro-tutorial-804855817a6b

class Generator(nn.Module):
    def __init__(self, output_activation=nn.ReLU()):
        """A generator for mapping a latent space to a sample space.
        Args:
            latent_dim (int): latent dimension ("noise vector")
            layers (List[int]): A list of layer widths including output width
            output_activation: torch activation function or None
        """
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(batch_size*31*21, 25)
        self.linear2 = nn.Linear(25, 59)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.linear3 = nn.Linear(59, batch_size*31*21)
        self.output_activation = output_activation
       
    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        intermediate = self.linear1(input_tensor)
        intermediate = self.softmax(intermediate)
        intermediate = self.linear2(intermediate)
        intermediate = self.relu(intermediate)
        intermediate = self.dropout(intermediate)
        intermediate = self.sigmoid(intermediate)
        intermediate = self.dropout(intermediate)
        intermediate = self.linear3(intermediate)
        
        if self.output_activation is not None:
            intermediate = self.output_activation(intermediate)
            
        # intermediate = self.relu(intermediate)

    
        
        intermediate = intermediate.view((batch_size, 31, 21))
        return intermediate
    
    
class VanillaGAN():
    def __init__(self, generator, discriminator, noise_fn, batch_size=batch_size, device = device, lr_g=lr):
        """A GAN class for holding and training a generator and discriminator
        Args:
            generator: a Ganerator network
            discriminator: A Discriminator network
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)

            batch_size: training batch size
            device: cpu or CUDA

            lr_g: learning rate for the generator
        """
        self.generator = generator
        self.generator = self.generator.to(device)
        self.discriminator = discriminator
        self.noise_fn = noise_fn
        self.batch_size = batch_size
        self.device = device
        self.criterion = nn.BCELoss()

        self.optim_g = optim.Adam(generator.parameters(),
                                  lr=lr_g, betas=(0.5, 0.989))
        
        self.target_ones = torch.ones((batch_size, 1)).to(device)


    def train_step_generator(self):
        """Train the generator one step and return the loss."""
        self.generator.zero_grad()

        latent_vec = self.noise_fn(self.batch_size * 31 * 21)
        generated = self.generator(latent_vec)
        classifications, mat = self.discriminator(batch_size, matrix, generated, data, free, filled, groups_empty_space, \
                                                  teachers_empty_space, subjects_order, file)

        classifications = torch.tensor(classifications).float().to(device)
        
        loss = self.criterion(classifications.clone().detach().requires_grad_(True), self.target_ones)
        
        loss.backward()
        self.optim_g.step()
        return loss.item()


    def train_step(self):
        """Train both networks and return the losses."""

        loss_g = self.train_step_generator()
        return loss_g
    
    
c = 0
    
def make_discriminator_model(BATCH_SIZE, matrix, generated_timetables, data, free, filled, groups_empty_space, teachers_empty_space, subjects_order, file):
        

    #  Generator - a solution is generated (matrix, data etc.)
    #  generate -> evolution -> harden -> check the solution meets hard constraints 
    # -> return 0 or 1 to the generator accordingly
    
    global c

    _ = initial_population(data, matrix, free, filled, groups_empty_space, teachers_empty_space, subjects_order)
    first_sol = np.abs(generated_timetables.cpu().detach().numpy().tolist())
    first_sol = (first_sol/first_sol.max()) * 384

    first_sol = first_sol.astype('int')

    free = []
    for i in range(BATCH_SIZE):
        free.append([])

    filled = []
    for i in range(BATCH_SIZE):
        filled.append([])

    for i in range(BATCH_SIZE):

        first_sol = np.array(first_sol)

        all_values = first_sol[i].flatten()



        duplicated_list = form_duplicated_list(all_values)
        unique_list = np.unique(all_values).tolist()
        missing_list = check_all_modules(first_sol[i])[1]



        all_values = append_missing(missing_list, all_values)
        all_values = delete_duplicated(all_values, missing_list, duplicated_list)
        all_values, missed = fix_unique_range(all_values)

        duplicated_list2 = form_duplicated_list(all_values)

        all_values = delete_duplicated(all_values, missed, duplicated_list2)
        all_values, missed2 = fix_unique_range(all_values)
        duplicated_list3 = form_duplicated_list(all_values)
        all_values = delete_duplicated(all_values, missed2, duplicated_list3)



        all_values = assign_free_slots(all_values, duplicated_list3, 120)
        first_sol = first_sol.tolist()
        first_sol[i] = np.reshape(all_values.tolist(), (31, 21)).tolist()


        for index1, inner_l in enumerate(first_sol[i]):
            for index2, item in enumerate(inner_l):
                if item == None:
                    free[i].append((index1, index2))

        filled[i] = process_list(first_sol[i])
        
    matrix = first_sol
        
        
    output = []
    original_matrix = matrix
    seed = 13
        
    for i in range(BATCH_SIZE):
        
        
        random.seed(seed)   
        seed += 1
        validity = 0

        t = check_all_modules(original_matrix[i])

        if t == False:

            output.append([validity])

        else:
            
            total, _, _, _, _ = hard_constraints_cost(original_matrix[i], data)
            print('Initial cost of hard constraints: {}'.format(total))

            matrix2, data, free2, filled2, groups_empty_space2, teachers_empty_space2, subjects_order2 \
            = evolutionary_algorithm(original_matrix[i], data, free[i], filled[i], groups_empty_space, teachers_empty_space, subjects_order)

            matrix3, data, free3, filled3, groups_empty_space3, teachers_empty_space3, subjects_order3 \
            = simulated_hardening(matrix2, data, free2, filled2, groups_empty_space2, teachers_empty_space2, subjects_order2, file)

            validity, perc_hard = show_statistics(matrix3, data, subjects_order3, groups_empty_space3, teachers_empty_space3)
            
            print("The validity and % hard constraints satisfied are: {}, {}".format(validity, perc_hard * 100))

            output.append([perc_hard])
            
            if validity == 1:
            
                print('TIMETABLE AFTER HARDENING:')
                
                show_timetable(matrix3)
                write_solution_to_file(matrix3, data, filled3, file, groups_empty_space3, teachers_empty_space3, subjects_order3, c)
                z = pd.DataFrame(matrix3)
                z.to_excel("output_valid_solution_{}.xlsx".format(c))
                
                c += 1
                print(c)

    return output, matrix3

number_of_timetables = 60
 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   
def main():
    from time import time

    epochs = num_epochs
    batches = number_of_timetables // batch_size
    generator = Generator()
    discriminator = make_discriminator_model
#     discriminator = Discriminator(1, [64, 32, 1])
    noise_fn = lambda x: torch.randn(1, x, device = device)
#     data_fn = lambda x: torch.randn((x, 1), device)
    gan = VanillaGAN(generator, discriminator, noise_fn, batch_size=batch_size, device = device)
    loss_g = []
    start = time()
    for epoch in range(epochs):
        loss_g_running = 0
        for batch in range(batches):
            lg_ = gan.train_step()
            loss_g_running += lg_

        loss_g.append(loss_g_running / batches)

        print(f"Epoch {epoch+1}/{epochs} ({int(time() - start)}s):"
              f" G={loss_g[-1]:.3f}")
    print(loss_g)
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.plot(loss_g, '-o')

if __name__ == "__main__":
    main()
    

# Testing the generator code - run after training

# filled = {}
# subjects_order = {}
# groups_empty_space = {} 
# teachers_empty_space = {}
# file = 'Week-Room Sample Module Data.txt'

    


# data = load_data('test_files/' + file, teachers_empty_space, groups_empty_space, subjects_order)
# matrix, free = set_up(len(data.rooms))
# batch_size = 1
# class Generator(nn.Module):
#     def __init__(self, output_activation=nn.ReLU()):
#         """A generator for mapping a latent space to a sample space.
#         Args:
#             latent_dim (int): latent dimension ("noise vector")
#             layers (List[int]): A list of layer widths including output width
#             output_activation: torch activation function or None
#         """
#         super(Generator, self).__init__()
#         self.linear1 = nn.Linear(batch_size*31*21, 25)
#         self.linear2 = nn.Linear(25, 59)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.7)
#         self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()
#         self.linear3 = nn.Linear(59, batch_size*31*21)
#         self.output_activation = output_activation
       
#     def forward(self, input_tensor):
#         """Forward pass; map latent vectors to samples."""
#         intermediate = self.linear1(input_tensor)
#         intermediate = self.softmax(intermediate)
#         intermediate = self.linear2(intermediate)
#         intermediate = self.relu(intermediate)
#         intermediate = self.dropout(intermediate)
#         intermediate = self.sigmoid(intermediate)
#         intermediate = self.dropout(intermediate)
#         intermediate = self.linear3(intermediate)
        
#         if self.output_activation is not None:
#             intermediate = self.output_activation(intermediate)
            
#         # intermediate = self.relu(intermediate)

    
        
#         intermediate = intermediate.view((batch_size, 31, 21))
#         return intermediate
# noise = torch.rand(1, batch_size*31*21)
# generator = Generator()
# generated = generator(noise)
# classifications, mat = make_discriminator_model(batch_size, matrix, generated, data, free, filled, groups_empty_space, teachers_empty_space, subjects_order, file)
# classifications
    
