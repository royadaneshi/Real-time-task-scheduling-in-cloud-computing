import random
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from collections import defaultdict
import matplotlib.table as tbl

import pandas as pd


# Task class
class Task:
    tasks = {}

    def __init__(self, id, release_time, execution_time, deadline, utilization, priority, predecessors):
        self.id = id
        self.release_time = release_time
        self.execution_time = execution_time
        self.deadline = deadline
        self.remaining_execution_time = execution_time
        self.utilization = utilization
        self.priority = priority
        self.successors = []
        self.predecessors = predecessors
        Task.tasks[id] = self

    @staticmethod
    def get_task(id):
        return Task.tasks[id]

    def add_successor(self, successor):
        self.successors.append(successor)

    def __lt__(self, other):
        # Define the comparison based on laxity for PriorityQueue
        return (self.deadline - self.remaining_execution_time) < (other.deadline - other.remaining_execution_time)


# Functions to generate tasks
def uunifast(num_tasks, total_utilization):
    utilizations = []
    sumU = total_utilization
    for i in range(1, num_tasks):
        nextSumU = sumU * (1 - random.random() ** (1.0 / (num_tasks - i)))
        utilizations.append(sumU - nextSumU)
        sumU = nextSumU
    utilizations.append(sumU)
    return utilizations


def generate_non_periodic_tasks(num_tasks, total_utilization, phase):
    utilizations = uunifast(num_tasks, total_utilization)
    tasks = []
    for i, util in enumerate(utilizations):
        execution_time = random.uniform(1, 10)
        deadline = execution_time + random.uniform(5, 15)
        max_release_time = deadline - execution_time
        release_time = random.uniform(0, max_release_time) if max_release_time > 0 else 0
        tasks.append(Task(i + 1, int(release_time), int(execution_time), int(deadline), util, 0, []))
    tasks.sort(key=lambda x: x.deadline)
    for i, task in enumerate(tasks):
        tasks[i].priority = i + 1
    return add_precedence(tasks, phase)


def add_precedence(tasks, phase):
    num_tasks = len(tasks)
    for i in range(1, num_tasks):
        num_predecessors = random.randint(0, i)
        if num_predecessors > 0:
            predecessors1 = random.sample(range(i), num_predecessors)
            for pred_index in predecessors1:
                predecessor_task = tasks[pred_index]
                tasks[i].predecessors.append(predecessor_task)
                predecessor_task.add_successor(tasks[i])
    if phase != 'schedulability':
        generate_table(tasks)
    return tasks


def generate_table(tasks):
    tasks.sort(key=lambda x: x.id)
    task_details = []
    for task in tasks:
        predecessor = []
        for pred in task.predecessors:
            predecessor.append(pred.id)
        task_details.append({
            "id": task.id,
            "release_time": task.release_time,
            "execution_time": task.execution_time,
            "deadline": task.deadline,
            # "precedence": predecessor
        })

    df = pd.DataFrame(task_details)

    fig, ax = plt.subplots(figsize=(20, 1 + len(df) * 0.2))

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    table = tbl.table(ax, cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.show()


# Assign tasks to cores based on successors
def assign_cores(tasks, num_cores):
    sorted_tasks = sorted(tasks, key=lambda t: len(t.successors), reverse=True)
    core_tasks = [[] for _ in range(num_cores)]
    for i, task in enumerate(sorted_tasks):
        core_tasks[i % num_cores].append(task)
    return core_tasks


# LLF scheduling function with QoS calculation
def llf_schedule(core_tasks):
    time = 0
    global_schedule = []
    core_queues = [PriorityQueue() for _ in core_tasks]
    QoS_list = []
    time_list = []
    slack_time = None
    # Initialize a list to keep track of tasks for each core
    core_queues = [PriorityQueue() for _ in core_tasks]

    # Fill the initial task queues based on release times
    released_tasks = [[] for _ in range(len(core_tasks))]
    for i, tasks in enumerate(core_tasks):
        for task in tasks:
            if task.release_time == 0 and task.remaining_execution_time > 0:
                laxity = task.deadline - task.remaining_execution_time
                core_queues[i].put((laxity, task))
            else:
                released_tasks[i].append(task)

    while any(not core_queues[i].empty() or released_tasks[i] for i in range(len(core_tasks))):
        current_schedule = []
        QoS = 1
        # Release tasks whose release time has come
        for i in range(len(core_tasks)):
            new_released_tasks = []
            for task in released_tasks[i]:
                if task.release_time <= time:
                    if task.remaining_execution_time > 0:
                        laxity = task.deadline - task.remaining_execution_time
                        core_queues[i].put((laxity, task))
                else:
                    new_released_tasks.append(task)
            released_tasks[i] = new_released_tasks

        for i in range(len(core_tasks)):
            if not core_queues[i].empty():
                laxity, current_task = core_queues[i].get()
                if current_task.remaining_execution_time > 0:
                    current_task.remaining_execution_time -= 1
                    current_schedule.append((time, current_task.id, i + 1))

                    # Recalculate laxity and re-add the task to the queue if it's not finished
                    if current_task.remaining_execution_time > 0:
                        laxity = current_task.deadline - time - current_task.remaining_execution_time
                        core_queues[i].put((laxity, current_task))

        for _, task_id, _ in current_schedule:
            task = Task.get_task(task_id)
            slack = task.deadline - time - task.remaining_execution_time
            if slack < 0:
                QoS += slack / time
                QoS = QoS % 1
            if slack <= -task.deadline:
                slack_time = slack

        global_schedule.extend(current_schedule)
        QoS_list.append(QoS)
        time_list.append(time)
        time += 1

    return QoS_list, time_list, global_schedule, slack_time


class Ant:
    def __init__(self, tasks, alpha, beta, pheromone, heuristic, num_cores):
        self.tasks = tasks
        self.alpha = alpha
        self.beta = beta
        self.pheromone = pheromone
        self.heuristic = heuristic
        self.visited = [False] * len(tasks)
        self.path = []
        self.schedule = [[] for _ in range(num_cores)]
        self.total_time = 0
        self.current_times = [0] * num_cores
        self.num_cores = num_cores

    def select_next_task(self):
        unvisited_tasks = [task for task in self.tasks if not self.visited[task.id - 1] and all(
            self.visited[pred.id - 1] for pred in task.predecessors)]
        if not unvisited_tasks:
            return None, None

        probabilities = np.zeros(len(unvisited_tasks))

        for index, task in enumerate(unvisited_tasks):
            pheromone_level = self.pheromone[self.path[-1].id - 1][task.id - 1] if self.path else self.pheromone[0][
                task.id - 1]
            heuristic_value = self.heuristic[self.path[-1].id - 1][task.id - 1] if self.path else self.heuristic[0][
                task.id - 1]
            probabilities[index] = (pheromone_level ** self.alpha) * (heuristic_value ** self.beta)

        probabilities /= probabilities.sum()
        next_task = np.random.choice(unvisited_tasks, p=probabilities)
        self.visited[next_task.id - 1] = True
        self.path.append(next_task)

        # assign task to the earliest available core
        core = np.argmin(self.current_times)
        start_time = max(self.current_times[core], next_task.release_time)
        finish_time = start_time + next_task.execution_time
        self.schedule[core].append((next_task.id, start_time, finish_time))
        self.current_times[core] = finish_time
        self.total_time = max(self.current_times)
        return next_task, finish_time

    def update_pheromone(self, decay):
        for i in range(len(self.tasks)):
            for j in range(len(self.tasks)):
                self.pheromone[i][j] *= decay
        for i in range(len(self.path) - 1):
            a, b = self.path[i].id - 1, self.path[i + 1].id - 1
            self.pheromone[a][b] += 1.0 / self.total_time


class ACO:
    def __init__(self, num_ants, num_iterations, alpha, beta, rho, tasks, heuristic, num_cores):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.tasks = tasks
        self.pheromone = np.ones((len(tasks), len(tasks))) / len(tasks)
        self.heuristic = heuristic
        self.best_path = None
        self.best_time = np.inf
        self.best_schedule = None
        self.num_cores = num_cores

    def run(self):
        for i in range(self.num_iterations):
            ants = [Ant(self.tasks, self.alpha, self.beta, self.pheromone, self.heuristic, self.num_cores) for _ in
                    range(self.num_ants)]
            QoS_list = {}
            for ant in ants:
                QoS = 1
                schedulable = True
                while len(ant.path) < len(self.tasks):
                    next_task, finish_time = ant.select_next_task()
                    if not next_task:
                        break
                    if finish_time > 2 * next_task.deadline:
                        laxity = finish_time - next_task.deadline
                        QoS -= laxity / (i + 1)
                        QoS = QoS % 1
                    QoS_list[i + 1] = QoS
                    if finish_time >= 2 * next_task.deadline:
                        schedulable = False

                ant.update_pheromone(self.rho)
                if ant.total_time < self.best_time:
                    self.best_time = ant.total_time
                    self.best_path = [task.id for task in ant.path]
                    self.best_schedule = ant.schedule
        return self.best_path, self.best_time, self.best_schedule, QoS_list, schedulable


def plot_gantt_chart(schedule, num_cores, title):
    fig, gnt = plt.subplots()
    fig.suptitle(title)
    gnt.set_xlabel('Time')
    gnt.set_ylabel('Cores')

    gnt.set_yticks([i + 1 for i in range(num_cores)])
    gnt.set_yticklabels([f'Core {i + 1}' for i in range(num_cores)])
    gnt.set_ylim(0.5, num_cores + 0.5)

    max_finish_time = 0

    for core in range(num_cores):
        for task_id, start_time, finish_time in schedule[core]:
            gnt.broken_barh([(start_time, finish_time - start_time)], (core + 0.5, 0.8),
                            facecolors=(f'C{task_id % 10}'))
            gnt.text(start_time + (finish_time - start_time) / 2, core + 1, f'Task {task_id}', va='center',
                     ha='center',
                     color='black', fontsize=8)
            if finish_time > max_finish_time:
                max_finish_time = finish_time

    ## add the makespan line
    gnt.axvline(x=max_finish_time, color='red', linestyle='--', label='Makespan')
    gnt.text(max_finish_time, num_cores + 0.4, f'Makespan: {max_finish_time}', color='red', ha='center')

    gnt.legend()

    plt.show()


def plot_gantt_chart_with_makespan(schedule, num_cores, title):
    fig, gnt = plt.subplots()
    fig.suptitle(title)
    gnt.set_xlabel('Time')
    gnt.set_ylabel('Cores')

    gnt.set_yticks([i + 1 for i in range(num_cores)])
    gnt.set_yticklabels([f'Core {i + 1}' for i in range(num_cores)])
    gnt.set_ylim(0.5, num_cores + 0.5)

    task_start_finish_times = {}
    for time, task_id, core in schedule:
        if task_id not in task_start_finish_times:
            task_start_finish_times[task_id] = [time, time]
        else:
            task_start_finish_times[task_id][1] = time

    max_finish_time = 0

    core_schedules = [[] for _ in range(num_cores)]
    for time, task_id, core in schedule:
        core_schedules[core - 1].append((task_id, time))

    for core in range(num_cores):
        for task_id, start_time in core_schedules[core]:
            finish_time = task_start_finish_times[task_id][1] + 1
            duration = finish_time - task_start_finish_times[task_id][0]
            gnt.broken_barh([(task_start_finish_times[task_id][0], duration)], (core + 0.5, 0.8),
                            facecolors=(f'C{task_id % 10}'))
            gnt.text(task_start_finish_times[task_id][0] + duration / 2, core + 1, f'Task {task_id}', va='center',
                     ha='center',
                     color='black', fontsize=8)
            if finish_time > max_finish_time:
                max_finish_time = finish_time

    # add the makespan line
    gnt.axvline(x=max_finish_time, color='red', linestyle='--', label='Makespan')
    gnt.text(max_finish_time, num_cores + 0.4, f'Makespan: {max_finish_time}', color='red', ha='center')

    gnt.legend()

    plt.show()


def schedulability():
    schedulable_16_25_LLF = 100
    schedulable_16_05_LLF = 100
    schedulable_32_3_LLF = 100
    schedulable_32_7_LLF = 100

    schedulable_16_25_ACO = 100
    schedulable_16_05_ACO = 100
    schedulable_32_3_ACO = 100
    schedulable_32_7_ACO = 100
    for i in range(0, 20):
        num_tasks = 50
        total_utilization = 0.25
        tasks = generate_non_periodic_tasks(num_tasks, total_utilization, 'schedulability')
        cores1 = assign_cores(tasks, 16)
        _, _, _, slack = llf_schedule(cores1)
        _, _, schedulability = plot_ACO(tasks, num_tasks, 16, "schedulability", 0.25)
        if not schedulability:
            schedulable_16_25_ACO -= 5
        if slack:
            schedulable_16_25_LLF -= 5
    for i in range(0, 20):
        num_tasks = 50
        total_utilization = 0.5
        tasks = generate_non_periodic_tasks(num_tasks, total_utilization, 'schedulability')
        cores1 = assign_cores(tasks, 16)
        _, _, _, slack = llf_schedule(cores1)
        _, _, schedulability = plot_ACO(tasks, num_tasks, 16, "schedulability", 0.5)
        if not schedulability:
            schedulable_16_05_ACO -= 5
        if slack:
            schedulable_16_05_LLF -= 5

    for i in range(0, 20):
        num_tasks = 50
        total_utilization = 0.3
        tasks = generate_non_periodic_tasks(num_tasks, total_utilization, 'schedulability')
        cores1 = assign_cores(tasks, 32)
        _, _, _, slack = llf_schedule(cores1)
        _, _, schedulability = plot_ACO(tasks, num_tasks, 32, "schedulability", 0.3)
        if not schedulability:
            schedulable_32_3_ACO -= 5
        if slack:
            schedulable_32_3_LLF -= 5

    for i in range(0, 20):
        num_tasks = 50
        total_utilization = 0.7
        tasks = generate_non_periodic_tasks(num_tasks, total_utilization, 'schedulability')
        cores1 = assign_cores(tasks, 32)
        _, _, _, slack = llf_schedule(cores1)
        _, _, schedulability = plot_ACO(tasks, num_tasks, 32, "schedulability", 0.7)
        if not schedulability:
            schedulable_32_7_ACO -= 5
        if slack:
            schedulable_32_7_LLF -= 5

    fig = plt.figure(figsize=(10, 5))
    type = ['16 core with 0.25 utilization', '16 core with 0.5 utilization', '32 core with 0.3 utilization',
            '32 core with 0.7 utilization']
    systems = [schedulable_16_25_LLF, schedulable_16_05_LLF, schedulable_32_3_LLF, schedulable_32_7_LLF]
    plt.bar(type, systems, color=['#FF796C', '#F97306', '#DBB40C', 'pink'], width=0.4)
    plt.grid()

    plt.xlabel("System type")
    plt.ylabel("Number of scheulable task sets")
    plt.title("Schedulable tasks in LLF")
    plt.show()

    fig = plt.figure(figsize=(10, 5))
    type = ['16 core with 0.25 utilization', '16 core with 0.5 utilization', '32 core with 0.3 utilization',
            '32 core with 0.7 utilization']
    systems = [schedulable_16_25_ACO, schedulable_16_05_ACO, schedulable_32_3_ACO, schedulable_32_7_ACO]
    plt.bar(type, systems, color=['#FF796C', '#F97306', '#DBB40C', 'pink'], width=0.4)
    plt.grid()

    plt.xlabel("System type")
    plt.ylabel("Number of scheulable task sets")
    plt.title("Schedulable tasks in ACO")
    plt.show()


def Qos():
    global tasks
    num_tasks = 50
    total_utilization = 0.25
    tasks = generate_non_periodic_tasks(num_tasks, total_utilization, "QoS")
    cores1 = assign_cores(tasks, 16)
    QoS_list_16_1_LLF, time_list_16_1_LLF, global_schedule_16_1, _ = llf_schedule(cores1)
    print("ACO: tasks on system with 16 cores and utilization of 0.25:")
    QoS_list_16_1_ACO, time_list_16_1_ACO, _ = plot_ACO(tasks, num_tasks, 16, "QoS", 0.25)
    print("LLF: tasks on system with 16 cores and utilization of 0.25:")
    for time, task_name, core_id in global_schedule_16_1:
        print(f"Time {time}: Task {task_name} on Core {core_id}")
    plot_gantt_chart_with_makespan(global_schedule_16_1, 16, "LLF: Core:16, Utilization0.25")

    total_utilization = 0.5
    tasks2 = generate_non_periodic_tasks(num_tasks, total_utilization, "QoS")
    cores2 = assign_cores(tasks2, 16)
    QoS_list_16_2_LLF, time_list_16_2_LLF, global_schedule_16_2, _ = llf_schedule(cores2)
    print("ACO: tasks on system with 16 cores and utilization of 0.5:")
    QoS_list_16_2_ACO, time_list_16_2_ACO, _ = plot_ACO(tasks, num_tasks, 16, "QoS", 0.5)
    print("LLF: tasks on system with 16 cores and utilization of 0.5:")
    for time, task_name, core_id in global_schedule_16_2:
        print(f"Time {time}: Task {task_name} on Core {core_id}")
    plot_gantt_chart_with_makespan(global_schedule_16_2, 16, "LLF: Core:16, Utilization0.5")

    total_utilization = 0.3
    tasks3 = generate_non_periodic_tasks(num_tasks, total_utilization, "QoS")
    cores3 = assign_cores(tasks3, 32)
    QoS_list_32_1_LLF, time_list_32_1_LLF, global_schedule_32_1, _ = llf_schedule(cores3)
    print("ACO: tasks on system with 32 cores and utilization of 0.3:")
    QoS_list_32_1_ACO, time_list_32_1_ACO, _ = plot_ACO(tasks, num_tasks, 32, "QoS", 0.3)
    print("LLF: tasks on system with 32 cores and utilization of 0.3:")
    for time, task_name, core_id in global_schedule_32_1:
        print(f"Time {time}: Task {task_name} on Core {core_id}")
    plot_gantt_chart_with_makespan(global_schedule_32_1, 32, "LLF: Core:32, Utilization0.3")

    total_utilization = 0.7
    tasks4 = generate_non_periodic_tasks(num_tasks, total_utilization, "QoS")
    cores4 = assign_cores(tasks4, 32)
    QoS_list_32_2_LLF, time_list_32_2_LLF, global_schedule_32_2, _ = llf_schedule(cores4)
    print("ACO: tasks on system with 32 cores and utilization of 0.7:")
    QoS_list_32_2_ACO, time_list_32_2_ACO, _ = plot_ACO(tasks, num_tasks, 32, "QoS", 0.7)
    print("LLF: tasks on system with 32 cores and utilization of 0.7:")
    for time, task_name, core_id in global_schedule_32_2:
        print(f"Time {time}: Task {task_name} on Core {core_id}")
    plot_gantt_chart_with_makespan(global_schedule_32_2, 32, "LLF: Core:32, Utilization0.7")

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(time_list_16_1_LLF, QoS_list_16_1_LLF, label='QoS', marker='o', color='c')
    plt.xlabel('Time')
    plt.ylabel('QoS')
    plt.title('QoS of 16 core system with utilization=0.25 in LLF')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(time_list_16_2_LLF, QoS_list_16_2_LLF, label='QoS', marker='o', color='m')
    plt.xlabel('Time')
    plt.ylabel('QoS')
    plt.title('QoS of 16 core system with utilization=0.5 in LLF')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(time_list_32_1_LLF, QoS_list_32_1_LLF, label='QoS', marker='o', color='g')
    plt.xlabel('Time')
    plt.ylabel('QoS')
    plt.title('QoS of 32 core system with utilization=0.3 in LLF')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(time_list_32_2_LLF, QoS_list_32_2_LLF, label='QoS', marker='o', color='b')
    plt.xlabel('Time')
    plt.ylabel('QoS')
    plt.title('QoS of 32 core system with utilization=0.7 in LLF')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(time_list_16_1_ACO, QoS_list_16_1_ACO, label='QoS', marker='o', color='c')
    plt.xlabel('Time')
    plt.ylabel('QoS')
    plt.title('QoS of 16 core system with utilization=0.25 in ACO')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(time_list_16_2_ACO, QoS_list_16_2_ACO, label='QoS', marker='o', color='m')
    plt.xlabel('Time')
    plt.ylabel('QoS')
    plt.title('QoS of 16 core system with utilization=0.5 in ACO')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(time_list_32_1_ACO, QoS_list_32_1_ACO, label='QoS', marker='o', color='g')
    plt.xlabel('Time')
    plt.ylabel('QoS')
    plt.title('QoS of 32 core system with utilization=0.3 in ACO')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(time_list_32_2_ACO, QoS_list_32_2_ACO, label='QoS', marker='o', color='b')
    plt.xlabel('Time')
    plt.ylabel('QoS')
    plt.title('QoS of 32 core system with utilization=0.7 in ACO')
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_ACO(tasks, num_tasks, num_cores):
    heuristic = np.zeros((num_tasks, num_tasks))
    for i, task_i in enumerate(tasks):
        for j, task_j in enumerate(tasks):
            heuristic[i][j] = 1 / (task_j.execution_time + 1e-6)  # avoid division by zero

    aco = ACO(num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, rho=0.5, tasks=tasks, heuristic=heuristic,
              num_cores=num_cores)
    best_path, best_time, best_schedule, QoS, schedulablity = aco.run()
    return best_path, best_time, best_schedule, QoS, schedulablity


def plot_ACO(tasks, num_tasks, num_cores, method, utilization):
    best_path, best_time, best_schedule, QoS, schedulablity = run_ACO(tasks, num_tasks, num_cores)
    task_executions = []
    for core_schedule in best_schedule:
        for task_id, start_time, _ in core_schedule:
            task_executions.append((start_time, task_id))

    # sort by time
    task_executions.sort()

    task_finish = {}
    for time, task_id in task_executions:
        core = next(core for core, core_schedule in enumerate(best_schedule) if
                    any(task_id == t_id for t_id, _, _ in core_schedule))
        if method == "QoS":
            print(f"Time {time}: Task {task_id} on Core {core + 1} deadline {Task.get_task(task_id).deadline}")
        if task_id not in task_finish:
            task_finish[task_id] = Task.get_task(task_id).execution_time + time

    Qos = 1
    Qos_list = []
    schedulable = True
    for task in task_finish.keys():
        if task_finish[task] > 2 * Task.get_task(task).deadline:
            schedulable = False

        if task_finish[task] > Task.get_task(task).deadline:
            laxity = task_finish[task] - Task.get_task(task).deadline
            Qos -= laxity / task_finish[task]
            Qos = Qos % 1
        Qos_list.append((Qos, task_finish[task]))
    Qos_list.sort(key=lambda x: x[1])

    # Generate a new list of tuples with the unique second values and their corresponding average first values
    sums_and_counts = defaultdict(lambda: [0, 0])
    for value, key in Qos_list:
        sums_and_counts[key][0] += value
        sums_and_counts[key][1] += 1
    averages = {key: sums_and_counts[key][0] / sums_and_counts[key][1] for key in sums_and_counts}
    averaged_data = [(averages[key], key) for key in averages]

    if method == "QoS":
        plot_gantt_chart(best_schedule, num_cores, f"ACO, Cores:{num_cores}, Utilization:{utilization}")
    return [t[0] for t in averaged_data], [t[1] for t in averaged_data], schedulable


if __name__ == '__main__':
    print("LLF scheduling:")
    Qos()
    schedulability()
