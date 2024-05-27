import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# llf for Aperiodic independent task on uniprocessor:
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
        self.predecessors = []
        Task.tasks[id] = self
        self.successors = []
        self.core = None

        # QoS metrics
        self.latency = 0
        self.jitter = 0
        self.execution_start_time = None
        self.execution_end_time = None

    @staticmethod
    def get_task(id):
        return Task.tasks[id]

    def add_successor(self, successor):
        self.successors.append(successor)


def uunifast(num_tasks, total_utilization):
    utilizations = []
    sumU = total_utilization
    for i in range(1, num_tasks):
        nextSumU = sumU * (1 - random.random() ** (1.0 / (num_tasks - i)))
        utilizations.append(sumU - nextSumU)
        sumU = nextSumU
    utilizations.append(sumU)
    return utilizations


def generate_non_periodic_tasks(num_tasks, total_utilization):
    utilizations = uunifast(num_tasks, total_utilization)
    tasks = []
    for i, util in enumerate(utilizations):
        execution_time = random.uniform(1, 10)
        deadline = execution_time + random.uniform(5, 15)

        # Ensure that release time is always less than deadline
        max_release_time = deadline - execution_time
        release_time = random.uniform(0, max_release_time) if max_release_time > 0 else 0

        tasks.append(Task(i + 1, int(release_time), int(execution_time), int(deadline), util, 0, []))

    # Sort tasks by deadlines for priority assignment (Deadline Monotonic)
    tasks.sort(key=lambda x: x.deadline)
    for i, task in enumerate(tasks):
        # Lower number means higher priority
        tasks[i].priority = i + 1
    return add_precedence(tasks)


def add_precedence(tasks):
    num_tasks = len(tasks)
    for i in range(1, num_tasks):
        num_predecessors = random.randint(0, i)
        if num_predecessors > 0:
            predecessors1 = random.sample(range(i), num_predecessors)
            for pred_index in predecessors1:
                predecessor_task = tasks[pred_index]
                tasks[i].predecessors.append(predecessor_task)
                predecessor_task.add_successor(tasks[i])
    return tasks


def calculate_mobility(tasks):
    for task in tasks:
        task.mobility = len(task.successors)
    tasks.sort(key=lambda t: t.mobility)


def assign_tasks_to_cores(tasks, num_cores=16):
    calculate_mobility(tasks)
    cores = [[] for _ in range(num_cores)]
    for i, task in enumerate(tasks):
        core_index = i % num_cores
        task.core = core_index
        cores[core_index].append(task)
    return cores


def least_laxity_first(cores):
    current_time = 0
    QoS_list = []
    jitter_list = []
    time_list = []

    while any(task.remaining_execution_time > 0 for core in cores for task in core):
        available_tasks = []
        for core in cores:
            for task in core:
                if task.release_time <= current_time and task.remaining_execution_time > 0:
                    if all(pred.remaining_execution_time == 0 for pred in task.predecessors):
                        available_tasks.append(task)

        if not available_tasks:
            current_time += 1
            continue  # No tasks available, move to next time step

        laxities = {task: task.deadline - current_time - task.remaining_execution_time for task in available_tasks}
        min_laxity_task = min(laxities, key=laxities.get)
        QoS = 1

        for slack in laxities.values():
            if slack < 0:
                QoS = QoS * 0.5

        if min_laxity_task.execution_start_time is None:
            min_laxity_task.execution_start_time = current_time

        print(f"Executing Task {min_laxity_task.id} on Core {min_laxity_task.core} at time {current_time}")
        min_laxity_task.remaining_execution_time -= 1

        QoS_list.append(QoS)
        time_list.append(current_time)

        current_time += 1

    return QoS_list, time_list


if __name__ == '__main__':
    num_tasks = 50
    total_utilization = 0.25
    tasks = generate_non_periodic_tasks(num_tasks, total_utilization)
    cores1 = assign_tasks_to_cores(tasks, 16)
    QoS_list_16_1, time_list_16_1 = least_laxity_first(cores1)

    total_utilization = 0.5
    tasks2 = generate_non_periodic_tasks(num_tasks, total_utilization)
    cores = assign_tasks_to_cores(tasks2, 16)
    QoS_list_16_2, time_list_16_2 = least_laxity_first(cores)

    total_utilization = 0.3
    tasks2 = generate_non_periodic_tasks(num_tasks, total_utilization)
    cores = assign_tasks_to_cores(tasks2, 32)
    QoS_list_32_1, time_list_32_1 = least_laxity_first(cores)

    total_utilization = 0.7
    tasks2 = generate_non_periodic_tasks(num_tasks, total_utilization)
    cores = assign_tasks_to_cores(tasks2, 32)
    QoS_list_32_2, time_list_32_2 = least_laxity_first(cores)

    plt.figure(figsize=(24, 18))
    plt.subplot(2, 2, 1)
    plt.bar(time_list_16_1, QoS_list_16_1, width=0.5, label='QoS')
    plt.xlabel('Time')
    plt.ylabel('time_list_16_1')
    plt.title(' QoS_list_16_1 over Time')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.bar(time_list_16_2, QoS_list_16_2, width=0.5, label='QoS')
    plt.xlabel('Time')
    plt.ylabel('QoS_list_16_2')
    plt.title('QoS_list_16_2 over Time')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.bar(time_list_32_1, QoS_list_32_1, width=0.5, label='QoS')
    plt.xlabel('Time')
    plt.ylabel('QoS_list_32_1')
    plt.title(' QoS_list_32_1 over Time')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.bar(time_list_32_2, QoS_list_32_2, width=0.5, label='QoS')
    plt.xlabel('Time')
    plt.ylabel('QoS_list_32_2')
    plt.title(' QoS_list_32_2 over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()
