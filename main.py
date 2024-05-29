import random
import matplotlib.pyplot as plt
from queue import PriorityQueue


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


def generate_non_periodic_tasks(num_tasks, total_utilization):
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
    for i, tasks in enumerate(core_tasks):
        for task in tasks:
            if task.remaining_execution_time > 0:
                laxity = task.deadline - task.remaining_execution_time
                core_queues[i].put((laxity, task))

    while any(not core_queues[i].empty() for i in range(len(core_tasks))):
        current_schedule = []
        QoS = 1

        for i in range(len(core_tasks)):
            if not core_queues[i].empty():
                laxity, current_task = core_queues[i].get()
                if current_task.remaining_execution_time > 0:
                    current_task.remaining_execution_time -= 1
                    current_schedule.append((time, current_task.id, i + 1))
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


# Function to generate and schedule tasks
def schedulability():
    schedulable_16_25 = 100
    schedulable_16_05 = 100
    schedulable_32_3 = 100
    schedulable_32_7 = 100
    for i in range(0, 100):
        num_tasks = 50
        total_utilization = 0.25
        tasks = generate_non_periodic_tasks(num_tasks, total_utilization)
        cores1 = assign_cores(tasks, 16)
        _, _, _, slack = llf_schedule(cores1)
        if slack:
            schedulable_16_25 -= 1
    for i in range(0, 100):
        num_tasks = 50
        total_utilization = 0.5
        tasks = generate_non_periodic_tasks(num_tasks, total_utilization)
        cores1 = assign_cores(tasks, 16)
        _, _, _, slack = llf_schedule(cores1)
        if slack:
            schedulable_16_05 -= 1

    for i in range(0, 100):
        num_tasks = 50
        total_utilization = 0.3
        tasks = generate_non_periodic_tasks(num_tasks, total_utilization)
        cores1 = assign_cores(tasks, 32)
        _, _, _, slack = llf_schedule(cores1)
        if slack:
            schedulable_32_3 -= 1

    for i in range(0, 100):
        num_tasks = 50
        total_utilization = 0.7
        tasks = generate_non_periodic_tasks(num_tasks, total_utilization)
        cores1 = assign_cores(tasks, 32)
        _, _, _, slack = llf_schedule(cores1)
        if slack:
            schedulable_32_7 -= 1

    fig = plt.figure(figsize=(10, 5))
    type = ['16 core with 0.25 utilization', '16 core with 0.5 utilization', '32 core with 0.3 utilization',
            '32 core with 0.7 utilization']
    systems = [schedulable_16_25, schedulable_16_05, schedulable_32_3, schedulable_32_7]
    plt.bar(type, systems, color=['#FF796C', '#F97306', '#DBB40C', 'pink'], width=0.4)
    plt.grid()

    plt.xlabel("System type")
    plt.ylabel("Number of scheulable task sets")
    plt.title("Schedulable tasks")
    plt.show()


def Qos():
    global tasks
    num_tasks = 50
    total_utilization = 0.25
    tasks = generate_non_periodic_tasks(num_tasks, total_utilization)
    cores1 = assign_cores(tasks, 16)
    QoS_list_16_1, time_list_16_1, global_schedule_16_1, _ = llf_schedule(cores1)
    print("tasks on system with 16 cores and utilization of 0.25:")
    for time, task_name, core_id in global_schedule_16_1:
        print(f"Time {time}: Task {task_name} on Core {core_id}")
    total_utilization = 0.5
    tasks2 = generate_non_periodic_tasks(num_tasks, total_utilization)
    cores2 = assign_cores(tasks2, 16)
    QoS_list_16_2, time_list_16_2, global_schedule_16_2, _ = llf_schedule(cores2)
    print("tasks on system with 16 cores and utilization of 0.5:")
    for time, task_name, core_id in global_schedule_16_2:
        print(f"Time {time}: Task {task_name} on Core {core_id}")
    total_utilization = 0.3
    tasks3 = generate_non_periodic_tasks(num_tasks, total_utilization)
    cores3 = assign_cores(tasks3, 32)
    QoS_list_32_1, time_list_32_1, global_schedule_32_1, _ = llf_schedule(cores3)
    print("tasks on system with 32 cores and utilization of 0.3:")
    for time, task_name, core_id in global_schedule_32_1:
        print(f"Time {time}: Task {task_name} on Core {core_id}")
    total_utilization = 0.7
    tasks4 = generate_non_periodic_tasks(num_tasks, total_utilization)
    cores4 = assign_cores(tasks4, 32)
    QoS_list_32_2, time_list_32_2, global_schedule_32_2, _ = llf_schedule(cores4)
    print("tasks on system with 32 cores and utilization of 0.7:")
    for time, task_name, core_id in global_schedule_32_2:
        print(f"Time {time}: Task {task_name} on Core {core_id}")
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(time_list_16_1, QoS_list_16_1, label='QoS', marker='o', color='c')
    plt.xlabel('Time')
    plt.ylabel('QoS')
    plt.title('QoS of 16 core system with utilization=0.25')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(time_list_16_2, QoS_list_16_2, label='QoS', marker='o', color='m')
    plt.xlabel('Time')
    plt.ylabel('QoS')
    plt.title('QoS of 16 core system with utilization=0.5')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(time_list_32_1, QoS_list_32_1, label='QoS', marker='o', color='g')
    plt.xlabel('Time')
    plt.ylabel('QoS')
    plt.title('QoS of 32 core system with utilization=0.3')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(time_list_32_2, QoS_list_32_2, label='QoS', marker='o', color='b')
    plt.xlabel('Time')
    plt.ylabel('QoS')
    plt.title('QoS of 32 core system with utilization=0.7')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    Qos()
    schedulability()
