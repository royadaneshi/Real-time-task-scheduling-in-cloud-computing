import random


def uunifast(num_tasks, total_utilization):
    utilizations = []
    sumU = total_utilization
    for i in range(1, num_tasks):
        nextSumU = sumU * (random.random() ** (1.0 / (num_tasks - i)))
        utilizations.append(sumU - nextSumU)
        sumU = nextSumU
    utilizations.append(sumU)

    tasks = []
    for i, u in enumerate(utilizations):
        period = random.randint(10, 100)
        execution_time = u * period
        deadline = period
        priority = num_tasks - i
        tasks.append({
            'task_id': i + 1,
            'execution_time': execution_time,
            'deadline': deadline,
            'period': period,
            'utilization': u,
            'priority': priority
        })

    return tasks


# this function returns task set for each total utilization in dictionary
def get_task_by_utilization():
    num_task = 50
    utility_task = {}
    total_utilization = [0.25, 0.5, 0.3, 0.7]
    for u in total_utilization:
        tasks = uunifast(num_task, u)
        utility_task[str(u)] = tasks
    return utility_task


if __name__ == '__main__':

    u_tasks = get_task_by_utilization()
    # example of usage:
    tasks = u_tasks["0.25"]
    for task in tasks:
        print(
            f"Task {task['task_id']}: Execution Time = {task['execution_time']:.2f}, Deadline = {task['deadline']}, Period = {task['period']}, Utilization = {task['utilization']:.2f}, Priority = {task['priority']}")

# llf for periodic independent tasks on multicore system:
class Task:
    def __init__(self, id, release_time, execution_time, period, deadline):
        self.id = id
        self.release_time = release_time
        self.execution_time = execution_time
        self.period = period
        self.deadline = deadline
        self.remaining_execution_time = execution_time
        self.next_release_time = release_time + period

class Core:
    def __init__(self, id):
        self.id = id
        self.ready_queue = []
        self.is_idle = True

def least_laxity_first_periodic(tasks, num_cores, efficiency):
    # Adjust execution time based on number of cores and efficiency
    #for task in tasks:
     #   task.execution_time = task.execution_time / (num_cores * efficiency)

    # Create cores
    cores = [Core(id) for id in range(num_cores)]

    # Calculate the hyperperiod
    hyperperiod = tasks[0].period
    for task in tasks[1:]:
        hyperperiod = lcm(hyperperiod, task.period)

    current_time = 0
    while current_time < hyperperiod:
        # Check for active tasks and assign to idle cores
        active_tasks = [task for task in tasks if current_time >= task.release_time and task.remaining_execution_time > 0]
        idle_cores = [core for core in cores if core.is_idle]

        if active_tasks and len(active_tasks) <= len(idle_cores):
            # Assign each active task to an idle core
            for task, core in zip(active_tasks, idle_cores):
                core.ready_queue.append(task)
                core.is_idle = False
                print(f"Task {task.id} added to Core {core.id} ready queue at time {current_time}")
        elif active_tasks and len(active_tasks) > len(idle_cores):
            # Sort active tasks by least laxity first
            active_tasks.sort(key=lambda x: x.deadline - current_time - x.remaining_execution_time)
            # Assign tasks to idle cores using LLF
            for task in active_tasks[:len(idle_cores)]:
                core = idle_cores.pop(0)
                core.ready_queue.append(task)
                core.is_idle = False
                print(f"Task {task.id} added to Core {core.id} ready queue at time {current_time}")

        # Execute tasks from each core's ready queue based on least laxity first
        for core in cores:
            if core.ready_queue:
                min_laxity_task = min(core.ready_queue, key=lambda x: x.deadline - current_time - x.remaining_execution_time)
                core.ready_queue.remove(min_laxity_task)
                print(f"Executing Task {min_laxity_task.id} on Core {core.id} at time {current_time}")
                min_laxity_task.remaining_execution_time -= 1
                #added
                core.is_idle = True
                # Check if task is completed and update release time and deadline
                if min_laxity_task.remaining_execution_time == 0:
                    min_laxity_task.release_time = min_laxity_task.next_release_time
                    min_laxity_task.next_release_time += min_laxity_task.period
                    min_laxity_task.deadline += min_laxity_task.period
                    min_laxity_task.remaining_execution_time = min_laxity_task.execution_time
                    core.is_idle = True

        # Move to the next time step
        current_time += 1
        
# Function to calculate the least common multiple
def lcm(x, y):
    from math import gcd
    return x * y // gcd(x, y)

# Example:
tasks = [
    Task(id=1, release_time=0, execution_time=2, period=5, deadline=5),
    Task(id=2, release_time=0, execution_time=1, period=3, deadline=2),
    Task(id=3, release_time=0, execution_time=3, period=6, deadline=4)
]

num_cores = 16
efficiency = 0.25

least_laxity_first_periodic(tasks, num_cores, efficiency)
