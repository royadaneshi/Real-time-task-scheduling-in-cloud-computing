# rtc project without charts:
import random


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

        tasks.append(Task(i+1, int(release_time), int(execution_time), int(deadline), util, 0, []))

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
            #predecessors = [Task.get_task(pre+1) for pre in predecessors1]
            # print(predecessors)
            #tasks[i].predecessors = predecessors
            # set successors
            #for pred in tasks[i].predecessors:
             #   pred.successors.append(tasks[i])
            for pred_index in predecessors1:
                predecessor_task = tasks[pred_index]
                tasks[i].predecessors.append(predecessor_task)
                predecessor_task.add_successor(tasks[i])
        #else:
         #   tasks[i].predecessors = []

    return tasks

# Mobility function to map tasks to cores
def calculate_mobility(tasks):
    for task in tasks:
        task.mobility = len(task.successors)
    tasks.sort(key=lambda t: t.mobility)

# Assign tasks to cores based on mobility (assuming 16 cores for this example)
def assign_tasks_to_cores(tasks, num_cores=16):
    calculate_mobility(tasks)
    cores = [[] for _ in range(num_cores)]
    for i, task in enumerate(tasks):
        core_index = i % num_cores
        task.core = core_index
        cores[core_index].append(task)
    return cores

# Schedule tasks on each core using LLF
def least_laxity_first(cores):
    current_time = 0
    while True:
        available_tasks = []
        for core in cores:
            for task in core:
                if task.release_time <= current_time and task.remaining_execution_time > 0:
                    if all(pred.remaining_execution_time == 0 for pred in task.predecessors):
                        available_tasks.append(task)

        if not available_tasks:
            break  # No tasks available

        # Calculate laxity for each task
        laxities = {task: task.deadline - current_time - task.remaining_execution_time for task in available_tasks}

        # Find the task with the least laxity
        min_laxity_task = min(laxities, key=laxities.get)

        # Execute the task for one time step
        print(f"Executing Task {min_laxity_task.id} on Core {min_laxity_task.core} at time {current_time}")
        min_laxity_task.remaining_execution_time -= 1

        # Move to the next time step
        current_time += 1

if __name__ == '__main__':
    num_tasks = 50
    total_utilization = 0.25
    # Generate a list of task objects
    tasks = generate_non_periodic_tasks(num_tasks, total_utilization)
    # Assign tasks to cores
    cores = assign_tasks_to_cores(tasks)
    # Schedule tasks on cores
    least_laxity_first(cores)
