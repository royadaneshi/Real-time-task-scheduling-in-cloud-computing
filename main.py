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
        self.predecessors = predecessors
        Task.tasks[id] = self

    def get_task(id):
        return Task.tasks[id]


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

        tasks.append(Task(i, int(release_time), int(execution_time), int(deadline), util, 0, []))

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
            predecessors = [Task.get_task(pre) for pre in predecessors1]
            # print(predecessors)
            tasks[i].predecessors = predecessors

        else:
            tasks[i].predecessors = []

    return tasks


def least_laxity_first(tasks):
    current_time = 0
    while True:
        # Calculate laxity for each task
        laxities = {}
        for task in tasks:
            if task.release_time <= current_time and task.remaining_execution_time > 0:
                laxity = task.deadline - current_time - task.remaining_execution_time
                laxities[task] = laxity

        if not laxities:
            break  # No tasks available

        # Find the task with the least laxity
        min_laxity_task = min(laxities, key=laxities.get)

        # Execute the task for one time step
        print(f"Executing Task {min_laxity_task.id} at time {current_time}")
        min_laxity_task.remaining_execution_time -= 1

        # Move to the next time step
        current_time += 1


if __name__ == '__main__':
    num_tasks = 50
    total_utilization = 0.25
    # gives a list of task objects
    tasks = generate_non_periodic_tasks(num_tasks, total_utilization)
    least_laxity_first(tasks)
    # print(tasks)
    # for task in tasks:
    #     print(task.predecessors)

# # Example usage:
# tasks = [
#     Task(id=1, release_time=0, execution_time=10, deadline=33),
#     Task(id=2, release_time=4, execution_time=3, deadline=28),
#     Task(id=3, release_time=5, execution_time=10, deadline=29)
# ]
#
# least_laxity_first(tasks)
