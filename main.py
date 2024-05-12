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

# llf for Aperiodic independent task on uniprocessor:
class Task:
    def __init__(self, id, release_time, execution_time, deadline):
        self.id = id
        self.release_time = release_time
        self.execution_time = execution_time
        self.deadline = deadline
        self.remaining_execution_time = execution_time

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

# Example usage:
tasks = [
    Task(id=1, release_time=0, execution_time=10, deadline=33),
    Task(id=2, release_time=4, execution_time=3, deadline=28),
    Task(id=3, release_time=5, execution_time=10, deadline=29)
]

least_laxity_first(tasks)

