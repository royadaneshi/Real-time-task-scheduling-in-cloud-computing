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
