from specmodel.tasks import sql, story

TASKS = {"story": story, "sql": sql}


def get_task(name):
    return TASKS[name]
