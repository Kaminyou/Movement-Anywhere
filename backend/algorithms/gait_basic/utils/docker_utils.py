import typing as t


def run_container(
    client,
    image: str,
    command: str,
    working_dir: str,
    volumes: t.Optional[t.Union[list, dict]] = None,
    device_requests: t.Optional[list] = None,
):

    container = client.containers.run(
        image,
        command=command,
        working_dir=working_dir,
        volumes=volumes,
        device_requests=device_requests,
        auto_remove=True,
        detach=True,
        shm_size='512g',
    )

    for log in container.logs(stream=True):
        print(log.decode('utf-8'))
