import docker
from typing import Tuple
from config import CONFIG

class DockerSandbox:
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.client = docker.from_env()
        self.container = None
        self.image_name = self._resolve_image_name(instance_id)
        
        print("=" * 100)
        print(f"Docker client: {self.client}")
        print(f"Docker client containers: {self.client.containers}")
        print(f"Docker image: {self.image_name}")
        print("=" * 100)
        print("\n")

    def _resolve_image_name(self, instance_id: str) -> str:
        # Standard SWE-bench docker image naming convention
        sanitized_id = instance_id.replace("__", "_1776_")
        return f"docker.io/swebench/sweb.eval.x86_64.{sanitized_id}:latest".lower()
    
    def _ensure_image_exists(self):
        """Ensures the image is locally cached before starting the container."""
        try:
            self.client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            print(f"[{self.instance_id}] Image not found locally. Pulling {self.image_name}...")
            try:
                self.client.images.pull(self.image_name)
                print("Image pulled seccesfully")
            except:
                raise RuntimeError("Unable to pull image")

    def start(self):
        print("=" * 100)
        print(f"[{self.instance_id}] Starting isolated container: {self.image_name}")
        self._ensure_image_exists()

        container_kwargs = {
            "image": self.image_name,
            "command": "/bin/bash",
            "working_dir": "/testbed",  # Start directly inside the repository
            "detach": True,
            "tty": True,
            "mem_limit": CONFIG.MAX_DOCKER_MEMORY,
            "memswap_limit": CONFIG.MAX_DOCKER_SWAP,
            "pids_limit": 256,
            "network_mode": "none",
            "labels": {"owner": "swebench_harness", "instance_id": self.instance_id},
            "remove": False,
        }

        try:
            self.container = self.client.containers.run(**container_kwargs)
            print(f"Succesfully started container:\n    CONTAINER ID: {self.container.short_id}\n    CONTAINER NAME: {self.container.name}\n    CONTAINER STATUS: {self.container.status}")
        except docker.errors.DockerException as err:
            raise RuntimeError(f"[{self.instance_id}] Failed to start container: {err}") from err
        print("=" * 100)
        print("\n\n")
    
    def exec_cmd(self, cmd: str) -> Tuple[int, str]:
        if not self.container:
            raise RuntimeError("Sandbox container is not running.")
        
        # Pass as a list so Docker invokes execve directly with zero quote mangling
        wrapped_script = f"source /opt/miniconda3/bin/activate testbed && cd /testbed && {cmd}"
        exec_res = self.container.exec_run(
            cmd=["/bin/bash", "-c", wrapped_script],
            workdir="/testbed",
            stdout=True,
            stderr=True,
            demux=False,
        )
        output = exec_res.output.decode("utf-8", errors="replace") if exec_res.output else ""
        return exec_res.exit_code, output

    def get_git_diff(self) -> str:
        # Ensure we capture diff inside /testbed cleanly
        exec_res = self.container.exec_run(
            cmd=["git", "diff"],
            workdir="/testbed",
            stdout=True,
            stderr=True,
            demux=False,
        )
        diff = exec_res.output.decode("utf-8", errors="replace") if exec_res.output else ""
        return diff if exec_res.exit_code == 0 else ""

    def cleanup(self):
        """Idempotent teardown without 404 race conditions."""
        if self.container:
            try:
                # Force kill and remove in one clean call
                print("=" * 100)
                print(f"Removing container {self.container.name} [{self.container.short_id}]")
                self.container.remove(force=True)
            except docker.errors.NotFound:
                pass
            except Exception as e:
                print(f"[{self.instance_id}] Warning during container cleanup: {e}")
            finally:
                print("Succesfully removed container")
                self.container = None
            print("=" * 100)
            print("\n\n")