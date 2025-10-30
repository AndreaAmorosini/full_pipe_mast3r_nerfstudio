import toml
import logging
import jinja2
import yaml
import os
import platform
import tempfile
from pathlib import Path
from .utils import run_command
# L'import viene fatto localmente per evitare dipendenze circolari
# from .uninstaller import MethodUninstaller


class MethodInstaller:
    """Installa metodi creando ambienti Conda locali in ./.envs/"""

    def __init__(self, method_config_path: Path):
        self.config_path = method_config_path.resolve()
        self.config = toml.load(method_config_path)
        self.name = self.config["name"]
        self.logger = logging.getLogger(f"Installer.{self.name}")
        self.install_config = self.config.get("installation", {})

        # La root del progetto è la directory genitore di 'full_pipe_v2'
        self.project_root = self.config_path.parent.parent.parent.parent
        self.pipe_root = self.project_root / "full_pipe_v2"
        self.envs_dir = self.pipe_root / ".envs"
        self.envs_dir.mkdir(exist_ok=True)
        self.jinja_env = jinja2.Environment(loader=jinja2.BaseLoader())

    def _get_env_path(self) -> Path | None:
        """Restituisce il percorso assoluto dell'ambiente locale."""
        env_name = self.install_config.get("conda_env_name")
        if env_name:
            return (self.envs_dir / env_name).resolve()
        return None

    def _render_template(self, template_str: str, vars: dict) -> str:
        template = self.jinja_env.from_string(template_str)
        return template.render(vars)

    def _cleanup(self):
        """
        Esegue la pulizia completa del metodo in caso di installazione fallita.
        Riutilizza la logica di MethodUninstaller.
        """
        self.logger.warning(
            f"Installazione di '{self.name}' fallita o interrotta. Avvio pulizia..."
        )
        try:
            # Import locale per evitare dipendenze circolari
            from .uninstaller import MethodUninstaller

            uninstaller = MethodUninstaller(self.config_path)
            uninstaller.uninstall()
            self.logger.info(f"Pulizia per '{self.name}' completata.")
        except Exception as e:
            self.logger.error(f"Pulizia per '{self.name}' fallita: {e}")

    def install(self, verbose=False):
        self.logger.info(f"Inizio installazione di '{self.name}'...")
        env_path = self._get_env_path()
        method_vendor_dir = self.pipe_root / "vendor" / self.name

        # Controlla se l'ambiente esiste già. Se sì, consideriamo il metodo installato.
        if env_path and env_path.exists():
            self.logger.info(
                f"Ambiente Conda '{env_path.name}' esiste già. Considero il metodo installato."
            )
            return

        try:
            template_vars = {
                "env_path": str(env_path) if env_path else "",
                "method_vendor_dir": str(method_vendor_dir),
            }

            # 1. Git Repos (SEMPRE PRIMA)
            for repo in self.install_config.get("git_repos", []):
                method_vendor_dir.mkdir(parents=True, exist_ok=True)
                path = method_vendor_dir / repo["path"]
                if not path.exists():
                    self.logger.info(f"Clonazione repository da {repo['url']}...")
                    recursive_flag = (
                        "--recursive" if repo.get("recursive", False) else ""
                    )
                    cmd = f"git clone --branch {repo['branch']} {recursive_flag} {repo['url']} {path}"
                    run_command(cmd, self.logger.name, verbose, shell=True)

            if env_path:
                # --- CASO 1: AMBIENTE DEDICATO ---
                #TODO: da cancellare caso di environment.yml
                env_file_template = self.install_config.get("conda_env_file")

                if env_file_template:
                    # Sottocaso A: Installazione da file environment.yml in 2 FASI
                    env_file_path_str = self._render_template(
                        env_file_template, template_vars
                    )
                    env_file_path = Path(env_file_path_str)

                    if not env_file_path.is_file():
                        raise FileNotFoundError(
                            f"Il file di ambiente specificato non è stato trovato: {env_file_path}"
                        )

                    self.logger.info(f"Lettura del file di ambiente: {env_file_path}")
                    with open(env_file_path, "r") as f:
                        env_data = yaml.safe_load(f)

                    # Separa dipendenze Conda e Pip
                    conda_deps = []
                    pip_deps = []
                    channels = env_data.get("channels", [])
                    for dep in env_data.get("dependencies", []):
                        if isinstance(dep, dict) and "pip" in dep:
                            pip_deps.extend(dep["pip"])
                        else:
                            conda_deps.append(dep)

                    # --- INIZIO MODIFICA: Logica di installazione compilatore ---
                    if platform.system() == "Linux":
                        self.logger.info(
                            "Aggiunta dei compilatori nativi (c-compiler, cxx-compiler) e cudatoolkit-dev."
                        )

                        cuda_version_str = self.install_config.get(
                            "CUDA_VERSION", "11.8"
                        )

                        # Mappa delle versioni CUDA -> GCC
                        CUDA_TO_COMPILER_VERSION = {
                            "11.6": "9.*",  # Per PyTorch 1.12/1.13
                            "11.7": "11.*",
                            "11.8": "11.*",  # Per PyTorch 2.x
                            "12.0": "12.*",
                        }

                        compiler_version = CUDA_TO_COMPILER_VERSION.get(
                            str(cuda_version_str)
                        )

                        if compiler_version:
                            self.logger.info(
                                f"Blocco compilatori alla versione {compiler_version} per CUDA {cuda_version_str}."
                            )
                            # Usa i meta-pacchetti 'c-compiler' e 'cxx-compiler' di conda-forge
                            conda_deps.append(f"c-compiler *_{compiler_version}")
                            conda_deps.append(f"cxx-compiler *_{compiler_version}")
                        else:
                            self.logger.warning(
                                f"Nessuna versione GCC mappata per CUDA_VERSION='{cuda_version_str}'. Uso meta-pacchetti generici."
                            )
                            conda_deps.append("c-compiler")
                            conda_deps.append("cxx-compiler")

                        conda_deps.append(f"cudatoolkit-dev={cuda_version_str}")

                        if "libxcrypt" not in conda_deps:
                            conda_deps.append("libxcrypt")

                        if "conda-forge" not in channels:
                            channels.insert(0, "conda-forge")
                    # --- FINE MODIFICA ---

                    # FASE 1: Crea ambiente e installa TUTTI i pacchetti CONDA
                    self.logger.info(
                        f"FASE 1: Creazione ambiente e installazione pacchetti Conda: {conda_deps}"
                    )
                    channels_str = " ".join([f"-c {c}" for c in channels])
                    deps_str = " ".join(f'"{d}"' for d in conda_deps)
                    cmd = (
                        f"conda create --prefix {env_path} {channels_str} {deps_str} -y"
                    )
                    run_command(cmd, self.logger.name, verbose, shell=True)

                    # FASE 2: Installa pacchetti PIP nell'ambiente appena creato
                    if pip_deps:
                        self.logger.info(
                            f"FASE 2: Installazione pacchetti Pip: {pip_deps}"
                        )
                        cwd = env_file_path.parent

                        # --- INIZIO MODIFICA: Isolamento ambiente per pip ---
                        env_vars = os.environ.copy()
                        env_bin_path = env_path / "bin"
                        env_lib_path = env_path / "lib"
                        env_include_path = env_path / "include"

                        original_path = env_vars.get("PATH", "")
                        env_vars["PATH"] = f"{env_bin_path}{os.pathsep}{original_path}"

                        env_vars["CUDA_HOME"] = str(env_path)
                        env_vars["TORCH_CUDA_ARCH_LIST"] = "7.5 8.6 8.9"

                        original_include = env_vars.get("CPLUS_INCLUDE_PATH", "")
                        env_vars["CPLUS_INCLUDE_PATH"] = (
                            f"{env_include_path}{os.pathsep}{original_include}"
                        )

                        original_lib = env_vars.get("LIBRARY_PATH", "")
                        env_vars["LIBRARY_PATH"] = (
                            f"{env_lib_path}{os.pathsep}{original_lib}"
                        )

                        original_ld_lib = env_vars.get("LD_LIBRARY_PATH", "")
                        env_vars["LD_LIBRARY_PATH"] = (
                            f"{env_lib_path}{os.pathsep}{original_ld_lib}"
                        )

                        # --- MODIFICA CHIAVE 1: Azzeramento PYTHONPATH ---
                        env_vars["PYTHONPATH"] = ""

                        self.logger.debug(
                            f"Variabile CUDA_HOME forzata a: {env_vars['CUDA_HOME']}"
                        )
                        self.logger.debug(
                            f"Variabile CPLUS_INCLUDE_PATH forzata a: {env_vars.get('CPLUS_INCLUDE_PATH')}"
                        )
                        self.logger.debug(
                            f"Variabile LIBRARY_PATH forzata a: {env_vars.get('LIBRARY_PATH')}"
                        )
                        self.logger.debug(f"Variabile PYTHONPATH azzerata.")
                        self.logger.debug(f"Nuova variabile PATH: {env_vars['PATH']}")
                        # --- FINE MODIFICA ---

                        for pkg in pip_deps:
                            pkg_path = self._render_template(pkg, template_vars)
                            self.logger.info(f"Installazione pip: {pkg_path}")

                            python_executable = env_path / "bin" / "python"

                            # --- MODIFICA CHIAVE 2: Aggiunta flag -s ---
                            # -s = Non aggiungere il site-packages dell'utente a sys.path
                            cmd = f'"{python_executable}" -s -u -m pip install -v "{pkg_path}"'

                            run_command(
                                cmd,
                                self.logger.name,
                                verbose,
                                shell=True,
                                cwd=cwd,
                                env=env_vars,
                            )
                else:
                    # Sottocaso B: Installazione da liste nel .toml (per metodi semplici)
                    self.logger.info(
                        f"Creazione ambiente Conda da .toml: {env_path.name}"
                    )
                    channels = " ".join(
                        [
                            f"-c {c}"
                            for c in self.install_config.get("conda_channels", [])
                        ]
                    )

                    packages_list = self.install_config.get("conda_packages", [])
                    if platform.system() == "Linux":
                        self.logger.info(
                            "Aggiunta dei compilatori nativi (c-compiler, cxx-compiler)."
                        )
                        packages_list.extend(["c-compiler", "cxx-compiler"])
                        if "conda-forge" not in channels:
                            channels = f"-c conda-forge {channels}"
                    packages = " ".join(f'"{p}"' for p in packages_list)

                    if packages:
                        cmd = (
                            f"conda create --prefix {env_path} {channels} {packages} -y"
                        )
                        run_command(cmd, self.logger.name, verbose, shell=True)

                for pkg_template in self.install_config.get("pip_packages", []):
                    pkg_full_string = self._render_template(pkg_template, template_vars)
                    self.logger.info(f"Installazione pacchetto Pip: {pkg_full_string}")

                    # --- INIZIO MODIFICA: Parsing dei flag pip ---
                    import shlex
                    parts = shlex.split(pkg_full_string)
                    
                    packages_to_install = []
                    pip_flags = []

                    for part in parts:
                        if part.startswith('--'):
                            # Se la parte è un flag (es. --index-url), la aggiungiamo ai flag
                            # e assumiamo che la parte successiva sia il suo valore
                            pip_flags.append(part)
                        elif pip_flags and pip_flags[-1].startswith('--'):
                            # Se l'elemento precedente era un flag, questo è il suo valore
                            pip_flags.append(part)
                        else:
                            # Altrimenti, è un nome di pacchetto
                            packages_to_install.append(part)
                    
                    python_executable = env_path / "bin" / "python"
                    
                    # Ricostruisci il comando correttamente
                    cmd_list = [
                        str(python_executable),
                        "-s","-u", "-m", "pip", "install", "-v"
                    ]
                    cmd_list.extend(packages_to_install)
                    cmd_list.extend(pip_flags)
                    
                    # Converti la lista in una stringa per run_command con shell=True
                    cmd = " ".join(f'"{part}"' if " " in part else part for part in cmd_list)
                    # --- FINE MODIFICA ---

                    env_vars = os.environ.copy()
                    env_vars["PATH"] = (
                        f"{env_path / 'bin'}{os.pathsep}{env_vars.get('PATH', '')}"
                    )                        
                    # env_vars["CUDA_HOME"] = str(env_path)
                    # env_vars["LD_LIBRARY_PATH"] = (
                    #     f"{env_path / 'lib'}{os.pathsep}{env_vars.get('LD_LIBRARY_PATH', '')}"
                    # )
                    # --- MODIFICA CHIAVE 1: Azzeramento PYTHONPATH ---
                    env_vars["PYTHONPATH"] = ""

                    run_command(
                        cmd, self.logger.name, verbose, shell=True, env=env_vars
                    )

            else:
                # --- CASO 2: AMBIENTE ATTIVO (BASE) ---
                self.logger.info(
                    f"Installazione di '{self.name}' nell'ambiente Conda attivo..."
                )
                conda_packages = self.install_config.get("conda_packages", [])
                if conda_packages:
                    self.logger.info(f"Installazione pacchetti Conda: {conda_packages}")
                    packages_str = " ".join(f'"{p}"' for p in conda_packages)
                    cmd = f"conda install {packages_str} -y"
                    run_command(cmd, self.logger.name, verbose, shell=True)

                pip_packages = self.install_config.get("pip_packages", [])
                for pkg_template in pip_packages:
                    pkg = self._render_template(pkg_template, {})
                    self.logger.info(f"Installazione pacchetto Pip: {pkg}")
                    # --- MODIFICA CHIAVE 2: Aggiunta --ignore-installed ---
                    # Anche qui, -s per sicurezza
                    cmd = f'python -s -u -m pip install "{pkg}"'
                    run_command(cmd, self.logger.name, verbose, shell=True)

            # Esegui comandi di build finali, se presenti
            for cmd_template in self.install_config.get("build_commands", []):
                cmd = self._render_template(cmd_template, template_vars)
                self.logger.info(f"Esecuzione comando di build: {cmd}")
                if env_path:
                    # 'conda run' è ok qui perché le estensioni sono già compilate.
                    run_cmd = f"conda run --prefix {env_path} {cmd}"
                    run_command(
                        run_cmd,
                        self.logger.name,
                        verbose,
                        shell=True,
                        cwd=self.pipe_root,
                    )
                else:
                    run_command(
                        cmd, self.logger.name, verbose, shell=True, cwd=self.pipe_root
                    )

            self.logger.info(f"Installazione di '{self.name}' completata con successo.")

        except (Exception, KeyboardInterrupt) as e:
            self.logger.error(f"Errore durante l'installazione di '{self.name}': {e}")
            self._cleanup()
            raise
