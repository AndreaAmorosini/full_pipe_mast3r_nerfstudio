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
                            
                    # if platform.system() == "Linux":
                    #     self.logger.info("Aggiunta dei compilatori Conda (gxx_linux-64)")
                    #     conda_deps.append("gxx_linux-64")
                    #     conda_deps.append("gcc_linux-64")
                    #     conda_deps.append("libxcrypt")

                    # FASE 1: Crea ambiente e installa pacchetti CONDA
                    self.logger.info(
                        f"FASE 1: Creazione ambiente e installazione pacchetti Conda: {conda_deps}"
                    )
                    channels_str = " ".join([f"-c {c}" for c in channels])
                    deps_str = " ".join(f'"{d}"' for d in conda_deps)
                    cmd = (
                        f"conda create --prefix {env_path} {channels_str} {deps_str} -y"
                    )
                    run_command(cmd, self.logger.name, verbose, shell=True)

                    # --- NUOVA LOGICA: Installazione separata dei compilatori ---
                    if platform.system() == "Linux":
                        self.logger.info(
                            "FASE 1.5: Installazione separata dei compilatori Conda (gxx_linux-64, gcc_linux-64)..."
                        )
                        # compilers = ["gxx_linux-64", "gcc_linux-64", "libxcrypt"]
                        compilers = ["compilers", "libxcrypt"]
                        compilers_str = " ".join(f'"{c}"' for c in compilers)
                        cmd_compilers = f"conda install --prefix {env_path} {channels_str} {compilers_str} -y"
                        try:
                            run_command(
                                cmd_compilers, self.logger.name, verbose, shell=True
                            )
                            gxx_path = env_path / "bin" / "g++"
                            if not gxx_path.exists():
                                self.logger.warning(f"Compilatore g++ non trovato in {gxx_path} dopo l'installazione.")
                                self.logger.warning("La build potrebbe fallire. Potrebbe essere un problema di sincronizzazione con Conda.")
                            else:
                                self.logger.info(f"Compilatore g++ verificato con successo in: {gxx_path}")
                        except Exception as e:
                            self.logger.warning(
                                f"Installazione dei compilatori fallita: {e}. La build potrebbe non riuscire."
                            )
                    # --- FINE NUOVA LOGICA ---

                    # --- NUOVA LOGICA: assicurarsi che nvcc sia disponibile ---
                    # Determina versione CUDA richiesta dalle dipendenze (se presente)
                    # desired_cuda = "11.6"
                    desired_cuda = self.install_config.get("CUDA_VERSION")
                    for d in conda_deps:
                        if "cudatoolkit" in str(d):
                            parts = str(d).split("=")
                            if len(parts) > 1 and parts[1].strip():
                                desired_cuda = parts[1].split()[0].strip()
                                break

                    nvcc_path = env_path / "bin" / "nvcc"
                    if not nvcc_path.exists():
                        self.logger.info(
                            f"nvcc non trovato in {nvcc_path}. Provo ad installare 'cudatoolkit-dev={desired_cuda}' nel nuovo ambiente..."
                        )
                        # Usa i canali già dichiarati (channels_str) per installare cudatoolkit-dev
                        try:
                            cmd_ctd = f"conda install --prefix {env_path} {channels_str} -y cudatoolkit-dev={desired_cuda}"
                            run_command(cmd_ctd, self.logger.name, verbose, shell=True)
                        except Exception as e:
                            self.logger.warning(
                                f"Installazione automatica di cudatoolkit-dev fallita: {e}"
                            )

                    # Verifica finale nvcc
                    if not nvcc_path.exists():
                        self.logger.warning(
                            "nvcc ancora non trovato nell'ambiente. Le build CUDA richiedono un nvcc compatibile.\n"
                            "Opzioni:\n"
                            " - Installare il pacchetto conda 'cudatoolkit-dev' nella stessa env (es. conda install --prefix <env> -c conda-forge cudatoolkit-dev=11.6)\n"
                            " - Oppure installare system-wide CUDA Toolkit 11.6 e assicurarsi che il suo bin sia in PATH quando si costruisce.\n"
                            "Nel frattempo l'install fallirà finché nvcc non è disponibile."
                        )
                    else:
                        self.logger.info(f"nvcc trovato: {nvcc_path}")
                    # --- fine nuova logica ---

                    # FASE 2: Installa pacchetti PIP nell'ambiente appena creato
                    if pip_deps:
                        self.logger.info(
                            f"FASE 2: Installazione pacchetti Pip: {pip_deps}"
                        )
                        # I percorsi nei file .yml sono spesso relativi, quindi impostiamo la CWD
                        cwd = env_file_path.parent
                        
                        # --- NUOVA LOGICA PER FORZARE LE VARIABILI D'AMBIENTE ---
                        # Costruisce le variabili d'ambiente per isolare CUDA
                        env_vars = os.environ.copy()
                        env_bin_path = env_path / "bin"
                        original_path = env_vars.get("PATH", "")
                        
                        # Prepend del path dell'ambiente a PATH
                        env_vars["PATH"] = f"{env_bin_path}{os.pathsep}{original_path}"
                        # Imposta CUDA_HOME in modo esplicito
                        env_vars["CUDA_HOME"] = str(env_path)
                        
                        if platform.system() == "Linux":
                            env_vars["CC"] = str(env_bin_path / "gcc")
                            env_vars["CXX"] = str(env_bin_path / "g++")
                        
                        self.logger.debug(f"Variabile CUDA_HOME forzata a: {env_vars['CUDA_HOME']}")
                        self.logger.debug(f"Variabile CC forzata a: {env_vars.get('CC')}")
                        self.logger.debug(f"Nuova Variabile CXX forzata a: {env_vars.get('CXX')}")
                        self.logger.debug(f"Nuova variabile PATH: {env_vars['PATH']}")
                        # --- FINE NUOVA LOGICA ---
                        
                        for pkg in pip_deps:
                            # Renderizza eventuali template nel path del pacchetto pip
                            pkg_path = self._render_template(pkg, template_vars)
                            self.logger.info(f"Installazione pip: {pkg_path}")
                            # cmd = f'conda run --prefix {env_path} python -m pip install "{pkg_path}"'
                            
                            python_executable = env_path / "bin" / "python"
                            cmd = f'"{python_executable}" -u -m pip install -v "{pkg_path}"'
                            
                            run_command(
                                cmd, self.logger.name, verbose, shell=True, cwd=cwd, env=env_vars
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
                    packages = " ".join(self.install_config.get("conda_packages", []))
                    if packages:
                        cmd = (
                            f"conda create --prefix {env_path} {channels} {packages} -y"
                        )
                        run_command(cmd, self.logger.name, verbose, shell=True)

                    for pkg_template in self.install_config.get("pip_packages", []):
                        pkg = self._render_template(pkg_template, template_vars)
                        self.logger.info(f"Installazione pacchetto Pip: {pkg}")
                        cmd = f'conda run --prefix {env_path} pip install "{pkg}"'
                        run_command(cmd, self.logger.name, verbose, shell=True)

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
                    cmd = f'python -u -m pip install "{pkg}"'
                    run_command(cmd, self.logger.name, verbose, shell=True)

            # Esegui comandi di build finali, se presenti
            for cmd_template in self.install_config.get("build_commands", []):
                cmd = self._render_template(cmd_template, template_vars)
                self.logger.info(f"Esecuzione comando di build: {cmd}")
                # Usa conda run se l'ambiente è dedicato
                if env_path:
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
