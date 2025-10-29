import logging
from pathlib import Path


class PipelineContext:
    """
    Gestisce lo stato e il passaggio di dati tra gli step della pipeline.
    Si occupa anche di creare le directory di output in modo strutturato.
    """

    def __init__(self, initial_config: dict):
        self.data = initial_config
        self.output_dir = Path(self.data.get("output_dir", "outputs/default_run"))
        self.step_dirs = {}
        self._setup_dirs()
        self.logger = logging.getLogger("PipelineContext")

    def _setup_dirs(self):
        """Crea la directory di output principale."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(
            f"Directory di output principale: {self.output_dir.resolve()}"
        )

    def get(self, key: str, default=None):
        """Ottiene un valore dal contesto."""
        return self.data.get(key, default)

    def set(self, key: str, value):
        """Imposta un valore nel contesto."""
        self.logger.debug(f"Contesto aggiornato: {key} = {value}")
        self.data[key] = value

    def get_output_dir(self) -> Path:
        """Ritorna la directory di output radice."""
        return self.output_dir

    def get_step_output_dir(self, step_name: str) -> Path:
        """
        Crea e ritorna una sub-directory dedicata per uno step (es. /output/sfm/).
        Questo mantiene l'output pulito e organizzato.
        """
        # Pulisce il nome per evitare problemi di percorso
        safe_step_name = step_name.replace(":", "_").replace("/", "_")

        if safe_step_name not in self.step_dirs:
            step_dir = self.output_dir / safe_step_name
            step_dir.mkdir(parents=True, exist_ok=True)
            self.step_dirs[safe_step_name] = step_dir
            self.logger.debug(
                f"Creata directory per lo step '{safe_step_name}': {step_dir}"
            )
        return self.step_dirs[safe_step_name]
