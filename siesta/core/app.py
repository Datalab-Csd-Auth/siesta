import importlib
import pkgutil
from typing import List, Type, Dict
from siesta.core.interfaces import SiestaModule, StorageManager
import siesta.modules as modules
import siesta.core.sparkManager as SparkManager
from siesta.core.storageFactory import StorageManagerFactory, set_storage_manager
from siesta.core.config import initialize_config, load_config
import argparse
import logging

logger = logging.getLogger(__name__)


class Siesta:

    def __init__(self, config_path: str|None = None) -> None:
        self.config = load_config(config_path)
        self.storage_manager: StorageManager | None = None
        self.registered_routes: Dict[str, SiestaModule.ApiRoutes|None] = {}
        self.module_instances: Dict[str, SiestaModule] = {}

    @classmethod
    def with_args(cls, args:List[str]):
        """Alternative constructor to initialize Siesta with command-line arguments.
        
        Args:
            args: List of command-line arguments
            
        Returns:
            An instance of Siesta
        """
        logging.basicConfig(level=logging.INFO)
        parser = argparse.ArgumentParser(description="Siesta Framework Initialization")
        parser.add_argument('--config', type=str, help='Path to configuration JSON file', required=False)
        parser.add_argument('module', type=str, help='Module to run')
        
        parsed_args, unknown_args = parser.parse_known_args(args)
        
        app = cls(config_path=parsed_args.config if parsed_args.config else None)
        app.startup()

        # Reuse the already-initialized module instance
        if parsed_args.module in app.module_instances:
            mod = app.module_instances[parsed_args.module]
            mod.cli_run(unknown_args)
        else:
            logger.error(f"Module {parsed_args.module} not found")
        
        return app
    
    def discover_modules(self) -> List[Type[SiestaModule]]:
        discovered: set[Type[SiestaModule]] = set()

        for module_info in pkgutil.iter_modules(modules.__path__):
            module_name = f"siesta.modules.{module_info.name}.main"

            mod = importlib.import_module(module_name)

            for obj in vars(mod).values():
                if (
                    isinstance(obj, type)
                    and issubclass(obj, SiestaModule)
                    and obj is not SiestaModule
                ):
                    discovered.add(obj)

        return list(discovered)

    def startup(self) -> None:
        """
        Start the Siesta framework.
        """
        logger.info("Starting Framework")
        logging.basicConfig(level=logging.INFO)
        
        # Initialize global config accessor
        initialize_config(self.config)
        
        # Start Spark Manager
        SparkManager.startup(self.config)
        
        # Setup Storage Manager and set global accessor
        self.storage_manager = StorageManagerFactory.create_storage_manager(self.config)
        set_storage_manager(self.storage_manager)
        
        # Initialize Modules
        self.discovered_modules = self.discover_modules()

        logger.info(f"Discovered Modules: {[mod.__name__ for mod in self.discovered_modules]}")
        for mod_class in self.discovered_modules:
            mod_instance = mod_class()
            logger.info(f"Starting Module: {mod_instance.name} v{mod_instance.version}")
            mod_instance.startup()
            # Store instance for reuse in CLI mode
            self.module_instances[mod_instance.name] = mod_instance

        logger.info("Framework Started")
    
    def get_registered_routes(self) -> Dict[str, SiestaModule.ApiRoutes|None]:
        if not self.discovered_modules:
            raise RuntimeError("Modules not discovered. Call startup() first.")

        for mod_class in self.discovered_modules:
            mod_instance = mod_class()
            routes = mod_instance.register_routes()
            if routes:
                self.registered_routes[mod_class.__name__.lower()] = routes
        
        return self.registered_routes